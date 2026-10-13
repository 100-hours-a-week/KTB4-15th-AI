"""GPU(nvJPEG) 로 JPEG 를 decode·resize 하는 fast path.

GPU 가 Pillow 와 같은 결과를 낸다고 확인한 JPEG 만 여기서 처리하고, 나머지는 호출자가 기존 Pillow
load_image 로 처리한다. 반환 이미지는 load_image 와 같은 처리용 RGB PIL 이미지(긴 변
MAX_PROCESSING_IMAGE_SIDE 이하)이므로 이후 Person/Pose/밝기/rembg 는 경로를 구분하지 않는다.

torch/torchvision 은 GPU 를 켠 배포에서만 설치되므로 이 모듈 안에서만 지연 import 한다.
"""

import logging
import threading

from PIL import ExifTags, Image

from app.body_image_validation.image_io import processing_size
from app.config import body_image_validation as config

logger = logging.getLogger(__name__)

# nvJPEG 의 RGB 출력이 Pillow 의 RGB 변환과 같다고 확인한 JPEG mode. CMYK(YCCK 포함)는 decode 에
# 실패하므로 Pillow 로 보낸다. L 은 R=G=B 로 펼쳐지고 Pillow convert("RGB") 와 차이가 최대 1 이다.
GPU_JPEG_MODES = frozenset({"RGB", "L"})
_JPEG_EOI = b"\xff\xd9"
# ImageOps.exif_transpose 가 픽셀을 움직이는 Orientation 과 같은 변환. 텐서는 CHW 이다.
#   flip(-1): 좌우 반전, flip(-2): 상하 반전, transpose(-1, -2): 주대각선 기준 전치(H/W 교환)
_ORIENTATION_TRANSFORMS = {
    2: lambda t: t.flip(-1),  # FLIP_LEFT_RIGHT
    3: lambda t: t.flip(-1).flip(-2),  # ROTATE_180
    4: lambda t: t.flip(-2),  # FLIP_TOP_BOTTOM
    5: lambda t: t.transpose(-1, -2),  # TRANSPOSE
    6: lambda t: t.transpose(-1, -2).flip(-1),  # ROTATE_270 (시계 방향 90도)
    7: lambda t: t.transpose(-1, -2).flip(-1).flip(-2),  # TRANSVERSE
    8: lambda t: t.transpose(-1, -2).flip(-2),  # ROTATE_90 (반시계 방향 90도)
}


class GpuPreprocessingUnavailableError(RuntimeError):
    """torch/CUDA 가 없어 GPU 경로를 만들 수 없다. runtime 은 Pillow 만으로 동작한다."""


class GpuPreprocessingError(RuntimeError):
    """GPU decode·변환이 이 입력에서 실패했다. 같은 입력을 Pillow 로 다시 처리해도 된다."""


class GpuJpegDecoder:
    """JPEG bytes → nvJPEG decode → EXIF 회전 → resize(BICUBIC, antialias) → HxWx3 uint8."""

    def __init__(self) -> None:
        try:
            import torch
            from torchvision.io import ImageReadMode, decode_jpeg
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.v2 import functional as transforms
        except (ImportError, OSError) as error:  # OSError: CUDA 공유 라이브러리를 못 찾음
            raise GpuPreprocessingUnavailableError("torch/torchvision 을 불러오지 못했습니다.") from error
        if not torch.cuda.is_available():
            raise GpuPreprocessingUnavailableError("CUDA 를 사용할 수 없습니다.")
        try:
            self._device = torch.device("cuda")
            torch.zeros(1, device=self._device)  # CUDA context 를 요청 전에 만든다.
        except RuntimeError as error:
            raise GpuPreprocessingUnavailableError("CUDA 초기화에 실패했습니다.") from error
        self._torch = torch
        self._decode_jpeg = decode_jpeg
        self._rgb = ImageReadMode.RGB
        self._bicubic = InterpolationMode.BICUBIC
        self._resize = transforms.resize
        self._lock = threading.Lock()

    def decode(self, body: bytes, header_size: tuple[int, int], orientation: int):
        """처리용 크기의 HxWx3 uint8 numpy 배열. GPU 쪽 실패는 GpuPreprocessingError 로 바꾼다."""
        with self._lock:
            try:
                data = self._torch.frombuffer(bytearray(body), dtype=self._torch.uint8)
                # CUDA decode 는 apply_exif_orientation 을 반영하지 않는다(torchvision 0.29 확인).
                # 나중에 반영되더라도 두 번 돌지 않도록 끄고 아래에서 직접 적용한다.
                tensor = self._decode_jpeg(
                    data, mode=self._rgb, device=self._device, apply_exif_orientation=False
                )
                decoded_size = (int(tensor.shape[2]), int(tensor.shape[1]))
                if decoded_size != header_size:
                    raise GpuPreprocessingError(
                        f"decode 크기 {decoded_size} 가 헤더 크기 {header_size} 와 다릅니다."
                    )
                transform = _ORIENTATION_TRANSFORMS.get(orientation)
                if transform is not None:
                    tensor = transform(tensor)
                size = (int(tensor.shape[2]), int(tensor.shape[1]))
                width, height = processing_size(size, config.MAX_PROCESSING_IMAGE_SIDE)
                if (width, height) != size:
                    tensor = self._resize(
                        tensor, [height, width], interpolation=self._bicubic, antialias=True
                    )
                return tensor.permute(1, 2, 0).contiguous().cpu().numpy()
            except (NotImplementedError, RecursionError):
                raise  # RuntimeError 의 하위 클래스지만 입력 문제가 아니라 코드 문제다.
            except GpuPreprocessingError:
                raise
            except RuntimeError as error:
                # nvJPEG decode 실패, CUDA 오류, GPU 메모리 부족은 torch 에서 RuntimeError 로 온다.
                raise GpuPreprocessingError(str(error)) from error


class GpuImagePreprocessor:
    """GPU 로 처리할 수 있는 JPEG 만 GPU 로 처리한다. 처리하지 않으면 None 을 돌려준다.

    None 이면 호출자가 같은 source 로 기존 Pillow load_image 를 부른다. source 는 헤더만 읽은
    상태 그대로 두고 닫지 않는다. 돌려준 이미지는 source 와 다른 객체이며 호출자가 닫는다.
    """

    def __init__(self, decoder: GpuJpegDecoder) -> None:
        self._decoder = decoder

    @staticmethod
    def unsupported_reason(source: Image.Image, body: bytes) -> str | None:
        if source.format != "JPEG":
            return "format"
        if source.mode not in GPU_JPEG_MODES:
            return "mode"
        # 끝이 잘린 JPEG 를 nvJPEG 는 오류 없이 decode 하지만 Pillow 는 IMAGE_DECODE_FAILED 로
        # 거절한다. EOI 로 끝나지 않으면 기존 동작을 그대로 받도록 Pillow 로 보낸다.
        if not body.endswith(_JPEG_EOI):
            return "no-eoi"
        return None

    def try_load(self, source: Image.Image, body: bytes) -> Image.Image | None:
        reason = self.unsupported_reason(source, body)
        if reason is None:
            try:
                orientation = source.getexif().get(ExifTags.Base.Orientation, 1)
            except (OSError, ValueError, UserWarning):
                reason = "exif"  # load_image 가 같은 오류를 IMAGE_DECODE_FAILED 로 다룬다.
        if reason is not None:
            logger.debug(
                "body image GPU preprocessing skipped: reason=%s format=%s mode=%s",
                reason, source.format, source.mode,
            )
            return None
        try:
            array = self._decoder.decode(body, source.size, orientation)
        except GpuPreprocessingError as error:
            logger.warning(
                "body image GPU preprocessing failed, falling back to Pillow: "
                "error=%s cause=%s format=%s mode=%s size=%s orientation=%s",
                type(error).__name__,
                type(error.__cause__).__name__ if error.__cause__ else None,
                source.format, source.mode, source.size, orientation,
            )
            return None
        return Image.fromarray(array)
