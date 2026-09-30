"""업로드 이미지의 크기·포맷·decode 검증과 표준화."""

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from app.body_image_validation.exceptions import user_error
from app.config import body_image_validation as config

SUPPORTED_FORMATS = {"JPEG", "PNG"}


def processing_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    """긴 변이 max_side 를 넘으면 비율을 유지해 max_side 로 줄인 크기. 확대는 하지 않는다."""
    width, height = size
    long_side = max(width, height)
    if long_side <= max_side:
        return size
    scale = max_side / long_side
    return max(1, round(width * scale)), max(1, round(height * scale))


def open_image(body: bytes) -> Image.Image:
    """헤더만 읽어 입력을 검증한다. 픽셀은 아직 메모리에 펼치지 않는다.

    돌려준 이미지는 lazy 상태이며 호출자가 닫는다. 검증에 실패하면 여기서 닫는다.
    """
    if not body:
        raise user_error("IMAGE_EMPTY")
    if len(body) > config.MAX_IMAGE_BYTES:
        raise user_error("IMAGE_TOO_LARGE")

    # Image.open 은 헤더만 읽는다. 여기서 실패하면 애초에 이미지 파일로 볼 수 없다.
    try:
        source = Image.open(BytesIO(body))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        # Pillow 자체 한도는 Image.open 에서 걸린다. Error 는 항상, Warning 은 경고를 예외로
        # 다루는 실행 환경에서만 여기로 온다. Pillow 보호 설정은 바꾸지 않는다.
        raise user_error("IMAGE_RESOLUTION_TOO_LARGE") from error
    except (UnidentifiedImageError, OSError, ValueError, UserWarning) as error:
        raise user_error("INVALID_IMAGE") from error

    try:
        if source.format not in SUPPORTED_FORMATS:
            raise user_error("IMAGE_FORMAT_UNSUPPORTED")
        # 크기 검사는 헤더 값만으로 끝낸다. EXIF 회전은 픽셀 수와 짧은 변을 바꾸지 않는다.
        width, height = source.size
        if width * height > config.MAX_IMAGE_PIXELS:
            raise user_error("IMAGE_RESOLUTION_TOO_LARGE")
        if min(width, height) < config.MIN_SHORT_SIDE:
            raise user_error("IMAGE_RESOLUTION_TOO_SMALL")
    except BaseException:
        source.close()
        raise
    return source


def load_image(source: Image.Image) -> Image.Image:
    """open_image 로 연 이미지의 픽셀을 실제로 읽어 처리용 RGB 이미지로 만든다.

    메모리를 크게 쓰는 단계라 동시 실행 제한 안에서 부른다. source 는 호출자가 닫는다.
    중간에 생기는 이미지는 여기서 닫고, 돌려준 이미지는 호출자가 닫는다.
    """
    # 전체 decode. Pillow 가 본문 손상으로 내는 예외만 사용자 파일 문제로 본다.
    try:
        transposed = ImageOps.exif_transpose(source)
    except (OSError, ValueError, UserWarning) as error:
        raise user_error("IMAGE_DECODE_FAILED") from error
    try:
        image = transposed.convert("RGB")
        image.load()
    except (OSError, ValueError, UserWarning) as error:
        raise user_error("IMAGE_DECODE_FAILED") from error
    finally:
        if transposed is not source:
            transposed.close()

    # 입력 검증(최소 480px / 최대 25MP)은 원본 크기로 끝냈다. 이후 모델·rembg 는 줄인 이미지로
    # 돌려 메모리를 아낀다.
    target_size = processing_size(image.size, config.MAX_PROCESSING_IMAGE_SIDE)
    if target_size == image.size:
        return image
    try:
        return image.resize(target_size, Image.Resampling.LANCZOS)
    finally:
        image.close()


def decode_image(body: bytes) -> Image.Image:
    """open_image + load_image. 돌려준 이미지는 호출자가 닫는다."""
    source = open_image(body)
    try:
        return load_image(source)
    finally:
        source.close()
