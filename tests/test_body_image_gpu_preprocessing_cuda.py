"""실제 nvJPEG(CUDA) 결과를 기존 Pillow load_image 와 비교한다. CUDA 가 없으면 건너뛴다."""

from io import BytesIO

import numpy as np
import pytest
from PIL import ExifTags, Image, ImageOps

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
if not torch.cuda.is_available():
    pytest.skip("CUDA 가 없는 환경", allow_module_level=True)

from app.body_image_validation import service as service_module
from app.body_image_validation.exceptions import UserImageValidationError
from app.body_image_validation.gpu_preprocessing import (
    GpuImagePreprocessor,
    GpuJpegDecoder,
    GpuPreprocessingError,
)
from app.body_image_validation.image_io import decode_image, open_image
from app.body_image_validation.service import BodyImageValidationService
from tests.test_body_image_validation_service import (
    FakeBackgroundRemover,
    FakePersonDetector,
    FakePoseDetector,
    FakeStorage,
)


@pytest.fixture(scope="module")
def preprocessor():
    return GpuImagePreprocessor(GpuJpegDecoder())


def scene(size):
    """방향이 틀리면 크게 어긋나도록 비대칭이고, decoder 간 차이는 작도록 매끄러운 이미지."""
    width, height = size
    x = np.linspace(0, 1, width)[None, :]
    y = np.linspace(0, 1, height)[:, None]
    array = np.stack(
        [np.broadcast_to(220 * x, (height, width)), np.broadcast_to(220 * y, (height, width)),
         np.broadcast_to(60 + 100 * x * y, (height, width))], axis=2,
    ).astype(np.uint8)
    array[: height // 4, : width // 3] = (250, 30, 30)  # 왼쪽 위 빨간 블록
    return Image.fromarray(array)


def jpeg(size=(600, 800), orientation=None, mode="RGB"):
    exif = Image.Exif()
    if orientation is not None:
        exif[ExifTags.Base.Orientation] = orientation
    output = BytesIO()
    scene(size).convert(mode).save(output, format="JPEG", quality=95, exif=exif)
    return output.getvalue()


def gpu_and_pillow(preprocessor, body):
    source = open_image(body)
    try:
        gpu = preprocessor.try_load(source, body)
    finally:
        source.close()
    assert gpu is not None, "GPU 경로를 타야 하는 입력이다"
    return np.asarray(gpu).astype(np.int16), np.asarray(decode_image(body)).astype(np.int16)


@pytest.mark.parametrize("orientation", [None, 1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("size", [(600, 800), (1800, 2400)], ids=["no-resize", "resize"])
def test_exif_orientation_matches_pillow_exif_transpose(preprocessor, orientation, size):
    body = jpeg(size, orientation)

    gpu, pillow = gpu_and_pillow(preprocessor, body)

    assert gpu.shape == pillow.shape  # 회전 후 width/height
    assert np.abs(gpu - pillow).mean() < 1.0  # 방향이 틀리면 수십 단위로 벌어진다
    if orientation not in (None, 1):
        # 회전을 빼먹은 결과와는 크기나 픽셀이 확실히 다르다(판별력 확인).
        with Image.open(BytesIO(body)) as stored:
            unrotated = np.asarray(stored.convert("RGB").resize(
                (pillow.shape[1], pillow.shape[0]) if orientation < 5 else (pillow.shape[0], pillow.shape[1])
            )).astype(np.int16)
        assert unrotated.shape != gpu.shape or np.abs(gpu - unrotated).mean() > 20


def test_grayscale_jpeg_is_decoded_on_the_gpu_like_pillow(preprocessor):
    gpu, pillow = gpu_and_pillow(preprocessor, jpeg(mode="L"))

    assert gpu.shape == pillow.shape
    assert np.abs(gpu - pillow).max() <= 1
    assert (gpu[..., 0] == gpu[..., 1]).all() and (gpu[..., 1] == gpu[..., 2]).all()


def test_real_nvjpeg_failure_becomes_a_fallback_error():
    body = jpeg(mode="CMYK")  # 경로 선택을 거치지 않고 decoder 에 직접 넣는다

    with pytest.raises(GpuPreprocessingError) as exc_info:
        GpuJpegDecoder().decode(body, (600, 800), 1)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def service_with(preprocessor):
    return BodyImageValidationService(
        FakePersonDetector(), FakePoseDetector(), FakeBackgroundRemover(), FakeStorage(),
        gpu_preprocessor=preprocessor,
    )


def test_gpu_decode_exception_recovers_through_pillow(monkeypatch, preprocessor):
    calls = []
    real = service_module.load_image
    monkeypatch.setattr(service_module, "load_image", lambda source: (calls.append(1), real(source))[1])
    decoder = GpuJpegDecoder()

    def broken_decode(*args, **kwargs):
        raise RuntimeError("nvjpeg: forced failure")

    monkeypatch.setattr(decoder, "_decode_jpeg", broken_decode)

    result = service_with(GpuImagePreprocessor(decoder)).validate(jpeg())

    assert calls == [1]
    assert result.s3_key.startswith("body-images/")


def test_cmyk_and_png_are_processed_by_pillow(monkeypatch, preprocessor):
    calls = []
    real = service_module.load_image
    monkeypatch.setattr(service_module, "load_image", lambda source: (calls.append(source.format), real(source))[1])
    png = BytesIO()
    scene((600, 800)).save(png, format="PNG")

    service_with(preprocessor).validate(jpeg(mode="CMYK"))
    service_with(preprocessor).validate(png.getvalue())

    assert calls == ["JPEG", "PNG"]


def test_truncated_jpeg_is_still_decode_failed(preprocessor):
    body = jpeg()

    with pytest.raises(UserImageValidationError) as exc_info:
        service_with(preprocessor).validate(body[: len(body) // 2])

    assert exc_info.value.code == "IMAGE_DECODE_FAILED"


def test_programming_error_inside_the_gpu_path_is_not_hidden(monkeypatch):
    decoder = GpuJpegDecoder()

    def broken_resize(*args, **kwargs):
        raise TypeError("bug in resize call")

    monkeypatch.setattr(decoder, "_resize", broken_resize)

    with pytest.raises(TypeError):
        service_with(GpuImagePreprocessor(decoder)).validate(jpeg((1800, 2400)))


def test_result_image_is_independent_of_gpu_memory(preprocessor):
    body = jpeg()
    source = open_image(body)
    image = preprocessor.try_load(source, body)
    source.close()
    torch.cuda.empty_cache()

    assert image.mode == "RGB"
    assert np.asarray(image).flags["C_CONTIGUOUS"]
    assert ImageOps.exif_transpose(image).size == image.size
