"""GPU fast path 의 입력별 경로 선택과 Pillow fallback. GPU decoder 는 fake 로 바꿔 CPU 에서 돈다.

실제 nvJPEG 결과(회전 방향·픽셀 차이)는 test_body_image_gpu_preprocessing_cuda.py 가 CUDA 에서 확인한다.
"""

import logging
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import ExifTags, Image, ImageOps

from app.body_image_validation import inference
from app.body_image_validation import service as service_module
from app.body_image_validation.exceptions import UserImageValidationError
from app.body_image_validation.gpu_preprocessing import (
    GpuImagePreprocessor,
    GpuJpegDecoder,
    GpuPreprocessingError,
    GpuPreprocessingUnavailableError,
)
from app.body_image_validation.image_io import processing_size
from app.body_image_validation.service import BodyImageValidationService
from app.config import body_image_validation as config
from tests.test_body_image_validation_service import (
    FakeBackgroundRemover,
    FakePersonDetector,
    FakePoseDetector,
    FakeStorage,
)


def encoded(format_name="JPEG", mode="RGB", size=(600, 800), orientation=None):
    image = Image.new("RGB", size, (200, 180, 160)).convert(mode)
    output = BytesIO()
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[ExifTags.Base.Orientation] = orientation
        kwargs["exif"] = exif
    image.save(output, format=format_name, **kwargs)
    return output.getvalue()


class FakeDecoder:
    """GpuJpegDecoder 와 같은 인터페이스. Pillow 로 같은 모양의 결과를 만든다."""

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def decode(self, body, header_size, orientation):
        self.calls.append((header_size, orientation))
        if self.error:
            raise self.error
        image = ImageOps.exif_transpose(Image.open(BytesIO(body))).convert("RGB")
        return np.asarray(image.resize(processing_size(image.size, config.MAX_PROCESSING_IMAGE_SIDE)))


@pytest.fixture
def pillow_calls(monkeypatch):
    """서비스가 기존 Pillow load_image 를 부른 횟수."""
    calls = []
    real = service_module.load_image

    def load_image(source):
        calls.append(source.format)
        return real(source)

    monkeypatch.setattr(service_module, "load_image", load_image)
    return calls


def gpu_service(decoder, storage=None):
    return BodyImageValidationService(
        FakePersonDetector(),
        FakePoseDetector(),
        FakeBackgroundRemover(),
        storage or FakeStorage(),
        gpu_preprocessor=GpuImagePreprocessor(decoder),
    )


def test_rgb_jpeg_uses_the_gpu_fast_path_without_pillow(pillow_calls):
    decoder = FakeDecoder()
    storage = FakeStorage()

    result = gpu_service(decoder, storage).validate(encoded())

    assert decoder.calls == [((600, 800), 1)]
    assert pillow_calls == []
    assert result.s3_key.startswith("body-images/")
    assert len(storage.uploads) == 1


@pytest.mark.parametrize("orientation", [None, 1, 2, 3, 4, 5, 6, 7, 8])
def test_exif_jpeg_uses_the_gpu_with_the_header_orientation(pillow_calls, orientation):
    decoder = FakeDecoder()

    gpu_service(decoder).validate(encoded(orientation=orientation))

    assert decoder.calls == [((600, 800), orientation or 1)]
    assert pillow_calls == []


def test_png_is_sent_to_pillow_without_calling_the_gpu(pillow_calls):
    decoder = FakeDecoder()

    result = gpu_service(decoder).validate(encoded("PNG"))

    assert decoder.calls == []
    assert pillow_calls == ["PNG"]
    assert result.s3_key.startswith("body-images/")


def test_cmyk_jpeg_is_sent_to_pillow_and_converted_to_rgb(monkeypatch, pillow_calls):
    decoder = FakeDecoder()
    seen = []
    target = gpu_service(decoder)
    real_check = target._check_body
    monkeypatch.setattr(target, "_check_body", lambda image: (seen.append(image.mode), real_check(image)))

    target.validate(encoded(mode="CMYK"))

    assert decoder.calls == []
    assert pillow_calls == ["JPEG"]
    assert seen == ["RGB"]


def test_grayscale_jpeg_uses_the_gpu(pillow_calls):
    decoder = FakeDecoder()

    gpu_service(decoder).validate(encoded(mode="L"))

    assert len(decoder.calls) == 1
    assert pillow_calls == []


def test_truncated_jpeg_keeps_the_pillow_decode_failed_response(pillow_calls):
    # nvJPEG 는 끝이 잘린 JPEG 를 오류 없이 decode 한다. 기존 400 응답을 유지해야 한다.
    decoder = FakeDecoder()
    body = encoded()
    body = body[: len(body) * 2 // 3]

    with pytest.raises(UserImageValidationError) as exc_info:
        gpu_service(decoder).validate(body)

    assert exc_info.value.code == "IMAGE_DECODE_FAILED"
    assert decoder.calls == []
    assert pillow_calls == ["JPEG"]


def test_gpu_decode_failure_falls_back_to_pillow_and_logs_it(pillow_calls, caplog):
    cause = RuntimeError("nvjpeg failed")
    error = GpuPreprocessingError("decode failed")
    error.__cause__ = cause
    decoder = FakeDecoder(error=error)
    storage = FakeStorage()

    with caplog.at_level(logging.WARNING, logger="app.body_image_validation.gpu_preprocessing"):
        result = gpu_service(decoder, storage).validate(encoded())

    assert len(decoder.calls) == 1
    assert pillow_calls == ["JPEG"]
    assert result.s3_key.startswith("body-images/")
    assert len(storage.uploads) == 1
    [record] = caplog.records
    message = record.getMessage()
    assert "falling back to Pillow" in message
    for expected in ("GpuPreprocessingError", "RuntimeError", "format=JPEG", "mode=RGB"):
        assert expected in message


@pytest.mark.parametrize("error", [TypeError("bug"), AttributeError("bug"), KeyError("bug")])
def test_programming_errors_in_the_gpu_path_are_not_hidden_by_the_fallback(pillow_calls, error):
    decoder = FakeDecoder(error=error)

    with pytest.raises(type(error)) as exc_info:
        gpu_service(decoder).validate(encoded())

    assert exc_info.value is error
    assert pillow_calls == []


def test_without_a_gpu_preprocessor_every_image_uses_pillow(pillow_calls):
    target = BodyImageValidationService(
        FakePersonDetector(), FakePoseDetector(), FakeBackgroundRemover(), FakeStorage()
    )

    target.validate(encoded())
    target.validate(encoded("PNG"))

    assert pillow_calls == ["JPEG", "PNG"]


def test_unreadable_exif_is_left_to_pillow(monkeypatch, pillow_calls):
    decoder = FakeDecoder()
    monkeypatch.setattr(Image.Image, "getexif", lambda self: (_ for _ in ()).throw(ValueError("bad")))
    source = Image.open(BytesIO(encoded()))

    assert GpuImagePreprocessor(decoder).try_load(source, encoded()) is None
    assert decoder.calls == []


# --- GpuJpegDecoder 의 예외 분류. torch 없이 decode 단계만 fake 로 바꾼다. ---


def fake_torch_decoder(decode_error=None, output_shape=(3, 800, 600)):
    decoder = GpuJpegDecoder.__new__(GpuJpegDecoder)

    class Tensor:
        def __init__(self, shape):
            self.shape = shape

        def permute(self, *dims):
            return self

        def contiguous(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((self.shape[1], self.shape[2], 3), np.uint8)

    def decode_jpeg(data, **kwargs):
        assert kwargs["apply_exif_orientation"] is False  # 직접 회전하므로 꺼야 한다
        if decode_error:
            raise decode_error
        return Tensor(output_shape)

    decoder._torch = SimpleNamespace(frombuffer=lambda data, dtype: data, uint8="uint8")
    decoder._decode_jpeg = decode_jpeg
    decoder._device = "cuda"
    decoder._rgb = "RGB"
    decoder._bicubic = "BICUBIC"
    decoder._resize = lambda tensor, size, **kwargs: Tensor((3, *size))
    decoder._lock = __import__("threading").Lock()
    return decoder


@pytest.mark.parametrize(
    "error", [RuntimeError("nvjpeg"), type("OutOfMemoryError", (RuntimeError,), {})("oom")]
)
def test_decoder_turns_gpu_runtime_errors_into_fallback_errors(error):
    with pytest.raises(GpuPreprocessingError) as exc_info:
        fake_torch_decoder(error).decode(b"jpeg", (600, 800), 1)

    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize("error", [NotImplementedError("x"), TypeError("x"), ValueError("x")])
def test_decoder_does_not_turn_code_errors_into_fallback_errors(error):
    with pytest.raises(type(error)):
        fake_torch_decoder(error).decode(b"jpeg", (600, 800), 1)


def test_decoder_rejects_a_decode_size_that_differs_from_the_header():
    with pytest.raises(GpuPreprocessingError):
        fake_torch_decoder(output_shape=(3, 700, 600)).decode(b"jpeg", (600, 800), 1)


def test_decoder_resizes_to_the_production_processing_size():
    array = fake_torch_decoder(output_shape=(3, 4000, 3000)).decode(b"jpeg", (3000, 4000), 1)

    assert array.shape == (1600, 1200, 3)


# --- runtime 설정 ---


def test_gpu_preprocessing_is_off_by_default(monkeypatch):
    # Modal T4 worker 이미지는 이 변수를 true 로 두므로, 기본값은 변수를 지운 채 다시 읽어 확인한다.
    import importlib

    monkeypatch.delenv("BODY_IMAGE_GPU_PREPROCESSING", raising=False)
    try:
        assert importlib.reload(config).GPU_PREPROCESSING is False
        assert inference.build_gpu_preprocessor() is None
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_unavailable_gpu_leaves_the_runtime_on_pillow(monkeypatch, caplog):
    def unavailable():
        raise GpuPreprocessingUnavailableError("CUDA 를 사용할 수 없습니다.")

    monkeypatch.setattr(config, "GPU_PREPROCESSING", True)
    monkeypatch.setattr(inference, "GpuJpegDecoder", unavailable)

    with caplog.at_level(logging.WARNING, logger="app.body_image_validation.inference"):
        assert inference.build_gpu_preprocessor() is None
    assert "using Pillow only" in caplog.text


def test_enabled_gpu_is_passed_to_the_service(monkeypatch):
    monkeypatch.setattr(config, "GPU_PREPROCESSING", True)
    monkeypatch.setattr(inference, "GpuJpegDecoder", FakeDecoder)

    preprocessor = inference.build_gpu_preprocessor()

    assert isinstance(preprocessor, GpuImagePreprocessor)
