import struct
import zlib
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image, ImageFile, ImageOps

from app.body_image_validation.exceptions import UserImageValidationError
from app.body_image_validation.image_io import decode_image, processing_size
from app.config import body_image_validation as config


def image_bytes(format_name="PNG", size=(480, 640), **save_kwargs):
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format=format_name, **save_kwargs)
    return output.getvalue()


@pytest.mark.parametrize("format_name", ["PNG", "JPEG"])
def test_supported_image_is_decoded_as_rgb(format_name):
    image = decode_image(image_bytes(format_name))

    assert image.mode == "RGB"
    assert image.size == (480, 640)


def test_exif_orientation_is_applied_before_validation():
    exif = Image.Exif()
    exif[274] = 6

    image = decode_image(image_bytes("JPEG", size=(480, 640), exif=exif))

    assert image.size == (640, 480)


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"", "IMAGE_EMPTY"),
        (b"not-an-image", "INVALID_IMAGE"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "INVALID_IMAGE"),
        (image_bytes("JPEG")[:600] + b"\x00" * 2000, "INVALID_IMAGE"),
        (image_bytes("GIF"), "IMAGE_FORMAT_UNSUPPORTED"),
        (image_bytes(size=(479, 640)), "IMAGE_RESOLUTION_TOO_SMALL"),
    ],
)
def test_invalid_images_have_stable_codes(body, code):
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(body)

    assert exc_info.value.code == code


def test_size_limit_is_inclusive(monkeypatch):
    body = image_bytes()
    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", len(body))
    assert decode_image(body).size == (480, 640)

    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", len(body) - 1)
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(body)
    assert exc_info.value.code == "IMAGE_TOO_LARGE"


# --- 최대 픽셀 수: 픽셀 데이터 없이 IHDR 로 크기만 선언한 PNG 를 쓴다 ---


def png_header_only(width, height):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", b"") + chunk(b"IEND", b"")


@pytest.fixture
def decode_spies():
    """전체 decode(exif_transpose / convert / load)를 원래 동작 그대로 감싸 호출만 기록한다."""
    with (
        patch.object(ImageOps, "exif_transpose", wraps=ImageOps.exif_transpose) as exif_transpose,
        patch.object(
            Image.Image, "convert", autospec=True, side_effect=Image.Image.convert
        ) as convert,
        patch.object(
            ImageFile.ImageFile, "load", autospec=True, side_effect=ImageFile.ImageFile.load
        ) as load,
    ):
        yield SimpleNamespace(exif_transpose=exif_transpose, convert=convert, load=load)


def assert_full_decode_not_called(spies):
    spies.exif_transpose.assert_not_called()
    spies.convert.assert_not_called()
    spies.load.assert_not_called()


def test_header_only_fixture_is_tiny_and_declares_the_size():
    body = png_header_only(15055, 10000)

    assert len(body) < 100
    with Image.open(BytesIO(body)) as image:
        assert image.size == (15055, 10000)


@pytest.mark.parametrize(
    "size", [(6000, 4000), (5000, 5000)], ids=["24MP", "exactly-25MP"]
)
def test_images_up_to_the_pixel_limit_go_on_to_full_decode(decode_spies, size):
    assert config.MAX_IMAGE_PIXELS == 25_000_000
    # 크기 검사를 통과했는지만 본다. 실제 25MP 버퍼를 만들지 않도록 decode 첫 단계에서 멈춘다.
    decode_spies.exif_transpose.side_effect = OSError("stop before allocating pixels")

    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(png_header_only(*size))

    assert exc_info.value.code == "IMAGE_DECODE_FAILED"
    decode_spies.exif_transpose.assert_called_once()
    assert decode_spies.exif_transpose.call_args.args[0].size == size


@pytest.mark.parametrize(
    "size",
    [(5001, 5000), (15055, 10000)],
    ids=["over-25MP", "150_550_000-pixels-incident"],
)
def test_too_many_pixels_is_rejected_before_full_decode(decode_spies, size):
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(png_header_only(*size))

    error = exc_info.value
    assert error.code == "IMAGE_RESOLUTION_TOO_LARGE"
    assert error.status_code == 422
    assert_full_decode_not_called(decode_spies)


def test_pixel_limit_is_inclusive_for_a_real_image(monkeypatch):
    body = image_bytes(size=(480, 640))
    monkeypatch.setattr(config, "MAX_IMAGE_PIXELS", 480 * 640)
    assert decode_image(body).size == (480, 640)

    monkeypatch.setattr(config, "MAX_IMAGE_PIXELS", 480 * 640 - 1)
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(body)
    assert exc_info.value.code == "IMAGE_RESOLUTION_TOO_LARGE"


def test_pillow_decompression_bomb_error_is_a_validation_failure(decode_spies):
    # 2 * Image.MAX_IMAGE_PIXELS 를 넘으면 Pillow 가 Image.open 에서 바로 예외를 낸다.
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(png_header_only(20000, 10000))

    assert exc_info.value.code == "IMAGE_RESOLUTION_TOO_LARGE"
    assert isinstance(exc_info.value.__cause__, Image.DecompressionBombError)
    assert_full_decode_not_called(decode_spies)


@pytest.mark.filterwarnings("error::PIL.Image.DecompressionBombWarning")
def test_pillow_decompression_bomb_warning_as_error_is_a_validation_failure(decode_spies):
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(png_header_only(15055, 10000))

    assert exc_info.value.code == "IMAGE_RESOLUTION_TOO_LARGE"
    assert isinstance(exc_info.value.__cause__, Image.DecompressionBombWarning)
    assert_full_decode_not_called(decode_spies)


def test_pillow_protection_is_left_enabled():
    decode_image(image_bytes())

    assert Image.MAX_IMAGE_PIXELS == 89_478_485


def test_small_file_with_huge_resolution_is_422_not_the_413_file_size_error():
    body = png_header_only(15055, 10000)
    assert len(body) <= config.MAX_IMAGE_BYTES

    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(body)

    assert exc_info.value.code == "IMAGE_RESOLUTION_TOO_LARGE"
    assert exc_info.value.status_code == 422


# --- 헤더 문제 / 본문 손상 / 서버 내부 처리 문제 구분 ---


@pytest.mark.parametrize("size", [(479, 640), (640, 479)], ids=["portrait", "landscape"])
def test_short_side_below_minimum_is_rejected_before_full_decode(decode_spies, size):
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(png_header_only(*size))

    assert exc_info.value.code == "IMAGE_RESOLUTION_TOO_SMALL"
    assert_full_decode_not_called(decode_spies)


def _corrupt_png_data():
    body = image_bytes("PNG")
    index = body.index(b"IDAT") + 10
    return body[:index] + bytes([body[index] ^ 0xFF]) + body[index + 1 :]


@pytest.mark.parametrize(
    "body",
    [
        image_bytes("PNG")[: len(image_bytes("PNG")) // 2],
        _corrupt_png_data(),
        image_bytes("JPEG")[: len(image_bytes("JPEG")) // 3],
        png_header_only(480, 640),
    ],
    ids=["png-truncated", "png-corrupt-data", "jpeg-truncated", "png-no-pixel-data"],
)
def test_valid_header_with_damaged_body_is_decode_failed(body):
    with Image.open(BytesIO(body)) as source:
        assert source.format in {"PNG", "JPEG"}  # 헤더는 정상이다

    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(body)

    error = exc_info.value
    assert error.code == "IMAGE_DECODE_FAILED"
    assert error.status_code == 400
    assert error.message == "이미지가 손상되어 처리할 수 없습니다. 다른 사진을 업로드해주세요."


def test_invalid_image_and_decode_failed_have_different_messages():
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(b"not-an-image")

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "이미지 파일을 확인할 수 없습니다. 다른 사진을 업로드해주세요."


@pytest.mark.parametrize("error", [RuntimeError("bug"), TypeError("bad arg"), KeyError("x")])
def test_unexpected_error_on_a_valid_image_is_not_turned_into_a_user_error(monkeypatch, error):
    # 분류되지 않은 예외는 그대로 올라가 전역 핸들러가 500 INTERNAL_SERVER_ERROR 로 응답한다.
    def broken_exif_transpose(image):
        raise error

    monkeypatch.setattr(ImageOps, "exif_transpose", broken_exif_transpose)

    with pytest.raises(type(error)) as exc_info:
        decode_image(image_bytes())

    assert exc_info.value is error


# --- 긴급 안정화: 파일 크기 10MB, 처리용 resize 긴 변 1600px ---


def test_default_limits_follow_the_stabilization_policy():
    assert config.MAX_IMAGE_SIZE_MB == 10
    assert config.MAX_IMAGE_BYTES == 10 * 1024 * 1024
    assert config.MAX_IMAGE_PIXELS == 25_000_000
    assert config.MAX_PROCESSING_IMAGE_SIDE == 1600


def test_file_over_10mb_is_image_too_large_with_a_10mb_message():
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(b"\x00" * (10 * 1024 * 1024 + 1))

    error = exc_info.value
    assert (error.status_code, error.code) == (413, "IMAGE_TOO_LARGE")
    assert error.message == "이미지 크기는 10MB 이하여야 합니다."


def test_file_of_exactly_10mb_is_not_rejected_for_size():
    # 크기 검사는 통과하고 다음 단계(헤더 판독)에서 판단된다.
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(b"\x00" * (10 * 1024 * 1024))

    assert exc_info.value.code == "INVALID_IMAGE"


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((6000, 4000), (1600, 1067)),
        ((4000, 6000), (1067, 1600)),
        ((3200, 1600), (1600, 800)),
        ((1600, 1200), (1600, 1200)),  # 긴 변이 정확히 1600 이면 그대로
        ((480, 640), (480, 640)),  # 작은 이미지는 확대하지 않는다
    ],
    ids=["landscape-24MP", "portrait-24MP", "2x1", "exactly-1600", "small"],
)
def test_processing_size_caps_the_long_side_and_keeps_the_aspect_ratio(size, expected):
    result = processing_size(size, 1600)

    assert result == expected
    assert max(result) <= 1600
    assert abs(result[0] / result[1] - size[0] / size[1]) < 0.01


def test_decoded_image_is_resized_when_the_long_side_is_over_the_limit(monkeypatch):
    # 실제 1600px 넘는 이미지를 만들지 않도록 한도를 줄여서 본다.
    monkeypatch.setattr(config, "MAX_PROCESSING_IMAGE_SIDE", 320)

    image = decode_image(image_bytes(size=(480, 640)))

    assert image.size == (240, 320)
    assert image.mode == "RGB"


def test_decoded_image_is_not_resized_within_the_limit():
    assert decode_image(image_bytes(size=(480, 640))).size == (480, 640)


def test_minimum_resolution_is_checked_on_the_original_size_before_resize(monkeypatch):
    # 원본 짧은 변 480 은 통과하고, 줄인 뒤 480 보다 작아지는 것은 거절 사유가 아니다.
    monkeypatch.setattr(config, "MAX_PROCESSING_IMAGE_SIDE", 320)

    assert decode_image(image_bytes(size=(480, 640))).size == (240, 320)
