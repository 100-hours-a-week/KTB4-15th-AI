import random
import struct
import zlib
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import ExifTags, Image, ImageFile, ImageOps

from app.body_image_validation.exceptions import UserImageValidationError
from app.body_image_validation.image_io import (
    decode_image,
    load_image,
    open_image,
    processing_size,
)
from app.config import body_image_validation as config


def image_bytes(format_name="PNG", size=(480, 640), **save_kwargs):
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format=format_name, **save_kwargs)
    return output.getvalue()


def noise_image(mode="RGB", size=(480, 640)):
    """회전·반전·채널 차이가 픽셀에 드러나도록 좌우·상하가 비대칭인 이미지."""
    data = random.Random(0).randbytes(size[0] * size[1] * 3)
    image = Image.frombytes("RGB", size, data)
    if mode == "RGBA":
        image.putalpha(Image.linear_gradient("L").resize(size))
        return image
    return image if mode == "RGB" else image.convert(mode)


def jpeg_with_orientation(orientation, mode="RGB", size=(480, 640)):
    exif = Image.Exif()
    if orientation is not None:
        exif[ExifTags.Base.Orientation] = orientation
    output = BytesIO()
    noise_image(mode, size).save(output, format="JPEG", exif=exif)
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
    decode_spies.load.side_effect = OSError("stop before allocating pixels")

    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(png_header_only(*size))

    assert exc_info.value.code == "IMAGE_DECODE_FAILED"
    decode_spies.load.assert_called_once()
    assert decode_spies.load.call_args.args[0].size == size


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
@pytest.mark.parametrize(
    ("owner", "name", "orientation"),
    [(ImageFile.ImageFile, "load", None), (ImageOps, "exif_transpose", 6)],
    ids=["decode", "exif-transpose"],
)
def test_unexpected_error_on_a_valid_image_is_not_turned_into_a_user_error(
    monkeypatch, error, owner, name, orientation
):
    # 분류되지 않은 예외는 그대로 올라가 전역 핸들러가 500 INTERNAL_SERVER_ERROR 로 응답한다.
    def broken(*args, **kwargs):
        raise error

    body = jpeg_with_orientation(orientation)
    monkeypatch.setattr(owner, name, broken)

    with pytest.raises(type(error)) as exc_info:
        decode_image(body)

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


# --- load_image fast path: 회전·RGB 변환이 필요 없으면 원본 크기 사본을 만들지 않는다 ---


def previous_pipeline(body):
    """수정 전 load_image 와 같은 처리. 항상 exif_transpose → convert("RGB") → 필요하면 resize."""
    with Image.open(BytesIO(body)) as source, ImageOps.exif_transpose(source) as transposed:
        image = transposed.convert("RGB")
    size = processing_size(image.size, config.MAX_PROCESSING_IMAGE_SIDE)
    if size == image.size:
        return image
    with image:
        return image.resize(size, Image.Resampling.LANCZOS)


def png_of(mode):
    output = BytesIO()
    noise_image(mode).save(output, format="PNG")
    return output.getvalue()


def assert_same_as_previous_pipeline(body):
    expected = previous_pipeline(body)
    result = decode_image(body)

    assert result.mode == "RGB"
    assert result.size == expected.size
    assert result.tobytes() == expected.tobytes()


ORIENTATIONS = [None, 1, 2, 3, 4, 5, 6, 7, 8]
MODE_INPUTS = {
    "RGB": lambda: png_of("RGB"),
    "RGBA": lambda: png_of("RGBA"),
    "L": lambda: png_of("L"),
    "LA": lambda: png_of("LA"),
    "P": lambda: png_of("P"),
    "1": lambda: png_of("1"),
    "CMYK-jpeg": lambda: jpeg_with_orientation(None, mode="CMYK"),
    "L-jpeg": lambda: jpeg_with_orientation(None, mode="L"),
}


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_every_exif_orientation_gives_the_same_pixels_as_before(orientation):
    assert_same_as_previous_pipeline(jpeg_with_orientation(orientation))


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_exif_transpose_runs_only_for_orientations_that_move_pixels(decode_spies, orientation):
    body = jpeg_with_orientation(orientation)
    decode_spies.convert.reset_mock()  # 입력을 만들면서 부른 convert 는 세지 않는다

    decode_image(body)

    assert decode_spies.exif_transpose.call_count == (1 if orientation in range(2, 9) else 0)
    decode_spies.convert.assert_not_called()  # 이미 RGB


@pytest.mark.parametrize("name", MODE_INPUTS)
def test_every_input_mode_gives_the_same_rgb_pixels_as_before(name):
    body = MODE_INPUTS[name]()
    with Image.open(BytesIO(body)) as source:
        assert source.mode == name.removesuffix("-jpeg")

    assert_same_as_previous_pipeline(body)


@pytest.mark.parametrize("name", MODE_INPUTS)
def test_convert_runs_only_when_the_mode_is_not_rgb(decode_spies, name):
    body = MODE_INPUTS[name]()
    decode_spies.convert.reset_mock()  # 입력을 만들면서 부른 convert 는 세지 않는다

    decode_image(body)

    assert decode_spies.convert.call_count == (0 if name == "RGB" else 1)
    assert decode_spies.convert.call_args is None or decode_spies.convert.call_args.args[1] == "RGB"


@pytest.mark.parametrize(("orientation", "mode"), [(6, "CMYK"), (8, "L"), (3, "CMYK"), (2, "L")])
def test_orientation_and_mode_conversion_together_give_the_same_pixels_as_before(orientation, mode):
    assert_same_as_previous_pipeline(jpeg_with_orientation(orientation, mode=mode))


@pytest.mark.parametrize(
    ("orientation", "mode"), [(None, "RGB"), (1, "RGB"), (6, "RGB"), (1, "CMYK"), (6, "CMYK")]
)
def test_resized_result_is_the_same_as_before(monkeypatch, orientation, mode):
    monkeypatch.setattr(config, "MAX_PROCESSING_IMAGE_SIDE", 320)

    assert_same_as_previous_pipeline(jpeg_with_orientation(orientation, mode=mode))


# Case A~D 와 resize 유무. 최종 결과만 남고 load_image 가 만든 중간 이미지는 모두 닫혀야 한다.
LIFETIME_CASES = [
    pytest.param(1, "RGB", 320, id="A-orientation1-rgb-resize"),
    pytest.param(1, "RGB", 1600, id="A-orientation1-rgb-no-resize"),
    pytest.param(6, "RGB", 320, id="B-orientation6-rgb-resize"),
    pytest.param(6, "RGB", 1600, id="B-orientation6-rgb-no-resize"),
    pytest.param(1, "CMYK", 320, id="C-orientation1-cmyk-resize"),
    pytest.param(1, "CMYK", 1600, id="C-orientation1-cmyk-no-resize"),
    pytest.param(6, "CMYK", 320, id="D-orientation6-cmyk-resize"),
    pytest.param(6, "CMYK", 1600, id="D-orientation6-cmyk-no-resize"),
]


def is_closed(image):
    try:
        image.getpixel((0, 0))
    except ValueError:
        return True
    return False


@pytest.fixture
def created_images(monkeypatch):
    """load_image 안에서 새로 만들어진 이미지(회전본·변환본·resize 결과·사본)를 모은다."""
    created = []

    def record(owner, name):
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            created.append(result)
            return result

        monkeypatch.setattr(owner, name, wrapper)

    record(ImageOps, "exif_transpose")
    for name in ("convert", "resize", "copy"):
        record(Image.Image, name)
    return created


@pytest.mark.parametrize(("orientation", "mode", "max_side"), LIFETIME_CASES)
def test_result_outlives_the_source_and_temporary_images_are_closed(
    monkeypatch, created_images, orientation, mode, max_side
):
    monkeypatch.setattr(config, "MAX_PROCESSING_IMAGE_SIDE", max_side)
    source = open_image(jpeg_with_orientation(orientation, mode=mode))
    created_images.clear()  # 입력을 만들면서 생긴 이미지는 load_image 와 무관하다

    result = load_image(source)

    # source 는 호출자 것이라 load_image 가 닫지 않는다. 결과는 source 와 다른 객체다.
    assert result is not source
    assert not is_closed(source)
    temporaries = [image for image in created_images if image is not result]
    assert all(is_closed(image) for image in temporaries)

    # 호출자(service)처럼 source 를 먼저 닫아도 결과는 그대로 쓸 수 있다.
    source.close()
    result.load()
    assert result.getpixel((0, 0)) is not None
    assert result.mode == "RGB"
    assert max(result.size) <= max_side

    result.close()
    assert is_closed(result)
    assert is_closed(source)


# Case A~D 에서 load_image 가 새로 만드는 원본 해상도 이미지 수. 수정 전에는 항상 2장이었다.
FULL_SIZE_COPY_CASES = [
    pytest.param(1, "RGB", 0, id="A-orientation1-rgb"),
    pytest.param(6, "RGB", 1, id="B-orientation6-rgb"),
    pytest.param(1, "CMYK", 1, id="C-orientation1-cmyk"),
    pytest.param(6, "CMYK", 2, id="D-orientation6-cmyk"),
]


@pytest.mark.parametrize(("orientation", "mode", "expected"), FULL_SIZE_COPY_CASES)
def test_full_size_copies_are_made_only_when_needed(
    monkeypatch, created_images, orientation, mode, expected
):
    monkeypatch.setattr(config, "MAX_PROCESSING_IMAGE_SIDE", 320)
    source = open_image(jpeg_with_orientation(orientation, mode=mode))
    created_images.clear()  # 입력을 만들면서 생긴 이미지는 load_image 와 무관하다
    full_pixels = source.width * source.height

    result = load_image(source)

    full_size = [image for image in created_images if image.width * image.height == full_pixels]
    assert len(full_size) == expected
    assert result.size in {(240, 320), (320, 240)}
    result.close()
    source.close()


@pytest.mark.parametrize(("orientation", "mode", "expected"), FULL_SIZE_COPY_CASES)
def test_temporary_images_are_closed_when_resize_fails(
    monkeypatch, created_images, orientation, mode, expected
):
    monkeypatch.setattr(config, "MAX_PROCESSING_IMAGE_SIDE", 320)

    def failing_resize(self, *args, **kwargs):
        raise MemoryError("resize")

    source = open_image(jpeg_with_orientation(orientation, mode=mode))
    created_images.clear()  # 입력을 만들면서 생긴 이미지는 load_image 와 무관하다
    monkeypatch.setattr(Image.Image, "resize", failing_resize)

    with pytest.raises(MemoryError):
        load_image(source)

    assert not is_closed(source)
    assert all(is_closed(image) for image in created_images)
    source.close()


@pytest.mark.parametrize(("orientation", "mode", "expected"), FULL_SIZE_COPY_CASES)
def test_only_the_image_being_resized_and_the_source_are_open_during_resize(
    monkeypatch, created_images, orientation, mode, expected
):
    # Case D: 회전본은 RGB 변환 직후 닫혀, resize 중에는 source + 변환본만 남는다.
    monkeypatch.setattr(config, "MAX_PROCESSING_IMAGE_SIDE", 320)
    source = open_image(jpeg_with_orientation(orientation, mode=mode))
    created_images.clear()  # 입력을 만들면서 생긴 이미지는 load_image 와 무관하다
    open_during_resize = []
    recording_resize = Image.Image.resize

    def checking_resize(self, *args, **kwargs):
        open_during_resize.extend(
            image for image in created_images if image is not self and not is_closed(image)
        )
        return recording_resize(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", checking_resize)

    load_image(source).close()

    assert open_during_resize == []
    source.close()
