from io import BytesIO

import pytest
from PIL import Image

from app.body_image_validation.exceptions import UserImageValidationError
from app.body_image_validation.image_io import decode_image
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
    ("body", "reason_code"),
    [
        (b"", "IMAGE_EMPTY"),
        (b"not-an-image", "IMAGE_DECODE_FAILED"),
        (image_bytes("GIF"), "IMAGE_FORMAT_UNSUPPORTED"),
        (image_bytes(size=(479, 640)), "IMAGE_RESOLUTION_TOO_SMALL"),
    ],
)
def test_invalid_images_have_stable_reason_codes(body, reason_code):
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(body)

    assert exc_info.value.reason_code == reason_code


def test_size_limit_is_inclusive(monkeypatch):
    body = image_bytes()
    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", len(body))
    assert decode_image(body).size == (480, 640)

    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", len(body) - 1)
    with pytest.raises(UserImageValidationError) as exc_info:
        decode_image(body)
    assert exc_info.value.reason_code == "IMAGE_TOO_LARGE"
