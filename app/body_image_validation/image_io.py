"""업로드 이미지의 크기·포맷·decode 검증과 표준화."""

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from app.body_image_validation.exceptions import user_error
from app.config import body_image_validation as config

SUPPORTED_FORMATS = {"JPEG", "PNG"}


def decode_image(body: bytes) -> Image.Image:
    if not body:
        raise user_error("IMAGE_EMPTY")
    if len(body) > config.MAX_IMAGE_BYTES:
        raise user_error("IMAGE_TOO_LARGE")
    try:
        with Image.open(BytesIO(body)) as source:
            if source.format not in SUPPORTED_FORMATS:
                raise user_error("IMAGE_FORMAT_UNSUPPORTED")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except UserWarning as error:
        raise user_error("IMAGE_DECODE_FAILED") from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise user_error("IMAGE_DECODE_FAILED") from error
    if min(image.size) < config.MIN_SHORT_SIDE:
        raise user_error("IMAGE_RESOLUTION_TOO_SMALL")
    return image
