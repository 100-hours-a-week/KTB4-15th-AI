"""사람 bbox 영역의 HSV V 평균을 이용한 비차단성 밝기 warning."""

import numpy as np
from PIL import Image

from app.body_image_validation.models import BoundingBox
from app.config import body_image_validation as config

TOO_DARK_WARNING = "IMAGE_TOO_DARK"


class BrightnessCheckError(RuntimeError):
    """밝기 warning 계산만 건너뛸 수 있는 오류."""


def mean_value_channel(image: Image.Image, box: BoundingBox) -> float:
    try:
        import cv2
    except ImportError as error:
        raise BrightnessCheckError("OpenCV를 불러오지 못했습니다.") from error

    array = np.asarray(image)
    height, width = array.shape[:2]
    x1 = max(0, min(box.origin_x, width))
    y1 = max(0, min(box.origin_y, height))
    x2 = max(x1, min(box.origin_x + box.width, width))
    y2 = max(y1, min(box.origin_y + box.height, height))
    crop = array[y1:y2, x1:x2]
    if crop.size == 0:
        raise BrightnessCheckError("person bounding box가 이미지 영역과 겹치지 않습니다.")
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    except (cv2.error, ValueError) as error:
        raise BrightnessCheckError("HSV 변환에 실패했습니다.") from error
    return float(hsv[:, :, 2].mean())


def brightness_warnings(image: Image.Image, box: BoundingBox) -> list[str]:
    return [TOO_DARK_WARNING] if mean_value_channel(image, box) < config.DARK_THRESHOLD else []
