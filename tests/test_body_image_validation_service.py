from io import BytesIO

import pytest
from PIL import Image

from app.body_image_validation.brightness import BrightnessCheckError
from app.body_image_validation.exceptions import BodyImageSystemError
from app.body_image_validation.models import BoundingBox, Landmark, PersonDetection
from app.body_image_validation.service import BodyImageValidationService
from app.clients.s3 import ImageStorageError


def image_bytes():
    output = BytesIO()
    Image.new("RGB", (600, 800), "white").save(output, format="PNG")
    return output.getvalue()


def valid_landmarks():
    names = (
        "NOSE",
        "LEFT_EYE",
        "RIGHT_EYE",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX",
    )
    values = {name: Landmark(0.5, 0.5, 1.0, 1.0) for name in names}
    values.update(
        {
            "NOSE": Landmark(0.5, 0.1, 1.0, 1.0),
            "LEFT_SHOULDER": Landmark(0.3, 0.25, 1.0, 1.0),
            "RIGHT_SHOULDER": Landmark(0.7, 0.25, 1.0, 1.0),
            "LEFT_HIP": Landmark(0.4, 0.5, 1.0, 1.0),
            "RIGHT_HIP": Landmark(0.6, 0.5, 1.0, 1.0),
            "LEFT_HEEL": Landmark(0.4, 0.9, 1.0, 1.0),
            "RIGHT_HEEL": Landmark(0.6, 0.9, 1.0, 1.0),
            "LEFT_FOOT_INDEX": Landmark(0.4, 0.92, 1.0, 1.0),
            "RIGHT_FOOT_INDEX": Landmark(0.6, 0.92, 1.0, 1.0),
        }
    )
    return values


class FakePersonDetector:
    def __init__(self, error=None):
        self.error = error

    def detect(self, image):
        if self.error:
            raise self.error
        return [PersonDetection(BoundingBox(50, 50, 500, 700), 0.9)]


class FakePoseDetector:
    def __init__(self, error=None):
        self.error = error

    def detect(self, image):
        if self.error:
            raise self.error
        return valid_landmarks()


class FakeBackgroundRemover:
    def __init__(self, error=None):
        self.error = error

    def remove(self, image):
        if self.error:
            raise self.error
        return image.convert("RGBA")


class FakeStorage:
    def __init__(self, error=None):
        self.error = error
        self.uploads = []

    def upload_bytes(self, body, key, content_type):
        if self.error:
            raise self.error
        self.uploads.append((body, key, content_type))
        return key


def service(person=None, pose=None, remover=None, storage=None):
    return BodyImageValidationService(
        person or FakePersonDetector(),
        pose or FakePoseDetector(),
        remover or FakeBackgroundRemover(),
        storage or FakeStorage(),
    )


def test_success_removes_background_and_uploads_rgba_png():
    storage = FakeStorage()

    result = service(storage=storage).validate(42, image_bytes())

    assert result.s3_key.startswith("users/42/body-images/")
    assert result.s3_key.endswith(".png")
    assert result.warnings == []
    [(body, key, content_type)] = storage.uploads
    assert key == result.s3_key
    assert content_type == "image/png"
    assert Image.open(BytesIO(body)).mode == "RGBA"


def test_brightness_failure_is_logged_and_skipped(monkeypatch, caplog):
    def fail_brightness(*args):
        raise BrightnessCheckError("opencv failed")

    monkeypatch.setattr(
        "app.body_image_validation.service.brightness_warnings", fail_brightness
    )

    result = service().validate(42, image_bytes())

    assert result.warnings == []
    assert "brightness check skipped" in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "reason_code"),
    [
        ({"person": FakePersonDetector(RuntimeError("detector"))}, "PERSON_DETECTION_FAILED"),
        ({"pose": FakePoseDetector(RuntimeError("pose"))}, "POSE_ESTIMATION_FAILED"),
        (
            {"remover": FakeBackgroundRemover(RuntimeError("remove"))},
            "BACKGROUND_REMOVAL_FAILED",
        ),
        ({"storage": FakeStorage(ImageStorageError("s3"))}, "IMAGE_UPLOAD_FAILED"),
    ],
)
def test_system_failures_have_specific_reason_codes(kwargs, reason_code):
    with pytest.raises(BodyImageSystemError) as exc_info:
        service(**kwargs).validate(42, image_bytes())

    assert exc_info.value.reason_code == reason_code
