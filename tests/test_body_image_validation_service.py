from io import BytesIO

import pytest
from PIL import Image

from app.body_image_validation.brightness import BrightnessCheckError
from app.body_image_validation.exceptions import (
    BodyImageSystemError,
    ServerBusyError,
    UserImageValidationError,
    user_error,
)
from app.body_image_validation.models import BoundingBox, Landmark, PersonDetection
from app.body_image_validation.service import BodyImageValidationService
from app.clients.s3 import ImageStorageError
from app.config import body_image_validation as config


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

    result = service(storage=storage).validate(image_bytes())

    assert result.s3_key.startswith("body-images/")
    assert result.s3_key.endswith(".png")
    [(body, key, content_type)] = storage.uploads
    assert key == result.s3_key
    assert content_type == "image/png"
    assert Image.open(BytesIO(body)).mode == "RGBA"


def test_brightness_check_failure_is_system_failure(monkeypatch):
    storage = FakeStorage()

    def fail_brightness(*args):
        raise BrightnessCheckError("opencv failed")

    monkeypatch.setattr(
        "app.body_image_validation.service.validate_brightness", fail_brightness
    )

    with pytest.raises(BodyImageSystemError) as exc_info:
        service(storage=storage).validate(image_bytes())

    assert exc_info.value.code == "BRIGHTNESS_CHECK_FAILED"
    assert storage.uploads == []


def test_too_dark_image_fails_before_background_removal(monkeypatch):
    remover = FakeBackgroundRemover()
    storage = FakeStorage()

    def too_dark(*args):
        raise user_error("IMAGE_TOO_DARK")

    monkeypatch.setattr("app.body_image_validation.service.validate_brightness", too_dark)

    with pytest.raises(UserImageValidationError) as exc_info:
        service(remover=remover, storage=storage).validate(image_bytes())

    assert exc_info.value.code == "IMAGE_TOO_DARK"
    assert storage.uploads == []


@pytest.mark.parametrize(
    ("kwargs", "code"),
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
def test_system_failures_have_specific_codes(kwargs, code):
    with pytest.raises(BodyImageSystemError) as exc_info:
        service(**kwargs).validate(image_bytes())

    assert exc_info.value.code == code


# --- 긴급 안정화: 처리용 resize 와 동시 실행 제한 ---


class SizeRecordingPerson(FakePersonDetector):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def detect(self, image):
        self.sizes.append(("person", image.size))
        return super().detect(image)


class SizeRecordingPose(FakePoseDetector):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def detect(self, image):
        self.sizes.append(("pose", image.size))
        return super().detect(image)


class SizeRecordingRemover(FakeBackgroundRemover):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def remove(self, image):
        self.sizes.append(("rembg", image.size))
        return super().remove(image)


def test_resized_image_is_what_person_pose_rembg_and_storage_receive(monkeypatch):
    # 600x800 입력을 긴 변 700 으로 줄이면 525x700 이 된다(실제 1600 은 테스트 이미지가 너무 크다).
    monkeypatch.setattr(config, "MAX_PROCESSING_IMAGE_SIDE", 700)
    sizes = []
    storage = FakeStorage()

    BodyImageValidationService(
        SizeRecordingPerson(sizes),
        SizeRecordingPose(sizes),
        SizeRecordingRemover(sizes),
        storage,
    ).validate(image_bytes())

    assert sizes == [("person", (525, 700)), ("pose", (525, 700)), ("rembg", (525, 700))]
    [(body, _, _)] = storage.uploads
    assert Image.open(BytesIO(body)).size == (525, 700)


class FullLimiter:
    """실행 1 + 대기 1 이 모두 찬 상태를 흉내 낸다."""

    def __init__(self):
        self.entered = 0

    def slot(self):
        self.entered += 1
        raise ServerBusyError()


def test_busy_limiter_rejects_before_any_heavy_stage():
    sizes = []
    storage = FakeStorage()
    limiter = FullLimiter()

    with pytest.raises(ServerBusyError):
        BodyImageValidationService(
            SizeRecordingPerson(sizes),
            SizeRecordingPose(sizes),
            SizeRecordingRemover(sizes),
            storage,
            limiter=limiter,
        ).validate(image_bytes())

    assert limiter.entered == 1
    assert sizes == []
    assert storage.uploads == []


def test_invalid_image_is_rejected_before_taking_a_limiter_slot():
    limiter = FullLimiter()

    with pytest.raises(UserImageValidationError) as exc_info:
        service_with_limiter = BodyImageValidationService(
            FakePersonDetector(),
            FakePoseDetector(),
            FakeBackgroundRemover(),
            FakeStorage(),
            limiter=limiter,
        )
        service_with_limiter.validate(b"")

    assert exc_info.value.code == "IMAGE_EMPTY"
    assert limiter.entered == 0  # 잘못된 이미지는 대기열 자리를 쓰지 않는다


def test_slot_is_released_when_a_heavy_stage_fails():
    service_instance = service(person=FakePersonDetector(RuntimeError("detector crashed")))

    with pytest.raises(BodyImageSystemError):
        service_instance.validate(image_bytes())

    assert (service_instance.limiter.running, service_instance.limiter.waiting) == (0, 0)
