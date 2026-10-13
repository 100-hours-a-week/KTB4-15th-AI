"""Person·Pose·밝기 검사가 요청마다 한 번 만든 NumPy 배열과 mp.Image 를 같이 쓰는지."""

import random
import threading
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
import pytest
from PIL import Image

from app.body_image_validation.brightness import mean_value_channel
from app.body_image_validation.detectors import (
    LANDMARK_NAMES,
    DetectionImage,
    MediaPipePersonDetector,
    MediaPipePoseDetector,
)
from app.body_image_validation.exceptions import UserImageValidationError
from app.body_image_validation.models import BoundingBox, Landmark
from app.body_image_validation.service import BodyImageValidationService
from tests.test_body_image_validation_service import (
    FakeBackgroundRemover,
    FakeStorage,
    image_bytes,
    valid_landmarks,
)


def noise_image(size=(600, 800)):
    data = random.Random(0).randbytes(size[0] * size[1] * 3)
    return Image.frombytes("RGB", size, data)


# --- DetectionImage ---


def test_array_is_a_read_only_rgb_copy_that_outlives_the_pil_image():
    image = noise_image()
    expected = np.asarray(image).copy()

    detection = DetectionImage(image)
    image.close()

    array = detection.array
    assert array.dtype == np.uint8
    assert array.shape == (800, 600, 3)
    assert array.flags["C_CONTIGUOUS"]
    assert not array.flags["WRITEABLE"]
    assert np.array_equal(array, expected)


def test_mediapipe_image_is_created_once_from_the_same_srgb_pixels():
    detection = DetectionImage(noise_image())

    first = detection.mediapipe_image()
    second = detection.mediapipe_image()

    assert first is second
    assert first.image_format == mp.ImageFormat.SRGB
    assert np.array_equal(first.numpy_view(), detection.array)


def test_each_request_gets_its_own_arrays():
    image = noise_image()

    first, second = DetectionImage(image), DetectionImage(image)

    assert not np.shares_memory(first.array, second.array)
    assert first.mediapipe_image() is not second.mediapipe_image()


@pytest.mark.parametrize(
    "box",
    [
        BoundingBox(50, 60, 300, 500),
        BoundingBox(-20, -10, 200, 300),
        BoundingBox(500, 700, 400, 400),
    ],
    ids=["inside", "past-top-left", "past-bottom-right"],
)
def test_brightness_on_the_shared_array_matches_the_previous_pil_input(box):
    image = noise_image()
    # 수정 전 mean_value_channel 은 PIL 이미지를 받아 직접 np.asarray 했다.
    array = np.asarray(image)
    x1, y1 = max(0, box.origin_x), max(0, box.origin_y)
    x2 = min(box.origin_x + box.width, array.shape[1])
    y2 = min(box.origin_y + box.height, array.shape[0])
    previous = float(cv2.cvtColor(array[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)[:, :, 2].mean())

    detection = DetectionImage(image)
    before = detection.array.copy()

    assert mean_value_channel(detection.array, box) == previous
    assert np.array_equal(detection.array, before)  # 잘라 쓰기만 하고 바꾸지 않는다


# --- 한 요청의 변환 횟수: MediaPipe task 만 대신하고 adapter·밝기 검사는 실제 코드로 돈다 ---


class FakeTask:
    """MediaPipe task 대신 결과를 돌려주고, 받은 mp.Image 와 그 픽셀을 기록한다."""

    def __init__(self, result):
        self.result = result
        self.inputs = []

    def detect(self, mp_image):
        self.inputs.append((mp_image, mp_image.numpy_view().copy()))
        return self.result


def adapter(cls, result):
    detector = cls.__new__(cls)
    detector._detector = FakeTask(result)
    detector._lock = threading.Lock()
    return detector


def person_result():
    box = SimpleNamespace(origin_x=50, origin_y=50, width=500, height=700)
    return SimpleNamespace(
        detections=[SimpleNamespace(categories=[SimpleNamespace(score=0.9)], bounding_box=box)]
    )


def pose_result():
    values = valid_landmarks()
    default = Landmark(0.5, 0.5, 1.0, 1.0)
    points = [values.get(name, default) for name in LANDMARK_NAMES]
    return SimpleNamespace(
        pose_landmarks=[
            [
                SimpleNamespace(x=p.x, y=p.y, visibility=p.visibility, presence=p.presence)
                for p in points
            ]
        ]
    )


@pytest.fixture
def conversions(monkeypatch):
    """PIL → NumPy(np.asarray 가 읽는 __array_interface__)와 mp.Image 생성 횟수를 센다."""
    counts = {"pil_to_numpy": 0, "mediapipe_image": 0}
    array_interface = Image.Image.__array_interface__
    image_init = mp.Image.__init__

    def counting_array_interface(self):
        counts["pil_to_numpy"] += 1
        return array_interface.fget(self)

    def counting_image_init(self, *args, **kwargs):
        counts["mediapipe_image"] += 1
        image_init(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "__array_interface__", property(counting_array_interface))
    monkeypatch.setattr(mp.Image, "__init__", counting_image_init)
    # 세는 방법 자체가 맞는지: np.asarray(PIL) 한 번이 정확히 1 로 세어진다.
    np.asarray(noise_image((4, 4)))
    assert counts["pil_to_numpy"] == 1
    counts["pil_to_numpy"] = 0
    return counts


def test_one_request_converts_to_numpy_and_mediapipe_once(conversions):
    person = adapter(MediaPipePersonDetector, person_result())
    pose = adapter(MediaPipePoseDetector, pose_result())
    service = BodyImageValidationService(person, pose, FakeBackgroundRemover(), FakeStorage())

    service.validate(image_bytes())

    assert conversions == {"pil_to_numpy": 1, "mediapipe_image": 1}
    [(person_input, person_pixels)] = person._detector.inputs
    [(pose_input, pose_pixels)] = pose._detector.inputs
    assert person_input is pose_input
    assert np.array_equal(person_pixels, pose_pixels)


def test_rejected_person_stops_before_any_more_conversions(conversions):
    no_person = adapter(MediaPipePersonDetector, SimpleNamespace(detections=[]))
    pose = adapter(MediaPipePoseDetector, pose_result())
    service = BodyImageValidationService(no_person, pose, FakeBackgroundRemover(), FakeStorage())

    with pytest.raises(UserImageValidationError) as exc_info:
        service.validate(image_bytes())

    assert exc_info.value.code == "PERSON_NOT_FOUND"

    assert conversions == {"pil_to_numpy": 1, "mediapipe_image": 1}
    assert pose._detector.inputs == []


def test_requests_do_not_share_detection_images():
    person = adapter(MediaPipePersonDetector, person_result())
    pose = adapter(MediaPipePoseDetector, pose_result())
    service = BodyImageValidationService(person, pose, FakeBackgroundRemover(), FakeStorage())

    service.validate(image_bytes())
    service.validate(image_bytes())

    first, second = (mp_image for mp_image, _ in person._detector.inputs)
    assert first is not second
