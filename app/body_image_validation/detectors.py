"""MediaPipe Person Detector와 Pose Landmarker adapter."""

import threading
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

from app.body_image_validation.models import BoundingBox, Landmark, PersonDetection
from app.config import body_image_validation as config


class DetectionImage:
    """한 요청의 처리용 RGB 이미지와, detector·밝기 검사가 같이 쓰는 변환 결과.

    np.asarray(PIL) 와 mp.Image 생성은 부를 때마다 이미지 전체를 복사한다. Person·Pose·밝기 검사가
    요청마다 이 객체 하나를 새로 만들어 같이 쓰고, 요청 밖에 보관하지 않는다.
    """

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        # PIL 과 메모리를 공유하지 않는 읽기 전용 HxWx3 uint8 배열이라 image 를 닫아도 쓸 수 있다.
        self.array = np.asarray(image)
        self._mediapipe_image = None

    def mediapipe_image(self):
        """처음 부를 때 한 번 만든다. mp.Image 는 만들 때 픽셀을 복사한 뒤 바뀌지 않고, detect 는
        포인터만 빌려 쓰므로 두 detector 가 같은 객체를 써도 된다."""
        if self._mediapipe_image is None:
            import mediapipe as mp

            self._mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.array)
        return self._mediapipe_image


@runtime_checkable
class PersonDetector(Protocol):
    def detect(self, image: DetectionImage) -> list[PersonDetection]: ...


@runtime_checkable
class PoseDetector(Protocol):
    def detect(self, image: DetectionImage) -> Mapping[str, Landmark] | None: ...


class MediaPipePersonDetector:
    def __init__(self, model_path: str) -> None:
        import mediapipe as mp

        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=model_path,
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            score_threshold=config.PERSON_SCORE_THRESHOLD,
            category_allowlist=["person"],
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._detector = mp.tasks.vision.ObjectDetector.create_from_options(options)
        self._lock = threading.Lock()

    def detect(self, image: DetectionImage) -> list[PersonDetection]:
        mp_image = image.mediapipe_image()
        with self._lock:
            result = self._detector.detect(mp_image)
        detections = []
        for item in result.detections:
            category = item.categories[0]
            box = item.bounding_box
            detections.append(
                PersonDetection(
                    bounding_box=BoundingBox(box.origin_x, box.origin_y, box.width, box.height),
                    score=float(category.score),
                )
            )
        return detections

    def close(self) -> None:
        self._detector.close()


LANDMARK_NAMES: Sequence[str] = (
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
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


class MediaPipePoseDetector:
    def __init__(self, model_path: str) -> None:
        import mediapipe as mp

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=model_path,
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.POSE_PRESENCE_CONFIDENCE,
        )
        self._detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._lock = threading.Lock()

    def detect(self, image: DetectionImage) -> Mapping[str, Landmark] | None:
        mp_image = image.mediapipe_image()
        with self._lock:
            result = self._detector.detect(mp_image)
        if not result.pose_landmarks:
            return None
        return {
            name: Landmark(
                x=float(point.x or 0.0),
                y=float(point.y or 0.0),
                visibility=float(point.visibility or 0.0),
                presence=float(point.presence or 0.0),
            )
            for name, point in zip(LANDMARK_NAMES, result.pose_landmarks[0], strict=True)
        }

    def close(self) -> None:
        self._detector.close()
