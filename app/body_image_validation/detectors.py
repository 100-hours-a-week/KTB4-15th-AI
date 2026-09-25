"""MediaPipe Person Detector와 Pose Landmarker adapter."""

import threading
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

from app.body_image_validation.models import BoundingBox, Landmark, PersonDetection
from app.config import body_image_validation as config


@runtime_checkable
class PersonDetector(Protocol):
    def detect(self, image: Image.Image) -> list[PersonDetection]: ...


@runtime_checkable
class PoseDetector(Protocol):
    def detect(self, image: Image.Image) -> Mapping[str, Landmark] | None: ...


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
        self._mp = mp
        self._detector = mp.tasks.vision.ObjectDetector.create_from_options(options)
        self._lock = threading.Lock()

    def detect(self, image: Image.Image) -> list[PersonDetection]:
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.asarray(image),
        )
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
        self._mp = mp
        self._detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._lock = threading.Lock()

    def detect(self, image: Image.Image) -> Mapping[str, Landmark] | None:
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.asarray(image),
        )
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
