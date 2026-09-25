"""전신 이미지 fail-fast 검증, 전처리, S3 저장 orchestration."""

import logging
import uuid
from typing import Protocol

from PIL import Image

from app.body_image_validation import validation_rules
from app.body_image_validation.background import encode_png
from app.body_image_validation.brightness import BrightnessCheckError, brightness_warnings
from app.body_image_validation.detectors import PersonDetector, PoseDetector
from app.body_image_validation.exceptions import BodyImageSystemError
from app.body_image_validation.image_io import decode_image
from app.body_image_validation.models import BodyImageValidationResult
from app.clients.s3 import ImageStorage, ImageStorageError

logger = logging.getLogger(__name__)


class BackgroundRemover(Protocol):
    def remove(self, image: Image.Image) -> Image.Image: ...


class BodyImageValidationService:
    def __init__(
        self,
        person_detector: PersonDetector,
        pose_detector: PoseDetector,
        background_remover: BackgroundRemover,
        image_storage: ImageStorage,
    ) -> None:
        self.person_detector = person_detector
        self.pose_detector = pose_detector
        self.background_remover = background_remover
        self.image_storage = image_storage

    def validate(self, user_id: int, image_body: bytes) -> BodyImageValidationResult:
        image = decode_image(image_body)
        try:
            detections = self.person_detector.detect(image)
        except Exception as error:
            raise BodyImageSystemError("PERSON_DETECTION_FAILED") from error
        person_box = validation_rules.select_single_person(detections, image.height)

        try:
            landmarks = self.pose_detector.detect(image)
        except Exception as error:
            raise BodyImageSystemError("POSE_ESTIMATION_FAILED") from error
        if landmarks is None:
            from app.body_image_validation.exceptions import user_error

            raise user_error("FULL_BODY_NOT_VISIBLE")

        validation_rules.validate_full_body(landmarks)
        validation_rules.validate_arms(landmarks)
        validation_rules.validate_legs(landmarks)
        validation_rules.validate_frontal_pose(landmarks, person_box, image.width)

        try:
            warnings = brightness_warnings(image, person_box)
        except BrightnessCheckError as error:
            logger.warning("brightness check skipped: %s", error)
            warnings = []

        try:
            result = self.background_remover.remove(image)
        except Exception as error:
            raise BodyImageSystemError("BACKGROUND_REMOVAL_FAILED") from error
        try:
            png = encode_png(result)
        except (OSError, ValueError) as error:
            raise BodyImageSystemError("IMAGE_ENCODING_FAILED") from error
        key = f"users/{user_id}/body-images/{uuid.uuid4()}.png"
        try:
            self.image_storage.upload_bytes(png, key, "image/png")
        except ImageStorageError as error:
            raise BodyImageSystemError("IMAGE_UPLOAD_FAILED") from error
        return BodyImageValidationResult(s3_key=key, warnings=warnings)
