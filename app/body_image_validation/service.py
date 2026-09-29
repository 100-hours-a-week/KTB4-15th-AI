"""전신 이미지 fail-fast 검증, 전처리, S3 저장 orchestration."""

import uuid
from typing import Protocol

from PIL import Image

from app.body_image_validation import validation_rules
from app.body_image_validation.background import encode_png
from app.body_image_validation.brightness import BrightnessCheckError, validate_brightness
from app.body_image_validation.concurrency import BodyValidationLimiter
from app.body_image_validation.detectors import PersonDetector, PoseDetector
from app.body_image_validation.exceptions import BodyImageSystemError
from app.body_image_validation.image_io import load_image, open_image
from app.body_image_validation.models import BodyImageValidationResult
from app.clients.s3 import ImageStorage, ImageStorageError
from app.config import body_image_validation as config


class BackgroundRemover(Protocol):
    def remove(self, image: Image.Image) -> Image.Image: ...


class BodyImageValidationService:
    def __init__(
        self,
        person_detector: PersonDetector,
        pose_detector: PoseDetector,
        background_remover: BackgroundRemover,
        image_storage: ImageStorage,
        limiter: BodyValidationLimiter | None = None,
    ) -> None:
        self.person_detector = person_detector
        self.pose_detector = pose_detector
        self.background_remover = background_remover
        self.image_storage = image_storage
        self.limiter = limiter or BodyValidationLimiter(
            config.MAX_CONCURRENT_BODY_VALIDATIONS, config.MAX_WAITING_BODY_VALIDATIONS
        )

    def validate(self, image_body: bytes) -> BodyImageValidationResult:
        # 헤더만 읽는 입력 검증은 슬롯 밖에서 한다. 잘못된 이미지가 대기열을 차지하지 않게 하기
        # 위해서다. 픽셀 decode·resize 부터는 메모리를 크게 쓰므로 슬롯 안에서만 한다.
        # PIL 의 `with image:` 는 픽셀 메모리를 놓지 않으므로 close() 를 직접 부른다.
        source = open_image(image_body)
        try:
            with self.limiter.slot():
                image = load_image(source)
                source.close()  # decode 에 쓴 원본 픽셀과 입력 버퍼를 먼저 놓는다.
                try:
                    return self._validate_decoded(image)
                finally:
                    image.close()
        finally:
            source.close()

    def _validate_decoded(self, image: Image.Image) -> BodyImageValidationResult:
        """image 는 호출자가 닫는다. 여기서 만든 배경 제거 결과는 여기서 닫는다."""
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
            validate_brightness(image, person_box)
        except BrightnessCheckError as error:
            raise BodyImageSystemError("BRIGHTNESS_CHECK_FAILED") from error

        try:
            result = self.background_remover.remove(image)
        except Exception as error:
            raise BodyImageSystemError("BACKGROUND_REMOVAL_FAILED") from error
        # RGBA 결과는 PNG 로 인코딩한 직후 닫는다. 업로드 동안에는 PNG bytes 만 남긴다.
        try:
            png = encode_png(result)
        except (OSError, ValueError) as error:
            raise BodyImageSystemError("IMAGE_ENCODING_FAILED") from error
        finally:
            result.close()
        key = f"body-images/{uuid.uuid4()}.png"
        try:
            self.image_storage.upload_bytes(png, key, "image/png")
        except ImageStorageError as error:
            raise BodyImageSystemError("IMAGE_UPLOAD_FAILED") from error
        return BodyImageValidationResult(s3_key=key)
