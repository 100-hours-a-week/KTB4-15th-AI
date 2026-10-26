"""전신 이미지 fail-fast 검증, 전처리, S3 저장 orchestration."""

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Protocol

from PIL import Image

from app.body_image_validation import deadline, validation_rules
from app.body_image_validation.background import encode_png
from app.body_image_validation.brightness import BrightnessCheckError, validate_brightness
from app.body_image_validation.concurrency import BodyValidationLimiter
from app.body_image_validation.detectors import DetectionImage, PersonDetector, PoseDetector
from app.body_image_validation.exceptions import BodyImageSystemError
from app.body_image_validation.gpu_preprocessing import GpuImagePreprocessor
from app.body_image_validation.image_io import load_image, open_image
from app.body_image_validation.models import BodyImageValidationResult
from app.clients.s3 import ImageStorage, ImageStorageError
from app.config import body_image_validation as config

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
        limiter: BodyValidationLimiter | None = None,
        gpu_preprocessor: GpuImagePreprocessor | None = None,
        diagnostics: bool = False,
    ) -> None:
        self.person_detector = person_detector
        self.pose_detector = pose_detector
        self.background_remover = background_remover
        self.image_storage = image_storage
        # local 모드는 기존 정책 그대로 대기 시간 제한이 없다(대기열 크기만 제한). 대기 시간 제한은 remote 모드의
        # deadline 과 함께 runtime.RemoteBodyImageRuntime 이 건다.
        self.limiter = limiter or BodyValidationLimiter(
            config.MAX_CONCURRENT_BODY_VALIDATIONS, config.MAX_WAITING_BODY_VALIDATIONS
        )
        # None 이면 기존처럼 Pillow 로만 처리한다.
        self.gpu_preprocessor = gpu_preprocessor
        # 진단(기본 꺼짐). 켜면 요청마다 단계별 처리 시간(ms)을 로그 한 줄로 남긴다. 요청별 기록은
        # thread 마다 따로 둔다(endpoint 는 threadpool 에서 돈다).
        self._diagnostics = threading.local() if diagnostics else None

    def validate(self, image_body: bytes) -> BodyImageValidationResult:
        if self._diagnostics is None:
            return self._validate(image_body)
        timings = self._diagnostics.timings = {}
        started = time.perf_counter()
        outcome = "error"
        try:
            result = self._validate(image_body)
            outcome = "success"
            return result
        except Exception as error:
            outcome = getattr(error, "code", None) or type(error).__name__
            raise
        finally:
            timings["total"] = (time.perf_counter() - started) * 1000
            self._diagnostics.timings = None
            logger.info(
                "body image validation timings: outcome=%s ms=%s",
                outcome, {name: round(value, 1) for name, value in timings.items()},
            )

    def _validate(self, image_body: bytes) -> BodyImageValidationResult:
        # 헤더만 읽는 입력 검증은 슬롯 밖에서 한다. 잘못된 이미지가 대기열을 차지하지 않게 하기
        # 위해서다. 픽셀 decode·resize 부터는 메모리를 크게 쓰므로 슬롯 안에서만 한다.
        # PIL 의 `with image:` 는 픽셀 메모리를 놓지 않으므로 close() 를 직접 부른다.
        source = open_image(image_body)
        try:
            with self.limiter.slot():
                # Modal worker 에서만 의미가 있다(AI Server 가 deadline 을 보냈을 때). 이미 지났으면 GPU 작업 전에 멈춘다.
                deadline.check("before_processing")
                with self._timed("preprocess"):
                    image = self._load(source, image_body)
                source.close()  # decode 에 쓴 원본 픽셀과 입력 버퍼를 먼저 놓는다.
                try:
                    return self._validate_decoded(image)
                finally:
                    image.close()
        finally:
            source.close()

    @contextmanager
    def _timed(self, stage: str):
        """진단이 켜져 있으면 이 블록의 경과 시간(ms)을 stage 이름으로 기록한다."""
        timings = getattr(self._diagnostics, "timings", None) if self._diagnostics else None
        if timings is None:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            timings[stage] = (time.perf_counter() - started) * 1000

    def _load(self, source: Image.Image, image_body: bytes) -> Image.Image:
        """GPU 로 처리할 수 있는 JPEG 는 GPU 로, 나머지와 GPU 실패는 기존 Pillow load_image 로.

        두 경로 모두 source 와 다른 처리용 RGB 이미지를 돌려주고 호출자가 닫는다.
        """
        if self.gpu_preprocessor is not None:
            image = self.gpu_preprocessor.try_load(source, image_body)
            if image is not None:
                self._mark("preprocess_gpu")
                return image
        return load_image(source)

    def _mark(self, stage: str) -> None:
        """진단이 켜져 있으면 해당 경로를 탔다는 표시(1)를 남긴다."""
        timings = getattr(self._diagnostics, "timings", None) if self._diagnostics else None
        if timings is not None:
            timings[stage] = 1

    def _validate_decoded(self, image: Image.Image) -> BodyImageValidationResult:
        """image 는 호출자가 닫는다. 여기서 만든 배경 제거 결과는 여기서 닫는다."""
        self._check_body(image)

        deadline.check("before_background_removal")
        try:
            with self._timed("rembg"):
                result = self.background_remover.remove(image)
        except Exception as error:
            raise BodyImageSystemError("BACKGROUND_REMOVAL_FAILED") from error
        # RGBA 결과는 PNG 로 인코딩한 직후 닫는다. 업로드 동안에는 PNG bytes 만 남긴다.
        try:
            with self._timed("png"):
                png = encode_png(result)
        except (OSError, ValueError) as error:
            raise BodyImageSystemError("IMAGE_ENCODING_FAILED") from error
        finally:
            result.close()
        # AI Server 가 결과를 받을 시간이 남지 않았으면 올리지 않는다(아무도 받지 않을 객체를 만들지 않는다).
        deadline.check("before_upload", margin=deadline.UPLOAD_MARGIN_SECONDS)
        key = f"body-images/{uuid.uuid4()}.png"
        try:
            with self._timed("s3_upload"):
                self.image_storage.upload_bytes(png, key, "image/png")
        except ImageStorageError as error:
            raise BodyImageSystemError("IMAGE_UPLOAD_FAILED") from error
        return BodyImageValidationResult(s3_key=key)

    def _check_body(self, image: Image.Image) -> None:
        """사람·자세·밝기 검사. 변환 결과(detection)는 이 안에서만 쓰여 배경 제거 전에 놓인다."""
        try:
            with self._timed("person"):
                detection = DetectionImage(image)
                detections = self.person_detector.detect(detection)
        except Exception as error:
            raise BodyImageSystemError("PERSON_DETECTION_FAILED") from error
        person_box = validation_rules.select_single_person(detections, image.height)

        try:
            with self._timed("pose"):
                landmarks = self.pose_detector.detect(detection)
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
            with self._timed("brightness"):
                validate_brightness(detection.array, person_box)
        except BrightnessCheckError as error:
            raise BodyImageSystemError("BRIGHTNESS_CHECK_FAILED") from error
