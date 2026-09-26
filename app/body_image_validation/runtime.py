"""무거운 모델/session을 프로세스 생명주기 동안 한 번만 준비한다."""

import logging
import threading
from pathlib import Path

from app.body_image_validation.background import RembgBackgroundRemover
from app.body_image_validation.detectors import MediaPipePersonDetector, MediaPipePoseDetector
from app.body_image_validation.service import BodyImageValidationService
from app.clients.s3 import S3ConfigError, S3ImageStorage, get_s3_bucket
from app.config import body_image_validation as config

logger = logging.getLogger(__name__)


class BodyImageRuntimeError(RuntimeError):
    """모델 파일 또는 runtime 초기화 실패."""


class BodyImageRuntime:
    def __init__(self) -> None:
        paths = (
            config.PERSON_DETECTOR_MODEL,
            config.POSE_LANDMARKER_MODEL,
            config.REMBG_MODEL,
        )
        missing = [str(path) for path in paths if not Path(path).is_file()]
        if missing:
            raise BodyImageRuntimeError(f"전신 이미지 모델 파일이 없습니다: {missing}")
        person_detector = None
        pose_detector = None
        try:
            storage = S3ImageStorage()
            person_detector = MediaPipePersonDetector(str(config.PERSON_DETECTOR_MODEL))
            pose_detector = MediaPipePoseDetector(str(config.POSE_LANDMARKER_MODEL))
            background_remover = RembgBackgroundRemover(str(config.REMBG_MODEL))
        # MediaPipe, ONNX Runtime, boto3가 서로 다른 예외 계층을 사용하므로 초기화 경계에서
        # 하나의 BodyImageRuntimeError로 묶고 이미 열린 native resource를 정리한다.
        except Exception as error:
            if person_detector is not None:
                person_detector.close()
            if pose_detector is not None:
                pose_detector.close()
            raise BodyImageRuntimeError("전신 이미지 runtime 초기화에 실패했습니다.") from error
        self.person_detector = person_detector
        self.pose_detector = pose_detector
        self.background_remover = background_remover
        self.service = BodyImageValidationService(
            self.person_detector,
            self.pose_detector,
            self.background_remover,
            storage,
        )

    def close(self) -> None:
        self.person_detector.close()
        self.pose_detector.close()


class BodyImageRuntimeManager:
    def __init__(self) -> None:
        self._runtime: BodyImageRuntime | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._runtime is not None:
                return
            try:
                get_s3_bucket()
                self._runtime = BodyImageRuntime()
                self._error = None
            except (S3ConfigError, BodyImageRuntimeError) as error:
                self._error = error
                logger.warning("body image validation runtime is unavailable: %s", error)

    def get_service(self) -> BodyImageValidationService:
        if self._runtime is None:
            self.start()
        if self._runtime is None:
            if isinstance(self._error, S3ConfigError):
                raise S3ConfigError(str(self._error)) from self._error
            raise BodyImageRuntimeError("전신 이미지 검증 runtime을 초기화하지 못했습니다.") from self._error
        return self._runtime.service

    def close(self) -> None:
        with self._lock:
            if self._runtime is not None:
                self._runtime.close()
            self._runtime = None
            self._error = None


runtime_manager = BodyImageRuntimeManager()
