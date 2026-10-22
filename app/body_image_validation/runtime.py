"""무거운 모델/session을 프로세스 생명주기 동안 한 번만 준비한다."""

import logging
import threading
from pathlib import Path

from app.body_image_validation.background import RembgBackgroundRemover
from app.body_image_validation.concurrency import BodyValidationLimiter
from app.body_image_validation.detectors import MediaPipePersonDetector, MediaPipePoseDetector
from app.body_image_validation.gpu_preprocessing import (
    GpuImagePreprocessor,
    GpuJpegDecoder,
    GpuPreprocessingUnavailableError,
)
from app.body_image_validation.service import BodyImageValidationService
from app.clients.s3 import S3ConfigError, S3ImageStorage, get_s3_bucket
from app.config import body_image_validation as config

logger = logging.getLogger(__name__)


class BodyImageRuntimeError(RuntimeError):
    """모델 파일 또는 runtime 초기화 실패."""


def build_gpu_preprocessor() -> GpuImagePreprocessor | None:
    """GPU 경로는 선택 사항이다. 켰는데 쓸 수 없으면 경고만 남기고 Pillow 로만 동작한다."""
    if not config.GPU_PREPROCESSING:
        return None
    try:
        return GpuImagePreprocessor(GpuJpegDecoder())
    except GpuPreprocessingUnavailableError as error:
        logger.warning("body image GPU preprocessing is unavailable, using Pillow only: %s", error)
        return None


class BodyImageRuntime:
    def __init__(
        self,
        person_detector: MediaPipePersonDetector | None = None,
        pose_detector: MediaPipePoseDetector | None = None,
    ) -> None:
        """detector 를 넘기면 새로 만들지 않고 쓴다. 넘긴 detector 도 이 runtime 이 닫는다.

        Modal worker 는 CPU 전용인 MediaPipe detector 를 memory snapshot 전에 만들어 넘기고,
        GPU 를 쓰는 rembg session·GPU 전처리·S3 client 는 snapshot 복원 뒤 여기서 만든다.
        """
        paths = (
            config.PERSON_DETECTOR_MODEL,
            config.POSE_LANDMARKER_MODEL,
            config.REMBG_MODEL,
        )
        missing = [str(path) for path in paths if not Path(path).is_file()]
        if missing:
            raise BodyImageRuntimeError(f"전신 이미지 모델 파일이 없습니다: {missing}")
        try:
            storage = S3ImageStorage()
            if person_detector is None:
                person_detector = MediaPipePersonDetector(str(config.PERSON_DETECTOR_MODEL))
            if pose_detector is None:
                pose_detector = MediaPipePoseDetector(str(config.POSE_LANDMARKER_MODEL))
            remover_options = {"enable_cpu_mem_arena": config.REMBG_ENABLE_CPU_MEM_ARENA}
            if config.REMBG_PROVIDERS:
                remover_options["providers"] = list(config.REMBG_PROVIDERS)
            background_remover = RembgBackgroundRemover(str(config.REMBG_MODEL), **remover_options)
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
            gpu_preprocessor=build_gpu_preprocessor(),
        )

    def close(self) -> None:
        self.person_detector.close()
        self.pose_detector.close()


class RemoteBodyImageRuntime:
    """remote 모드(AI Server). 모델·GPU·S3 client 를 만들지 않고 Modal worker 를 부르는 service 만 둔다."""

    def __init__(self) -> None:
        from app.body_image_validation.remote import (
            RemoteBodyImageValidationService,
            RemoteWorkerConfigError,
        )
        from app.config import settings

        try:
            self.service = RemoteBodyImageValidationService(
                config.REMOTE_URL,
                proxy_key=settings.BODY_IMAGE_REMOTE_PROXY_KEY,
                proxy_secret=settings.BODY_IMAGE_REMOTE_PROXY_SECRET,
                api_key=settings.BODY_IMAGE_REMOTE_API_KEY,
                timeout=config.REMOTE_TIMEOUT_SECONDS,
                total_deadline=config.REMOTE_TOTAL_DEADLINE_SECONDS,
                limiter=BodyValidationLimiter(
                    config.MAX_CONCURRENT_BODY_VALIDATIONS,
                    config.MAX_WAITING_BODY_VALIDATIONS,
                    max_wait_seconds=config.MAX_QUEUE_WAIT_SECONDS,
                ),
            )
        except RemoteWorkerConfigError as error:
            raise BodyImageRuntimeError(str(error)) from error

    def close(self) -> None:
        pass


class BodyImageRuntimeManager:
    def __init__(self) -> None:
        self._runtime: BodyImageRuntime | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()

    def start(self, **runtime_options) -> None:
        """runtime_options 는 BodyImageRuntime 에 그대로 넘긴다(Modal worker 의 미리 만든 detector)."""
        with self._lock:
            if self._runtime is not None:
                return
            try:
                if config.VALIDATION_MODE == "remote":
                    # 업로드는 worker 가 하므로 S3 설정을 보지 않는다. detector 를 넘기는 쪽은 worker 뿐이다.
                    if runtime_options:
                        raise BodyImageRuntimeError("remote 모드에는 미리 만든 모델을 넘길 수 없습니다.")
                    self._runtime = RemoteBodyImageRuntime()
                elif config.VALIDATION_MODE == "local":
                    get_s3_bucket()
                    self._runtime = BodyImageRuntime(**runtime_options)
                else:
                    raise BodyImageRuntimeError(
                        f"BODY_IMAGE_VALIDATION_MODE 는 local 또는 remote 여야 합니다: "
                        f"{config.VALIDATION_MODE!r}"
                    )
                self._error = None
            except (S3ConfigError, BodyImageRuntimeError) as error:
                self._error = error
                logger.warning("body image validation runtime is unavailable: %s", error)

    @property
    def runtime(self) -> BodyImageRuntime | RemoteBodyImageRuntime | None:
        """준비된 runtime. 아직 없거나 초기화에 실패했으면 None."""
        return self._runtime

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
