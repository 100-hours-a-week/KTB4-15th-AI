"""무거운 모델/session을 프로세스 생명주기 동안 한 번만 준비한다.

모델(inference.BodyImageModels)과 연결·요청 상태(S3 client, service 의 limiter)를 나눠 만든다.
모델은 secret 없이 먼저 만들 수 있고, 이 모듈은 S3 client 를 만들며 settings(secret)를 읽는다.
"""

import logging
import threading

from app.body_image_validation.concurrency import BodyValidationLimiter
from app.body_image_validation.inference import BodyImageModels, BodyImageRuntimeError
from app.body_image_validation.service import BodyImageValidationService
from app.clients.s3 import ImageStorage, S3ConfigError, S3ImageStorage, get_s3_bucket
from app.config import body_image_validation as config

logger = logging.getLogger(__name__)


class BodyImageRuntime:
    """local 모드 runtime: 미리 만든 모델 + S3 저장소 + 요청을 처리하는 service."""

    def __init__(self, models: BodyImageModels, storage: ImageStorage) -> None:
        """모델 소유권을 넘겨받는다(close() 가 닫는다). service 와 그 limiter(Condition)는 여기서 새로 만든다."""
        self.models = models
        self.person_detector = models.person_detector
        self.pose_detector = models.pose_detector
        self.background_remover = models.background_remover
        self.service = BodyImageValidationService(
            models.person_detector,
            models.pose_detector,
            models.background_remover,
            storage,
            gpu_preprocessor=models.gpu_preprocessor,
            diagnostics=config.DIAGNOSTICS,
        )

    @classmethod
    def create(cls, models: BodyImageModels | None = None) -> "BodyImageRuntime":
        """S3 client 를 만들고 runtime 을 조립한다.

        models 를 넘기면(Modal worker) 그 모델을 쓰고, 없으면(EC2 local) S3 client 다음에 여기서 만든다.
        어느 쪽이든 실패하면 BodyImageRuntimeError 이고, 넘겨받은 모델도 닫는다.
        """
        try:
            storage = S3ImageStorage()
        # boto3 예외도 초기화 경계에서 BodyImageRuntimeError 로 묶는다.
        except Exception as error:
            if models is not None:
                models.close()
            raise BodyImageRuntimeError("전신 이미지 runtime 초기화에 실패했습니다.") from error
        if models is None:
            models = BodyImageModels.load()
        return cls(models, storage)

    def close(self) -> None:
        self.models.close()


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

    def start(self, models: BodyImageModels | None = None) -> None:
        """runtime 을 한 번만 만든다. models 는 Modal worker 가 미리 만든 모델이다(없으면 local 에서 만든다).

        이미 runtime 이 있으면 아무것도 하지 않는다(넘긴 models 의 정리는 호출자 몫이다).
        """
        with self._lock:
            if self._runtime is not None:
                return
            try:
                if config.VALIDATION_MODE == "remote":
                    # 업로드는 worker 가 하므로 S3 설정을 보지 않는다. 모델을 넘기는 쪽은 worker 뿐이다.
                    if models is not None:
                        raise BodyImageRuntimeError("remote 모드에는 미리 만든 모델을 넘길 수 없습니다.")
                    self._runtime = RemoteBodyImageRuntime()
                elif config.VALIDATION_MODE == "local":
                    get_s3_bucket()
                    self._runtime = BodyImageRuntime.create(models)
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
