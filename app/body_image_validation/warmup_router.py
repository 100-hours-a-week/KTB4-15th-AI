import logging
import math

from fastapi import APIRouter, Depends, FastAPI, Request, status

from app.body_image_validation.remote import RemoteWorkerConfigError
from app.body_image_validation.runtime import RemoteBodyImageRuntime
from app.body_image_validation.warmup_manager import (
    BodyImageWarmupManager,
    WarmupStatus,
    create_warmup_manager,
)
from app.config import body_image_validation as config
from app.errors import ApiResponse, error_response
from app.security import verify_internal_key

logger = logging.getLogger(__name__)

ACCEPTED = "BODY_IMAGE_WARMUP_ACCEPTED"
RATE_LIMITED = "BODY_IMAGE_WARMUP_RATE_LIMITED"
UNAVAILABLE = "BODY_IMAGE_WARMUP_UNAVAILABLE"
ACCEPTED_MESSAGE = "전신 이미지 검증 사전 활성화 요청을 접수했습니다."
RATE_LIMITED_MESSAGE = "사전 활성화 요청이 너무 많습니다. 잠시 후 다시 시도해주세요."
UNAVAILABLE_MESSAGE = "현재 사전 활성화 요청을 처리할 수 없습니다."

_ACCEPTED_STATUSES = frozenset(
    {WarmupStatus.STARTED, WarmupStatus.IN_PROGRESS, WarmupStatus.COOLDOWN}
)

router = APIRouter(
    tags=["body-image-validation"],
    dependencies=[Depends(verify_internal_key)],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ApiResponse},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ApiResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiResponse},
    },
)


@router.post(
    "/api/v1/validation/body-image/warmup",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ApiResponse,
)
async def warm_up_body_image_worker(request: Request):
    # async def: manager 의 Lock·Task 가 lifespan 과 같은 서버 event loop 에서 돈다.
    manager: BodyImageWarmupManager | None = getattr(request.app.state, "body_image_warmup", None)
    if manager is None:
        return error_response(status.HTTP_503_SERVICE_UNAVAILABLE, UNAVAILABLE, UNAVAILABLE_MESSAGE)
    acceptance = await manager.request()
    if acceptance.status in _ACCEPTED_STATUSES:
        return ApiResponse(code=ACCEPTED, message=ACCEPTED_MESSAGE)
    if acceptance.status is WarmupStatus.RATE_LIMITED:
        response = error_response(
            status.HTTP_429_TOO_MANY_REQUESTS, RATE_LIMITED, RATE_LIMITED_MESSAGE
        )
        retry_after = _retry_after_header(acceptance.retry_after)
        if retry_after is not None:
            response.headers["Retry-After"] = retry_after
        return response
    return error_response(status.HTTP_503_SERVICE_UNAVAILABLE, UNAVAILABLE, UNAVAILABLE_MESSAGE)


def _retry_after_header(seconds: float | None) -> str | None:
    """양의 유한한 초만 올림한 정수 문자열로. 그 외(None·0·음수·NaN·inf)는 헤더를 붙이지 않는다."""
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return None
    return str(math.ceil(seconds))


def start_warmup(app: FastAPI, runtime) -> None:
    """lifespan 시작에서 부른다. 조건이 맞으면 manager 를 만들어 app.state 에 둔다(아니면 None → API 는 503).

    runtime 은 runtime_manager.start() 뒤의 runtime_manager.runtime 이다. 검증 runtime 의 시작 정책은 바꾸지 않는다.
    warm-up 만의 설정 오류(RemoteWorkerConfigError)는 로그로 남기고 서버는 계속 뜬다. 그 밖의 예외는 숨기지 않는다.
    """
    app.state.body_image_warmup = _create_manager(runtime)


def _create_manager(runtime) -> BodyImageWarmupManager | None:
    if config.VALIDATION_MODE != "remote":
        logger.info("body image warm-up is not used in %r mode", config.VALIDATION_MODE)
        return None
    if not isinstance(runtime, RemoteBodyImageRuntime):
        # 검증 요청도 처리할 수 없는 상태라 GPU 를 깨우지 않는다(원인은 runtime 로그에 있다).
        logger.warning("body image warm-up is disabled: remote runtime is unavailable")
        return None
    try:
        return create_warmup_manager()
    except RemoteWorkerConfigError as error:
        # 메시지에는 설정 이름만 있고 URL·키 값은 없다.
        logger.warning("body image warm-up is disabled: %s", error)
        return None


async def close_warmup(app: FastAPI) -> None:
    """lifespan 종료에서 부른다. 새 접수를 먼저 막고(503) 진행 중인 호출을 취소해 끝날 때까지 기다린다.

    실패해도 예외를 내지 않는다(뒤의 runtime·DB pool 정리가 이어져야 한다).
    """
    manager = getattr(app.state, "body_image_warmup", None)
    app.state.body_image_warmup = None
    if manager is None:
        return
    try:
        await manager.aclose()
    except Exception as error:  # noqa: BLE001 — 종료 정리 실패가 다른 자원 정리를 막지 않게
        logger.error("body image warm-up close failed: error=%s", type(error).__name__)
