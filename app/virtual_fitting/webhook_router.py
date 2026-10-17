"""Runware → AI Webhook 엔드포인트 (9단계). POST /api/v2/webhooks/runware?token=<WEBHOOK_TOKEN>

내부 API 인증(verify_internal_key)이 아니라 URL 의 token 을 settings.WEBHOOK_TOKEN 과 비교한다(hmac.compare_digest).
확인되지 않은 HMAC 서명은 요구하지 않는다. 응답은 공통 형식({"code", "data", "message"})이며 BE 결과 JSON 과 별개다.

  200 — 처리했거나 이미 안전하게 처리된 중복·늦은 알림(가상피팅 성공이라는 뜻이 아님)
  400 INVALID_REQUEST      — JSON·필수 식별자·형식 오류, 지원하지 않는 본문
  400 WEBHOOK_UNKNOWN_TASK — 등록되지 않은 task 이거나 호출 시작 기록 없음(부작용 없음)
  401 UNAUTHORIZED         — token 누락·불일치(DB·SQS·비용·자리 변경 없음)
  500 WEBHOOK_NOT_CONFIGURED / SQS_CONFIG_ERROR / DATABASE_ERROR — 설정·DB 오류
  503 WEBHOOK_FOLLOW_UP_FAILED — 기록 뒤 후속 큐 발행 실패·UNKNOWN(재전송 때 이어서 처리)
  503 WEBHOOK_FOLLOW_UP_NOT_READY — 기록·정리는 했지만 남은 일(결과 발행 PUBLISH_RESULT·운영 확인/복구)을 이어 갈
                             실제 경로가 없다(FollowUpPaths). 결과는 보존되고 재전송 때 다시 확인한다.
  503 WEBHOOK_TIMEOUT      — 처리 시간(WEBHOOK_DEADLINE_SECONDS) 초과. 처리 thread 는 계속 돌 수 있고 모든 단계가
                             반복해도 같은 결과라 재전송이 안전하다. 200 을 먼저 주고 메모리에서만 처리하지 않는다.

토큰·URL·query·본문 전문은 로그에 남기지 않는다(접근 로그의 token 은 logging_config 가 가린다).
"""

import hmac
import logging
from collections.abc import Iterator
from contextlib import contextmanager

import anyio
import psycopg
from anyio import to_thread
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.clients.sqs import SqsConfigError, SqsQueueClient, get_webhook_sqs_client
from app.config import settings
from app.config.database import DatabaseConfigError, get_connection_pool
from app.errors import INVALID_REQUEST, ApiResponse, error_response
from app.virtual_fitting.exceptions import FittingDatabaseError, VirtualFittingError
from app.virtual_fitting.pruna_submitter import FollowUpPaths, NoFollowUpPaths
from app.virtual_fitting.runware_webhook import (
    WEBHOOK_PATH,
    EventResult,
    RunwareWebhookService,
    WebhookFollowUpNotReadyError,
    WebhookPayloadError,
    parse_webhook,
)

logger = logging.getLogger(__name__)

# Runware 권장 응답 시간(약 5초)보다 조금 짧게. DB·SQS 대기 전체에 건다.
WEBHOOK_DEADLINE_SECONDS = 4.5
# 동시에 Webhook 처리에 쓰는 worker thread 수. 시간 초과로 버려진 처리가 쌓여도 다른 요청 thread 를 다 쓰지 않게 한다.
WEBHOOK_THREADS = 8
WEBHOOK_PROCESSED = "WEBHOOK_PROCESSED"

router = APIRouter(
    tags=["virtual-fitting-webhook"],
    responses={
        code: {"model": ApiResponse}
        for code in (
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    },
)


def webhook_follow_up_paths() -> FollowUpPaths:
    """Webhook 뒤에 남는 일(결과 발행 14단계·Watchdog 복구 15단계)을 이어 갈 실제 경로.

    아직 구현되지 않았으므로 항상 NoFollowUpPaths 다. Router 가 있거나 큐 URL·환경변수가 설정됐다는 이유로 준비됐다고
    보지 않는다. 해당 단계를 구현할 때 코드로 연결한다.
    """
    return NoFollowUpPaths()


@contextmanager
def open_webhook_service() -> Iterator[RunwareWebhookService]:
    """요청 하나가 쓸 Service 를 조립한다(blocking: DB pool·boto3). worker thread 에서 부른다."""
    from app.virtual_fitting.providers.runware_async import DEFAULT_SUBMIT_TIMEOUT
    from app.virtual_fitting.repositories.async_job_repository import AsyncJobRepository
    from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository
    from app.virtual_fitting.repositories.pruna_limiter import PrunaLimiter, TaskCostRepository

    try:
        pool = get_connection_pool()
    except (DatabaseConfigError, psycopg.Error) as error:
        raise FittingDatabaseError("AI PostgreSQL 연결에 실패했습니다.") from error
    yield RunwareWebhookService(
        jobs=AsyncJobRepository(pool),
        state=AsyncJobStateRepository(pool),
        # 자리 반환만 쓴다. 제한 값은 요청 Worker 와 같은 설정을 넘긴다(반환에는 쓰이지 않음).
        limiter=PrunaLimiter(
            pool,
            max_in_flight=settings.VIRTUAL_FITTING_PRUNA_MAX_IN_FLIGHT,
            max_concurrent_sends=settings.VIRTUAL_FITTING_PRUNA_MAX_CONCURRENT_SENDS,
            min_send_interval_seconds=settings.VIRTUAL_FITTING_PRUNA_MIN_SEND_INTERVAL_SECONDS,
            send_hold_seconds=DEFAULT_SUBMIT_TIMEOUT + 30,
        ),
        costs=TaskCostRepository(pool),
        queue=SqsQueueClient(get_webhook_sqs_client()),
        follow_up=webhook_follow_up_paths(),
    )


def _process(events) -> list[EventResult]:
    with open_webhook_service() as service:
        return service.handle(events)


def _results(results) -> list[dict]:
    return [
        {
            "task_uuid": result.task_uuid,
            "outcome": result.outcome.value,
            "published": result.published.value if result.published else None,
        }
        for result in results
    ]


def _token_ok(request: Request, expected: str) -> bool:
    tokens = request.query_params.getlist("token")
    if len(tokens) != 1:
        return False
    return hmac.compare_digest(tokens[0].encode(), expected.encode())


def _thread_limiter(request: Request) -> anyio.CapacityLimiter:
    limiter = getattr(request.app.state, "webhook_thread_limiter", None)
    if limiter is None:
        limiter = anyio.CapacityLimiter(WEBHOOK_THREADS)
        request.app.state.webhook_thread_limiter = limiter
    return limiter


@router.post(WEBHOOK_PATH, response_model=ApiResponse)
async def runware_webhook(request: Request):
    expected = settings.WEBHOOK_TOKEN
    if not expected:
        logger.error("runware webhook rejected: WEBHOOK_TOKEN is not configured")
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "WEBHOOK_NOT_CONFIGURED",
            "Webhook 인증 설정이 없습니다.",
        )
    if not _token_ok(request, expected):
        logger.warning("runware webhook rejected: invalid token")
        return error_response(status.HTTP_401_UNAUTHORIZED, "UNAUTHORIZED", "인증에 실패했습니다.")

    try:
        events = parse_webhook(await request.body())
    except WebhookPayloadError as error:
        logger.warning("runware webhook rejected: unsupported body reason=%s", error.reason)
        return error_response(
            status.HTTP_400_BAD_REQUEST, INVALID_REQUEST, "지원하지 않는 Webhook 본문입니다."
        )

    try:
        with anyio.fail_after(WEBHOOK_DEADLINE_SECONDS):
            results = await to_thread.run_sync(
                _process, events, abandon_on_cancel=True, limiter=_thread_limiter(request)
            )
    except TimeoutError:
        logger.error("runware webhook timed out after %.1fs", WEBHOOK_DEADLINE_SECONDS)
        return error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "WEBHOOK_TIMEOUT",
            "알림 처리 시간이 초과되었습니다.",
        )
    except WebhookFollowUpNotReadyError as error:
        return error_response(
            error.status_code,
            error.code,
            error.message,
            data={
                "results": _results(error.results),
                "pending": [{"task_uuid": task, "next_step": step} for task, step in error.pending],
            },
        )
    except SqsConfigError:
        logger.exception("runware webhook follow-up queue is not configured")
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR, SqsConfigError.code, "후속 큐 설정이 없습니다."
        )
    except VirtualFittingError as error:
        if error.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            logger.error("runware webhook failed: %s", error.code)
        return error_response(error.status_code, error.code, error.message)

    data = {"results": _results(results)}
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ApiResponse(
            code=WEBHOOK_PROCESSED, data=data, message="알림을 처리했습니다."
        ).model_dump(),
    )
