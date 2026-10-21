import hmac
import json
import logging
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import anyio
import psycopg
from anyio import to_thread
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.clients.sqs import SqsConfigError, SqsQueueClient, get_webhook_sqs_client
from app.config import settings
from app.config.database import DatabaseConfigError, get_connection_pool
from app.errors import INVALID_REQUEST, ApiResponse, error_response
from app.virtual_fitting import diagnostics as diag
from app.virtual_fitting.exceptions import FittingDatabaseError, VirtualFittingError
from app.virtual_fitting.pruna_submitter import FollowUpPaths
from app.virtual_fitting.runware_webhook import (
    WEBHOOK_PATH,
    EventResult,
    RunwareWebhookService,
    WebhookFollowUpNotReadyError,
    WebhookPayloadError,
    parse_webhook,
)

logger = logging.getLogger(__name__)

# Runware 권장 응답 시간(약 5초)보다 조금 짧게. DB·연결 풀·SQS 대기를 모두 포함한 처리 전체에 건다.
# 넘기면 503 WEBHOOK_TIMEOUT 으로 먼저 응답하고, 처리 thread 는 계속 돌 수 있다(모든 단계가 반복해도 같은 결과).
# Webhook 전용 SQS client 의 timeout(app.clients.sqs, 연결 1초·응답 2초)은 이보다 짧게 맞춘다.
WEBHOOK_DEADLINE_SECONDS = 4.5
# 동시에 Webhook 처리에 쓰는 worker thread 수. 시간 초과로 버려진 처리가 쌓여도 다른 요청 thread 를 다 쓰지 않게 한다.
WEBHOOK_THREADS = 8
WEBHOOK_PROCESSED = "WEBHOOK_PROCESSED"
# 진단: 이 HTTP 요청의 내부 식별자. worker thread 로도 이어진다(anyio 가 context 를 복사). 외부 값이 아니다.
_REQUEST_ID: ContextVar[str | None] = ContextVar("vf_webhook_request_id", default=None)

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
    """Webhook 뒤에 남는 일을 이어 갈 실제 경로.

    결과 발행(PUBLISH_RESULT, 14단계)은 결과 전송 루프가 실제로 돌고 있고 그 결과가 PENDING 일 때만 준비됐다고 본다
    (ResultPublishPaths). 루프가 꺼져 있거나 멈췄으면 준비되지 않음. 운영 확인·복구(RECOVER_UNKNOWN, 15단계 Watchdog)는
    아직 준비되지 않음. 큐 URL·환경변수 설정만으로 준비됐다고 보지 않는다.
    """
    from app.virtual_fitting.watchdog import follow_up_paths

    return follow_up_paths()


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
        source="webhook",
        request_id=_REQUEST_ID.get(),
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


def _validation(request_id: str, stage: str, result: str, reason: str, **fields) -> None:
    diag.event(
        "webhook_validation",
        logging.INFO if result == "passed" else logging.WARNING,
        component="webhook",
        source="webhook",
        request_id=request_id,
        stage=stage,
        result=result,
        reason=reason,
        **fields,
    )


# 진단: 수신 헤더 관찰(이름·존재 여부만). Runware 가 실제로 어떤 인증·서명 헤더를 보내는지 확인하려는 것이며
# 헤더가 있다고 Runware 발신이나 서명이 유효하다는 뜻은 아니다. 인증은 기존 쿼리 token 검증만 한다.
_SIGNAL_HEADERS = (
    "authorization",
    "x-webhook-token",
    "x-webhook-signature",
    "x-signature",
    "webhook-signature",
    "webhook-id",
    "webhook-timestamp",
)
# HTTP header 이름(token 문자)만. 그 밖(제어문자·공백·비 ASCII 등)은 이름도 남기지 않고 invalid 로 센다.
_HEADER_NAME = re.compile(r"[a-z0-9!#$%&'*+.^_`|~-]{1,64}")
_MAX_HEADER_NAMES = 40
_HEADER_NAMES_BUDGET = 230  # diag 값 길이 상한(240) 안에서 이름 중간이 잘리지 않게


def _header_fields(request: Request) -> dict:
    """수신 헤더의 이름 목록·대표 인증/서명 헤더 존재 여부·쿼리 token 개수. 헤더 값·쿼리 값은 읽지 않는다."""
    try:
        names = []
        for raw_name, _value in request.headers.raw:  # 값은 쓰지 않는다
            name = raw_name.decode("latin-1").lower()
            names.append(name if _HEADER_NAME.fullmatch(name) else "invalid")
        unique = list(dict.fromkeys(names))
        shown, used = [], 0
        for name in unique[:_MAX_HEADER_NAMES]:
            if used + len(name) + 1 > _HEADER_NAMES_BUDGET:
                break
            shown.append(name)
            used += len(name) + 1
        present = set(unique)
        tokens = len(request.query_params.getlist("token"))
        return {
            "header_count": len(names),
            "header_names": ",".join(shown) or "none",
            "header_names_truncated": len(shown) < len(unique),
            # <이름>_present 형식(has_<이름>= 이면 "token=" 같은 비밀값 패턴처럼 보여 검사·검색에 걸린다)
            **{f"{name.replace('-', '_')}_present": name in present for name in _SIGNAL_HEADERS},
            "query_token_present": tokens > 0,
            "query_token_count": tokens,
        }
    except Exception:  # noqa: BLE001 — 진단 값만 포기
        return {"header_names": "unavailable"}


def _token_reason(request: Request) -> str:
    count = len(request.query_params.getlist("token"))  # 개수만 본다(값은 남기지 않음)
    return "token_missing" if count == 0 else "token_multiple" if count > 1 else "token_mismatch"


def _response_code(response) -> str:
    """이 엔드포인트가 만든 응답의 code(우리가 만든 JSON). 읽지 못하면 unknown."""
    try:
        return str(json.loads(response.body).get("code") or "unknown")
    except Exception:  # noqa: BLE001 — 진단 값만 포기
        return "unknown"


@router.post(WEBHOOK_PATH, response_model=ApiResponse)
async def runware_webhook(request: Request):
    """진단: 수신(webhook_http_received) → 검증(webhook_validation) → 처리(runware_result_processed, thread)
    → 응답(webhook_http_completed) 을 내부 request_id 하나로 잇는다. 인증·본문 검증 전에 수신을 남긴다."""
    request_id, started = uuid.uuid4().hex[:16], time.monotonic()
    context = _REQUEST_ID.set(request_id)
    diag.event(
        "webhook_http_received",
        component="webhook",
        source="webhook",
        request_id=request_id,
        **_header_fields(request),
    )
    http_status, code = status.HTTP_500_INTERNAL_SERVER_ERROR, "unhandled_exception"
    try:
        response = await _handle(request, request_id)
        http_status, code = response.status_code, _response_code(response)
        return response
    finally:
        diag.event(
            "webhook_http_completed",
            logging.INFO
            if http_status < status.HTTP_500_INTERNAL_SERVER_ERROR
            else logging.WARNING,
            component="webhook",
            source="webhook",
            request_id=request_id,
            http_status=http_status,
            code=code,
            elapsed_ms=diag.elapsed_ms(started),
        )
        _REQUEST_ID.reset(context)


async def _handle(request: Request, request_id: str):
    expected = settings.WEBHOOK_TOKEN
    if not expected:
        logger.error("runware webhook rejected: WEBHOOK_TOKEN is not configured")
        _validation(request_id, "config", "rejected", "token_not_configured")
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "WEBHOOK_NOT_CONFIGURED",
            "Webhook 인증 설정이 없습니다.",
        )
    if not _token_ok(request, expected):
        logger.warning("runware webhook rejected: invalid token")
        _validation(request_id, "token", "rejected", _token_reason(request))
        return error_response(status.HTTP_401_UNAUTHORIZED, "UNAUTHORIZED", "인증에 실패했습니다.")

    try:
        events = parse_webhook(await request.body())
    except WebhookPayloadError as error:
        logger.warning("runware webhook rejected: unsupported body reason=%s", error.reason)
        _validation(request_id, "body", "rejected", error.reason)
        return error_response(
            status.HTTP_400_BAD_REQUEST, INVALID_REQUEST, "지원하지 않는 Webhook 본문입니다."
        )

    tasks = list(dict.fromkeys(event.task_uuid for event in events))  # 검증된 UUID 만(정규화된 값)
    _validation(
        request_id, "body", "passed", "ok", event_count=len(events), task_uuid=",".join(tasks)
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
