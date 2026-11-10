import json
import logging
import re
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC
from decimal import Decimal
from enum import Enum
from typing import Any
from urllib.parse import parse_qs, urlsplit

import psycopg

from app.clients.sqs import FittingQueue, SqsConfigError, get_queue_url
from app.virtual_fitting import diagnostics as diag
from app.virtual_fitting.exceptions import (
    FittingDatabaseError,
    FittingModelError,
    VirtualFittingError,
)
from app.virtual_fitting.providers.runware_async import parse_retry_after
from app.virtual_fitting.providers.runware_errors import (  # noqa: F401 (기존 import 경로 유지용 재노출)
    AUTH_ERROR_CODES,
    FALLBACK_ERROR_CODES,
    TaskErrorKind,
    classify_task_error,
)
from app.virtual_fitting.providers.runware_status import image_ref
from app.virtual_fitting.pruna_submitter import (
    FollowUpPaths,
    NextStep,
    NoFollowUpPaths,
    auth_hold_active,
    awaiting_result_url,
    failure_payload,
    next_step_for,
)
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncFittingJob,
    AsyncJobStatus,
    NextAction,
    PrunaAttempt,
)
from app.virtual_fitting.repositories.async_job_state import (
    NotAcceptedOutcome,
    NotAcceptedSource,
)

logger = logging.getLogger(__name__)

WEBHOOK_PATH = "/api/v2/webhooks/runware"
TASK_TYPE = "imageInference"
MAX_BODY_BYTES = 256 * 1024
# 오류 코드 분류는 providers/runware_errors.py 한 곳(제출 응답·Webhook·getResponse 공통, 공식 문서 근거만).
# FALLBACK_ERROR_CODES(timeoutProvider)·AUTH_ERROR_CODES(invalidApiKey)는 기존 이름 그대로 다시 내보낸다.
# retryable 표시·retryAfter 만으로는 유료 fallback 을 허용하지 않는다. message 문자열로 원인을 추측하지 않는다.
_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
# Webhook 의 DB 처리(작업 조회, 성공 결과의 상태·참조 기록) 재시도: 최초 1회 + 재시도 3회. 대기는 짧게(응답 시한 4.5초 안).
DB_ATTEMPTS = 4
DB_RETRY_DELAYS = (0.05, 0.1, 0.2)
# 다시 하면 풀릴 수 있는 PostgreSQL 오류(SQLSTATE). 연결 예외(08xxx)는 접두사로 본다.
_TRANSIENT_SQLSTATES = frozenset(
    {
        "40001",  # serialization_failure
        "40P01",  # deadlock_detected
        "55P03",  # lock_not_available (기존 lock_timeout 5초 초과)
        "57014",  # query_canceled (statement timeout 등)
        "57P01",  # admin_shutdown
        "57P02",  # crash_shutdown
        "57P03",  # cannot_connect_now
        "53300",  # too_many_connections
    }
)


def transient_db_error(error: BaseException) -> bool:
    """DB 오류가 일시 장애인지. 저장소가 감싼 FittingDatabaseError 의 원인(psycopg 예외)의 SQLSTATE 로 판단한다.

    일시: 위 SQLSTATE·연결 예외(08xxx), SQLSTATE 없는 OperationalError(연결 끊김·commit 응답 유실·pool 대기 초과
    PoolTimeout). 그 밖(스키마·SQL 문법·권한·제약 위반·데이터 오류, 원인 없는 오류)은 반복해도 같아 재시도하지 않는다.
    """
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, psycopg.Error):
            state = current.sqlstate
            if state is None:
                return isinstance(current, psycopg.OperationalError)
            return state.startswith("08") or state in _TRANSIENT_SQLSTATES
        current = current.__cause__
    return False


class EventKind(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PROCESSING = "processing"


@dataclass(frozen=True)
class WebhookEvent:
    """내부 처리용 공통 구조. 외부 본문 전문·오류 message 는 담지 않는다."""

    task_uuid: str
    kind: EventKind
    error_code: str | None = None
    # 명시적인 유효 비용(USD). 없거나 잘못됐으면 None(0 으로 확정하지 않음).
    cost: Decimal | None = None
    cost_invalid: bool = False
    # 성공 결과 참조(imageURL·imageUUID). DB 저장용이며 로그·repr 에 URL 을 남기지 않는다.
    # 저장할 수 없는 값이면 None 이고 image_ref_invalid=True(성공 판정은 그대로).
    image_url: str | None = field(default=None, repr=False)
    image_uuid: str | None = None
    image_ref_invalid: bool = False
    # 내부 호출 경로 표시: Watchdog 이 우리 인증으로 이 task 를 직접 조회(getResponse)한 결과다. parse_webhook(외부 본문)은
    # 이 값을 정하지 않는다(항상 False). AI 저장 작업의 빈 참조를 이 근거로 처음 채울 수 있다.
    from_status_query: bool = False
    # 이 task 오류 항목의 대기 안내(retryAfter, 초). 확실한 미접수의 다음 재요청 예약에만 쓴다(접수·실패 판단 근거 아님).
    # 없거나 무효면 None(기본 대기), 유효하지만 표현할 수 없게 크면 retry_after_unrepresentable(보류).
    retry_after: float | None = None
    retry_after_unrepresentable: bool = False

    @property
    def error_kind(self) -> TaskErrorKind | None:
        """생성 task 결과(이 task 에 연결된 Webhook·getResponse 의 errors)의 분류. 실패 이벤트가 아니면 None."""
        return classify_task_error(self.error_code) if self.kind is EventKind.FAILED else None

    @property
    def recoverable(self) -> bool:
        return self.error_kind is TaskErrorKind.GENERATION_FALLBACK

    @property
    def auth_error(self) -> bool:
        return self.error_kind is TaskErrorKind.AUTH

    @property
    def unresolved(self) -> bool:
        """오류를 받았지만 생성 실패로 확정할 근거가 없다(확실한 미접수 또는 근거 부족). 상태를 바꾸지 않는다."""
        return self.error_kind in (TaskErrorKind.NOT_ACCEPTED, TaskErrorKind.UNCLEAR)


class WebhookPayloadError(ValueError):
    """지원하지 않는 본문. reason 은 로그용 짧은 코드다(본문 내용은 담지 않음)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class WebhookUnknownTaskError(VirtualFittingError):
    """등록되지 않은 task 이거나 호출 시작 기록이 없다. 유효한 완료로 처리하지 않는다(부작용 없음)."""

    status_code = 400
    code = "WEBHOOK_UNKNOWN_TASK"
    message = "등록되지 않은 작업의 알림입니다."


class WebhookFollowUpError(VirtualFittingError):
    """기록은 했지만 후속 큐 발행이 실패하거나 결과를 모른다. 5xx 로 답해 Runware 가 다시 보내게 한다."""

    status_code = 503
    code = "WEBHOOK_FOLLOW_UP_FAILED"
    message = "알림의 후속 작업을 등록하지 못했습니다."


class WebhookResultRefNotSavedError(FittingDatabaseError):
    """AI 저장 작업이 받은 유효한 결과 URL 을 DB 에 저장하지 못했다(저장됐는지도 확인 못 함). 다운로드에 꼭 필요한 기록이라
    완료로 답하지 않고 503 으로 답해 Runware 가 다시 보내게 한다. 충돌·근거 부족·허용하지 않는 URL 은 여기에 해당하지 않는다."""

    status_code = 503
    code = "WEBHOOK_RESULT_REF_NOT_SAVED"
    message = "알림의 결과 참조를 저장하지 못했습니다."

    def __init__(self) -> None:
        super().__init__(self.message)


class WebhookFollowUpNotReadyError(VirtualFittingError):
    """기록은 끝났지만 남은 일(결과 발행·운영 확인/복구)을 이어 갈 실제 경로가 없다. 완료(200)로 응답하지 않는다."""

    status_code = 503
    code = "WEBHOOK_FOLLOW_UP_NOT_READY"
    message = "알림은 기록했지만 이어서 처리할 경로가 아직 준비되지 않았습니다."

    def __init__(
        self, results: Sequence["EventResult"], pending: Sequence[tuple[str, str]]
    ) -> None:
        super().__init__(self.message)
        self.results = list(results)
        # (task_uuid, 준비되지 않은 다음 일) 목록
        self.pending = list(pending)


class _NonFinite:
    """본문의 NaN·Infinity 표기(숫자로 보지 않음)."""

    def __init__(self, text: str) -> None:
        self.text = text


# --- 본문 해석 ---


def _load(body: bytes) -> Any:
    if len(body) > MAX_BODY_BYTES:
        raise WebhookPayloadError("body_too_large")
    try:
        # 금액은 float 를 거치지 않고 Decimal 로 읽는다.
        return json.loads(body, parse_float=Decimal, parse_int=Decimal, parse_constant=_NonFinite)
    except (ValueError, UnicodeDecodeError) as error:
        raise WebhookPayloadError("invalid_json") from error


def _task_uuid(item: dict) -> str:
    value = item.get("taskUUID")
    if not isinstance(value, str):
        raise WebhookPayloadError("invalid_task_uuid")
    try:
        return str(uuid.UUID(value))
    except ValueError as error:
        raise WebhookPayloadError("invalid_task_uuid") from error


def _check_task_type(item: dict, *, required: bool) -> None:
    task_type = item.get("taskType")
    if task_type is None and not required:
        return
    if task_type != TASK_TYPE:
        raise WebhookPayloadError("unsupported_task_type")


def _cost(item: dict) -> tuple[Decimal | None, bool]:
    if "cost" not in item or item["cost"] is None:
        return None, False
    value = item["cost"]
    if isinstance(value, bool) or not isinstance(value, Decimal | int):
        return None, True
    cost = Decimal(value)
    if not cost.is_finite() or cost < 0:
        return None, True
    return cost, False


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _result_event(item: Any) -> WebhookEvent:
    if not isinstance(item, dict):
        raise WebhookPayloadError("invalid_item")
    _check_task_type(item, required=True)
    task = _task_uuid(item)
    status = item.get("status")
    cost, cost_invalid = _cost(item)
    if status == "processing":
        return WebhookEvent(task, EventKind.PROCESSING, cost=None)
    if status not in (None, "success"):
        raise WebhookPayloadError("unsupported_status")
    if not (_non_empty_str(item.get("imageURL")) or _non_empty_str(item.get("imageUUID"))):
        raise WebhookPayloadError("unclear_completion")
    url, image_uuid, invalid = image_ref(item)
    return WebhookEvent(
        task,
        EventKind.SUCCEEDED,
        cost=cost,
        cost_invalid=cost_invalid,
        image_url=url,
        image_uuid=image_uuid,
        image_ref_invalid=invalid,
    )


def _error_event(item: Any) -> WebhookEvent:
    if not isinstance(item, dict):
        raise WebhookPayloadError("invalid_item")
    _check_task_type(item, required=False)
    task = _task_uuid(item)
    code = item.get("code")
    if not isinstance(code, str) or not _CODE_PATTERN.match(code):
        raise WebhookPayloadError("invalid_error_code")
    cost, cost_invalid = _cost(item)
    retry = parse_retry_after([item["retryAfter"]] if item.get("retryAfter") is not None else [])
    return WebhookEvent(
        task,
        EventKind.FAILED,
        code,
        cost=cost,
        cost_invalid=cost_invalid,
        retry_after=retry.seconds,
        retry_after_unrepresentable=retry.unrepresentable,
    )


def parse_webhook(body: bytes) -> tuple[WebhookEvent, ...]:
    """지원하는 세 형식만 공통 구조로 바꾼다. 하나라도 지원하지 않는 항목이 있으면 전체를 거절한다(부분 처리 없음)."""
    data = _load(body)
    if not isinstance(data, dict):
        raise WebhookPayloadError("not_object")
    if "data" not in data and "errors" not in data:
        return (_result_event(data),)
    events: list[WebhookEvent] = []
    for key, convert in (("data", _result_event), ("errors", _error_event)):
        if key not in data:
            continue
        items = data[key]
        if not isinstance(items, list):
            raise WebhookPayloadError("unsupported_shape")
        events.extend(convert(item) for item in items)
    if not events:
        raise WebhookPayloadError("empty")
    return _without_ambiguous_refs(_without_conflicting_results(events))


def _without_conflicting_results(events: Sequence[WebhookEvent]) -> list[WebhookEvent]:
    """한 본문에 같은 task 의 최종 결과가 서로 어긋나면(성공과 실패가 함께, 또는 서로 다른 실패 코드) 어느 쪽도 고르지
    않는다. 그 task 의 이벤트를 상태를 바꾸지 않는 진행 중 알림 하나로 바꾸고(같은 task 의 getResponse 조회가 확인한다)
    경고만 남긴다. 다른 task 의 이벤트는 그대로 둔다."""
    finals: dict[str, set] = {}
    for event in events:
        if event.kind is not EventKind.PROCESSING:
            finals.setdefault(event.task_uuid, set()).add((event.kind, event.error_code))
    conflicting = {task for task, results in finals.items() if len(results) > 1}
    processing = {event.task_uuid for event in events if event.kind is EventKind.PROCESSING}
    conflicting.update(
        event.task_uuid
        for event in events
        if event.kind is EventKind.FAILED and event.task_uuid in processing
    )
    if not conflicting:
        return list(events)
    logger.warning(
        "runware webhook conflicting results ignored task_uuid=%s", ",".join(sorted(conflicting))
    )
    kept: list[WebhookEvent] = []
    for event in events:
        if event.task_uuid not in conflicting:
            kept.append(event)
        elif not any(e.task_uuid == event.task_uuid for e in kept):
            kept.append(WebhookEvent(event.task_uuid, EventKind.PROCESSING))
    return kept


def _without_ambiguous_refs(events: Sequence[WebhookEvent]) -> tuple[WebhookEvent, ...]:
    """한 본문에 같은 task 의 성공 결과가 서로 다른 참조로 여럿 있으면(요청은 1개) 첫 항목을 고르지 않는다.
    그 task 의 성공 이벤트 참조를 모두 비우고 image_ref_invalid 로 표시한다(성공 판정은 그대로)."""
    seen: dict[str, set] = {}
    for event in events:
        if event.kind is EventKind.SUCCEEDED and (event.image_url or event.image_uuid):
            seen.setdefault(event.task_uuid, set()).add((event.image_url, event.image_uuid))
    ambiguous = {task for task, refs in seen.items() if len(refs) > 1}
    return tuple(
        replace(event, image_url=None, image_uuid=None, image_ref_invalid=True)
        if event.kind is EventKind.SUCCEEDED and event.task_uuid in ambiguous
        else event
        for event in events
    )


# --- 토큰·URL ---


def webhook_url_matches_token(url: str, token: str) -> bool:
    """8단계가 Runware 에 넘기는 URL 이 이 서버의 Webhook 경로이고 token 이 서버 검증값과 같은지."""
    import hmac

    if not url or not token:
        return False
    parts = urlsplit(url)
    tokens = parse_qs(parts.query, keep_blank_values=True).get("token", [])
    return (
        parts.scheme == "https"
        and bool(parts.netloc)
        and parts.path == WEBHOOK_PATH
        and len(tokens) == 1
        and hmac.compare_digest(tokens[0].encode(), token.encode())
    )


# --- 처리 ---


class EventOutcome(str, Enum):
    APPLIED = "applied"  # 이번 알림으로 상태를 바꿨다
    DUPLICATE = "duplicate"  # 같은 결과가 이미 기록돼 있다
    IGNORED = "ignored"  # 이미 다른 경로로 진행됐거나 종료됐다(상태·결과 보존)
    HELD = "held"  # Runware 인증 오류: 상태를 바꾸지 않고 운영 확인 대상으로 보류
    PROCESSING = "processing"  # 진행 중 알림(아무것도 바꾸지 않음)
    # 오류 알림이지만 생성 실패로 확정할 근거가 없다(확실한 미접수·근거 부족 코드). 상태·결과를 바꾸지 않는다.
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class EventResult:
    task_uuid: str
    outcome: EventOutcome
    published: NextAction | None = None


def _follow_up_action(event: WebhookEvent, job: AsyncFittingJob | None) -> NextAction | None:
    """현재 DB 상태가 이 이벤트의 후속 작업을 기다리면 그 action. PUBLISH_RESULT 는 여기서 발행하지 않는다(14단계)."""
    if job is None or job.status in (AsyncJobStatus.COMPLETED, AsyncJobStatus.FAILED):
        return None
    if (
        event.kind is EventKind.SUCCEEDED
        and job.status is AsyncJobStatus.POSTPROCESSING
        and job.next_action is NextAction.POSTPROCESS
    ):
        return NextAction.POSTPROCESS
    if (
        event.recoverable
        and job.status is AsyncJobStatus.FALLBACK_RUNNING
        and job.next_action is NextAction.RUN_FALLBACK
    ):
        return NextAction.RUN_FALLBACK
    return None


def _diag_job(job: AsyncFittingJob | None) -> dict[str, str]:
    """DB 가 돌려준 현재 상태(추가 조회 없음)."""
    if job is None:
        return {"job_status": "unknown", "next_action": "unknown"}
    return {
        "job_status": job.status.value,
        "next_action": job.next_action.value if job.next_action else "none",
    }


def _already_recorded(event: WebhookEvent, job: AsyncFittingJob | None) -> bool:
    if job is None:
        return False
    if event.kind is EventKind.SUCCEEDED:
        return job.status is AsyncJobStatus.POSTPROCESSING
    if event.recoverable:
        return job.status is AsyncJobStatus.FALLBACK_RUNNING
    payload = job.result_payload or {}
    return job.next_action is NextAction.PUBLISH_RESULT and payload.get("status") == "FAILED"


def _needs_result_url(job: AsyncFittingJob | None) -> bool:
    """AI 저장 작업이고 아직 결과 이미지 저장(verified_image_key)이 끝나지 않았다: 결과 URL 이 다운로드에 필요하다."""
    return job is not None and job.ai_stores_result and job.verified_image_key is None


def _awaiting_result_url(job: AsyncFittingJob | None) -> bool:
    """AI 저장 작업이 후처리 대기 중인데 결과 URL 이 없다(판단은 pruna_submitter.awaiting_result_url 한 곳)."""
    return awaiting_result_url(job)


def _image_ref_result(event: WebhookEvent, applied: bool, job: AsyncFittingJob | None) -> str:
    """성공 알림의 결과 참조가 DB 에 어떻게 남았는지(로그·진단용 코드, URL 은 담지 않음).

    stored      — 이번 성공 전환과 함께 저장했다
    absent      — 알림에 저장할 결과 참조가 없다
    same        — 전환되지 않았고(중복 등) 이미 같은 값이 있다(멱등)
    conflict    — 전환되지 않았고 이미 다른 값이 있다(덮어쓰지 않음, 운영 확인 대상)
    unconfirmed — 빈 컬럼에 넣을 값은 있지만 이미 있는 참조와 일치하는 값이 없어 같은 결과라는 근거가 없다(연결하지 않음)
    not_stored  — 빈 컬럼에 넣을 값이 있다. 후처리 중이면 보완을 시도하고(filled·fill_failed), 아니면(늦은 알림·
                  완료·fallback 등) 상태를 되돌리지 않으려고 저장하지 않는다
    """
    incoming = {"url": event.image_url, "uuid": event.image_uuid}
    if all(value is None for value in incoming.values()):
        return "absent"
    if applied:
        return "stored"
    stored = {
        "url": job.runware_image_url if job else None,
        "uuid": job.runware_image_uuid if job else None,
    }
    pairs = [(incoming[k], stored[k]) for k in incoming if incoming[k] is not None]
    if any(old is not None and old != new for new, old in pairs):
        return "conflict"
    if any(old is None for _, old in pairs):
        anchored = any(old == new for new, old in pairs)
        has_stored = any(value is not None for value in stored.values())
        return "unconfirmed" if has_stored and not anchored else "not_stored"
    return "same"


def follow_up_message(job: AsyncFittingJob, action: NextAction) -> dict[str, Any]:
    """ai-fitting-response 메시지. 이미지 Key·입력 등은 후처리 Worker 가 DB 에서 조회한다."""
    if action not in (NextAction.POSTPROCESS, NextAction.RUN_FALLBACK):
        raise ValueError("ai-fitting-response 는 POSTPROCESS 또는 RUN_FALLBACK 만 보냅니다.")
    return {
        "fitting_job_id": job.fitting_job_id,
        "task_uuid": job.task_uuid,
        "action": action.value,
    }


def _awaiting_task_result(job: AsyncFittingJob) -> bool:
    """아직 이 비동기 task 의 결과를 기다리는 중이다(결과 저장·후속 단계 진행 전)."""
    return (
        job.status in (AsyncJobStatus.RECEIVED, AsyncJobStatus.WAITING_PRUNA)
        and job.result_payload is None
        and job.next_action in (None, NextAction.SUBMIT_PRUNA)
    )


def _pending_step(event: WebhookEvent, job: AsyncFittingJob | None) -> NextStep | None:
    """이 알림 뒤에 남은 일 중 실제 경로가 있어야 하는 것. 후처리 큐로 넘긴 일(POSTPROCESS·RUN_FALLBACK)은 제외."""
    if job is None:
        return None
    if event.kind is EventKind.SUCCEEDED and _awaiting_result_url(job):
        return NextStep.RECOVER_RESULT_REF  # Watchdog 이 실제로 돌 때만 완료(2xx)로 답한다
    step = next_step_for(job)
    if step in (NextStep.PUBLISH_RESULT, NextStep.OPERATOR_REVIEW, NextStep.RESUBMIT_PRUNA):
        # OPERATOR_REVIEW: DB 의 인증 오류 보류 기록(auth_hold_*)으로 판단한다. 요청 Worker 재수신과 같은 기록·같은 판단.
        # Watchdog 이 돌아도 인증 오류가 해결된 것은 아니므로 자동 경로가 준비됐다고 보지 않는다.
        return step
    return None


# 진단(vf_diag runware_result_processed)의 outcome. 이 알림 한 건의 처리 결과이며 가상피팅 완료를 뜻하지 않는다
# (남은 일은 next_action 으로 본다). 기존 EventOutcome·후속 발행·경로 확인 결과로만 정한다.
#   applied  — 이번 알림으로 상태를 바꿨고, 필요한 후속 발행도 성공했다
#   ignored  — 이미 같은 결과가 있거나(duplicate) 다른 경로로 진행·종료됐다(상태·결과 보존)
#   deferred — 진행 중 알림, 인증 오류 보류, 또는 기록은 했지만 남은 일을 이어 갈 경로가 아직 없다
#   failed   — 처리 중 오류. state_change·follow_up 으로 어디까지 됐는지 구분한다(예: 상태 기록 뒤 발행 실패)
_DIAG_OUTCOME = {
    EventOutcome.APPLIED: ("applied", "state_recorded"),
    EventOutcome.DUPLICATE: ("ignored", "duplicate"),
    EventOutcome.IGNORED: ("ignored", "already_progressed"),
    EventOutcome.HELD: ("deferred", "auth_hold"),
    EventOutcome.PROCESSING: ("deferred", "still_processing"),
    EventOutcome.UNRESOLVED: ("deferred", "error_unresolved"),
}


def _diag_failure_reason(error: Exception) -> str:
    if isinstance(error, WebhookFollowUpError):
        return "follow_up_publish_failed"
    if isinstance(error, VirtualFittingError):
        return error.code.lower()
    return type(error).__name__


class RunwareWebhookService:
    def __init__(
        self,
        *,
        jobs: Any,
        state: Any,
        limiter: Any,
        costs: Any,
        queue: Any,
        follow_up: FollowUpPaths | None = None,
        source: str = "unknown",
        request_id: str | None = None,
        db_deadline_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._jobs, self._state, self._limiter = jobs, state, limiter
        self._costs, self._queue = costs, queue
        # 기본은 NoFollowUpPaths(결과 발행 14단계·Watchdog 15단계 미구현). 설정값으로 바꿀 수 없다.
        self._follow_up = follow_up or NoFollowUpPaths()
        # 진단용 출처. 조립하는 쪽이 명시한다(HTTP: webhook, Watchdog: watchdog). 그 밖은 unknown(추측하지 않음).
        self._source, self._request_id = source, request_id
        # DB 재시도: 이 시각(응답 시한) 뒤에는 새 시도를 시작하지 않는다(HTTP 는 시한 뒤 503 으로 이미 답했다). None 이면 횟수만.
        self._clock, self._sleep = clock, sleep
        self._db_deadline = clock() + db_deadline_seconds if db_deadline_seconds else None

    def _diag_processed(
        self, event: WebhookEvent, trace: dict, outcome: str, reason: str, started: float
    ) -> None:
        fields = {"request_id": self._request_id} if self._request_id else {}
        diag.event(
            "runware_result_processed",
            logging.WARNING if outcome == "failed" else logging.INFO,
            component="runware_result",
            source=self._source,
            **fields,
            fitting_job_id=trace.get("fitting_job_id", "unknown"),
            task_uuid=event.task_uuid,
            event_kind=event.kind.value,
            outcome=outcome,
            reason=reason,
            state_change=trace.get("state_change", "unknown"),
            follow_up=trace.get("follow_up", "unknown"),
            **(
                {"follow_up_message_id": trace["follow_up_message_id"]}
                if trace.get("follow_up_message_id")
                else {}
            ),
            job_status=trace.get("job_status", "unknown"),
            next_action=trace.get("next_action", "unknown"),
            **({"image_ref": trace["image_ref"]} if trace.get("image_ref") else {}),
            elapsed_ms=diag.elapsed_ms(started),
        )

    def handle(self, events: Sequence[WebhookEvent]) -> list[EventResult]:
        started = time.monotonic()
        jobs: dict[str, AsyncFittingJob] = {}
        # 같은 작업의 이전 전송 시도(확실한 미접수 뒤 새 task 로 재요청)의 늦은 알림. 현재 task 로 적용하지 않는다.
        past: dict[str, PrunaAttempt] = {}
        for event in events:  # 사전 확인: 조회만 한다
            if event.task_uuid in jobs:
                continue
            job = self._with_db_retry(
                "find_job", lambda event=event: self._jobs.find_by_task_uuid(event.task_uuid)
            )
            if job is None:
                attempt = self._with_db_retry(
                    "find_attempt", lambda event=event: self._jobs.find_attempt(event.task_uuid)
                )
                if attempt is not None:
                    job = self._with_db_retry(
                        "find_job",
                        lambda attempt=attempt: self._jobs.find_by_fitting_job_id(
                            attempt.fitting_job_id
                        ),
                    )
                    if job is not None and job.task_uuid != event.task_uuid:
                        past[event.task_uuid] = attempt
            if job is None or job.pruna_submit_started_at is None:
                logger.warning(
                    "runware webhook for unknown task task_uuid=%s registered=%s",
                    event.task_uuid,
                    job is not None,
                )
                trace = {"fitting_job_id": job.fitting_job_id if job else "none"}
                trace |= {"state_change": "none", "follow_up": "none"}
                self._diag_processed(event, trace, "failed", "unknown_task", started)
                raise WebhookUnknownTaskError()
            jobs[event.task_uuid] = job
        if any(event.kind is not EventKind.PROCESSING for event in events):
            try:
                get_queue_url(
                    FittingQueue.AI_FITTING_RESPONSE
                )  # 설정 오류면 기록 전에 멈춘다(SqsConfigError)
            except SqsConfigError:
                for event in events:
                    trace = {"fitting_job_id": jobs[event.task_uuid].fitting_job_id}
                    trace |= {"state_change": "none", "follow_up": "none"}
                    self._diag_processed(event, trace, "failed", "follow_up_queue_config", started)
                raise
        results, pending = [], []
        for event in events:
            trace: dict[str, Any] = {"fitting_job_id": jobs[event.task_uuid].fitting_job_id}
            try:
                if event.task_uuid in past:
                    result, current = self._apply_past(
                        event, past[event.task_uuid], jobs[event.task_uuid], trace
                    )
                else:
                    result, current = self._apply(event, jobs[event.task_uuid], trace)
            except Exception as error:
                self._diag_processed(event, trace, "failed", _diag_failure_reason(error), started)
                raise
            results.append(result)
            step = None if event.task_uuid in past else _pending_step(event, current)
            if step is not None and not self._follow_up.ready_for(current, step):
                pending.append((event.task_uuid, step.value))
                self._diag_processed(
                    event, trace, "deferred", f"follow_up_not_ready_{step.value}", started
                )
            else:
                self._diag_processed(event, trace, *_DIAG_OUTCOME[result.outcome], started)
        if pending:
            logger.warning(
                "runware webhook recorded but follow-up path not ready pending=%s",
                ",".join(f"{task}:{step}" for task, step in pending),
            )
            raise WebhookFollowUpNotReadyError(results, pending)
        return results

    def _apply(
        self, event: WebhookEvent, job: AsyncFittingJob, trace: dict | None = None
    ) -> tuple[EventResult, AsyncFittingJob | None]:
        # trace: 진단용. 어디까지 됐는지(상태 기록·후속 발행)를 단계마다 적어 두면 예외가 나도 handle 이 남긴다.
        trace = trace if trace is not None else {}
        job_id, task = job.fitting_job_id, event.task_uuid
        if event.kind is EventKind.PROCESSING:
            # 미접수로 끝났다고 기록한 시도에 진행 근거가 왔다. 기존 모순 기록으로 예약된 재요청을 보류한다.
            self._with_db_retry(
                "record_conflict",
                lambda: self._state.record_attempt_contradiction(task, "processing_after_not_accepted"),
            )
            job = self._jobs.find_by_fitting_job_id(job_id) or job
            logger.info("runware webhook processing fitting_job_id=%s task_uuid=%s", job_id, task)
            trace |= {"state_change": "processing", "follow_up": "not_needed"}
            trace |= _diag_job(job)
            return EventResult(task, EventOutcome.PROCESSING), job

        if event.auth_error:
            # 상태·결과·next_action 은 바꾸지 않는다. 결과를 기다리는 작업이면 보류 기록(auth_hold_*)만 남긴다(처음 한 번).
            # 기록이 실패하면 예외(FittingDatabaseError)로 올라가 비용·자리 정리 전에 멈추고 5xx 로 응답한다.
            change = self._state.record_auth_hold(task, event.error_code)
            current = change.job or job
            outcome = EventOutcome.HELD if auth_hold_active(current) else EventOutcome.IGNORED
            logger.error(
                "runware webhook external auth error fitting_job_id=%s task_uuid=%s code=%s "
                "outcome=%s; Runware API key / settings need operator check",
                job_id,
                task,
                event.error_code,
                outcome.value,
            )
        elif event.error_kind is TaskErrorKind.NOT_ACCEPTED:
            # 확실한 미접수: 제출 응답·상태 조회와 같은 기록 경로(미접수 기록·다음 시각 예약 또는 소진 결과·자리 반환을 한
            # 트랜잭션으로). 같은 시도의 반복 알림은 횟수·예약을 바꾸지 않는다.
            source = (
                NotAcceptedSource.STATUS_QUERY
                if event.from_status_query
                else NotAcceptedSource.WEBHOOK
            )
            change = self._with_db_retry(
                "record_not_accepted",
                lambda: self._state.record_not_accepted(
                    task,
                    event.error_code,
                    source,
                    retry_after=event.retry_after,
                    retry_after_unrepresentable=event.retry_after_unrepresentable,
                ),
            )
            current = change.job or job
            if change.applied:
                outcome = EventOutcome.APPLIED
            elif change.outcome is NotAcceptedOutcome.ALREADY_RECORDED:
                outcome = EventOutcome.DUPLICATE
            else:
                outcome = EventOutcome.IGNORED
            trace["not_accepted"] = change.outcome.value
            log = logger.error if change.outcome is NotAcceptedOutcome.CONTRADICTED else logger.info
            log(
                "runware result not accepted fitting_job_id=%s task_uuid=%s code=%s attempt=%s "
                "record=%s",
                job_id,
                task,
                event.error_code,
                change.attempt_no,
                change.outcome.value,
            )
        elif event.unresolved:
            # 확실한 미접수(재요청 후보, 자동 재요청은 아직 연결 안 됨) 또는 근거 부족 코드. fallback·FAILED 로 넘기지 않고
            # 상태·결과·호출 기록·진행 중 자리를 그대로 둔다(Watchdog 조회·운영 확인이 잇는다). 보고된 비용만 기록한다.
            # 이미 결과가 저장됐거나 다른 경로로 진행한 작업이면 늦은 알림이다(IGNORED, 결과 보존).
            current = job
            outcome = (
                EventOutcome.UNRESOLVED if _awaiting_task_result(job) else EventOutcome.IGNORED
            )
            logger.error(
                "runware result error not treated as a generation failure fitting_job_id=%s "
                "task_uuid=%s code=%s kind=%s outcome=%s; state kept",
                job_id,
                task,
                event.error_code,
                event.error_kind.value,
                outcome.value,
            )
        else:
            change, current, image_ref = self._record(event, job)
            if change.applied:
                outcome = EventOutcome.APPLIED
            elif _already_recorded(event, current):
                outcome = EventOutcome.DUPLICATE
            else:
                outcome = EventOutcome.IGNORED
            if event.kind is EventKind.SUCCEEDED:
                trace["image_ref"] = image_ref
                self._log_image_ref(event, job, image_ref)
            if change.applied and job.pruna_not_accepted_at is not None:
                # 미접수로 기록한 현재 시도에 생성 결과 근거가 왔다. 결과는 그대로 따르고(예약된 재요청은 상태 전환으로 멈춤)
                # 모순을 남긴다(운영 확인).
                self._state.record_attempt_contradiction(task, "result_after_not_accepted")
                logger.error(
                    "runware result for an attempt recorded as not accepted fitting_job_id=%s "
                    "task_uuid=%s kind=%s; operator check",
                    job_id,
                    task,
                    event.kind.value,
                )
        trace |= {"state_change": outcome.value} | _diag_job(current)
        self._record_cost(event, job)
        # 미접수의 자리 반환은 미접수 기록과 같은 트랜잭션에서 했다. 불명확한 오류는 자리를 그대로 둔다.
        released = not event.unresolved and self._limiter.release_in_flight(job_id, task_uuid=task)
        action = _follow_up_action(event, current)
        if action is NextAction.POSTPROCESS and _awaiting_result_url(current):
            action = None  # URL 이 저장된 뒤(Watchdog 참조 복구) 발행한다
        trace["follow_up"] = "not_needed"
        if action is not None:
            trace["follow_up"] = "unknown"  # 보내는 중 예외가 나면 큐에 들어갔는지 모른다
            sent = self._queue.send_json(
                FittingQueue.AI_FITTING_RESPONSE, follow_up_message(current, action)
            )
            outcome_value = getattr(getattr(sent, "outcome", None), "value", "unknown")
            trace["follow_up"] = f"{action.value.lower()}_{outcome_value}"
            if getattr(sent, "message_id", None):
                trace["follow_up_message_id"] = sent.message_id
            if not sent.succeeded:
                logger.error(
                    "runware webhook follow-up not published fitting_job_id=%s action=%s "
                    "outcome=%s reason=%s",
                    job_id,
                    action.value,
                    sent.outcome.value,
                    sent.reason,
                )
                raise WebhookFollowUpError()
        logger.info(
            "runware webhook fitting_job_id=%s task_uuid=%s kind=%s error_code=%s outcome=%s "
            "status=%s next_action=%s slot_released=%s published=%s image_ref=%s",
            job_id,
            task,
            event.kind.value,
            event.error_code,
            outcome.value,
            current.status.value,
            current.next_action.value if current.next_action else None,
            released,
            action.value if action else None,
            trace.get("image_ref"),
        )
        return EventResult(task, outcome, action), current

    def _apply_past(
        self, event: WebhookEvent, attempt: PrunaAttempt, job: AsyncFittingJob, trace: dict
    ) -> tuple[EventResult, AsyncFittingJob]:
        """이전 전송 시도(현재 task 아님)의 늦은 알림. 현재 시도의 상태·결과·자리에 적용하지 않는다(task_uuid 를 되돌리지
        않음). 비용은 그 시도의 task 와 시작 날짜로만 기록한다. 미접수로 기록한 시도에 생성 결과 근거(성공·생성 종료 실패)가
        오면 모순을 남기고, 아직 예약만 된 재요청은 보류한다(운영 확인)."""
        task = event.task_uuid
        if event.kind is EventKind.PROCESSING:
            if attempt.not_accepted_at is not None:
                self._with_db_retry(
                    "record_conflict",
                    lambda: self._state.record_attempt_contradiction(
                        task, "processing_after_not_accepted"
                    ),
                )
            outcome = EventOutcome.PROCESSING
        elif event.error_kind is TaskErrorKind.NOT_ACCEPTED and attempt.not_accepted_at is not None:
            outcome = EventOutcome.DUPLICATE
        else:
            outcome = EventOutcome.IGNORED
            evidence = event.kind is EventKind.SUCCEEDED or event.error_kind in (
                TaskErrorKind.GENERATION_FALLBACK,
                TaskErrorKind.GENERATION_FAILED,
            )
            if evidence and attempt.not_accepted_at is not None:
                self._with_db_retry(
                    "record_conflict",
                    lambda: self._state.record_attempt_contradiction(
                        task, "result_after_not_accepted"
                    ),
                )
                logger.error(
                    "runware result for a past attempt recorded as not accepted fitting_job_id=%s "
                    "task_uuid=%s attempt=%s kind=%s; not applied, operator check",
                    job.fitting_job_id,
                    task,
                    attempt.attempt_no,
                    event.kind.value,
                )
        self._record_cost(event, job, started_at=attempt.started_at)
        current = self._jobs.find_by_fitting_job_id(job.fitting_job_id) or job
        trace |= {"state_change": "past_attempt", "follow_up": "not_needed"} | _diag_job(current)
        logger.info(
            "runware webhook past attempt fitting_job_id=%s task_uuid=%s attempt=%s kind=%s "
            "error_code=%s outcome=%s",
            job.fitting_job_id,
            task,
            attempt.attempt_no,
            event.kind.value,
            event.error_code,
            outcome.value,
        )
        return EventResult(task, outcome), current

    def _log_image_ref(self, event: WebhookEvent, job: AsyncFittingJob, result: str) -> None:
        """결과 참조의 이상만 남긴다. URL·UUID 값은 로그에 넣지 않는다."""
        if event.image_ref_invalid:
            logger.warning(
                "runware result image ref ignored (invalid) fitting_job_id=%s task_uuid=%s",
                job.fitting_job_id,
                event.task_uuid,
            )
        if result in ("conflict", "unconfirmed", "fill_failed"):
            logger.warning(
                "runware result image ref %s kept existing fitting_job_id=%s task_uuid=%s",
                result,
                job.fitting_job_id,
                event.task_uuid,
            )

    def _with_db_retry(
        self, step: str, work: Callable[[], Any], attempts: int = DB_ATTEMPTS
    ) -> Any:
        """work(한 논리적 DB 처리)를 일시 장애일 때만 최대 attempts 번 한다. 다른 재시도 안에서 부르지 않는다(중첩 없음).

        work 안의 저장소 메서드는 각자 한 트랜잭션이다. 실패하면 그 트랜잭션은 rollback 되고 연결은 pool 로 돌아간다.
        다음 시도는 새 트랜잭션에서 최신 행을 잠근 뒤 조건부 UPDATE 로 다시 판단한다(이미 저장된 것은 다시 적용되지 않음).
        work 에는 SQS 발행·비용·자리 반환·외부 호출을 넣지 않는다.
        """
        for attempt in range(1, attempts + 1):
            try:
                return work()
            except FittingDatabaseError as error:
                if attempt == attempts or not transient_db_error(error):
                    raise
                delay = DB_RETRY_DELAYS[min(attempt, len(DB_RETRY_DELAYS)) - 1]
                if self._db_deadline is not None and self._clock() + delay >= self._db_deadline:
                    logger.warning(
                        "runware webhook db retry stopped at deadline step=%s attempt=%s/%s",
                        step,
                        attempt,
                        attempts,
                    )
                    raise
                cause = error.__cause__
                logger.warning(
                    "runware webhook db retry step=%s attempt=%s/%s error_type=%s sqlstate=%s",
                    step,
                    attempt,
                    attempts,
                    type(cause).__name__,
                    getattr(cause, "sqlstate", None),
                )
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _record(
        self, event: WebhookEvent, job: AsyncFittingJob
    ) -> tuple[Any, AsyncFittingJob, str | None]:
        """이벤트의 상태 기록 단계. 성공 알림은 [성공 전환(상태·참조 한 UPDATE) → 필요하면 늦은 참조 보완] 한 단위를
        일시 DB 장애일 때 최대 DB_ATTEMPTS 번 한다(실패 알림은 기존처럼 1번).

        commit 응답을 못 받았던 시도 뒤의 재시도는 이미 저장된 상태·참조를 조건부 UPDATE 로 다시 확인할 뿐 다시 적용하지
        않는다(중복·same). 소진되면: 상태 기록 자체가 확인되지 않았거나 AI 저장 작업의 필요한 결과 URL 을 저장하지 못했으면
        예외(AI 작업 503), 기존 방식 작업의 참조 보완만 실패했으면 fill_failed 로 기존 처리를 이어 간다.
        """
        progress: dict[str, Any] = {}

        def attempt() -> tuple[Any, AsyncFittingJob, str | None]:
            progress.clear()
            change = self._record_state(event, job)
            progress["change"] = change
            current = change.job or job
            if event.kind is not EventKind.SUCCEEDED:
                return change, current, None
            image_ref = _image_ref_result(event, change.applied, change.job)
            # 조회 근거는 AI 저장 작업에만 쓴다(기존 방식 작업의 보완 정책은 그대로).
            confirmed = event.from_status_query and current.ai_stores_result
            if (
                (image_ref == "not_stored" or (confirmed and image_ref == "unconfirmed"))
                and current.status is AsyncJobStatus.POSTPROCESSING
                and current.next_action is NextAction.POSTPROCESS
                and current.result_payload is None
            ):
                # 먼저 읽은 상태는 후보를 고르는 데만 쓴다. 실제 저장 조건은 UPDATE 시점에 다시 확인한다.
                image_ref, filled = self._fill_image_ref(event, confirmed)
                current = filled or current  # 참조 컬럼만 바뀐 최신 기록(상태는 그대로)
            return change, current, image_ref

        attempts = DB_ATTEMPTS if event.kind is EventKind.SUCCEEDED else 1
        try:
            return self._with_db_retry("record_result", attempt, attempts)
        except FittingDatabaseError as error:
            change = progress.get("change")
            if change is None:
                if (
                    event.kind is EventKind.SUCCEEDED
                    and event.image_url is not None
                    and _needs_result_url(job)
                ):
                    # 성공 전환과 결과 URL 은 한 UPDATE 다. 저장을 확인하지 못했다(503, 재수신 때 다시).
                    raise WebhookResultRefNotSavedError() from error
                raise
            # 상태 기록은 확인됐고 늦은 참조 보완만 실패했다.
            current = change.job or job
            if _needs_result_url(current) and event.image_url is not None:
                # AI 저장 작업은 아직 저장하지 않은 결과의 URL 이 다운로드에 꼭 필요하다(503, 재수신·Watchdog 조회 때 다시).
                raise WebhookResultRefNotSavedError() from error
            # 기존 방식 작업: 참조는 내부 기록이라 경고만 남기고 기존 처리를 이어 간다.
            return change, current, "fill_failed"

    def _fill_image_ref(
        self, event: WebhookEvent, confirmed: bool
    ) -> tuple[str, AsyncFittingJob | None]:
        """늦게 온 같은 결과의 참조로 빈 컬럼만 보완한다. DB 오류는 그대로 올린다(재시도·최종 처리는 _record)."""
        change = self._state.fill_pruna_image_ref(
            event.task_uuid,
            image_url=event.image_url,
            image_uuid=event.image_uuid,
            confirmed_by_query=confirmed,
        )
        if change.applied:
            return "filled", change.job
        return _image_ref_result(event, False, change.job), change.job

    def _record_state(self, event: WebhookEvent, job: AsyncFittingJob):
        if event.kind is EventKind.SUCCEEDED:
            return self._state.record_pruna_succeeded(
                event.task_uuid, image_url=event.image_url, image_uuid=event.image_uuid
            )
        if event.recoverable:
            return self._state.record_pruna_failed(event.task_uuid, recoverable=True)
        if event.error_kind is not TaskErrorKind.GENERATION_FAILED:  # 호출부(_apply)가 걸러 낸다
            raise ValueError("생성 실패로 확정된 오류만 실패 결과로 저장합니다.")
        # 근거로 확인한 생성 종료 실패이고 fallback 대상이 아니다: 최종 실패 결과(fallback 까지 고려한 최종 결과).
        return self._state.record_pruna_failed(
            event.task_uuid,
            recoverable=False,
            failure_payload=failure_payload(
                job.fitting_job_id, FittingModelError.code, FittingModelError.message
            ),
        )

    def _record_cost(
        self, event: WebhookEvent, job: AsyncFittingJob, *, started_at: Any = None
    ) -> None:
        if event.cost_invalid:
            logger.warning(
                "runware webhook cost ignored (invalid) fitting_job_id=%s task_uuid=%s",
                job.fitting_job_id,
                event.task_uuid,
            )
        if event.cost is None:
            return
        # 귀속 날짜는 이 생성 요청을 시작한 UTC 날짜로 고정한다(지연·중복 수신으로 바뀌지 않음, 기존 UTC 기준).
        usage_date = (started_at or job.pruna_submit_started_at).astimezone(UTC).date()
        self._costs.record(event.task_uuid, job.fitting_job_id, event.cost, usage_date)
