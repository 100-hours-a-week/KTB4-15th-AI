import json
import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from decimal import Decimal
from enum import Enum
from typing import Any
from urllib.parse import parse_qs, urlsplit

from app.clients.sqs import FittingQueue, SqsConfigError, get_queue_url
from app.virtual_fitting import diagnostics as diag
from app.virtual_fitting.exceptions import FittingModelError, VirtualFittingError
from app.virtual_fitting.pruna_submitter import (
    FollowUpPaths,
    NextStep,
    NoFollowUpPaths,
    auth_hold_active,
    failure_payload,
    next_step_for,
)
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncFittingJob,
    AsyncJobStatus,
    NextAction,
)

logger = logging.getLogger(__name__)

WEBHOOK_PATH = "/api/v2/webhooks/runware"
TASK_TYPE = "imageInference"
MAX_BODY_BYTES = 256 * 1024
# Runware 실패 중 V1(같은 Pruna 를 동기로 부르는 기존 경로)으로 다시 만들어 볼 수 있다고 확인한 코드만.
# timeoutProvider: 공급자가 시간 안에 응답하지 않음(설계 문서의 fallback 예시). 그 밖의 코드(알 수 없는 코드,
# providerRateLimitExceeded 처럼 같은 공급자를 다시 부르면 같은 원인일 수 있는 코드 포함)는 fallback 하지 않는다.
# retryable 표시·retryAfter 만으로는 유료 fallback 을 허용하지 않는다.
FALLBACK_ERROR_CODES = frozenset({"timeoutProvider"})
# Runware 공식 오류 문서에서 확인한 인증 오류 코드만. 운영 설정 확인 대상이며 사용자용 모델 실패로 저장하지 않는다.
# 권한(403)·잔액(402)은 문서에 HTTP 상태로만 나오고 Webhook 오류 코드명이 확인되지 않아 넣지 않았다(실제 응답 확인 뒤 확장).
# message 문자열로 원인을 추측하지 않는다.
AUTH_ERROR_CODES = frozenset({"invalidApiKey"})
_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class EventKind(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PROCESSING = "processing"


@dataclass(frozen=True)
class WebhookEvent:
    """내부 처리용 공통 구조. 외부 본문 전문·imageURL·오류 message 는 담지 않는다."""

    task_uuid: str
    kind: EventKind
    error_code: str | None = None
    # 명시적인 유효 비용(USD). 없거나 잘못됐으면 None(0 으로 확정하지 않음).
    cost: Decimal | None = None
    cost_invalid: bool = False

    @property
    def recoverable(self) -> bool:
        return self.kind is EventKind.FAILED and self.error_code in FALLBACK_ERROR_CODES

    @property
    def auth_error(self) -> bool:
        return self.kind is EventKind.FAILED and self.error_code in AUTH_ERROR_CODES


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
    return WebhookEvent(task, EventKind.SUCCEEDED, cost=cost, cost_invalid=cost_invalid)


def _error_event(item: Any) -> WebhookEvent:
    if not isinstance(item, dict):
        raise WebhookPayloadError("invalid_item")
    _check_task_type(item, required=False)
    task = _task_uuid(item)
    code = item.get("code")
    if not isinstance(code, str) or not _CODE_PATTERN.match(code):
        raise WebhookPayloadError("invalid_error_code")
    cost, cost_invalid = _cost(item)
    return WebhookEvent(task, EventKind.FAILED, code, cost=cost, cost_invalid=cost_invalid)


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
    return tuple(events)


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


def follow_up_message(job: AsyncFittingJob, action: NextAction) -> dict[str, Any]:
    """ai-fitting-response 메시지. 이미지 Key·입력 등은 후처리 Worker 가 DB 에서 조회한다."""
    if action not in (NextAction.POSTPROCESS, NextAction.RUN_FALLBACK):
        raise ValueError("ai-fitting-response 는 POSTPROCESS 또는 RUN_FALLBACK 만 보냅니다.")
    return {
        "fitting_job_id": job.fitting_job_id,
        "task_uuid": job.task_uuid,
        "action": action.value,
    }


def _pending_step(event: WebhookEvent, job: AsyncFittingJob | None) -> NextStep | None:
    """이 알림 뒤에 남은 일 중 실제 경로가 있어야 하는 것. 후처리 큐로 넘긴 일(POSTPROCESS·RUN_FALLBACK)은 제외."""
    if job is None:
        return None
    step = next_step_for(job)
    if step in (NextStep.PUBLISH_RESULT, NextStep.OPERATOR_REVIEW):
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
    ) -> None:
        self._jobs, self._state, self._limiter = jobs, state, limiter
        self._costs, self._queue = costs, queue
        # 기본은 NoFollowUpPaths(결과 발행 14단계·Watchdog 15단계 미구현). 설정값으로 바꿀 수 없다.
        self._follow_up = follow_up or NoFollowUpPaths()
        # 진단용 출처. 조립하는 쪽이 명시한다(HTTP: webhook, Watchdog: watchdog). 그 밖은 unknown(추측하지 않음).
        self._source, self._request_id = source, request_id

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
            elapsed_ms=diag.elapsed_ms(started),
        )

    def handle(self, events: Sequence[WebhookEvent]) -> list[EventResult]:
        started = time.monotonic()
        jobs: dict[str, AsyncFittingJob] = {}
        for event in events:  # 사전 확인: 조회만 한다
            if event.task_uuid in jobs:
                continue
            job = self._jobs.find_by_task_uuid(event.task_uuid)
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
                result, current = self._apply(event, jobs[event.task_uuid], trace)
            except Exception as error:
                self._diag_processed(event, trace, "failed", _diag_failure_reason(error), started)
                raise
            results.append(result)
            step = _pending_step(event, current)
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
        else:
            change = self._record_state(event, job)
            current = change.job or job
            if change.applied:
                outcome = EventOutcome.APPLIED
            elif _already_recorded(event, current):
                outcome = EventOutcome.DUPLICATE
            else:
                outcome = EventOutcome.IGNORED
        trace |= {"state_change": outcome.value} | _diag_job(current)
        self._record_cost(event, job)
        released = self._limiter.release_in_flight(job_id, task_uuid=task)
        action = _follow_up_action(event, current)
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
            "status=%s next_action=%s slot_released=%s published=%s",
            job_id,
            task,
            event.kind.value,
            event.error_code,
            outcome.value,
            current.status.value,
            current.next_action.value if current.next_action else None,
            released,
            action.value if action else None,
        )
        return EventResult(task, outcome, action), current

    def _record_state(self, event: WebhookEvent, job: AsyncFittingJob):
        if event.kind is EventKind.SUCCEEDED:
            return self._state.record_pruna_succeeded(event.task_uuid)
        if event.recoverable:
            return self._state.record_pruna_failed(event.task_uuid, recoverable=True)
        # 확인된 실패지만 V1 로 해결된다고 확인되지 않았다(알 수 없는 코드 포함): fallback 없이 실패 결과.
        return self._state.record_pruna_failed(
            event.task_uuid,
            recoverable=False,
            failure_payload=failure_payload(
                job.fitting_job_id, FittingModelError.code, FittingModelError.message
            ),
        )

    def _record_cost(self, event: WebhookEvent, job: AsyncFittingJob) -> None:
        if event.cost_invalid:
            logger.warning(
                "runware webhook cost ignored (invalid) fitting_job_id=%s task_uuid=%s",
                job.fitting_job_id,
                event.task_uuid,
            )
        if event.cost is None:
            return
        # 귀속 날짜는 이 생성 요청을 시작한 UTC 날짜로 고정한다(지연·중복 수신으로 바뀌지 않음, 기존 UTC 기준).
        usage_date = job.pruna_submit_started_at.astimezone(UTC).date()
        self._costs.record(event.task_uuid, job.fitting_job_id, event.cost, usage_date)
