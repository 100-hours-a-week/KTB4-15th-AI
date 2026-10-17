"""V2: Runware 완료 Webhook 처리 (9단계). POST /api/v2/webhooks/runware?token=<WEBHOOK_TOKEN>

이 모듈은 본문 해석(parse_webhook)과 처리(RunwareWebhookService)를 한다. HTTP·인증은 webhook_router.py 가 한다.
Webhook 안에서는 S3 HEAD·이미지 다운로드·OpenAI·V1·BE 결과 발행을 하지 않는다. 기록하고 후속 큐에 넘길 뿐이다.

지원하는 본문 (공식 문서 models-api/webhooks·errors, 기존 1단계 Provider 계약):
  1. 단일 성공 객체(공식 Webhook 형식):
       {"taskType": "imageInference", "taskUUID": ..., "imageUUID": ..., "imageURL": ..., "cost": 0.023}
  2. errors 배열(공식 오류 객체): {"errors": [{"taskUUID": ..., "code": "timeoutProvider", ...}]}
  3. data 배열(Runware 표준 API 응답 형식, 1단계 Provider 가 읽는 형식): {"data": [<1 의 객체>, ...]}
     data 와 errors 가 함께 있어도 된다.
  그 밖의 모양, taskType 이 imageInference 가 아닌 항목, taskUUID 가 UUID 문자열이 아닌 항목, 완료 여부가 불명확한
  항목(status 가 없거나 success 인데 결과 이미지(imageURL·imageUUID)가 없음, 알 수 없는 status)은 400 으로 거절한다.
  status=processing 은 진행 중이다. 완료로 처리하지 않고 자리도 반환하지 않는다.
  실제 Runware 계정에서 받은 본문으로는 아직 확인하지 않았다(공식 예시 기반).

처리 단계(이벤트마다, 순서대로). 단계마다 반복해도 같은 결과라 재수신 때 빠진 단계부터 이어진다.
  0. 사전 확인(부작용 없음): 모든 task 가 등록돼 있고 호출 시작 기록이 있는지, 후속 큐 설정이 있는지.
  1. 상태: 기존 조건부 전환(record_pruna_succeeded / record_pruna_failed). 이미 바뀌었으면 그대로 둔다.
  2. 비용: 명시적인 유효 비용이 있으면 TaskCostRepository.record(task 당 한 번).
  3. 자리: 이 task 가 잡은 진행 중 자리를 반환(release_in_flight, task 일치).
  4. 발행: 현재 DB 상태가 이 이벤트의 후속 작업(POSTPROCESS / RUN_FALLBACK)을 기다리면 ai-fitting-response 에 발행.
     상태가 이미 바뀌었다는 이유로 2·3·4 를 건너뛰지 않는다. 발행이 실패·UNKNOWN 이면 예외(5xx) — 재수신 때 다시 발행.
  5. 경로 확인: DB 에 다음 작업이 **기록됐다**는 것과 그 작업을 **실행할 경로가 있다**는 것은 다르다. 남은 일이
     결과 발행(PUBLISH_RESULT, 14단계)이거나 운영 확인·복구(RECOVER_UNKNOWN, 15단계 Watchdog)인데 FollowUpPaths 가
     그 경로를 준비됐다고 답하지 않으면 모든 이벤트를 처리한 뒤 WebhookFollowUpNotReadyError(503)를 낸다.
     기본 NoFollowUpPaths 는 항상 "없음"이라 지금은 503 이다(완료로 응답하지 않는 보호 조건).

Runware 인증 오류(본문 code 가 AUTH_ERROR_CODES): Runware 가 우리 API 키를 거절했다는 외부 오류다(Webhook token
인증 실패 401 과 다르다). 사용자 실패 결과를 저장하지 않고, fallback·재호출도 하지 않으며, 상태·기존 결과를 바꾸지
않는다. 운영 확인 대상으로 오류 로그를 남기고 그 task 의 비용·자리만 정리한다.
"""

import json
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from decimal import Decimal
from enum import Enum
from typing import Any
from urllib.parse import parse_qs, urlsplit

from app.clients.sqs import FittingQueue, get_queue_url
from app.virtual_fitting.exceptions import FittingModelError, VirtualFittingError
from app.virtual_fitting.pruna_submitter import (
    FollowUpPaths,
    NextStep,
    NoFollowUpPaths,
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


def _waiting_for_pruna(job: AsyncFittingJob) -> bool:
    """아직 Pruna 결과를 기다리는 중(결과·후속 작업 없음)."""
    return (
        job.status in (AsyncJobStatus.RECEIVED, AsyncJobStatus.WAITING_PRUNA)
        and job.result_payload is None
        and job.next_action in (None, NextAction.SUBMIT_PRUNA)
    )


def _pending_step(event: WebhookEvent, job: AsyncFittingJob | None) -> NextStep | None:
    """이 알림 뒤에 남은 일 중 실제 경로가 있어야 하는 것. 후처리 큐로 넘긴 일(POSTPROCESS·RUN_FALLBACK)은 제외."""
    if job is None:
        return None
    if next_step_for(job) is NextStep.PUBLISH_RESULT:
        return NextStep.PUBLISH_RESULT
    if event.auth_error and _waiting_for_pruna(job):
        return NextStep.RECOVER_UNKNOWN  # 운영 확인 뒤 Watchdog(15단계)이 이어 갈 일
    return None


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
    ) -> None:
        self._jobs, self._state, self._limiter = jobs, state, limiter
        self._costs, self._queue = costs, queue
        # 기본은 NoFollowUpPaths(결과 발행 14단계·Watchdog 15단계 미구현). 설정값으로 바꿀 수 없다.
        self._follow_up = follow_up or NoFollowUpPaths()

    def handle(self, events: Sequence[WebhookEvent]) -> list[EventResult]:
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
                raise WebhookUnknownTaskError()
            jobs[event.task_uuid] = job
        if any(event.kind is not EventKind.PROCESSING for event in events):
            get_queue_url(
                FittingQueue.AI_FITTING_RESPONSE
            )  # 설정 오류면 기록 전에 멈춘다(SqsConfigError)
        results, pending = [], []
        for event in events:
            result, current = self._apply(event, jobs[event.task_uuid])
            results.append(result)
            step = _pending_step(event, current)
            if step is not None and not self._follow_up.ready_for(current, step):
                pending.append((event.task_uuid, step.value))
        if pending:
            logger.warning(
                "runware webhook recorded but follow-up path not ready pending=%s",
                ",".join(f"{task}:{step}" for task, step in pending),
            )
            raise WebhookFollowUpNotReadyError(results, pending)
        return results

    def _apply(
        self, event: WebhookEvent, job: AsyncFittingJob
    ) -> tuple[EventResult, AsyncFittingJob | None]:
        job_id, task = job.fitting_job_id, event.task_uuid
        if event.kind is EventKind.PROCESSING:
            logger.info("runware webhook processing fitting_job_id=%s task_uuid=%s", job_id, task)
            return EventResult(task, EventOutcome.PROCESSING), job

        if event.auth_error:
            # 상태·결과를 바꾸지 않는다. 최신 기록으로 보류인지 늦은 알림인지 판단한다.
            current = self._jobs.find_by_task_uuid(task) or job
            outcome = EventOutcome.HELD if _waiting_for_pruna(current) else EventOutcome.IGNORED
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
        self._record_cost(event, job)
        released = self._limiter.release_in_flight(job_id, task_uuid=task)
        action = _follow_up_action(event, current)
        if action is not None:
            sent = self._queue.send_json(
                FittingQueue.AI_FITTING_RESPONSE, follow_up_message(current, action)
            )
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
