"""가상피팅 V2 AI 후처리 Worker (ai-fitting-response, 10단계).

FastAPI lifespan 이 시작·종료한다(별도 실행 명령 없음). VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED 기본 꺼짐.
켜도 POSTPROCESS·RUN_FALLBACK 핸들러가 모두 연결돼 있지 않으면 수신을 시작하지 않는다(not_ready). 13단계부터 둘 다
연결돼 있다(결과 발행 14단계·복구 15단계는 미구현).
미구현 단계 때문에 실제 큐 메시지를 받기만 하고 지우지 못해 DLQ 로 보내는 일을 막는다.

메시지 하나 처리(FollowUpProcessor.process):
  형식 검증(follow_up_queue) → 작업 조회 → task_uuid 일치 → 결과 발행까지 끝난 중복이면 삭제
  → status·next_action 이 action 과 맞는지 → 핸들러 연결 여부 → 실행 권한 선점(acquire_lease, 원자적)
  → 선점 뒤 최신 기록으로 다시 확인 → keepalive(lease·Visibility 주기 연장) 안에서 핸들러 호출
  → 최신 DB 기록으로 삭제 판단 → 자기 토큰의 실행 권한만 반납

| action       | 실행 대상(status / next_action)        | 핸들러(연결 단계) |
| POSTPROCESS  | POSTPROCESSING / POSTPROCESS           | 11단계(연결됨)    |
| RUN_FALLBACK | FALLBACK_RUNNING / RUN_FALLBACK        | 13단계(연결됨)    |

삭제는 (1) 결과 발행까지 끝난 같은 작업의 중복(COMPLETED/FAILED + PUBLISHED), 또는 (2) 핸들러가 완료를 알렸고
DB 기록상 그 단계가 끝났으며 남은 일의 실제 경로가 있을 때(deletion_allowed + FollowUpPaths)만 한다. 메시지 수신·실행 권한
확보·핸들러 호출·횟수 한도·상태 불일치·메모리의 성공 결과만으로는 지우지 않는다. 형식 오류·미등록 작업·task 불일치는
원인을 기록하고 남겨 둔다(Redrive Policy 에 따라 DLQ). 가짜 작업·실패 결과를 만들지 않는다.

시도 횟수(s3_check·llm_generation·fallback·result_message)는 각 단계(11·13·14)가 실제 시도 직전에 5단계 claim_* 로 늘린다.
이 Worker 는 횟수를 늘리지도 초기화하지도 않는다. ApproximateReceiveCount 는 로그에만 남긴다.
DB 트랜잭션·행 잠금은 5단계 메서드 안에서만 짧게 잡는다(핸들러 실행 중에는 잡지 않는다).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from app.clients.sqs import FittingQueue, SqsQueueClient
from app.virtual_fitting.follow_up_queue import InvalidFollowUp, ValidFollowUp
from app.virtual_fitting.pruna_submitter import (
    FollowUpPaths,
    KeepaliveState,
    NoFollowUpPaths,
    deletion_allowed,
    keepalive,
)
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncFittingJob,
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import ChangeReason
from app.virtual_fitting.request_worker import Heartbeat, RequestOutcome

logger = logging.getLogger(__name__)

EXPECTED_STATUS = {
    NextAction.POSTPROCESS: AsyncJobStatus.POSTPROCESSING,
    NextAction.RUN_FALLBACK: AsyncJobStatus.FALLBACK_RUNNING,
}


# --- 핸들러 인터페이스 ---


class HandlerStatus(str, Enum):
    # 이 단계의 결과를 DB 에 기록했다(핸들러의 주장). 삭제는 Worker 가 최신 DB 기록으로 다시 판단한다.
    DONE = "done"
    # 지금 끝낼 수 없어 보류(일시 오류·대기 등). 메시지를 남겨 다음 수신 때 이어 간다.
    DEFERRED = "deferred"
    # 처리 중 실패. 실패 결과를 저장했는지는 DB 기록으로 판단한다(이 값만으로 삭제하지 않음).
    FAILED = "failed"


@dataclass(frozen=True)
class HandlerResult:
    status: HandlerStatus
    reason: str = ""
    # 이 단계를 다른 action 의 처리로 넘겼다(예: POSTPROCESS 의 S3 확인 실패 → RUN_FALLBACK). Worker 가 그 action 의
    # 내부 큐 메시지를 발행하고, 발행이 확인되고 그 핸들러가 연결돼 있을 때만 현재 메시지를 지운다.
    handoff: NextAction | None = None


@dataclass
class HandlerContext:
    """핸들러가 받는 실행 환경. 새 외부 부작용(S3·OpenAI·V1 호출, 결과 저장)을 시작하기 직전에 may_continue() 를 본다."""

    heartbeat: Heartbeat
    alive: KeepaliveState
    should_stop: Callable[[], bool]
    # 수신 기록(참고용). 실행 횟수로 쓰지 않는다.
    approximate_receive_count: int | None = None

    def may_continue(self) -> bool:
        """종료 요청이 없고 lease·Visibility 연장이 실패하지 않았으면 True(필요하면 지금 연장한다)."""
        if self.should_stop() or self.alive.failed:
            return False
        if not self.heartbeat.beat():
            # 여기서 연장이 실패해도 keepalive 실패와 같이 기록한다(Worker 가 이 receipt 로 지우지 않게).
            self.alive.failed = True
            return False
        return True


class ActionHandler(Protocol):
    def __call__(self, job: AsyncFittingJob, token: str, context: HandlerContext) -> HandlerResult:
        """실행 권한(token)을 쥔 상태로 그 단계를 처리한다. DB 변경은 token 을 넘겨 5단계 메서드로 한다."""
        ...


@dataclass(frozen=True)
class PostprocessHandlers:
    postprocess: ActionHandler | None = None  # 11단계: S3 HEAD·제목·코멘트
    run_fallback: ActionHandler | None = None  # 13단계: V1 fallback

    def for_action(self, action: NextAction) -> ActionHandler | None:
        return {
            NextAction.POSTPROCESS: self.postprocess,
            NextAction.RUN_FALLBACK: self.run_fallback,
        }.get(action)

    def missing(self) -> list[str]:
        return [
            name
            for name, handler in (
                ("postprocess", self.postprocess),
                ("run_fallback", self.run_fallback),
            )
            if handler is None
        ]


def connected_postprocess_handlers() -> PostprocessHandlers:
    """실제 조립에 연결된 핸들러: POSTPROCESS(11단계 PostprocessHandler)와 RUN_FALLBACK(13단계 RunFallbackHandler).

    두 핸들러가 모두 연결돼 not_ready 는 풀린다. 그래도 Worker 는 VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED(기본 false)로만
    켜지고, 결과 발행(14단계)·복구(15단계) 경로는 여전히 없다(FollowUpPaths 미준비 → 결과를 저장해도 메시지를 지우지 않음).
    설정값으로 채우지 않는다. 두 핸들러는 처음 불릴 때 DB·S3·Runware client 를 만든다.
    """
    from app.virtual_fitting.fallback_handler import RunFallbackHandler
    from app.virtual_fitting.postprocess_handler import PostprocessHandler

    return PostprocessHandlers(postprocess=PostprocessHandler(), run_fallback=RunFallbackHandler())


# --- 메시지 하나 처리 ---


def _published(job: AsyncFittingJob | None) -> bool:
    return (
        job is not None
        and job.status in (AsyncJobStatus.COMPLETED, AsyncJobStatus.FAILED)
        and job.result_message_status is ResultMessageStatus.PUBLISHED
    )


def _waiting_for(job: AsyncFittingJob, action: NextAction) -> bool:
    return job.status is EXPECTED_STATUS[action] and job.next_action is action


class FollowUpProcessor:
    """동기. 메시지 하나를 처리하고 RequestOutcome 을 돌려준다(delete 가 True 일 때만 Worker 가 지운다). 예외를 밖으로 내지 않는다."""

    def __init__(
        self,
        jobs: Any,
        state: Any,
        sqs: SqsQueueClient,
        handlers: PostprocessHandlers,
        *,
        lease_seconds: float,
        visibility_seconds: int,
        heartbeat_interval_seconds: float | None = None,
        keepalive_seconds: float | None = None,
        follow_up: FollowUpPaths | None = None,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> None:
        self._jobs, self._state, self._sqs, self._handlers = jobs, state, sqs, handlers
        self._lease_seconds, self._visibility_seconds = lease_seconds, visibility_seconds
        self._heartbeat_interval = (
            lease_seconds / 3 if heartbeat_interval_seconds is None else heartbeat_interval_seconds
        )
        self._keepalive = lease_seconds / 3 if keepalive_seconds is None else keepalive_seconds
        # 남은 일(결과 발행 14단계·복구 15단계 등)을 이어 갈 실제 경로. 기본은 "없음".
        self._follow_up = follow_up or NoFollowUpPaths()
        self._should_stop = should_stop

    def process(self, item: ValidFollowUp | InvalidFollowUp) -> RequestOutcome:
        if isinstance(item, InvalidFollowUp):
            return RequestOutcome(item.message_id, item.fitting_job_id, "invalid_kept")
        try:
            return self._valid(item)
        except Exception:
            logger.exception(
                "postprocess worker failed fitting_job_id=%s message_id=%s",
                item.message.fitting_job_id,
                item.message_id,
            )
            return RequestOutcome(item.message_id, item.message.fitting_job_id, "error_kept")

    def _hand_off(self, job: AsyncFittingJob, action: NextAction, name: str) -> tuple[str, bool]:
        """다음 action 의 내부 큐 메시지를 발행한다(9단계와 같은 형식·비동기 task_uuid). DB 트랜잭션 밖에서 한다.

        그 action 의 핸들러가 연결돼 있고 발행이 SUCCEEDED 일 때만 현재 메시지를 지운다. 실패·UNKNOWN 이면 남겨 두고
        재수신 때 다시 발행한다(중복 발행 허용, 생성 1회는 DB 의 fallback 기록이 막는다).
        """
        if self._handlers.for_action(action) is None:
            logger.warning(
                "postprocess hand-off target not connected fitting_job_id=%s action=%s",
                job.fitting_job_id,
                action.value,
            )
            return f"{name}_target_not_connected_kept", False
        from app.virtual_fitting.follow_up_queue import publish_follow_up

        if not publish_follow_up(self._sqs, job, action).succeeded:
            return f"{name}_publish_failed_kept", False
        return f"{name}_deleted", True

    def _valid(self, item: ValidFollowUp) -> RequestOutcome:
        message = item.message
        job_id, action = message.fitting_job_id, message.action

        def outcome(name: str, delete: bool = False) -> RequestOutcome:
            return RequestOutcome(item.message_id, job_id, name, delete)

        job = self._jobs.find_by_fitting_job_id(job_id)
        if job is None:
            logger.error("ai-fitting-response for unknown job fitting_job_id=%s", job_id)
            return outcome("unknown_job_kept")
        if job.task_uuid != message.task_uuid:
            # 같은 작업 번호의 다른 생성 시도. 기존 작업을 바꾸지 않고 실행하지 않는다.
            logger.error(
                "ai-fitting-response task mismatch fitting_job_id=%s message_task=%s job_task=%s",
                job_id,
                message.task_uuid,
                job.task_uuid,
            )
            return outcome("task_mismatch_kept")
        if _published(job):
            return outcome("already_published_deleted", delete=True)
        if not _waiting_for(job, action):
            if action is NextAction.POSTPROCESS and _waiting_for(job, NextAction.RUN_FALLBACK):
                # POSTPROCESS 가 RUN_FALLBACK 으로 넘긴 뒤의 재수신(예: 이전 발행 실패). 다음 메시지를 다시 발행한다.
                return outcome(*self._hand_off(job, NextAction.RUN_FALLBACK, "republished"))
            # 오래된 메시지 등. 상태가 다르다고 완료로 단정하지 않는다(남은 일의 실제 경로가 있을 때만 삭제).
            return outcome("state_mismatch", delete=deletion_allowed(job, self._follow_up))
        handler = self._handlers.for_action(action)
        if handler is None:
            logger.warning(
                "ai-fitting-response handler not connected fitting_job_id=%s action=%s",
                job_id,
                action.value,
            )
            return outcome("handler_not_connected_kept")
        if self._should_stop():
            return outcome("stopping_kept")

        grant = self._state.acquire_lease(job_id, action, self._lease_seconds)
        if not grant.acquired:
            # 다른 Worker 가 처리 중(LEASE_HELD)이면 추가 실행하지 않고 남긴다.
            delete = False
            if grant.reason is ChangeReason.STATE_MISMATCH:
                latest = self._jobs.find_by_fitting_job_id(job_id)
                delete = _published(latest) or deletion_allowed(latest, self._follow_up)
            return outcome(f"lease_{grant.reason.value}", delete=delete)
        token = grant.token
        try:
            current = grant.job
            if (
                current is None
                or current.task_uuid != message.task_uuid
                or not _waiting_for(current, action)
            ):
                return outcome("state_changed_kept")
            heartbeat = Heartbeat(
                self._sqs,
                self._state,
                job_id,
                token,
                item.receipt_handle,
                lease_seconds=self._lease_seconds,
                visibility_seconds=self._visibility_seconds,
                interval_seconds=self._heartbeat_interval,
                queue=FittingQueue.AI_FITTING_RESPONSE,
                name="postprocess",
            )
            with keepalive(heartbeat, self._keepalive) as alive:
                context = HandlerContext(
                    heartbeat, alive, self._should_stop, item.approximate_receive_count
                )
                if not context.may_continue():
                    return outcome("not_started_kept")
                try:
                    result = handler(current, token, context)
                except Exception:
                    logger.exception(
                        "postprocess handler crashed fitting_job_id=%s action=%s",
                        job_id,
                        action.value,
                    )
                    return outcome("handler_error_kept")
            if alive.failed:
                # 처리 중 lease·Visibility 연장이 실패했다. 이 receipt 의 안전성을 모르므로 지우지 않는다.
                return outcome("keepalive_failed_kept")
            if not isinstance(result, HandlerResult):
                return outcome("handler_invalid_result_kept")
            if result.status is not HandlerStatus.DONE:
                return outcome(f"handler_{result.status.value}_kept")
            latest = self._jobs.find_by_fitting_job_id(job_id)
            if latest is None or (_waiting_for(latest, action) and not _published(latest)):
                # 핸들러는 완료라고 했지만 DB 에는 그 단계가 그대로 남아 있다.
                return outcome("handler_done_not_recorded_kept")
            if result.handoff is not None and _waiting_for(latest, result.handoff):
                return outcome(*self._hand_off(latest, result.handoff, "handed_off"))
            return outcome(
                "handler_done",
                delete=_published(latest) or deletion_allowed(latest, self._follow_up),
            )
        finally:
            released = self._state.release_lease(job_id, token)
            if not released.applied and released.reason is not ChangeReason.LEASE_INVALID:
                logger.warning(
                    "postprocess worker lease release failed fitting_job_id=%s reason=%s",
                    job_id,
                    released.reason.value,
                )


# --- 조립 ---


@dataclass(frozen=True)
class PostprocessSettings:
    lease_seconds: float
    visibility_seconds: int
    shutdown_timeout_seconds: float
    receive_wait_seconds: int = 20


def postprocess_settings() -> PostprocessSettings:
    """후처리 Worker 설정을 읽고 범위를 확인한다(Worker 를 켤 때만 부른다)."""
    from app.clients.sqs import MAX_VISIBILITY_TIMEOUT_SECONDS
    from app.config import settings
    from app.virtual_fitting.repositories.async_job_state import MAX_LEASE_SECONDS

    lease = settings.VIRTUAL_FITTING_POSTPROCESS_LEASE_SECONDS
    visibility = settings.VIRTUAL_FITTING_POSTPROCESS_VISIBILITY_SECONDS
    shutdown = settings.VIRTUAL_FITTING_POSTPROCESS_SHUTDOWN_TIMEOUT_SECONDS
    if not 0 < lease <= MAX_LEASE_SECONDS:
        raise ValueError(
            f"VIRTUAL_FITTING_POSTPROCESS_LEASE_SECONDS 는 0 초과 {MAX_LEASE_SECONDS} 이하여야 합니다."
        )
    if not 1 <= visibility <= MAX_VISIBILITY_TIMEOUT_SECONDS:
        raise ValueError(
            "VIRTUAL_FITTING_POSTPROCESS_VISIBILITY_SECONDS 는 1 이상 "
            f"{MAX_VISIBILITY_TIMEOUT_SECONDS} 이하여야 합니다."
        )
    if not shutdown > 0:
        raise ValueError(
            "VIRTUAL_FITTING_POSTPROCESS_SHUTDOWN_TIMEOUT_SECONDS 는 0 보다 커야 합니다."
        )
    return PostprocessSettings(float(lease), int(visibility), float(shutdown))


def create_postprocess_worker(handlers: PostprocessHandlers):
    """실제 의존성으로 조립한다(blocking: DB pool·boto3 client). lifespan 이 thread 에서 부른다.

    DB pool(get_connection_pool)과 boto3 SQS client(get_sqs_client, long polling 용 기본 timeout)는 요청 Worker 와
    같은 프로세스 전역 자원이다. Worker 는 닫지 않고, lifespan 이 두 Worker 를 모두 멈춘 뒤 pool 을 닫는다.
    Webhook 의 짧은 timeout client(get_webhook_sqs_client)는 쓰지 않는다.
    """
    from app.config.database import get_connection_pool
    from app.virtual_fitting.follow_up_queue import receive_follow_ups
    from app.virtual_fitting.repositories.async_job_repository import AsyncJobRepository
    from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository
    from app.virtual_fitting.request_worker import RequestWorker
    from app.virtual_fitting.watchdog import follow_up_paths

    missing = handlers.missing()
    if missing:
        raise ValueError(f"후처리 핸들러가 연결되지 않았습니다: {','.join(missing)}")
    config = postprocess_settings()
    pool = get_connection_pool()
    sqs = SqsQueueClient()
    worker: RequestWorker
    processor = FollowUpProcessor(
        AsyncJobRepository(pool),
        AsyncJobStateRepository(pool),
        sqs,
        handlers,
        lease_seconds=config.lease_seconds,
        visibility_seconds=config.visibility_seconds,
        # 실제로 도는 경로만 준비됐다고 본다(14단계 결과 전송 루프·15단계 Watchdog).
        follow_up=follow_up_paths(),
        should_stop=lambda: worker.stopping,
    )
    worker = RequestWorker(
        lambda: receive_follow_ups(
            sqs,
            max_messages=1,
            wait_time_seconds=config.receive_wait_seconds,
            visibility_timeout=config.visibility_seconds,
        ),
        processor.process,
        lambda receipt: sqs.delete(FittingQueue.AI_FITTING_RESPONSE, receipt),
        name="postprocess",
    )
    return worker
