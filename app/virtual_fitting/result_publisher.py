"""가상피팅 V2 결과 전송(14단계): DB 에 저장된 최종 결과를 be-fitting-response 로 보낸다.

실행 주체: FastAPI lifespan 이 시작·종료하는 결과 전송 루프(ResultPublisherWorker, VIRTUAL_FITTING_RESULT_PUBLISHER_ENABLED
기본 false). 내부 큐에 새 action 을 만들지 않고 **DB 를 조회**한다. 그래서 결과가 어느 경로(요청 Worker·Webhook·POSTPROCESS·
RUN_FALLBACK)에서 저장됐든, 저장 직후 프로세스가 죽었든 다음 실행에서 찾아 보낸다. 15단계 Watchdog(Pruna·S3·fallback 의 미완료
작업 조사)과 다르다 — 이미 저장된 결과를 보내는 일만 한다.

작업 하나(ResultPublishProcessor.publish):
  대상: next_action=PUBLISH_RESULT, result_payload 있음, result_message_status=PENDING, 종료 아님, 유효한 실행 권한 없음
  1. acquire_lease(PUBLISH_RESULT) — 여러 루프·프로세스가 같은 작업을 동시에 보내지 않는다(원자적 선점).
  2. 이미 3회를 썼으면(이전 시도 결과 불명확) 보내지 않고 RETRY_EXHAUSTED.
  3. 보내기 전 확인(횟수를 쓰지 않음): payload 형식(5단계 검증)·작업 번호·최종 이미지 Key 일치, 결과 큐 URL 설정.
  4. claim_result_message_attempt(최초 포함 최대 3회, < 3, 되돌리지 않음) → SendMessage 1번(SDK 재시도 없음).
  5. SUCCEEDED → mark_result_published(payload status 와 같은 최종 상태, next_action=NULL, lease 정리).
     FAILED·UNKNOWN·예외 → 남은 횟수가 있으면 실행 권한을 backoff 동안 쥔 채 두고(그동안 다른 루프가 선점하지 못함) 다음에
     같은 payload 로 다시 보낸다. 3회를 다 쓰면 mark_result_retry_exhausted(결과·횟수 보존, 자동 재전송 없음).
전송 장애를 가상피팅 실패로 바꾸지 않는다(결과를 새로 만들지 않음). Presigned URL·토큰은 payload 에 없다(5단계 검증).
AI 의 PUBLISHED 는 "SQS 가 등록 성공을 응답했다"는 뜻이다. BE 의 DB 반영·FE 알림(SSE)은 기다리지 않는다.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from anyio import CapacityLimiter, to_thread

from app.clients.sqs import FittingQueue, SqsConfigError, SqsOutcome, get_queue_url
from app.virtual_fitting import diagnostics as diag
from app.virtual_fitting.exceptions import VirtualFittingError
from app.virtual_fitting.pruna_submitter import NextStep
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncFittingJob,
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import (
    RESULT_MESSAGE_LIMIT,
    ChangeReason,
    _check_result_payload,
)

logger = logging.getLogger(__name__)

# 실패 뒤 다시 보내기까지(초). 시도 번호(1·2)마다. 큐 권한·없는 큐처럼 반복해도 같은 실패는 길게 둔다.
RETRY_BACKOFF_SECONDS = (5.0, 30.0)
CONFIG_BACKOFF_SECONDS = 300.0
_CONFIG_REASONS = frozenset({"access_denied", "queue_not_found"})


class PublishOutcome(str, Enum):
    PUBLISHED = "published"
    RETRY_LATER = "retry_later"
    EXHAUSTED = "exhausted"
    SKIPPED = "skipped"  # 선점 실패·이미 처리됨 등(아무것도 하지 않음)
    HELD = "held"  # 보내기 전 확인 실패(형식·설정). 횟수를 쓰지 않음


def payload_problem(job: AsyncFittingJob) -> str | None:
    """보내기 전에 확인할 수 있는 문제. 없으면 None. 저장된 payload 를 바꾸지 않는다."""
    payload = job.result_payload
    try:
        checked = _check_result_payload(payload)
    except (TypeError, ValueError):
        return "invalid_payload"
    if checked["fitting_job_id"] != job.fitting_job_id:
        return "job_id_mismatch"
    if checked["status"] == "COMPLETED" and job.result_image_key != checked["result_image_key"]:
        return "image_key_mismatch"
    if checked["status"] == "FAILED" and job.result_image_key is not None:
        return "image_key_on_failure"
    return None


class ResultPublishProcessor:
    """동기. 작업 하나의 결과를 보낸다. DB 잠금을 쥔 채 SQS 를 기다리지 않는다(5단계 메서드는 짧은 트랜잭션)."""

    def __init__(
        self,
        jobs: Any,
        state: Any,
        sqs: Any,
        *,
        lease_seconds: float = 180.0,
        retry_backoff_seconds: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
        config_backoff_seconds: float = CONFIG_BACKOFF_SECONDS,
    ) -> None:
        self._jobs, self._state, self._sqs = jobs, state, sqs
        self._lease = lease_seconds
        self._backoff, self._config_backoff = retry_backoff_seconds, config_backoff_seconds

    def publish(self, fitting_job_id: int) -> PublishOutcome:
        grant = self._state.acquire_lease(fitting_job_id, NextAction.PUBLISH_RESULT, self._lease)
        if not grant.acquired:
            return PublishOutcome.SKIPPED
        token, job = grant.token, grant.job
        release = True
        try:
            if (
                job.result_payload is None
                or job.result_message_status is not ResultMessageStatus.PENDING
            ):
                return PublishOutcome.SKIPPED
            if job.result_message_attempt_count >= RESULT_MESSAGE_LIMIT:
                # 횟수를 다 썼는데 성공 기록이 없다(이전 시도 결과 불명확). 다시 보내지 않는다.
                return self._exhaust(job, token, "attempts_already_used")
            problem = payload_problem(job)
            if problem is not None:
                logger.error(
                    "result publish held fitting_job_id=%s reason=%s; operator check required",
                    job.fitting_job_id,
                    problem,
                )
                release = not self._hold(job, token, self._config_backoff)
                return PublishOutcome.HELD
            try:
                get_queue_url(FittingQueue.BE_FITTING_RESPONSE)
            except SqsConfigError:
                logger.error(
                    "result publish held fitting_job_id=%s reason=queue_not_configured",
                    job.fitting_job_id,
                )
                release = not self._hold(job, token, self._config_backoff)
                return PublishOutcome.HELD
            claim = self._state.claim_result_message_attempt(job.fitting_job_id, token)
            if not claim.applied:
                if claim.reason is ChangeReason.LIMIT_REACHED:
                    return self._exhaust(job, token, "attempts_already_used")
                return PublishOutcome.SKIPPED
            attempt = claim.job.result_message_attempt_count
            outcome, reason, message_id = self._send(job)
            fields = {
                "component": "result_publisher",
                "fitting_job_id": job.fitting_job_id,
                "result_status": job.result_payload.get("status"),
                "attempt": attempt,
            }
            if outcome is SqsOutcome.SUCCEEDED:
                # SQS 가 MessageId 와 함께 성공을 응답했다(BE 처리 완료 뜻은 아님).
                diag.event(
                    "result_sent", **fields, sqs_result="succeeded", sqs_message_id=message_id
                )
                final = AsyncJobStatus(job.result_payload["status"])
                try:
                    marked = self._state.mark_result_published(job.fitting_job_id, token, final)
                except Exception as error:
                    diag.event(
                        "result_sent_not_recorded",
                        logging.ERROR,
                        **fields,
                        sqs_message_id=message_id,
                        error_type=type(error).__name__,
                    )
                    raise
                if marked.applied:
                    diag.event("result_publish_recorded", **fields, sqs_message_id=message_id)
                    release = False  # mark_result_published 가 lease 를 정리했다
                    logger.info(
                        "result published fitting_job_id=%s status=%s attempt=%s",
                        job.fitting_job_id,
                        final.value,
                        attempt,
                    )
                    return PublishOutcome.PUBLISHED
                # SQS 는 받았지만 성공 기록을 남기지 못했다. 정확히 한 번 전달됐다고 가정하지 않는다(BE 는 중복을 견딘다).
                logger.error(
                    "result sent but not recorded fitting_job_id=%s attempt=%s reason=%s",
                    job.fitting_job_id,
                    attempt,
                    marked.reason.value,
                )
                reason = f"sent_not_recorded_{marked.reason.value}"
                diag.event(
                    "result_sent_not_recorded",
                    logging.ERROR,
                    **fields,
                    sqs_message_id=message_id,
                    reason=marked.reason.value,
                )
            else:
                # FAILED(전달 안 됨) 또는 UNKNOWN(전달됐을 수 있음). 성공으로 표시하지 않는다.
                diag.event(
                    "result_send_failed",
                    logging.WARNING,
                    **fields,
                    sqs_result=outcome.value,
                    reason=reason,
                )
            if attempt >= RESULT_MESSAGE_LIMIT:
                return self._exhaust(job, token, reason)
            backoff = (
                self._config_backoff
                if reason in _CONFIG_REASONS
                else self._backoff[min(attempt, len(self._backoff)) - 1]
            )
            logger.warning(
                "result publish will retry fitting_job_id=%s attempt=%s outcome=%s reason=%s "
                "backoff=%.0fs",
                job.fitting_job_id,
                attempt,
                outcome.value,
                reason,
                backoff,
            )
            release = not self._hold(job, token, backoff)
            return PublishOutcome.RETRY_LATER
        finally:
            if release:
                released = self._state.release_lease(job.fitting_job_id, token)
                if not released.applied and released.reason is not ChangeReason.LEASE_INVALID:
                    logger.warning(
                        "result publisher lease release failed fitting_job_id=%s reason=%s",
                        job.fitting_job_id,
                        released.reason.value,
                    )

    def _send(self, job: AsyncFittingJob) -> tuple[SqsOutcome, str, str | None]:
        try:
            sent = self._sqs.send_json(FittingQueue.BE_FITTING_RESPONSE, dict(job.result_payload))
        except Exception:
            logger.exception("result publish crashed fitting_job_id=%s", job.fitting_job_id)
            return SqsOutcome.UNKNOWN, "send_crashed", None
        return sent.outcome, sent.reason, getattr(sent, "message_id", None)

    def _hold(self, job: AsyncFittingJob, token: str, seconds: float) -> bool:
        """실행 권한을 seconds 동안 쥔 채 둔다(그동안 다른 루프가 선점·재전송하지 않음). 성공하면 True."""
        renewed = self._state.renew_lease(job.fitting_job_id, token, seconds)
        return renewed.applied

    def _exhaust(self, job: AsyncFittingJob, token: str, reason: str) -> PublishOutcome:
        marked = self._state.mark_result_retry_exhausted(job.fitting_job_id, token)
        # BE 전달을 확인하지 못했다. 결과·이미지 Key·횟수는 그대로다. 자동으로 다시 보내지 않는다(운영 확인 대상).
        logger.error(
            "result publish retry exhausted fitting_job_id=%s attempts=%s last_reason=%s "
            "recorded=%s; delivery to BE unconfirmed, operator check required",
            job.fitting_job_id,
            RESULT_MESSAGE_LIMIT,
            reason,
            marked.applied,
        )
        return PublishOutcome.EXHAUSTED


# --- 반복 루프 ---


@dataclass
class PublisherStatus:
    running: bool = False
    stopped: bool = False
    crashed: bool = False
    last_error: str | None = None
    consecutive_errors: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)


class ResultPublisherWorker:
    """FastAPI 이벤트 루프의 Task. 대상 조회·전송은 전용 thread 하나에서 한다(다른 HTTP 요청을 막지 않음)."""

    def __init__(
        self,
        find_due: Callable[[], list[int]],
        processor: ResultPublishProcessor,
        *,
        poll_seconds: float = 2.0,
        backoff_initial_seconds: float = 1.0,
        backoff_max_seconds: float = 60.0,
    ) -> None:
        self._find_due, self._processor = find_due, processor
        self._poll = poll_seconds
        self._backoff_initial, self._backoff_max = backoff_initial_seconds, backoff_max_seconds
        self._limiter = CapacityLimiter(1)
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.status = PublisherStatus()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    @property
    def alive(self) -> bool:
        """실제로 돌고 있다(시작했고, 멈추는 중이 아니고, 비정상 종료하지 않음)."""
        return self.status.running and not self.status.crashed and not self._stop.is_set()

    def start(self) -> asyncio.Task:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="virtual-fitting-result-publisher")
            self._task.add_done_callback(self._finished)
        return self._task

    async def stop(self, timeout: float) -> None:
        """새 선점을 멈추고 진행 중인 전송이 끝나기를 기다린다(그 thread 는 버리지 않는다 — DB pool 은 그 뒤에 닫힌다)."""
        self._stop.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout)
        except TimeoutError:
            logger.warning(
                "result publisher did not stop within %ss; waiting for the current send", timeout
            )
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _finished(self, task: asyncio.Task) -> None:
        self.status.running = False
        if task.cancelled():
            self.status.stopped = True
            return
        error = task.exception()
        if error is not None:
            self.status.crashed = True
            self.status.last_error = type(error).__name__
            logger.error("result publisher crashed", exc_info=error)
            diag.event(
                "component_crashed",
                logging.ERROR,
                component="result_publisher",
                error_type=type(error).__name__,
            )
        else:
            self.status.stopped = True

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    def _cycle(self) -> int:
        """대상을 찾아 하나씩 보낸다(동기, thread 에서). 종료 요청이면 다음 작업을 선점하지 않는다."""
        done = 0
        for job_id in self._find_due():
            if self._stop.is_set():
                break
            outcome = self._processor.publish(job_id)
            self.status.outcomes[outcome.value] = self.status.outcomes.get(outcome.value, 0) + 1
            done += 1
        return done

    async def _run(self) -> None:
        self.status.running = True
        logger.info("result publisher started")
        diag.event("component_started", component="result_publisher")
        while not self._stop.is_set():
            try:
                done = await to_thread.run_sync(self._cycle, limiter=self._limiter)
            except VirtualFittingError as error:
                self.status.consecutive_errors += 1
                self.status.last_error = error.code
                backoff = min(
                    self._backoff_max,
                    self._backoff_initial * 2 ** max(0, self.status.consecutive_errors - 1),
                )
                logger.warning(
                    "result publisher cycle failed reason=%s; backing off %.1fs",
                    error.code,
                    backoff,
                )
                await self._sleep(backoff)
                continue
            self.status.consecutive_errors = 0
            if not done:
                await self._sleep(self._poll)
        logger.info("result publisher stopped")
        diag.event("component_stopped", component="result_publisher")


# --- 준비 여부(FollowUpPaths) ---

_registered: ResultPublisherWorker | None = None


def register_result_publisher(worker: ResultPublisherWorker | None) -> None:
    """lifespan 이 실제로 시작한 루프를 등록한다(멈출 때 None). 다른 경로의 준비 판단이 이 상태를 본다."""
    global _registered
    _registered = worker


class ResultPublishPaths:
    """PUBLISH_RESULT 만, 실제 결과 전송 루프가 돌고 있고 그 결과가 아직 보낼 대상(PENDING)일 때 준비됐다고 답한다.

    큐 URL 설정·Router 존재만으로 준비됐다고 보지 않는다. 루프가 꺼져 있거나 멈췄거나 비정상 종료했으면 준비되지 않음.
    BE_FITTING_RESPONSE_QUEUE_URL 이 기존 로컬 검증(get_queue_url: 누락·형식·리전·FIFO)을 통과하지 못해도 준비되지 않음.
    이 확인은 로컬 검증뿐이다(AWS 에 요청하지 않으므로 IAM 권한·큐 존재까지 확인한 것은 아니다). 전송 횟수를 쓰지 않는다.
    RETRY_EXHAUSTED(자동 전송 소진)는 운영 복구 대상이라 준비되지 않음. RECOVER_UNKNOWN·Watchdog(15단계)도 준비되지 않음.
    """

    def __init__(self, worker: Callable[[], ResultPublisherWorker | None] = lambda: _registered):
        self._worker = worker

    def ready_for(self, job: AsyncFittingJob | None, step: NextStep) -> bool:
        if step is not NextStep.PUBLISH_RESULT or job is None:
            return False
        worker = self._worker()
        return (
            worker is not None
            and worker.alive
            and job.next_action is NextAction.PUBLISH_RESULT
            and job.result_payload is not None
            and job.result_message_status is ResultMessageStatus.PENDING
            and _result_queue_configured()
        )


def _result_queue_configured() -> bool:
    """결과 큐 URL 이 기존 SqsQueueClient 설정 검증을 통과하는지(로컬, AWS 요청 없음). URL 은 로그에 남기지 않는다."""
    try:
        get_queue_url(FittingQueue.BE_FITTING_RESPONSE)
    except SqsConfigError:
        return False
    return True


def create_result_publisher() -> ResultPublisherWorker:
    """실제 의존성으로 조립한다(blocking: DB pool·boto3). DB pool·SQS client 는 Worker 들과 같은 프로세스 전역 자원이다.

    long polling 용과 같은 기본 SQS client(get_sqs_client, total_max_attempts=1)를 쓴다(Webhook 의 짧은 timeout client 아님).
    """
    from app.clients.sqs import SqsQueueClient
    from app.config import settings
    from app.config.database import get_connection_pool
    from app.virtual_fitting.repositories.async_job_repository import AsyncJobRepository
    from app.virtual_fitting.repositories.async_job_state import (
        MAX_LEASE_SECONDS,
        AsyncJobStateRepository,
    )

    lease = settings.VIRTUAL_FITTING_RESULT_PUBLISHER_LEASE_SECONDS
    poll = settings.VIRTUAL_FITTING_RESULT_PUBLISHER_POLL_SECONDS
    if not 0 < lease <= MAX_LEASE_SECONDS:
        raise ValueError(
            "VIRTUAL_FITTING_RESULT_PUBLISHER_LEASE_SECONDS 는 0 초과 3600 이하여야 합니다."
        )
    if not 0 < poll <= 300:
        raise ValueError(
            "VIRTUAL_FITTING_RESULT_PUBLISHER_POLL_SECONDS 는 0 초과 300 이하여야 합니다."
        )
    pool = get_connection_pool()
    jobs = AsyncJobRepository(pool)
    processor = ResultPublishProcessor(
        jobs, AsyncJobStateRepository(pool), SqsQueueClient(), lease_seconds=lease
    )
    return ResultPublisherWorker(
        lambda: jobs.find_result_publish_due(20), processor, poll_seconds=poll
    )


# 요청·후처리 Worker·Webhook 이 쓰는 기본 준비 판단(등록된 루프의 실제 상태를 그때그때 본다).
def result_publish_paths() -> ResultPublishPaths:
    return ResultPublishPaths()
