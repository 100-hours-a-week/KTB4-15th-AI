"""가상피팅 V2 8단계: 요청 Worker 의 submit 연결 지점에 붙는 Pruna 비동기 제출.

순서(한 작업, 실행 권한을 가진 Worker 하나만):
  1. task_uuid(UUID v4)·async_image_key 를 AI DB 에 먼저 저장(이미 있으면 재사용)
  2. 공유 제한 자리 대기(진행 중·동시 전송·최소 간격). 기다리는 동안 lease·Visibility 를 단계마다 연장
  3. 오래된 예산·잔액 확인은 다시 확인(recheck_after_seconds)
  4. Presigned GET(원본)·PUT(결과) 발급 — 전송 직전, 로컬 서명, DB·로그에 남기지 않음
  5. 자리 확보 + pruna_submit_started_at 기록을 한 트랜잭션으로 커밋(PrunaLimiter.begin_submission)
  6. 1단계 Provider 로 한 번 전송(자동 재전송 없음). 전송 중에는 keepalive thread 가 lease·Visibility 를 연장
  7. ACCEPTED / REJECTED / UNKNOWN 에 따라 기록하고, 메시지를 지워도 되는지 판단

pruna_submit_started_at 은 "보냈을 수 있다"는 기록이다(접수 증거가 아님). 이 기록이 있으면 재수신·Worker 변경·lease
만료에도 다시 보내지 않는다. 메시지 삭제는 deletion_allowed 가 판단한다: 접수 결과나 복구에 필요한 기록이 있고,
다음 일이 분명하고, FollowUpPaths 가 그 일을 이어 갈 실제 경로가 있다고 답할 때만. 기본 NoFollowUpPaths 는 항상
"없음"이다(Webhook 9단계·결과 발행 14단계·Watchdog 15단계 미구현). 환경변수로 바꿀 수 없다.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from app.virtual_fitting.exceptions import (
    FittingBalanceExhaustedError,
    FittingDailyBudgetExceededError,
    VirtualFittingError,
)
from app.virtual_fitting.providers.base import FittingInput
from app.virtual_fitting.providers.runware_async import SubmissionOutcome
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncFittingJob,
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.pruna_limiter import (
    HOLD_RETRY_AFTER_UNREPRESENTABLE,
    SubmissionReason,
)

logger = logging.getLogger(__name__)


class NextStep(str, Enum):
    """메시지를 지운 뒤 이 작업을 이어 갈 일."""

    AWAIT_WEBHOOK = "await_webhook"  # 접수됨, 완료 알림 대기(9단계 Webhook)
    POSTPROCESS = "postprocess"  # 10~11단계
    RUN_FALLBACK = "run_fallback"  # 13단계
    PUBLISH_RESULT = "publish_result"  # 14단계
    RECOVER_UNKNOWN = "recover_unknown"  # 보냈을 수 있지만 접수 여부 불명(15단계 Watchdog)


class FollowUpPaths(Protocol):
    def ready_for(self, job: AsyncFittingJob, step: NextStep) -> bool:
        """그 일을 실제로 이어 갈 경로(엔드포인트·Worker·Watchdog)가 이 프로세스·배포에 있으면 True."""
        ...


class NoFollowUpPaths:
    """기본값: 이어 갈 경로가 아직 하나도 없다. 그래서 8단계는 접수 뒤에도 메시지를 지우지 않는다."""

    def ready_for(self, job: AsyncFittingJob, step: NextStep) -> bool:
        return False


def next_step_for(job: AsyncFittingJob) -> NextStep | None:
    """최신 작업 기록에서 다음 일을 정한다. 분명하지 않으면 None."""
    if job.status in (AsyncJobStatus.COMPLETED, AsyncJobStatus.FAILED):
        return None
    if job.next_action is NextAction.PUBLISH_RESULT and job.result_payload is not None:
        return NextStep.PUBLISH_RESULT
    if job.next_action is NextAction.RUN_FALLBACK:
        return NextStep.RUN_FALLBACK
    if job.next_action is NextAction.POSTPROCESS:
        return NextStep.POSTPROCESS
    if (
        job.status is AsyncJobStatus.WAITING_PRUNA
        and job.next_action is None
        and job.runware_accepted_at is not None
    ):
        return NextStep.AWAIT_WEBHOOK
    if (
        job.status is AsyncJobStatus.RECEIVED
        and job.next_action is NextAction.SUBMIT_PRUNA
        and job.task_uuid is not None
        and job.pruna_submit_started_at is not None
        and job.runware_accepted_at is None
    ):
        return NextStep.RECOVER_UNKNOWN
    return None


def deletion_allowed(job: AsyncFittingJob | None, follow_up: FollowUpPaths) -> bool:
    """삭제 직전의 최신 기록으로 판단한다. 기록·다음 일·실제 경로가 모두 있어야 True."""
    if job is None:
        return False
    if (
        job.status in (AsyncJobStatus.COMPLETED, AsyncJobStatus.FAILED)
        and job.result_message_status is ResultMessageStatus.PUBLISHED
    ):
        return True  # 결과 발행까지 끝났다(7단계와 같음)
    step = next_step_for(job)
    if step is None:
        return False
    return bool(follow_up.ready_for(job, step))


class Heartbeat(Protocol):
    def beat(self) -> bool: ...


@dataclass
class KeepaliveState:
    failed: bool = False
    beats: int = 0


@contextmanager
def keepalive(heartbeat: Heartbeat, period_seconds: float) -> Iterator[KeepaliveState]:
    """블록이 도는 동안 별도 thread 가 period 마다 heartbeat.beat() 를 부른다(lease·Visibility 연장).

    한 번이라도 실패하면 state.failed 를 세우고 멈춘다. 블록을 나갈 때 thread 가 끝날 때까지 기다린다(닫힌
    자원을 쓰지 않도록). 연장이 실패해도 진행 중인 HTTP 요청을 멈추지는 않는다(취소됐다고 볼 수 없으므로).
    """
    state = KeepaliveState()
    stop = threading.Event()

    def run() -> None:
        while not stop.wait(period_seconds):
            try:
                ok = heartbeat.beat()
            except Exception:
                logger.exception("keepalive heartbeat crashed")
                ok = False
            state.beats += 1
            if not ok:
                state.failed = True
                return

    thread = threading.Thread(target=run, name="pruna-submit-keepalive", daemon=True)
    thread.start()
    try:
        yield state
    finally:
        stop.set()
        thread.join()


def failure_payload(fitting_job_id: int, error_code: str, error_message: str) -> dict[str, Any]:
    return {
        "fitting_job_id": fitting_job_id,
        "status": "FAILED",
        "error_code": error_code,
        "error_message": error_message,
    }


class PrunaSubmitter:
    """RequestProcessor 의 submit 연결 지점(SubmitHandler). True 를 돌려줄 때만 요청 메시지를 지운다."""

    def __init__(
        self,
        *,
        jobs: Any,
        state: Any,
        limiter: Any,
        provider: Any,
        storage: Any,
        budget: Any,
        balance: Any,
        webhook_url: str,
        follow_up: FollowUpPaths | None = None,
        follow_up_queue: Any = None,
        should_stop: Callable[[], bool] = lambda: False,
        max_wait_seconds: float = 60.0,
        poll_seconds: float = 2.0,
        recheck_after_seconds: float = 30.0,
        keepalive_seconds: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        new_task_uuid: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self._jobs, self._state, self._limiter = jobs, state, limiter
        self._provider, self._storage = provider, storage
        self._budget, self._balance = budget, balance
        self._webhook_url = webhook_url
        self._follow_up = follow_up or NoFollowUpPaths()
        # 제출 거절(재시도 가능 코드)로 RUN_FALLBACK 이 된 작업의 내부 큐 발행에 쓴다(SqsQueueClient). 없으면 발행하지 않는다.
        self._follow_up_queue = follow_up_queue
        self._should_stop = should_stop
        self._max_wait, self._poll = max_wait_seconds, poll_seconds
        self._recheck_after, self._keepalive = recheck_after_seconds, keepalive_seconds
        self._clock, self._sleep, self._new_task_uuid = clock, sleep, new_task_uuid

    def __call__(self, job, token, request, check, heartbeat) -> bool:
        from app.virtual_fitting.async_storage import async_result_key

        job_id = job.fitting_job_id
        if not self._webhook_url.startswith("https://"):
            logger.error(
                "pruna submit skipped fitting_job_id=%s: webhook url is not configured", job_id
            )
            return False

        prepared = self._state.prepare_pruna_submission(
            job_id, token, job.task_uuid or self._new_task_uuid(), async_result_key(job_id)
        )
        if not prepared.applied and prepared.reason.value != "already_recorded":
            logger.warning(
                "pruna submit not prepared fitting_job_id=%s reason=%s",
                job_id,
                prepared.reason.value,
            )
            return False
        job = prepared.job
        if job.pruna_submit_started_at is not None:
            # 보냈을 수 있는 작업이다. 다시 보내지 않고 복구 경로에 맡긴다.
            logger.warning("pruna submit already started fitting_job_id=%s; not resending", job_id)
            return self._decide(job_id)

        granted, fitting_input, upload_url = self._wait_for_slot(
            job, token, request, check, heartbeat
        )
        if not granted:
            return False
        return self._send(job, token, fitting_input, upload_url, heartbeat)

    # --- 자리 확보 ---

    def _wait_for_slot(self, job, token, request, check, heartbeat):
        from app.virtual_fitting.async_storage import result_upload_url, user_image_get_url
        from app.virtual_fitting.prompt import build_fitting_prompt

        started = self._clock()
        last_check = started
        while True:
            if self._should_stop() or not heartbeat.beat():
                return False, None, None
            if self._clock() - last_check >= self._recheck_after:
                if not self._recheck(job, token):
                    return False, None, None
                last_check = self._clock()
            # 전송 직전 발급(로컬 서명). URL 은 저장·로그하지 않는다.
            fitting_input = FittingInput(
                person_image_url=user_image_get_url(self._storage, request.user_image_key),
                garment_image_urls=[product.image_url for product in check.products],
                prompt=build_fitting_prompt(list(check.products)),
            )
            upload_url = result_upload_url(self._storage, job.async_image_key)
            grant = self._limiter.begin_submission(job.fitting_job_id, token, job.task_uuid)
            if grant.granted:
                return True, fitting_input, upload_url
            if grant.reason is SubmissionReason.SEND_HOLD:
                # 공유 보류 중. 기다려도 풀리지 않으므로(운영 확인 필요) 보내지 않고 메시지를 남긴 채 보류한다.
                logger.error(
                    "pruna submit held fitting_job_id=%s reason=%s; not sending",
                    job.fitting_job_id,
                    grant.reason.value,
                )
                return False, None, None
            if grant.reason not in (
                SubmissionReason.INTERVAL_WAIT,
                SubmissionReason.IN_FLIGHT_FULL,
                SubmissionReason.SENDS_FULL,
            ):
                logger.warning(
                    "pruna submit refused fitting_job_id=%s reason=%s",
                    job.fitting_job_id,
                    grant.reason.value,
                )
                return False, None, None
            waited = self._clock() - started
            if waited >= self._max_wait:
                logger.info(
                    "pruna submit postponed fitting_job_id=%s reason=%s waited=%.1fs",
                    job.fitting_job_id,
                    grant.reason.value,
                    waited,
                )
                return False, None, None
            self._sleep(min(self._poll, grant.wait_seconds or self._poll, self._max_wait - waited))

    def _recheck(self, job, token) -> bool:
        """기다리는 동안 오래된 예산·잔액 확인을 다시 한다. 확정 실패면 결과를 저장한다(fallback 없음)."""
        try:
            self._budget.ensure_available()
            self._balance.ensure_available()
        except (FittingDailyBudgetExceededError, FittingBalanceExhaustedError) as error:
            self._state.save_final_result(
                job.fitting_job_id,
                token,
                failure_payload(job.fitting_job_id, error.code, error.message),
            )
            return False
        except VirtualFittingError as error:  # 잔액 조회 실패·DB 오류 등 → 보류
            logger.warning(
                "pruna submit recheck postponed fitting_job_id=%s reason=%s",
                job.fitting_job_id,
                error.code,
            )
            return False
        return True

    # --- 전송과 결과 ---

    def _send(self, job, token, fitting_input, upload_url, heartbeat) -> bool:
        job_id, task_uuid = job.fitting_job_id, job.task_uuid
        invoked = False
        try:
            with keepalive(heartbeat, self._keepalive) as alive:
                try:
                    invoked = True
                    result = self._provider.submit(
                        fitting_input,
                        task_uuid=task_uuid,
                        webhook_url=self._webhook_url,
                        upload_endpoint=upload_url,
                    )
                except ValueError:
                    # Provider 는 입력 검증 오류를 전송 전에 낸다. 보내지 않은 것이 확실하다.
                    invoked = False
                    raise
        except ValueError:
            logger.exception("pruna submit input rejected before sending fitting_job_id=%s", job_id)
            self._limiter.cancel_unsent(job_id, token, task_uuid)
            return False
        except Exception:
            if not invoked:
                self._limiter.cancel_unsent(job_id, token, task_uuid)
                raise
            # 보냈을 수 있다. 시작 기록·진행 중 자리를 그대로 두고 복구 대상으로 남긴다.
            logger.exception("pruna submit crashed after invoking fitting_job_id=%s", job_id)
            return False
        finally:
            try:
                self._limiter.finish_send(job_id)
            except VirtualFittingError:
                logger.exception("pruna send slot release failed fitting_job_id=%s", job_id)

        # Runware 대기 안내(초, Provider 가 본문·헤더 중 유효한 긴 값으로 정함)를 줄이지 않고 공유 대기 시각에 저장한다.
        # 다른 Worker 도 이 시각 전에는 생성 요청 권한을 받지 못한다. 같은 작업을 다시 보내지는 않는다.
        # 표현할 수 없게 긴 안내면 기본 간격만 지키고 보내지 않도록 공유 보류를 기록한다.
        if result.retry_after_unrepresentable:
            logger.error(
                "pruna retry guidance unrepresentable fitting_job_id=%s; holding new sends", job_id
            )
            self._limiter.hold_sends(HOLD_RETRY_AFTER_UNREPRESENTABLE)
        elif result.retry_after is not None:
            self._limiter.defer_sends(result.retry_after)

        if result.outcome is SubmissionOutcome.ACCEPTED:
            recorded = self._state.record_submission_accepted(task_uuid)
            if not recorded.applied:
                logger.error(
                    "pruna accepted but not recorded fitting_job_id=%s reason=%s",
                    job_id,
                    recorded.reason.value,
                )
                return False
        elif result.outcome is SubmissionOutcome.REJECTED:
            self._rejected(job_id, token, task_uuid, result)
        else:
            logger.warning(
                "pruna submit outcome unknown fitting_job_id=%s reason=%s; kept for recovery",
                job_id,
                result.reason,
            )
        if alive.failed:
            return False  # lease·Visibility 연장이 실패했다. 이 receipt 로 지우지 않는다.
        return self._decide(job_id)

    def _rejected(self, job_id, token, task_uuid, result) -> None:
        # 명확한 미접수. 진행 중 자리를 놓는다(이미 놓였으면 아무 일 없음).
        self._limiter.release_in_flight(job_id)
        if result.reason == "balance_exhausted":
            error = FittingBalanceExhaustedError("Runware 잔액이 부족합니다.")
            self._state.save_final_result(
                job_id, token, failure_payload(job_id, error.code, error.message)
            )
            return
        if result.http_status in (401, 403):
            # 우리 설정·권한 문제다. 사용자 실패로 확정하지 않고 운영 확인에 맡긴다(다시 보내지도 않음).
            logger.error(
                "pruna submit rejected by auth fitting_job_id=%s status=%s",
                job_id,
                result.http_status,
            )
            return
        if result.retryable:
            # Runware 가 일시적인 원인을 알렸다. 허용된 다음 작업(V1 fallback, 13단계)으로 넘긴다.
            self._state.record_pruna_failed(task_uuid, recoverable=True)
            return
        error_code, error_message = "FITTING_MODEL_FAILED", "가상 피팅 이미지 생성에 실패했습니다."
        self._state.save_final_result(
            job_id, token, failure_payload(job_id, error_code, error_message)
        )

    def _decide(self, job_id: int) -> bool:
        from app.virtual_fitting.follow_up_queue import publish_follow_up, waiting_for_fallback

        latest = self._jobs.find_by_fitting_job_id(job_id)
        if waiting_for_fallback(latest) and self._follow_up_queue is not None:
            # 거절로 RUN_FALLBACK 이 됐다. DB 커밋 뒤(트랜잭션 밖) RUN_FALLBACK 을 발행하고, 등록이 확인될 때만 요청 메시지를
            # 지운다(다음 실행 경로 확보). 실패·UNKNOWN 이면 남겨 두고 재수신 때 다시 발행한다.
            published = publish_follow_up(
                self._follow_up_queue, latest, NextAction.RUN_FALLBACK
            ).succeeded
            logger.info(
                "pruna submit fallback hand-off fitting_job_id=%s published=%s", job_id, published
            )
            return published
        allowed = deletion_allowed(latest, self._follow_up)
        logger.info(
            "pruna submit message decision fitting_job_id=%s next_step=%s delete=%s",
            job_id,
            next_step_for(latest).value if latest and next_step_for(latest) else None,
            allowed,
        )
        return allowed
