import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol

from app.virtual_fitting.exceptions import (
    FittingBalanceExhaustedError,
    FittingDailyBudgetExceededError,
    FittingModelError,
    VirtualFittingError,
)
from app.virtual_fitting.providers.base import FittingInput
from app.virtual_fitting.providers.runware_async import SubmissionOutcome
from app.virtual_fitting.providers.runware_errors import AUTH_ERROR_CODES
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
    # 현재 시도가 확실한 미접수로 기록됐고 다음 재요청이 예약돼 있다(Watchdog 이 예약 시각 뒤 같은 Submitter 로 재요청).
    RESUBMIT_PRUNA = "resubmit_pruna"
    # AI 저장 작업이 생성 성공을 받았지만 결과 URL 이 없다. Watchdog 이 같은 task 를 조회해 참조를 복구한 뒤 POSTPROCESS 로 잇는다.
    RECOVER_RESULT_REF = "recover_result_ref"
    # 운영자가 원인(예: Runware 인증 오류)을 확인해야 이어 갈 수 있는 일. 자동 경로가 준비됐다고 답하지 않는다.
    OPERATOR_REVIEW = "operator_review"


class FollowUpPaths(Protocol):
    def ready_for(self, job: AsyncFittingJob, step: NextStep) -> bool:
        """그 일을 실제로 이어 갈 경로(엔드포인트·Worker·Watchdog)가 이 프로세스·배포에 있으면 True."""
        ...


class NoFollowUpPaths:
    """기본값: 이어 갈 경로가 아직 하나도 없다. 그래서 8단계는 접수 뒤에도 메시지를 지우지 않는다."""

    def ready_for(self, job: AsyncFittingJob, step: NextStep) -> bool:
        return False


def auth_hold_active(job: AsyncFittingJob) -> bool:
    """이 작업의 현재 비동기 생성 시도에 Runware 인증 오류 보류 기록(DB)이 있고, 아직 결과를 기다리는 중이다.

    기록은 지우지 않는다. 결과가 저장됐거나 다른 경로(POSTPROCESS·RUN_FALLBACK 등)로 진행했으면 보류로 보지 않는다.
    """
    return (
        job.auth_hold_task_uuid is not None
        and job.auth_hold_task_uuid == job.task_uuid
        and job.status in (AsyncJobStatus.RECEIVED, AsyncJobStatus.WAITING_PRUNA)
        and job.result_payload is None
        and job.next_action in (None, NextAction.SUBMIT_PRUNA)
    )


def awaiting_result_url(job: AsyncFittingJob | None) -> bool:
    """AI 저장 작업이 후처리 대기 중인데 결과를 내려받을 URL 이 없다(저장·결과 전). 후처리가 할 일이 없으므로 POSTPROCESS
    대신 Watchdog 의 같은 task 조회(참조 복구)가 이어 간다."""
    return (
        job is not None
        and job.ai_stores_result
        and job.verified_image_key is None
        and job.runware_image_url is None
        and job.status is AsyncJobStatus.POSTPROCESSING
        and job.next_action is NextAction.POSTPROCESS
        and job.result_payload is None
    )


def not_accepted_pending(job: AsyncFittingJob | None) -> bool:
    """현재 전송 시도가 확실한 미접수로 기록됐고, 결과·다른 진행 없이 다음 시도를 기다린다(예약 또는 보류)."""
    return (
        job is not None
        and job.status is AsyncJobStatus.RECEIVED
        and job.next_action is NextAction.SUBMIT_PRUNA
        and job.result_payload is None
        and job.pruna_not_accepted_at is not None
    )


def next_step_for(job: AsyncFittingJob) -> NextStep | None:
    """최신 작업 기록에서 다음 일을 정한다. 분명하지 않으면 None."""
    if job.status in (AsyncJobStatus.COMPLETED, AsyncJobStatus.FAILED):
        return None
    if auth_hold_active(job):
        # 인증 오류 보류는 일반 접수 대기·접수 불명확(await_webhook·recover_unknown)으로 보지 않는다(15단계 보완).
        return NextStep.OPERATOR_REVIEW
    if job.next_action is NextAction.PUBLISH_RESULT and job.result_payload is not None:
        return NextStep.PUBLISH_RESULT
    if job.next_action is NextAction.RUN_FALLBACK:
        return NextStep.RUN_FALLBACK
    if job.next_action is NextAction.POSTPROCESS:
        return NextStep.RECOVER_RESULT_REF if awaiting_result_url(job) else NextStep.POSTPROCESS
    if (
        job.status is AsyncJobStatus.WAITING_PRUNA
        and job.next_action is None
        and job.runware_accepted_at is not None
    ):
        return NextStep.AWAIT_WEBHOOK
    if not_accepted_pending(job):
        # 확실한 미접수: 접수 불명확(recover_unknown)이 아니다. 예약 시각이 없으면(표현 불가 대기·모순 근거) 운영 확인.
        return (
            NextStep.RESUBMIT_PRUNA if job.pruna_retry_at is not None else NextStep.OPERATOR_REVIEW
        )
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

        # 처음 준비하는 작업은 AI 저장 방식(uploadEndpoint 없이 요청, AI 가 결과 URL 에서 내려받아 저장)으로 확정한다.
        # task_uuid·결과 Key·저장 방식을 한 UPDATE 로 기록한다. 이미 준비된 작업(task_uuid 있음)은 바뀌지 않고 DB 기록을
        # 그대로 돌려받는다(기존 직접 업로드 작업은 false 유지). 이후 요청 조립은 이 DB 기록(prepared.job)만 따른다.
        prepared = self._state.prepare_pruna_submission(
            job_id,
            token,
            job.task_uuid or self._new_task_uuid(),
            job.async_image_key or async_result_key(job_id),
            ai_stores_result=True,
        )
        if not prepared.applied and prepared.reason.value != "already_recorded":
            logger.warning(
                "pruna submit not prepared fitting_job_id=%s reason=%s",
                job_id,
                prepared.reason.value,
            )
            return False
        job = prepared.job
        if not_accepted_pending(job):
            # 이전 시도가 확실한 미접수로 기록된 작업의 재요청(Watchdog 이 예약 시각 뒤에 부른다). 같은 작업·원래 입력·결과
            # Key·저장 방식으로 새 task_uuid 를 쓴다. 조건(예약 시각·남은 횟수·실행 권한)은 자리 확보 트랜잭션이 다시 확인한다.
            if job.pruna_retry_at is None:
                logger.warning("pruna resubmit held fitting_job_id=%s; operator check", job_id)
                return False
            return self._resubmit(job, token, request, check, heartbeat)
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

    def _resubmit(self, job, token, request, check, heartbeat) -> bool:
        new_task = self._new_task_uuid()
        granted, fitting_input, upload_url = self._wait_for_slot(
            job, token, request, check, heartbeat, retry_task=new_task
        )
        if not granted:
            return False
        logger.info(
            "pruna resubmit started fitting_job_id=%s previous_task_uuid=%s task_uuid=%s",
            job.fitting_job_id,
            job.task_uuid,
            new_task,
        )
        started = replace(job, task_uuid=new_task, pruna_not_accepted_at=None, pruna_retry_at=None)
        return self._send(started, token, fitting_input, upload_url, heartbeat)

    # --- 자리 확보 ---

    def _wait_for_slot(self, job, token, request, check, heartbeat, *, retry_task=None):
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
            # 결과 PUT URL 은 Runware 가 직접 올리는 기존 방식 작업에만 발급한다(AI 저장 작업은 발급하지 않음).
            upload_url = (
                None
                if job.ai_stores_result
                else result_upload_url(self._storage, job.async_image_key)
            )
            if retry_task is None:
                grant = self._limiter.begin_submission(job.fitting_job_id, token, job.task_uuid)
            else:
                grant = self._limiter.begin_retry_submission(
                    job.fitting_job_id, token, job.task_uuid, retry_task
                )
            if grant.granted:
                return True, fitting_input, upload_url
            if grant.reason in (SubmissionReason.RETRY_WAIT, SubmissionReason.RETRY_LIMIT_REACHED):
                # 예약 시각 전이거나 횟수를 다 썼다. 기다리지 않고 돌아간다(예약 시각에 Watchdog 이 다시 부른다).
                logger.info(
                    "pruna resubmit not started fitting_job_id=%s reason=%s",
                    job.fitting_job_id,
                    grant.reason.value,
                )
                return False, None, None
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
        from app.virtual_fitting.async_storage import async_output_format

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
                        output_format=async_output_format(job.async_image_key),
                        ai_stores_result=job.ai_stores_result,  # DB 에 확정된 저장 방식
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
        """명확한 거절(REJECTED). reason(runware_async.classify_response)별로 Webhook·getResponse 와 같은 기준을 쓴다.

        진행 중 자리는 이 task 의 것만 놓는다(이미 놓였으면 아무 일 없음). 호출 시작·task 기록은 지우지 않는다(재수신 때
        같은 요청을 무조건 다시 보내지 않음). fallback 은 생성 종료 실패 근거(generation_fallback)일 때만,
        FITTING_MODEL_FAILED 는 확정 생성 실패 근거(generation_failed)일 때만이다. 확실한 미접수(not_accepted)는 Webhook·
        상태 조회와 같은 기록 경로(record_not_accepted)가 미접수 기록·다음 시각 예약(또는 소진 결과)·자리 반환을 한
        트랜잭션으로 한다. 실제 재요청은 Watchdog(pruna_resubmit)이 그 기록을 읽어 같은 Submitter 로 한다.
        """
        if result.reason == "not_accepted":
            from app.virtual_fitting.providers.runware_errors import NOT_ACCEPTED_CODES
            from app.virtual_fitting.repositories.async_job_state import NotAcceptedSource

            code = next(code for code in result.error_codes if code in NOT_ACCEPTED_CODES)
            change = self._state.record_not_accepted(
                task_uuid,
                code,
                NotAcceptedSource.SUBMIT_RESPONSE,
                retry_after=result.retry_after,
                retry_after_unrepresentable=result.retry_after_unrepresentable,
            )
            logger.warning(
                "pruna submit not accepted fitting_job_id=%s task_uuid=%s code=%s attempt=%s "
                "outcome=%s retry_at=%s",
                job_id,
                task_uuid,
                code,
                change.attempt_no,
                change.outcome.value,
                change.job.pruna_retry_at.isoformat()
                if change.job and change.job.pruna_retry_at
                else None,
            )
            return
        self._limiter.release_in_flight(job_id, task_uuid=task_uuid)
        if result.reason == "generation_fallback":
            # 이 task 의 생성 종료 실패(timeoutProvider). Webhook 과 같은 기록으로 V1 fallback(13단계)으로 넘긴다.
            self._state.record_pruna_failed(task_uuid, recoverable=True)
            return
        if result.reason == "generation_failed":
            # 확정 생성 실패(fallback 대상 아님). Webhook 과 같은 기록·같은 최종 코드.
            self._state.record_pruna_failed(
                task_uuid,
                recoverable=False,
                failure_payload=failure_payload(
                    job_id, FittingModelError.code, FittingModelError.message
                ),
            )
            return
        if result.reason == "balance_exhausted":
            error = FittingBalanceExhaustedError("Runware 잔액이 부족합니다.")
            self._state.save_final_result(
                job_id, token, failure_payload(job_id, error.code, error.message)
            )
            return
        if result.reason == "auth_error" or result.http_status in (401, 403):
            # 우리 설정·권한 문제다. 사용자 실패로 확정하지 않고 운영 확인에 맡긴다(다시 보내지도 않음).
            # 재수신·재시작 뒤에도 같은 판단(operator_review)을 하도록 보류 기록을 남긴다. 기록이 실패하면 예외로
            # 올려 이 메시지를 지우지 않는다.
            reason = (
                f"http_{result.http_status}"
                if result.http_status in (401, 403)
                else next(code for code in result.error_codes if code in AUTH_ERROR_CODES)
            )
            change = self._state.record_auth_hold(task_uuid, reason)
            logger.error(
                "pruna submit rejected by auth fitting_job_id=%s status=%s hold=%s",
                job_id,
                result.http_status,
                change.reason.value,
            )
            return
        # 그 밖의 명확한 거절(입력·설정 등 4xx). 전용 정책이 없어 생성 실패 코드로 확정하지 않고 운영 확인으로 보류한다.
        logger.error(
            "pruna submit rejected without a dedicated policy fitting_job_id=%s http_status=%s codes=%s; "
            "held for operator check",
            job_id,
            result.http_status,
            ",".join(result.error_codes),
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
