import logging
import time
from collections.abc import Callable
from typing import Any

from app.virtual_fitting.async_request import FittingRequest
from app.virtual_fitting.pruna_submitter import failure_payload, not_accepted_pending
from app.virtual_fitting.repositories.async_job_repository import NextAction
from app.virtual_fitting.repositories.async_job_state import ChangeReason
from app.virtual_fitting.request_worker import PrecheckStatus

logger = logging.getLogger(__name__)


class LeaseHeartbeat:
    """DB 실행 권한만 연장한다(SQS 메시지가 없다). interval 마다 한 번, 실패하면 False."""

    def __init__(
        self,
        state: Any,
        fitting_job_id: int,
        token: str,
        *,
        lease_seconds: float,
        interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._state, self._job_id, self._token = state, fitting_job_id, token
        self._lease, self._interval, self._clock = lease_seconds, interval_seconds, clock
        self._last = clock()

    def beat(self) -> bool:
        if self._clock() - self._last < self._interval:
            return True
        renewed = self._state.renew_lease(self._job_id, self._token, self._lease)
        if not renewed.applied:
            logger.warning(
                "pruna resubmit lease renewal failed fitting_job_id=%s reason=%s",
                self._job_id,
                renewed.reason.value,
            )
            return False
        self._last = self._clock()
        return True


class PrunaResubmitter:
    """작업 번호 하나의 예약된 재요청을 한 번 시도하고 짧은 결과 코드를 돌려준다(Watchdog 보고용). 예외를 밖으로 내지 않는다."""

    def __init__(
        self,
        *,
        jobs: Any,
        state: Any,
        precheck: Any,
        submit: Callable[..., bool],
        lease_seconds: float,
        heartbeat_interval_seconds: float | None = None,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> None:
        self._jobs, self._state, self._precheck, self._submit = jobs, state, precheck, submit
        self._lease = lease_seconds
        self._interval = (
            lease_seconds / 3 if heartbeat_interval_seconds is None else heartbeat_interval_seconds
        )
        self._should_stop = should_stop

    def __call__(self, fitting_job_id: int) -> str:
        try:
            return self._run(fitting_job_id)
        except Exception:
            # 보낸 뒤의 기록은 Submitter 가 남긴다(보냈을 수 있으면 시작 기록이 남아 다시 보내지 않음). 다음 점검에서 다시 본다.
            logger.exception("pruna resubmit failed fitting_job_id=%s", fitting_job_id)
            return "error"

    def _run(self, job_id: int) -> str:
        if self._should_stop():
            return "stopping"
        grant = self._state.acquire_lease(job_id, NextAction.SUBMIT_PRUNA, self._lease)
        if not grant.acquired:
            return f"lease_{grant.reason.value}"  # 다른 Worker·Watchdog 이 처리 중이거나 상태가 바뀌었다
        token = grant.token
        try:
            job = grant.job
            if not not_accepted_pending(job) or job.pruna_retry_at is None:
                return "not_scheduled"
            if job.user_image_key is None or not job.products:
                # 원래 입력이 정리됐다. 다른 입력으로 만들지 않는다(운영 확인).
                logger.error("pruna resubmit input missing fitting_job_id=%s", job_id)
                return "input_missing"
            request = FittingRequest(
                fitting_job_id=job_id,
                requested_at=job.requested_at,
                user_image_key=job.user_image_key,
                product_codes=tuple(job.products),
            )
            heartbeat = LeaseHeartbeat(
                self._state,
                job_id,
                token,
                lease_seconds=self._lease,
                interval_seconds=self._interval,
            )
            check = self._precheck.run(
                request, before_step=lambda: not self._should_stop() and heartbeat.beat()
            )
            if check.status is PrecheckStatus.RETRY:
                return f"precheck_retry_{check.reason}"
            if check.status is PrecheckStatus.FAILED:
                # 요청 Worker 와 같은 정책: 확정된 사전 확인 실패(상품·사진·예산·잔액)는 그 결과로 끝낸다(fallback 없음).
                saved = self._state.save_final_result(
                    job_id, token, failure_payload(job_id, check.error_code, check.error_message)
                )
                if saved.reason not in (ChangeReason.APPLIED, ChangeReason.ALREADY_RECORDED):
                    return f"precheck_failure_not_saved_{saved.reason.value}"
                return f"precheck_failed_{check.error_code}"
            if not heartbeat.beat():
                return "heartbeat_failed"
            self._submit(job, token, request, check, heartbeat)
            latest = self._jobs.find_by_fitting_job_id(job_id)
            sent = latest is not None and latest.task_uuid != job.task_uuid
            return "sent" if sent else "not_sent"
        finally:
            released = self._state.release_lease(job_id, token)
            if not released.applied and released.reason is not ChangeReason.LEASE_INVALID:
                logger.warning(
                    "pruna resubmit lease release failed fitting_job_id=%s reason=%s",
                    job_id,
                    released.reason.value,
                )
