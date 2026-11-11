import logging
import math
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import psycopg

from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.repositories.async_job_repository import ATTEMPTS, _check_job_id
from app.virtual_fitting.repositories.async_job_repository import TABLE as JOBS
from app.virtual_fitting.repositories.usage_repository import ADD_SQL, add_params

logger = logging.getLogger(__name__)

LIMITER = "virtual_fitting_pruna_limiter"
IN_FLIGHT = "virtual_fitting_pruna_in_flight"
SENDS = "virtual_fitting_pruna_sends"
COSTS = "virtual_fitting_task_costs"
_LOCK_TIMEOUT = "SET LOCAL lock_timeout = '5s'"
# 공유 보류 원인: Provider 가 초(float)로 표현하지 못함 / DB 가 시각·간격으로 계산하지 못함
HOLD_RETRY_AFTER_UNREPRESENTABLE = "retry_after_unrepresentable"
HOLD_RETRY_AFTER_OUT_OF_RANGE = "retry_after_out_of_range"
# 시각·간격·숫자 범위 초과(22008·22015·22003)
OUT_OF_RANGE_ERRORS = (
    psycopg.errors.DatetimeFieldOverflow,
    psycopg.errors.IntervalFieldOverflow,
    psycopg.errors.NumericValueOutOfRange,
)
_OUT_OF_RANGE = OUT_OF_RANGE_ERRORS


class SubmissionReason(str, Enum):
    GRANTED = "granted"
    NOT_FOUND = "not_found"
    LEASE_INVALID = "lease_invalid"
    STATE_MISMATCH = "state_mismatch"
    # 이미 호출 시작 기록이 있다. 실제로 전송됐을 수 있으므로 다시 보내지 않는다(확인·복구 대상).
    ALREADY_STARTED = "already_started"
    IN_FLIGHT_FULL = "in_flight_full"
    INTERVAL_WAIT = "interval_wait"
    # 표현할 수 없게 긴 대기 안내로 새 생성 요청을 보류 중이다(자동 해제 없음).
    SEND_HOLD = "send_hold"
    # fallback: 이미 1회를 썼다 / 같은 작업 번호의 자리를 다른 생성 시도가 쥐고 있다.
    FALLBACK_LIMIT_REACHED = "fallback_limit_reached"
    OTHER_ATTEMPT_IN_FLIGHT = "other_attempt_in_flight"
    SENDS_FULL = "sends_full"
    # 재요청: 미접수 뒤 예약한 다음 전송 시각 전이다 / 최대 전송 횟수(4)를 다 썼다.
    RETRY_WAIT = "retry_wait"
    RETRY_LIMIT_REACHED = "retry_limit_reached"


@dataclass(frozen=True)
class SubmissionGrant:
    granted: bool
    reason: SubmissionReason
    # INTERVAL_WAIT 일 때 다음 전송 가능 시각까지 남은 초(DB 시각 기준)
    wait_seconds: float | None = None
    started_at: datetime | None = None


class PrunaLimiter:
    def __init__(
        self,
        connection_pool: Any,
        *,
        max_in_flight: int,
        max_concurrent_sends: int,
        min_send_interval_seconds: float,
        send_hold_seconds: float,
    ) -> None:
        if max_in_flight < 1 or max_concurrent_sends < 1:
            raise ValueError("진행 중·동시 전송 자리는 1 이상이어야 합니다.")
        if min_send_interval_seconds < 0 or send_hold_seconds <= 0:
            raise ValueError("간격은 0 이상, 전송 자리 유지 시간은 0 보다 커야 합니다.")
        self._pool = connection_pool
        self.max_in_flight = max_in_flight
        self.max_concurrent_sends = max_concurrent_sends
        self.min_send_interval_seconds = float(min_send_interval_seconds)
        self.send_hold_seconds = float(send_hold_seconds)

    def begin_submission(self, fitting_job_id: int, token: str, task_uuid: str) -> SubmissionGrant:
        job_id = _check_job_id(fitting_job_id)
        task = uuid.UUID(task_uuid)

        def work(cursor) -> SubmissionGrant:
            cursor.execute(f"SELECT send_hold_reason FROM {LIMITER} WHERE id = 1 FOR UPDATE")
            limiter = cursor.fetchone()
            if limiter is None:
                raise FittingDatabaseError("Pruna 제한 상태 행이 없습니다(마이그레이션 확인).")
            cursor.execute(
                f"""
                SELECT status, next_action, task_uuid, pruna_submit_started_at,
                       worker_lease_token, worker_lease_until
                FROM {JOBS} WHERE fitting_job_id = %s FOR UPDATE
                """,
                (job_id,),
            )
            job = cursor.fetchone()
            if job is None:
                return SubmissionGrant(False, SubmissionReason.NOT_FOUND)
            # 잠금을 모두 얻은 뒤의 시각. 이 문장이 시작된 실제 시각이다.
            cursor.execute("SELECT statement_timestamp()")
            (now,) = cursor.fetchone()
            status, next_action, job_task, started, lease_token, lease_until = job
            if lease_token != token or lease_until is None or lease_until <= now:
                return SubmissionGrant(False, SubmissionReason.LEASE_INVALID)
            if started is not None:
                return SubmissionGrant(False, SubmissionReason.ALREADY_STARTED)
            if status != "RECEIVED" or next_action != "SUBMIT_PRUNA" or job_task != task:
                return SubmissionGrant(False, SubmissionReason.STATE_MISMATCH)

            refused = self._take_slot(cursor, job_id, task, now, limiter[0])
            if refused is not None:
                return refused
            cursor.execute(
                f"UPDATE {JOBS} SET pruna_submit_started_at = %s, updated_at = %s "
                "WHERE fitting_job_id = %s",
                (now, now, job_id),
            )
            # 1번 전송 시도 기록(전송 시작·횟수 확정). 같은 트랜잭션이라 시작 기록과 따로 남지 않는다.
            cursor.execute(
                f"INSERT INTO {ATTEMPTS} (fitting_job_id, attempt_no, task_uuid, started_at) "
                "VALUES (%s, 1, %s, %s)",
                (job_id, task, now),
            )
            return SubmissionGrant(True, SubmissionReason.GRANTED, started_at=now)

        return self._transaction(work, "Pruna 생성 요청 자리 확보")

    def begin_retry_submission(
        self, fitting_job_id: int, token: str, previous_task_uuid: str, task_uuid: str
    ) -> SubmissionGrant:
        """확실한 미접수 뒤 같은 작업의 다음 전송 시도(새 task_uuid)를 시작한다. begin_submission 과 같은 공유 제한을 같은
        순서로 적용하고, 자리 확보·간격 갱신과 새 시도 시작(task_uuid·pruna_submit_started_at·시도 줄)·예약 해제를 한
        트랜잭션으로 커밋한다. 외부 호출·대기는 하지 않는다. 기다리기만 하면 횟수를 쓰지 않는다.

        조건(잠금 뒤 DB 시각): 유효한 lease, RECEIVED / SUBMIT_PRUNA, 결과·접수 기록·인증 보류 없음, 현재 task 가
        previous_task_uuid 이고 그 시도가 미접수로 기록됐으며(pruna_not_accepted_at), 예약 시각(pruna_retry_at)이 지났고,
        지금까지 시도 수 < 4. 이전 시도 기록은 그대로 둔다. 기존 최초 준비 경로(prepare_pruna_submission)는 쓰지 않는다.
        """
        from app.virtual_fitting.repositories.async_job_state import PRUNA_SUBMIT_LIMIT

        job_id = _check_job_id(fitting_job_id)
        previous, task = uuid.UUID(previous_task_uuid), uuid.UUID(task_uuid)

        def work(cursor) -> SubmissionGrant:
            cursor.execute(f"SELECT send_hold_reason FROM {LIMITER} WHERE id = 1 FOR UPDATE")
            limiter = cursor.fetchone()
            if limiter is None:
                raise FittingDatabaseError("Pruna 제한 상태 행이 없습니다(마이그레이션 확인).")
            cursor.execute(
                f"""
                SELECT status, next_action, task_uuid, pruna_submit_started_at, runware_accepted_at,
                       result_payload IS NULL, auth_hold_task_uuid IS NULL,
                       pruna_not_accepted_at IS NOT NULL, pruna_retry_at IS NOT NULL,
                       EXTRACT(EPOCH FROM pruna_retry_at - statement_timestamp()),
                       worker_lease_token, worker_lease_until
                FROM {JOBS} WHERE fitting_job_id = %s FOR UPDATE
                """,
                (job_id,),
            )
            job = cursor.fetchone()
            if job is None:
                return SubmissionGrant(False, SubmissionReason.NOT_FOUND)
            cursor.execute("SELECT statement_timestamp()")
            (now,) = cursor.fetchone()
            (
                status,
                next_action,
                job_task,
                started,
                accepted,
                no_result,
                no_auth_hold,
                not_accepted,
                scheduled,
                retry_wait,
                lease_token,
                lease_until,
            ) = job
            if lease_token != token or lease_until is None or lease_until <= now:
                return SubmissionGrant(False, SubmissionReason.LEASE_INVALID)
            if (
                status != "RECEIVED"
                or next_action != "SUBMIT_PRUNA"
                or job_task != previous
                or started is None
                or accepted is not None
                or not no_result
                or not no_auth_hold
                or not not_accepted
                or not scheduled
            ):
                return SubmissionGrant(False, SubmissionReason.STATE_MISMATCH)
            cursor.execute(
                f"SELECT max(attempt_no), bool_and(not_accepted_at IS NOT NULL) "
                f"FILTER (WHERE task_uuid = %s) FROM {ATTEMPTS} WHERE fitting_job_id = %s",
                (previous, job_id),
            )
            attempts, previous_recorded = cursor.fetchone()
            if attempts is None or not previous_recorded:
                return SubmissionGrant(False, SubmissionReason.STATE_MISMATCH)
            if attempts >= PRUNA_SUBMIT_LIMIT:
                return SubmissionGrant(False, SubmissionReason.RETRY_LIMIT_REACHED)
            if retry_wait is not None and retry_wait > 0:
                return SubmissionGrant(
                    False, SubmissionReason.RETRY_WAIT, wait_seconds=float(retry_wait)
                )
            cursor.execute(
                f"SELECT task_uuid FROM {IN_FLIGHT} WHERE fitting_job_id = %s", (job_id,)
            )
            held = cursor.fetchone()
            if held is not None and held[0] != task:
                return SubmissionGrant(False, SubmissionReason.OTHER_ATTEMPT_IN_FLIGHT)
            refused = self._take_slot(cursor, job_id, task, now, limiter[0])
            if refused is not None:
                return refused
            cursor.execute(
                f"""
                UPDATE {JOBS} SET task_uuid = %s, pruna_submit_started_at = %s,
                       pruna_not_accepted_at = NULL, pruna_retry_at = NULL, updated_at = %s
                WHERE fitting_job_id = %s
                """,
                (task, now, now, job_id),
            )
            cursor.execute(
                f"INSERT INTO {ATTEMPTS} (fitting_job_id, attempt_no, task_uuid, started_at) "
                "VALUES (%s, %s, %s, %s)",
                (job_id, attempts + 1, task, now),
            )
            return SubmissionGrant(True, SubmissionReason.GRANTED, started_at=now)

        return self._transaction(work, "Pruna 재요청 자리 확보")

    def begin_fallback(
        self, fitting_job_id: int, token: str, fallback_task_uuid: str
    ) -> SubmissionGrant:
        """V1 fallback 생성 요청(13단계)의 자리 확보. begin_submission 과 같은 공유 제한(진행 중·동시 전송·간격·보류)을
        같은 순서로 적용하고, 자리 확보·간격 갱신과 fallback 1회 확보(fallback_attempt_count + 1)·시작 기록
        (fallback_started_at)을 한 트랜잭션으로 커밋한다. 외부 호출·대기는 하지 않는다.

        조건(잠금 뒤 DB 시각): 유효한 lease, FALLBACK_RUNNING / RUN_FALLBACK, 결과 없음, 저장된 fallback_task_uuid 와 같음,
        시작 기록 없음, fallback_attempt_count < 1. 기다리기만 하면(INTERVAL_WAIT 등) 횟수를 쓰지 않는다.
        같은 작업 번호의 자리를 다른 생성 시도(비동기 task 등)가 쥐고 있으면 덮어쓰지 않고 거절한다.
        """
        from app.virtual_fitting.repositories.async_job_state import FALLBACK_LIMIT

        job_id = _check_job_id(fitting_job_id)
        task = uuid.UUID(fallback_task_uuid)

        def work(cursor) -> SubmissionGrant:
            cursor.execute(f"SELECT send_hold_reason FROM {LIMITER} WHERE id = 1 FOR UPDATE")
            limiter = cursor.fetchone()
            if limiter is None:
                raise FittingDatabaseError("Pruna 제한 상태 행이 없습니다(마이그레이션 확인).")
            cursor.execute(
                f"""
                SELECT status, next_action, fallback_task_uuid, fallback_started_at,
                       fallback_attempt_count, result_payload IS NULL,
                       worker_lease_token, worker_lease_until
                FROM {JOBS} WHERE fitting_job_id = %s FOR UPDATE
                """,
                (job_id,),
            )
            job = cursor.fetchone()
            if job is None:
                return SubmissionGrant(False, SubmissionReason.NOT_FOUND)
            cursor.execute("SELECT statement_timestamp()")
            (now,) = cursor.fetchone()
            (
                status,
                next_action,
                job_task,
                started,
                attempts,
                no_result,
                lease_token,
                lease_until,
            ) = job
            if lease_token != token or lease_until is None or lease_until <= now:
                return SubmissionGrant(False, SubmissionReason.LEASE_INVALID)
            if started is not None:
                return SubmissionGrant(False, SubmissionReason.ALREADY_STARTED)
            if attempts >= FALLBACK_LIMIT:
                return SubmissionGrant(False, SubmissionReason.FALLBACK_LIMIT_REACHED)
            if (
                status != "FALLBACK_RUNNING"
                or next_action != "RUN_FALLBACK"
                or not no_result
                or job_task != task
            ):
                return SubmissionGrant(False, SubmissionReason.STATE_MISMATCH)
            cursor.execute(
                f"SELECT task_uuid FROM {IN_FLIGHT} WHERE fitting_job_id = %s", (job_id,)
            )
            held = cursor.fetchone()
            if held is not None and held[0] != task:
                return SubmissionGrant(False, SubmissionReason.OTHER_ATTEMPT_IN_FLIGHT)
            refused = self._take_slot(cursor, job_id, task, now, limiter[0])
            if refused is not None:
                return refused
            cursor.execute(
                f"UPDATE {JOBS} SET fallback_attempt_count = fallback_attempt_count + 1, "
                "fallback_started_at = %s, updated_at = %s "
                "WHERE fitting_job_id = %s AND fallback_attempt_count < %s",
                (now, now, job_id, FALLBACK_LIMIT),
            )
            return SubmissionGrant(True, SubmissionReason.GRANTED, started_at=now)

        return self._transaction(work, "Pruna fallback 자리 확보")

    def _take_slot(
        self, cursor, job_id: int, task: uuid.UUID, now, hold_reason
    ) -> SubmissionGrant | None:
        """잠금을 쥔 트랜잭션 안에서 진행 중·보류·간격·동시 전송을 확인하고 자리를 잡는다. 거절이면 그 Grant, 잡았으면 None."""
        cursor.execute(
            f"SELECT count(*) FILTER (WHERE fitting_job_id <> %s), "
            f"coalesce(bool_or(fitting_job_id = %s), false) FROM {IN_FLIGHT}",
            (job_id, job_id),
        )
        others, holds = cursor.fetchone()
        if not holds and others >= self.max_in_flight:
            return SubmissionGrant(False, SubmissionReason.IN_FLIGHT_FULL)
        if hold_reason is not None:
            return SubmissionGrant(False, SubmissionReason.SEND_HOLD)
        # 남은 대기는 DB 에서 초로 계산한다. 아주 먼 next_send_at(9999년 이후)은 Python datetime 으로 읽을 수 없다.
        cursor.execute(
            f"SELECT EXTRACT(EPOCH FROM next_send_at - %s) FROM {LIMITER} WHERE id = 1", (now,)
        )
        (remaining,) = cursor.fetchone()
        if remaining > 0:
            return SubmissionGrant(
                False, SubmissionReason.INTERVAL_WAIT, wait_seconds=float(remaining)
            )
        cursor.execute(f"DELETE FROM {SENDS} WHERE held_until <= %s", (now,))
        cursor.execute(f"SELECT count(*) FROM {SENDS} WHERE fitting_job_id <> %s", (job_id,))
        if cursor.fetchone()[0] >= self.max_concurrent_sends:
            return SubmissionGrant(False, SubmissionReason.SENDS_FULL)
        cursor.execute(
            f"INSERT INTO {IN_FLIGHT} (fitting_job_id, task_uuid, acquired_at) VALUES (%s, %s, %s) "
            "ON CONFLICT (fitting_job_id) DO NOTHING",
            (job_id, task, now),
        )
        cursor.execute(
            f"INSERT INTO {SENDS} (fitting_job_id, held_until) "
            "VALUES (%s, %s + make_interval(secs => %s)) "
            "ON CONFLICT (fitting_job_id) DO UPDATE SET held_until = EXCLUDED.held_until",
            (job_id, now, self.send_hold_seconds),
        )
        cursor.execute(
            f"UPDATE {LIMITER} SET next_send_at = %s + make_interval(secs => %s) WHERE id = 1",
            (now, self.min_send_interval_seconds),
        )
        return None

    def finish_send(self, fitting_job_id: int) -> None:
        """HTTP 전송이 끝났다(결과와 관계없이). 동시 전송 자리만 놓는다. 진행 중 자리는 그대로다."""
        job_id = _check_job_id(fitting_job_id)
        self._transaction(
            lambda cursor: cursor.execute(
                f"DELETE FROM {SENDS} WHERE fitting_job_id = %s", (job_id,)
            ),
            "Pruna 전송 자리 반환",
        )

    def defer_sends(self, seconds: float) -> bool:
        """429 등 대기 안내가 있으면 다음 생성 요청 가능 시각을 그 뒤로 미룬다(앞당기지 않음, 줄이지 않음).

        DB 시각(statement_timestamp) 기준으로 저장하므로 모든 Worker 에 같이 적용된다. 유한한 양수만 반영한다.
        유효하지만 DB 가 시각으로 표현할 수 없게 큰 값이면 공유 보류를 기록한다(같은 트랜잭션). 반영·보류했으면 True.
        """
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, int | float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            return False

        def work(cursor) -> bool:
            try:
                # savepoint: 범위 오류면 이 UPDATE 만 되돌린다
                with cursor.connection.transaction():
                    cursor.execute(
                        f"UPDATE {LIMITER} SET next_send_at = GREATEST(next_send_at, "
                        "statement_timestamp() + make_interval(secs => %s)) WHERE id = 1",
                        (float(seconds),),
                    )
            except _OUT_OF_RANGE:
                logger.error(
                    "pruna retry guidance cannot be stored seconds=%s; holding new sends", seconds
                )
                self._hold(cursor, HOLD_RETRY_AFTER_OUT_OF_RANGE)
            return True

        return self._transaction(work, "Pruna 전송 대기 반영")

    def hold_sends(self, reason: str) -> None:
        """새 생성 요청을 모든 Worker 에서 보류한다(이미 보류 중이면 처음 기록을 유지). 자동으로 풀리지 않는다."""
        self._transaction(lambda cursor: self._hold(cursor, reason), "Pruna 전송 보류 기록")

    @staticmethod
    def _hold(cursor, reason: str) -> None:
        cursor.execute(
            f"UPDATE {LIMITER} SET send_hold_reason = coalesce(send_hold_reason, %s), "
            "send_hold_at = coalesce(send_hold_at, statement_timestamp()) WHERE id = 1",
            (reason,),
        )

    def release_in_flight(self, fitting_job_id: int, task_uuid: str | None = None) -> bool:
        """진행 중 자리를 놓는다. 생성이 끝났다는 근거(완료·확정 종료·명확한 미접수)가 있을 때만 부른다.

        task_uuid 를 주면 그 생성 시도(task)가 잡은 자리만 놓는다. 같은 작업 번호라도 다른 시도가 잡은 자리는
        놓지 않는다(완료 Webhook·Watchdog 경로). 이미 놓였으면 False(중복 종료 이벤트로 두 번 반환되지 않는다).
        """
        job_id = _check_job_id(fitting_job_id)
        task = uuid.UUID(task_uuid) if task_uuid is not None else None

        def work(cursor) -> bool:
            cursor.execute(
                f"DELETE FROM {IN_FLIGHT} WHERE fitting_job_id = %s "
                "AND (%s::uuid IS NULL OR task_uuid = %s::uuid) RETURNING fitting_job_id",
                (job_id, task, task),
            )
            return cursor.fetchone() is not None

        return self._transaction(work, "Pruna 진행 중 자리 반환")

    def cancel_unsent(self, fitting_job_id: int, token: str, task_uuid: str) -> bool:
        """자리를 잡고 호출 시작 기록을 남겼지만 HTTP 요청을 **보내기 전**에 멈춘 것이 확실할 때만 되돌린다.

        보냈을 가능성이 있으면 부르지 않는다. 유효한 lease·같은 task·아직 접수 기록 없음이 조건이다.
        """
        job_id = _check_job_id(fitting_job_id)
        task = uuid.UUID(task_uuid)

        def work(cursor) -> bool:
            cursor.execute(f"SELECT 1 FROM {LIMITER} WHERE id = 1 FOR UPDATE")
            # 보내지 않은 시도의 줄과 그 직전 시도(재요청이면). 재요청 시도를 되돌릴 때는 작업을 직전 시도의 미접수·예약
            # 상태로 되돌린다(보내지 않은 것이 확실하므로 횟수를 쓰지 않음). 직전 시도 기록은 그대로다.
            cursor.execute(
                f"""
                SELECT p.task_uuid, p.started_at, p.not_accepted_at, p.next_attempt_at
                FROM {ATTEMPTS} a JOIN {ATTEMPTS} p
                  ON p.fitting_job_id = a.fitting_job_id AND p.attempt_no = a.attempt_no - 1
                WHERE a.task_uuid = %s AND a.fitting_job_id = %s
                """,
                (task, job_id),
            )
            previous = cursor.fetchone()
            if previous is None:
                assignments, params = "pruna_submit_started_at = NULL", ()
            else:
                assignments = (
                    "task_uuid = %s, pruna_submit_started_at = %s, pruna_not_accepted_at = %s, "
                    "pruna_retry_at = COALESCE(%s, statement_timestamp())"
                )
                params = previous
            cursor.execute(
                f"""
                UPDATE {JOBS} SET {assignments}, updated_at = statement_timestamp()
                WHERE fitting_job_id = %s AND task_uuid = %s AND status = 'RECEIVED'
                  AND runware_accepted_at IS NULL AND pruna_submit_started_at IS NOT NULL
                  AND worker_lease_token = %s AND worker_lease_until > statement_timestamp()
                RETURNING fitting_job_id
                """,
                (*params, job_id, task, token),
            )
            if cursor.fetchone() is None:
                return False
            cursor.execute(f"DELETE FROM {ATTEMPTS} WHERE task_uuid = %s", (task,))
            cursor.execute(f"DELETE FROM {IN_FLIGHT} WHERE fitting_job_id = %s", (job_id,))
            cursor.execute(f"DELETE FROM {SENDS} WHERE fitting_job_id = %s", (job_id,))
            return True

        return self._transaction(work, "보내지 않은 Pruna 제출 되돌리기")

    def _transaction(self, work, purpose: str) -> Any:
        try:
            with self._pool.connection() as connection:
                connection.autocommit = True
                with connection.transaction(), connection.cursor() as cursor:
                    cursor.execute(_LOCK_TIMEOUT)
                    return work(cursor)
        except psycopg.Error as error:
            raise FittingDatabaseError(f"AI PostgreSQL {purpose}에 실패했습니다.") from error


class TaskCostRepository:
    """생성 작업(task_uuid)별 비용을 한 번만 하루 개별 기록 합계(usage_amount.recorded_amount)에 더한다.
    Webhook·상태 조회 단계(9단계 이후)가 쓴다. V1 budget.record 와 같은 ADD_SQL 을 같은 트랜잭션에서 쓴다.
    Runware 보정값(reconciled_amount)에는 더하지 않는다. 판단 금액은 두 합계 중 큰 값이다(UsageRepository).

    비용을 받지 못했으면 부르지 않는다(0 으로 확정하지 않음). 같은 task 를 다시 기록하면 아무것도 바꾸지 않는다.
    UNKNOWN·Worker 종료를 이유로 기록을 지우거나 환급하지 않는다.
    """

    def __init__(self, connection_pool: Any) -> None:
        self._pool = connection_pool

    def record(self, task_uuid: str, fitting_job_id: int, cost: Decimal, usage_date: date) -> bool:
        if not isinstance(cost, Decimal) or not cost.is_finite() or cost < 0:
            raise ValueError(
                "cost 는 0 이상의 유한한 Decimal 이어야 합니다(받지 못한 비용은 기록하지 않음)."
            )
        task = uuid.UUID(task_uuid)
        job_id = _check_job_id(fitting_job_id)
        try:
            with self._pool.connection() as connection:
                connection.autocommit = True
                with connection.transaction(), connection.cursor() as cursor:
                    # usage_amount 행 잠금(UPSERT) 대기도 limiter 와 같이 이 트랜잭션에서만 제한한다.
                    cursor.execute(_LOCK_TIMEOUT)
                    cursor.execute(
                        f"INSERT INTO {COSTS} (task_uuid, fitting_job_id, cost, usage_date) "
                        "VALUES (%s, %s, %s, %s) ON CONFLICT (task_uuid) DO NOTHING RETURNING task_uuid",
                        (task, job_id, cost, usage_date),
                    )
                    if cursor.fetchone() is None:
                        return False
                    cursor.execute(ADD_SQL, add_params(usage_date, cost))
                    return True
        except psycopg.Error as error:
            raise FittingDatabaseError("AI PostgreSQL 작업 비용 기록에 실패했습니다.") from error
