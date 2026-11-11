import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol

import psycopg
from anyio import CapacityLimiter, to_thread

from app.clients.s3 import S3ObjectState
from app.clients.sqs import FittingQueue, SqsConfigError, get_queue_url
from app.virtual_fitting import diagnostics as diag
from app.virtual_fitting.exceptions import FittingDatabaseError, VirtualFittingError
from app.virtual_fitting.follow_up_queue import publish_follow_up
from app.virtual_fitting.providers.runware_errors import TaskErrorKind, classify_task_error
from app.virtual_fitting.providers.runware_status import TaskStatusKind
from app.virtual_fitting.pruna_submitter import NextStep, auth_hold_active, not_accepted_pending
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncFittingJob,
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import S3_CHECK_LIMIT, ChangeReason
from app.virtual_fitting.runware_webhook import (
    EventKind,
    WebhookEvent,
    WebhookFollowUpError,
    WebhookFollowUpNotReadyError,
    WebhookUnknownTaskError,
)

logger = logging.getLogger(__name__)

CHECKS = "virtual_fitting_watchdog_checks"
ALERTS = "virtual_fitting_watchdog_alerts"
_LOCK_TIMEOUT = "SET LOCAL lock_timeout = '5s'"
# 객체 시각과 DB 시각 사이 허용 오차(초). 이보다 이전에 저장된 객체는 이 시도의 결과로 보지 않는다.
CLOCK_SKEW_SECONDS = 300


@dataclass(frozen=True)
class WatchdogSettings:
    check_interval_seconds: float = 60.0  # 같은 작업을 다시 점검하기까지
    batch_size: int = 20
    query_after_seconds: float = 180.0  # 접수 확인(또는 호출 시작) 후 상태 조회 시작
    alert_after_seconds: float = 300.0  # 장기 지연 알림
    stale_seconds: float = 600.0  # 후처리·fallback·자리·전송 장기 미진행
    republish_interval_seconds: float = 600.0
    realert_seconds: float = 900.0
    # 최종 결과 뒤 비용 누락 점검: 호출 시작 후 이 기간 안의 작업만(과거 종료 작업 전체를 매번 조회하지 않음),
    # 결론이 나지 않으면 이 간격으로만 다시 본다. Runware 결과 보관 기간은 문서에 없어 실측 전 초기값이다.
    cleanup_window_seconds: float = 86400.0
    cleanup_interval_seconds: float = 900.0


# --- 알림 ---


@dataclass(frozen=True)
class WatchdogAlert:
    """운영 이상 이벤트(Cloud 모니터링 연결용 형식). 토큰·URL·사진·상품 입력·외부 오류 원문을 담지 않는다."""

    reason: str
    stage: str
    fitting_job_id: int | None
    first_seen_at: str
    last_seen_at: str
    seen_count: int

    def as_event(self) -> dict[str, Any]:
        return {"event": "virtual_fitting.watchdog.alert", **self.__dict__}


class AlertSink(Protocol):
    def emit(self, alert: WatchdogAlert) -> None: ...


class LogAlertSink:
    """기본 알림: 구조화 로그 한 줄(JSON). 실제 Slack·이메일·SNS 전송은 연결돼 있지 않다(Cloud 로그 지표로 연결)."""

    def emit(self, alert: WatchdogAlert) -> None:
        logger.error("watchdog_alert %s", json.dumps(alert.as_event(), ensure_ascii=False))


# --- DB 기록 ---


class WatchdogRepository:
    """점검 선점·알림 중복 억제·대상 조회. 짧은 트랜잭션만, 5초 잠금 대기 제한."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def _run(self, work: Callable[[Any], Any]) -> Any:
        try:
            with self._pool.connection() as connection:
                connection.autocommit = True
                with connection.transaction(), connection.cursor() as cursor:
                    cursor.execute(_LOCK_TIMEOUT)
                    return work(cursor)
        except psycopg.Error as error:
            raise FittingDatabaseError("AI PostgreSQL Watchdog 기록에 실패했습니다.") from error

    def candidates(self, config: WatchdogSettings) -> list[int]:
        """점검할 작업 번호. 다음 점검 시각이 지난 것만, 아직 점검하지 않았거나 오래전에 점검한 것부터(굶주림 방지)."""

        def work(cursor):
            cursor.execute(
                f"""
                SELECT j.fitting_job_id FROM virtual_fitting_async_jobs j
                LEFT JOIN {CHECKS} c USING (fitting_job_id)
                WHERE (c.next_check_at IS NULL OR c.next_check_at <= statement_timestamp())
                  AND (
                    EXISTS (
                      SELECT 1 FROM virtual_fitting_pruna_in_flight f
                      WHERE f.fitting_job_id = j.fitting_job_id
                        AND f.acquired_at <= statement_timestamp() - make_interval(secs => %(stale)s)
                    )
                    OR (j.status NOT IN ('COMPLETED', 'FAILED') AND (
                      (j.status = 'WAITING_PRUNA' AND j.next_action IS NULL
                        AND j.runware_accepted_at <= statement_timestamp() - make_interval(secs => %(query)s))
                      OR (j.status = 'RECEIVED' AND j.next_action = 'SUBMIT_PRUNA'
                        AND j.runware_accepted_at IS NULL AND j.pruna_not_accepted_at IS NULL
                        AND j.pruna_submit_started_at
                            <= statement_timestamp() - make_interval(secs => %(query)s))
                      -- 확실한 미접수로 기록된 시도: 예약 시각이 지난 재요청(실행 권한이 비어 있을 때), 또는 보류(운영 확인 알림)
                      OR (j.status = 'RECEIVED' AND j.next_action = 'SUBMIT_PRUNA'
                        AND j.result_payload IS NULL AND j.pruna_not_accepted_at IS NOT NULL
                        AND (j.pruna_retry_at IS NULL OR (j.pruna_retry_at <= statement_timestamp()
                          AND (j.worker_lease_token IS NULL OR j.worker_lease_until <= statement_timestamp()))))
                      OR (j.next_action IN ('POSTPROCESS', 'RUN_FALLBACK')
                        AND (j.worker_lease_token IS NULL OR j.worker_lease_until <= statement_timestamp())
                        AND j.updated_at <= statement_timestamp() - make_interval(secs => %(stale)s))
                      -- AI 저장 작업인데 결과 URL 이 없다(UUID 만 받은 성공 등): 접수 대기와 같이 접수 3분 뒤부터 같은 task
                      -- 조회. updated_at 은 Worker 의 실행 권한 선점·반납마다 바뀌어(URL 이 없어 보류해도) 기준으로 쓰지 않는다.
                      OR (j.ai_stores_result AND j.status = 'POSTPROCESSING' AND j.next_action = 'POSTPROCESS'
                        AND j.result_payload IS NULL AND j.verified_image_key IS NULL
                        AND j.runware_image_url IS NULL
                        AND (j.worker_lease_token IS NULL OR j.worker_lease_until <= statement_timestamp())
                        AND j.runware_accepted_at <= statement_timestamp() - make_interval(secs => %(query)s))
                      -- AI 저장 작업의 후처리 장기 미진행: 단계 시작(성공 전환·저장 확인) 기준. lease·재수신으로 밀리지 않는다.
                      OR (j.ai_stores_result AND j.next_action = 'POSTPROCESS' AND j.result_payload IS NULL
                        AND (j.worker_lease_token IS NULL OR j.worker_lease_until <= statement_timestamp())
                        AND COALESCE(j.image_verified_at, j.runware_accepted_at) <= statement_timestamp() - make_interval(secs => %(stale)s))
                      OR (j.next_action = 'PUBLISH_RESULT' AND j.result_payload IS NOT NULL AND (
                        j.result_message_status = 'RETRY_EXHAUSTED'
                        OR j.updated_at <= statement_timestamp() - make_interval(secs => %(stale)s)))
                    ))
                    -- 최종 결과 뒤 비동기 task 의 비용 기록이 없음(접수 확인된 task, 기간 안, 점검 미완료만)
                    OR (j.status IN ('COMPLETED', 'FAILED') AND j.task_uuid IS NOT NULL
                      AND j.runware_accepted_at IS NOT NULL
                      AND j.pruna_submit_started_at
                          >= statement_timestamp() - make_interval(secs => %(window)s)
                      AND j.updated_at <= statement_timestamp() - make_interval(secs => %(query)s)
                      AND c.cleanup_settled_at IS NULL
                      AND NOT EXISTS (
                        SELECT 1 FROM virtual_fitting_task_costs t WHERE t.task_uuid = j.task_uuid
                      ))
                  )
                ORDER BY c.next_check_at NULLS FIRST, j.fitting_job_id
                LIMIT %(limit)s
                """,
                {
                    "stale": config.stale_seconds,
                    "query": config.query_after_seconds,
                    "window": config.cleanup_window_seconds,
                    "limit": config.batch_size,
                },
            )
            return [row[0] for row in cursor.fetchall()]

        return self._run(work)

    def claim_check(self, job_id: int, interval_seconds: float) -> bool:
        """이 작업의 이번 점검을 선점한다(여러 인스턴스 중 한 곳만). 작업 테이블은 바꾸지 않는다."""

        def work(cursor):
            cursor.execute(
                f"""
                INSERT INTO {CHECKS} (fitting_job_id, checked_at, next_check_at)
                VALUES (%(job)s, statement_timestamp(), statement_timestamp() + make_interval(secs => %(i)s))
                ON CONFLICT (fitting_job_id) DO UPDATE
                  SET checked_at = EXCLUDED.checked_at, next_check_at = EXCLUDED.next_check_at
                  WHERE {CHECKS}.next_check_at <= statement_timestamp()
                RETURNING fitting_job_id
                """,
                {"job": job_id, "i": interval_seconds},
            )
            return cursor.fetchone() is not None

        return self._run(work)

    def claim_republish(self, job_id: int, interval_seconds: float) -> bool:
        def work(cursor):
            cursor.execute(
                f"""
                UPDATE {CHECKS} SET follow_up_published_at = statement_timestamp()
                WHERE fitting_job_id = %(job)s AND (follow_up_published_at IS NULL
                  OR follow_up_published_at <= statement_timestamp() - make_interval(secs => %(i)s))
                RETURNING fitting_job_id
                """,
                {"job": job_id, "i": interval_seconds},
            )
            return cursor.fetchone() is not None

        return self._run(work)

    def claim_republish_once(self, job_id: int) -> bool:
        """AI 저장 작업의 지금 후처리 단계에서 아직 재발행하지 않았을 때만 선점한다(단계당 최대 한 번, DB 기록이라 재시작에도 유지).

        단계 시작 = COALESCE(j.image_verified_at, j.runware_accepted_at): 결과 저장 전에는 성공 전환(접수 시각 이후), 저장 확인 뒤에는 그 확인 시각.
        둘 다 실제 진척 기록이며 lease 선점·연장·반납·재수신(updated_at 만 바뀜)으로 바뀌지 않는다. 그래서 재발행한 메시지를
        Worker 가 받아 계속 실패해도(DLQ 로 가는 중) 제한이 다시 열리지 않는다. 발행이 실패·UNKNOWN 이면 forget_republish 로
        되돌린다(소비 실패가 아니라 발행 실패라서).
        """

        def work(cursor):
            cursor.execute(
                f"""
                UPDATE {CHECKS} c SET follow_up_published_at = statement_timestamp()
                FROM virtual_fitting_async_jobs j
                WHERE c.fitting_job_id = %(job)s AND j.fitting_job_id = c.fitting_job_id
                  AND (c.follow_up_published_at IS NULL
                       OR c.follow_up_published_at < COALESCE(j.image_verified_at, j.runware_accepted_at))
                RETURNING c.fitting_job_id
                """,
                {"job": job_id},
            )
            return cursor.fetchone() is not None

        return self._run(work)

    def defer_check(self, job_id: int, seconds: float) -> None:
        """이 작업의 다음 점검을 늦춘다(비용 누락처럼 급하지 않은 점검)."""
        self._run(
            lambda cursor: cursor.execute(
                f"UPDATE {CHECKS} SET next_check_at = statement_timestamp() + make_interval(secs => %s) "
                "WHERE fitting_job_id = %s",
                (seconds, job_id),
            )
        )

    def settle_cleanup(self, job_id: int) -> None:
        """최종 결과 뒤 비용 누락 점검을 끝낸다(비용을 기록하지 않음 — 0 으로 확정하는 것이 아니다)."""
        self._run(
            lambda cursor: cursor.execute(
                f"UPDATE {CHECKS} SET cleanup_settled_at = statement_timestamp() WHERE fitting_job_id = %s",
                (job_id,),
            )
        )

    def forget_republish(self, job_id: int) -> None:
        """발행이 실패·UNKNOWN 이면 다음 점검에서 다시 발행할 수 있게 한다."""
        self._run(
            lambda cursor: cursor.execute(
                f"UPDATE {CHECKS} SET follow_up_published_at = NULL WHERE fitting_job_id = %s",
                (job_id,),
            )
        )

    def timing(self, job_id: int) -> dict[str, float | None]:
        """DB 시각 기준 경과 초(접수 확인·호출 시작·마지막 변경·진행 중 자리·AI 후처리 단계 시작)."""

        def age(column: str) -> str:
            return f"EXTRACT(EPOCH FROM statement_timestamp() - {column})"

        def work(cursor):
            cursor.execute(
                f"""
                SELECT {age("j.runware_accepted_at")}, {age("j.pruna_submit_started_at")},
                       {age("j.updated_at")}, {age("f.acquired_at")}, f.task_uuid::text,
                       EXISTS (SELECT 1 FROM virtual_fitting_task_costs t WHERE t.task_uuid = j.task_uuid),
                       {age("COALESCE(j.image_verified_at, j.runware_accepted_at)")},
                       {age("j.pruna_retry_at")}
                FROM virtual_fitting_async_jobs j
                LEFT JOIN virtual_fitting_pruna_in_flight f USING (fitting_job_id)
                WHERE j.fitting_job_id = %s
                """,
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return {}
            keys = (
                "accepted",
                "started",
                "updated",
                "slot",
                "slot_task",
                "has_cost",
                "stage",
                "retry",  # 재요청 예약 시각부터 지난 초(음수면 아직 전)
            )
            return {
                k: (float(v) if v is not None and k not in ("slot_task", "has_cost") else v)
                for k, v in zip(keys, row, strict=True)
            }

        return self._run(work)

    def send_hold(self) -> str | None:
        def work(cursor):
            cursor.execute(
                "SELECT send_hold_reason FROM virtual_fitting_pruna_limiter WHERE id = 1"
            )
            row = cursor.fetchone()
            return row[0] if row else None

        return self._run(work)

    def record_alert(
        self, scope: str, reason: str, stage: str, job_id: int | None, realert_seconds: float
    ) -> WatchdogAlert | None:
        """같은 대상·원인을 한 줄로 모은다. 처음이거나 재알림 간격이 지났을 때만 알림 이벤트를 돌려준다(원자적)."""

        def work(cursor):
            cursor.execute(
                f"""
                INSERT INTO {ALERTS} AS a
                  (scope, reason, stage, fitting_job_id, first_seen_at, last_seen_at, last_notified_at)
                VALUES (%(scope)s, %(reason)s, %(stage)s, %(job)s,
                        statement_timestamp(), statement_timestamp(), statement_timestamp())
                ON CONFLICT (scope, reason) DO UPDATE SET
                  last_seen_at = statement_timestamp(), seen_count = a.seen_count + 1, stage = EXCLUDED.stage,
                  last_notified_at = CASE
                    WHEN a.last_notified_at IS NULL OR a.last_notified_at
                         <= statement_timestamp() - make_interval(secs => %(realert)s)
                    THEN statement_timestamp() ELSE a.last_notified_at END
                RETURNING a.last_notified_at = statement_timestamp(), a.first_seen_at, a.last_seen_at,
                          a.seen_count
                """,
                {
                    "scope": scope,
                    "reason": reason,
                    "stage": stage,
                    "job": job_id,
                    "realert": realert_seconds,
                },
            )
            notify, first, last, count = cursor.fetchone()
            if not notify:
                return None
            return WatchdogAlert(reason, stage, job_id, first.isoformat(), last.isoformat(), count)

        return self._run(work)


# --- 상태 조회 간격 ---


class StatusQueryGate:
    """상태 조회 사이 최소 간격과 Runware 대기 안내(retryAfter)를 지킨다(이 프로세스 안, 조회는 한 thread 에서 하나씩)."""

    def __init__(
        self,
        client: Any,
        *,
        min_interval_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client, self._min, self._clock, self._sleep = (
            client,
            min_interval_seconds,
            clock,
            sleep,
        )
        self._last: float | None = None
        self._not_before = 0.0
        # 진단: 마지막 query() 가 실제로 Runware 를 불렀는지(대기 안내 중이면 부르지 않는다)
        self.last_called = False

    def query(self, task_uuid: str):
        from app.virtual_fitting.providers.runware_status import TaskStatus

        self.last_called = False
        now = self._clock()
        if now < self._not_before:
            return TaskStatus(
                TaskStatusKind.RATE_LIMITED, task_uuid
            )  # 대기 안내 중: 조회하지 않는다
        if self._last is not None and now - self._last < self._min:
            self._sleep(self._min - (now - self._last))
        self._last = self._clock()
        self.last_called = True
        status = self._client.query(task_uuid)
        if status.kind is TaskStatusKind.RATE_LIMITED:
            self._not_before = self._clock() + (status.retry_after or 60.0)
        return status


# --- 점검 ---


@dataclass
class CheckReport:
    fitting_job_id: int
    actions: list[str] = field(default_factory=list)


class Watchdog:
    """동기. run_once() 가 전역 점검과 작업별 점검을 한 번 한다."""

    def __init__(
        self,
        *,
        jobs: Any,
        state: Any,
        repo: WatchdogRepository,
        status: StatusQueryGate,
        events: Any,
        sqs: Any,
        storage: Any,
        alerts: AlertSink | None = None,
        components: Callable[[], dict[str, Any]] = dict,
        publisher_ready: Callable[[], bool] = lambda: False,
        config: WatchdogSettings | None = None,
        should_stop: Callable[[], bool] = lambda: False,
        resubmit: Callable[[int], str] | None = None,
    ) -> None:
        # resubmit: 확실한 미접수 뒤 예약 시각이 지난 작업을 같은 Submitter 로 재요청한다(pruna_resubmit.PrunaResubmitter).
        # 없으면 재요청하지 않고 알린다(요청 메시지 삭제 판단도 준비되지 않음으로 본다).
        self._resubmit = resubmit
        self._jobs, self._state, self._repo = jobs, state, repo
        self._status, self._events, self._sqs, self._storage = status, events, sqs, storage
        self._alerts = alerts or LogAlertSink()
        self._components, self._publisher_ready = components, publisher_ready
        self._config = config or WatchdogSettings()
        self._should_stop = should_stop

    # 알림
    def _alert(self, reason: str, stage: str, job_id: int | None = None) -> None:
        scope = f"job:{job_id}" if job_id is not None else f"global:{stage}"
        alert = self._repo.record_alert(scope, reason, stage, job_id, self._config.realert_seconds)
        if alert is not None:
            self._alerts.emit(alert)

    def run_once(self) -> list[CheckReport]:
        self._global_checks()
        reports = []
        for job_id in self._repo.candidates(self._config):
            if self._should_stop():
                break
            if not self._repo.claim_check(job_id, self._config.check_interval_seconds):
                continue  # 다른 인스턴스가 점검 중
            try:
                reports.append(self.inspect(job_id))
            except FittingDatabaseError:
                # 이 작업만 다음 점검에서 다시(9단계 처리·복구 기록은 반복해도 같다). 다른 작업 점검은 계속한다.
                logger.warning("watchdog check failed fitting_job_id=%s reason=database", job_id)
                reports.append(CheckReport(job_id, ["database_error"]))
        logger.info("watchdog cycle checked=%s", len(reports))  # 외부 모니터링용 생존 신호
        return reports

    def _global_checks(self) -> None:
        hold = self._repo.send_hold()
        if hold is not None:
            # 자동 해제하지 않는다(수동 절차: virtual_fitting_pruna_limits.sql 주석).
            self._alert(f"send_hold_{hold}", "pruna_send_hold")
        if self._should_stop():
            return  # 종료 중에는 함께 멈추는 Worker 를 "중단"으로 알리지 않는다
        for name, worker in self._components().items():
            status = getattr(worker, "status", None)
            if worker is None or status is None:
                continue
            # Task 가 끝났거나(stopped) 비정상 종료(crashed)한 경우만. start() 직후 아직 돌기 전(running=False·stopped=False)은
            # 멈춘 것이 아니다(16단계에서 재현한 시작 직후 오탐).
            if getattr(status, "crashed", False) or getattr(status, "stopped", False):
                self._alert("component_stopped", f"component_{name}")

    def inspect(self, job_id: int) -> CheckReport:
        report = CheckReport(job_id)
        job = self._jobs.find_by_fitting_job_id(job_id)
        if job is None:
            return report
        timing = self._repo.timing(job_id)
        queried = None
        slot_age, slot_task = timing.get("slot"), timing.get("slot_task")
        if (
            slot_task is not None
            and slot_age is not None
            and slot_age >= self._config.stale_seconds
        ):
            if slot_task == job.task_uuid:
                queried = self._query(job, report, "slot")
            elif slot_task == job.fallback_task_uuid:
                report.actions.append("fallback_slot_unconfirmed")
                self._alert("fallback_slot_unconfirmed", "fallback", job_id)
            else:
                self._alert("slot_unknown_attempt", "slot", job_id)
            job = self._jobs.find_by_fitting_job_id(job_id) or job
        if job.status in (AsyncJobStatus.COMPLETED, AsyncJobStatus.FAILED):
            # 업무 결과는 끝났어도 외부 생성 시도의 정리(비용)가 남았을 수 있다. 결과·상태는 바꾸지 않는다.
            self._cleanup_cost(job, report, queried)
            return report
        if auth_hold_active(job):
            # Runware 인증 오류 보류(DB 기록). 같은 키로 조회해도 해결되지 않으므로 조회하지 않고 알리기만 한다.
            # 보류를 정상 대기로 바꾸거나 해제하지 않는다(운영 조치).
            report.actions.append("auth_hold")
            self._alert(f"pruna_auth_hold_{job.auth_hold_reason}", "pruna", job_id)
            return report
        if not_accepted_pending(job):
            self._resubmit_due(job, timing, report)
            return report
        if self._submit_unknown(job):
            age = timing.get("started") or 0.0
            if age >= self._config.query_after_seconds and queried is None:
                self._query(job, report, "submit_unknown")
            if age >= self._config.alert_after_seconds and self._still(
                job_id, self._submit_unknown
            ):
                self._alert("pruna_submit_unconfirmed", "pruna", job_id)
        elif self._awaiting_webhook(job):
            age = timing.get("accepted") or 0.0
            if age >= self._config.query_after_seconds and queried is None:
                self._query(job, report, "waiting")
            if age >= self._config.alert_after_seconds and self._still(
                job_id, self._awaiting_webhook
            ):
                self._alert("pruna_delayed", "pruna", job_id)
        elif (
            self._missing_result_url(job)
            and not self._leased(job)
            and (timing.get("accepted") or 0.0) >= self._config.query_after_seconds
        ):
            self._recover_result_url(job, report)
        elif job.next_action is NextAction.POSTPROCESS and self._stale(job, timing):
            self._postprocess(job, report)
        elif job.next_action is NextAction.RUN_FALLBACK and self._stale(job, timing):
            self._fallback(job, report)
        elif job.next_action is NextAction.PUBLISH_RESULT:
            self._publish(job, timing)
        return report

    def _resubmit_due(self, job: AsyncFittingJob, timing: dict, report: CheckReport) -> None:
        """확실한 미접수로 기록된 작업. 예약 시각(DB) 전에는 아무것도 하지 않고, 지났으면 같은 Submitter 로 재요청한다.

        재요청 조건(예약 시각·남은 횟수·실행 권한·공통 제한)은 Submitter 의 자리 확보 트랜잭션이 다시 확인한다. 예약이 없으면
        (표현할 수 없는 대기·모순 근거) 자동으로 보내지 않고 알린다.
        """
        job_id = job.fitting_job_id
        if job.pruna_retry_at is None:
            report.actions.append("resubmit_held")
            self._alert("pruna_resubmit_held", "pruna", job_id)
            return
        elapsed = timing.get("retry")
        if elapsed is None or elapsed < 0 or self._leased(job):
            report.actions.append("resubmit_waiting")
            return
        if self._resubmit is None:
            report.actions.append("resubmit_not_connected")
            self._alert("pruna_resubmit_not_connected", "pruna", job_id)
            return
        report.actions.append(f"resubmit_{self._resubmit(job_id)}")

    @staticmethod
    def _submit_unknown(job: AsyncFittingJob) -> bool:
        return (
            job.status is AsyncJobStatus.RECEIVED
            and job.next_action is NextAction.SUBMIT_PRUNA
            and job.pruna_submit_started_at is not None
            and job.runware_accepted_at is None
            and job.pruna_not_accepted_at is None
        )

    @staticmethod
    def _awaiting_webhook(job: AsyncFittingJob) -> bool:
        return job.status is AsyncJobStatus.WAITING_PRUNA and job.next_action is None

    def _still(self, job_id: int, predicate: Callable[[AsyncFittingJob], bool]) -> bool:
        latest = self._jobs.find_by_fitting_job_id(job_id)
        return latest is not None and predicate(latest)

    def _stale(self, job: AsyncFittingJob, timing: dict) -> bool:
        # AI 저장 작업의 후처리는 단계 시작(성공 전환·저장 확인) 기준: updated_at 은 lease 선점·연장·반납과 메시지 재수신마다
        # 바뀌어 진척이 없어도 판단이 계속 밀린다. 그 밖(기존 직접 업로드·fallback)은 기존대로 updated_at 기준이다.
        basis = (
            "stage"
            if job.ai_stores_result and job.next_action is NextAction.POSTPROCESS
            else "updated"
        )
        return (timing.get(basis) or 0.0) >= self._config.stale_seconds and not self._leased(job)

    @staticmethod
    def _leased(job: AsyncFittingJob) -> bool:
        from datetime import UTC, datetime

        return (
            job.worker_lease_token is not None
            and job.worker_lease_until is not None
            and job.worker_lease_until > datetime.now(UTC)
        )

    # 최종 결과 뒤 비용 누락 정리
    def _cleanup_cost(self, job: AsyncFittingJob, report: CheckReport, status) -> None:
        """종료 작업의 비동기 task 비용이 없을 때만. 조회의 success(실제 cost)·확정 실패만 9단계 처리로 정리한다.

        9단계 처리는 종료 작업의 상태·결과·횟수를 바꾸지 않고(IGNORED) 비용 1회 기록·그 task 자리 반환만 한다.
        processing·조회 실패·결과 없음은 아무것도 정리하지 않고 정리 간격 뒤 다시 본다(비용 0 기록 없음).
        fallback(동기) task 의 비용은 원격 조회 근거가 없어 여기서 다루지 않는다.
        """
        timing = self._repo.timing(job.fitting_job_id)
        started = timing.get("started")
        if (
            timing.get("has_cost")
            or job.task_uuid is None
            or job.runware_accepted_at is None
            or started is None
            or started > self._config.cleanup_window_seconds
        ):
            return
        if status is None:
            status = self._query(job, report, "cleanup", alert=False)
        if self._repo.timing(job.fitting_job_id).get("has_cost"):
            report.actions.append("cost_recorded")
            return
        if status.kind is TaskStatusKind.SUCCEEDED:
            # 생성은 끝났는데 조회에도 cost 가 없다. 0 으로 기록하지 않고 점검만 끝낸다(운영 확인).
            self._alert("cost_missing", "cost", job.fitting_job_id)
            self._repo.settle_cleanup(job.fitting_job_id)
            report.actions.append("cleanup_settled_without_cost")
        elif status.kind is TaskStatusKind.FAILED and classify_task_error(status.error_code) in (
            TaskErrorKind.GENERATION_FALLBACK,
            TaskErrorKind.GENERATION_FAILED,
        ):
            self._repo.settle_cleanup(job.fitting_job_id)  # 확정 생성 실패, 보고된 비용 없음
            report.actions.append("cleanup_settled_without_cost")
        else:
            self._repo.defer_check(job.fitting_job_id, self._config.cleanup_interval_seconds)
            report.actions.append("cleanup_deferred")

    # 비동기 Pruna 상태 조회 → 확인된 결과만 9단계 처리로
    def _query(self, job: AsyncFittingJob, report: CheckReport, why: str, *, alert: bool = True):
        started = time.monotonic()
        fields = {
            "component": "watchdog",
            "source": "watchdog",
            "fitting_job_id": job.fitting_job_id,
            "task_uuid": job.task_uuid,
            "check_reason": why,
        }
        try:
            status = self._status.query(job.task_uuid)
        except Exception as error:
            diag.event(
                "runware_status_checked",
                logging.WARNING,
                **fields,
                result="error",
                error_type=type(error).__name__,
                elapsed_ms=diag.elapsed_ms(started),
            )
            raise
        diag.event(
            "runware_status_checked",
            **fields,
            result=status.kind.value,
            called=getattr(self._status, "last_called", "unknown"),
            http_status=getattr(status, "http_status", None) or "unknown",
            error_code=diag.safe(getattr(status, "error_code", None), 64),
            elapsed_ms=diag.elapsed_ms(started),
        )
        report.actions.append(f"query_{why}_{status.kind.value}")
        if status.kind is TaskStatusKind.PROCESSING:
            # 조회 중 미접수 기록이 생겼을 수 있다. Webhook 과 같은 모순 보호로 다음 생성 재전송을 막는다.
            self._state.record_attempt_contradiction(
                job.task_uuid, "processing_after_not_accepted"
            )
        elif status.kind is TaskStatusKind.SUCCEEDED:
            # 결과 참조(imageURL·imageUUID)도 Webhook 과 같은 의미로 넘긴다(저장 조건은 9단계 처리가 정한다).
            event = WebhookEvent(
                job.task_uuid,
                EventKind.SUCCEEDED,
                cost=status.cost,
                image_url=status.image_url,
                image_uuid=status.image_uuid,
                image_ref_invalid=status.image_ref_invalid,
                from_status_query=True,  # 우리 인증으로 이 task 를 직접 조회한 결과(외부 본문이 아님)
            )
            self._apply(job, event, report)
        elif status.kind is TaskStatusKind.FAILED and status.error_code is not None:
            # 이 task 에 연결된 생성 task 의 오류 결과: Webhook 과 같은 처리(같은 분류·fallback 목록·보호 조건)로 넘긴다.
            event = WebhookEvent(
                job.task_uuid,
                EventKind.FAILED,
                status.error_code,
                cost=status.cost,
                from_status_query=True,  # 우리 인증으로 이 task 를 직접 조회한 결과(미접수 기록의 확인 경로)
                # 그 task 오류 항목의 안내만 넘긴다(조회 HTTP 자체의 Retry-After 는 status.retry_after 로 따로 있다).
                retry_after=status.task_retry_after,
                retry_after_unrepresentable=status.task_retry_after_unrepresentable,
            )
            self._apply(job, event, report)
            if event.unresolved and alert:
                # 확실한 미접수(자동 재요청 미연결)·근거 부족 코드: 상태를 바꾸지 않았다. 운영 확인으로 알린다.
                self._alert(f"pruna_error_{status.error_code}", "pruna", job.fitting_job_id)
        elif status.kind is TaskStatusKind.FAILED and alert:
            # 코드를 읽을 수 없거나 같은 task 의 실패 코드가 서로 다르다. 분류하지 않고 알린다(상태·자리 그대로).
            self._alert("pruna_error_unknown", "pruna", job.fitting_job_id)
        elif status.kind is TaskStatusKind.AUTH_ERROR:
            self._alert("status_query_auth", "pruna", job.fitting_job_id)
        return status

    def _apply(self, job: AsyncFittingJob, event: WebhookEvent, report: CheckReport) -> None:
        try:
            results = self._events.handle([event])
            report.actions.extend(f"applied_{r.outcome.value}" for r in results)
        except WebhookFollowUpNotReadyError as error:
            report.actions.extend(f"applied_{r.outcome.value}" for r in error.results)
        except WebhookFollowUpError:
            report.actions.append(
                "follow_up_publish_failed"
            )  # 다음 점검에서 다시(9단계 처리는 반복해도 같다)
        except WebhookUnknownTaskError:
            report.actions.append("unknown_task")
        except SqsConfigError:
            self._alert("follow_up_queue_config", "queue", job.fitting_job_id)

    # 후속 메시지 재발행(작업당 republish 간격에 한 번, 실패하면 다음 점검에서 다시)
    def _republish(
        self,
        job: AsyncFittingJob,
        action: NextAction,
        report: CheckReport,
        *,
        once_per_progress: bool = False,
    ) -> None:
        if once_per_progress:
            if not self._repo.claim_republish_once(job.fitting_job_id):
                report.actions.append(f"republish_{action.value.lower()}_held")
                self._alert(f"{action.value.lower()}_stalled", "postprocess", job.fitting_job_id)
                return
        elif not self._repo.claim_republish(
            job.fitting_job_id, self._config.republish_interval_seconds
        ):
            report.actions.append(f"republish_{action.value.lower()}_throttled")
            return
        latest = self._jobs.find_by_fitting_job_id(job.fitting_job_id)
        if latest is None or latest.next_action is not action or latest.result_payload is not None:
            self._repo.forget_republish(job.fitting_job_id)
            return
        try:
            sent = publish_follow_up(self._sqs, latest, action).succeeded
        except SqsConfigError:
            sent = False
            self._alert("follow_up_queue_config", "queue", job.fitting_job_id)
        if not sent:
            self._repo.forget_republish(job.fitting_job_id)
        report.actions.append(f"republish_{action.value.lower()}_{'ok' if sent else 'failed'}")

    def _recover_with_lease(self, job, action, record) -> bool:
        """복구 기록은 5단계 토큰 메서드로만 한다. 유효한 실행 권한이 없을 때만 잠깐 선점한다(빼앗지 않음)."""
        grant = self._state.acquire_lease(job.fitting_job_id, action, 60)
        if not grant.acquired:
            return False
        try:
            change = record(grant.token)
            return change.applied or change.reason is ChangeReason.ALREADY_RECORDED
        finally:
            self._state.release_lease(job.fitting_job_id, grant.token)

    def _fresh(self, check, since) -> bool:
        modified = getattr(check, "last_modified", None)
        if modified is None or since is None:
            return False
        return modified >= since - timedelta(seconds=CLOCK_SKEW_SECONDS)

    @staticmethod
    def _missing_result_url(job: AsyncFittingJob) -> bool:
        """AI 저장 작업이 후처리 대기 중인데 결과를 내려받을 URL 이 없다(저장·최종 결과·fallback 진행 기록 없음)."""
        return (
            job.ai_stores_result
            and job.status is AsyncJobStatus.POSTPROCESSING
            and job.next_action is NextAction.POSTPROCESS
            and job.result_payload is None
            and job.verified_image_key is None
            and job.runware_image_url is None
            and job.task_uuid is not None
            and job.pruna_submit_started_at is not None
        )

    def _recover_result_url(self, job: AsyncFittingJob, report: CheckReport) -> None:
        """같은 task 를 조회해 확인된 참조를 저장(9단계 처리 — 조건부 보완)한 뒤 POSTPROCESS 로 인계한다.

        조회 실패·미발견·참조 없음은 생성 실패가 아니다(상태를 바꾸지 않음). 운영 확인으로 알리고 정리 간격 뒤 다시 본다.
        """
        self._query(job, report, "result_ref")
        latest = self._jobs.find_by_fitting_job_id(job.fitting_job_id)
        if latest is not None and latest.runware_image_url is not None:
            # 참조 저장과 POSTPROCESS 발행은 9단계 처리가 같은 흐름에서 했다(DB 저장 뒤 발행). 발행이 실패했으면
            # 장기 미진행 점검(_postprocess)이 DB 기록으로 다시 발행한다.
            report.actions.append("result_ref_recovered")
            return
        report.actions.append("result_ref_unrecovered")
        self._alert("result_ref_missing", "postprocess", job.fitting_job_id)
        self._repo.defer_check(job.fitting_job_id, self._config.cleanup_interval_seconds)

    def _postprocess(self, job: AsyncFittingJob, report: CheckReport) -> None:
        if job.ai_stores_result:
            # AI 저장 작업은 S3 확인 횟수를 쓰지 않는다(객체 미존재·저장 실패를 fallback 근거로 쓰지 않음). 멈춘 메시지는
            # 후처리 단계(결과 저장 전 / 저장 확인 뒤 LLM·결과 저장)마다 한 번만 다시 발행하고, 그 뒤에도 진척이 없으면 운영
            # 확인으로 알린다(DLQ 우회 방지). 재발행 기록은 DB 에 남아 재시작에도 유지된다.
            self._republish(job, NextAction.POSTPROCESS, report, once_per_progress=True)
            return
        if job.verified_image_key is not None or job.s3_check_attempt_count < S3_CHECK_LIMIT:
            self._republish(
                job, NextAction.POSTPROCESS, report
            )  # 남은 단계는 POSTPROCESS 핸들러가 잇는다
            return
        # 2묶음을 썼는데 확인 결과가 없다(중단). 복구용 단발 HEAD — 일반 묶음 횟수와 별개, 횟수를 늘리지 않는다.
        check = self._storage.head_object(job.async_image_key)
        report.actions.append(f"recovery_head_{check.state.value}")
        if check.state is S3ObjectState.EXISTS and self._fresh(check, job.pruna_submit_started_at):
            recovered = self._recover_with_lease(
                job,
                NextAction.POSTPROCESS,
                lambda token: self._state.record_image_verified(
                    job.fitting_job_id, token, job.async_image_key
                ),
            )
            if recovered:
                self._republish(job, NextAction.POSTPROCESS, report)
                return
        # 404·403·일시 오류는 "끝까지 확인했고 없다"는 증거가 아니다. fallback 하지 않는다.
        self._alert("s3_check_unconfirmed", "postprocess", job.fitting_job_id)

    def _fallback(self, job: AsyncFittingJob, report: CheckReport) -> None:
        if job.fallback_image_key is not None or (
            job.fallback_started_at is None and job.fallback_attempt_count == 0
        ):
            self._republish(job, NextAction.RUN_FALLBACK, report)
            return
        # 생성 함수를 불렀을 수 있는데 이미지 기록이 없다. IDM-VTON(Modal)을 다시 부르지 않고 고정 Key 만 확인한다
        # (전환기 Pruna 동기 결과 pruna-sync.jpg 포함, idm_fallback_handler.inspect_fallback_result).
        from app.virtual_fitting.idm_fallback_handler import ResultState, inspect_fallback_result

        check = inspect_fallback_result(self._storage, job)
        report.actions.append(f"fallback_result_{check.state.value}")
        if check.state is ResultState.FOUND:
            # 고정 Key·작업·시도 메타데이터(또는 시작 뒤 저장 시각)가 맞는 객체만 그 시도의 결과로 기록한다.
            recovered = self._recover_with_lease(
                job,
                NextAction.RUN_FALLBACK,
                lambda token: self._state.record_fallback_image(
                    job.fitting_job_id, token, check.key
                ),
            )
            if recovered:
                self._republish(job, NextAction.RUN_FALLBACK, report)
                return
        elif check.state in (ResultState.ABSENT, ResultState.INVALID):
            # 더 올라올 수 없거나 쓸 수 없는 결과. 실패 결과 저장은 핸들러가 한다(메시지를 다시 보내 이어 가게 한다).
            self._republish(job, NextAction.RUN_FALLBACK, report)
            self._alert(f"fallback_result_{check.state.value}", "fallback", job.fitting_job_id)
            return
        self._alert("fallback_result_unknown", "fallback", job.fitting_job_id)

    def _publish(self, job: AsyncFittingJob, timing: dict) -> None:
        if job.result_message_status is ResultMessageStatus.RETRY_EXHAUSTED:
            # 자동 재발행하지 않는다(횟수 초기화 없음). 운영 확인.
            self._alert("result_publish_exhausted", "publish", job.fitting_job_id)
            return
        if job.result_message_status is not ResultMessageStatus.PENDING:
            return
        if not self._publisher_ready():
            self._alert("result_publisher_unavailable", "publish", job.fitting_job_id)
        elif (timing.get("updated") or 0.0) >= self._config.stale_seconds:
            self._alert("result_publish_stalled", "publish", job.fitting_job_id)


# --- 반복 루프 ---


@dataclass
class WatchdogStatus:
    running: bool = False
    stopped: bool = False
    crashed: bool = False
    last_error: str | None = None
    consecutive_errors: int = 0
    cycles: int = 0


class WatchdogWorker:
    """FastAPI 이벤트 루프의 Task. 점검은 전용 thread 하나에서 한다(다른 HTTP 요청을 막지 않음)."""

    def __init__(
        self,
        run_once: Callable[[], Any],
        *,
        interval_seconds: float = 60.0,
        backoff_max_seconds: float = 300.0,
    ) -> None:
        self._run_once, self._interval, self._backoff_max = (
            run_once,
            interval_seconds,
            backoff_max_seconds,
        )
        self._limiter = CapacityLimiter(1)
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.status = WatchdogStatus()
        # 확실한 미접수 재요청 경로가 조립됐는지(create_watchdog 이 정한다). False 면 RESUBMIT_PRUNA 를 준비됨으로 보지 않는다.
        self.resubmits = False

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    @property
    def alive(self) -> bool:
        return self.status.running and not self.status.crashed and not self._stop.is_set()

    def start(self) -> asyncio.Task:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="virtual-fitting-watchdog")
            self._task.add_done_callback(self._finished)
        return self._task

    async def stop(self, timeout: float) -> None:
        """새 점검을 멈추고 진행 중인 점검(thread)이 끝나기를 기다린다(버리지 않음 — DB pool 은 그 뒤에 닫힌다)."""
        self._stop.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout)
        except TimeoutError:
            logger.warning(
                "watchdog did not stop within %ss; waiting for the current check", timeout
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
            logger.error("watchdog crashed", exc_info=error)
            diag.event(
                "component_crashed",
                logging.ERROR,
                component="watchdog",
                error_type=type(error).__name__,
            )
        else:
            self.status.stopped = True

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def _run(self) -> None:
        self.status.running = True
        logger.info("watchdog started")
        diag.event("component_started", component="watchdog")
        while not self._stop.is_set():
            try:
                await to_thread.run_sync(self._run_once, limiter=self._limiter)
                self.status.cycles += 1
                self.status.consecutive_errors = 0
                await self._sleep(self._interval)
            except VirtualFittingError as error:
                self.status.consecutive_errors += 1
                self.status.last_error = error.code
                backoff = min(self._backoff_max, self._interval * self.status.consecutive_errors)
                logger.warning(
                    "watchdog cycle failed reason=%s; backing off %.0fs", error.code, backoff
                )
                await self._sleep(backoff)
        logger.info("watchdog stopped")
        diag.event("component_stopped", component="watchdog")


# --- 준비 여부(FollowUpPaths) ---

_registered: WatchdogWorker | None = None


def register_watchdog(worker: WatchdogWorker | None) -> None:
    global _registered
    _registered = worker


class WatchdogPaths:
    """Watchdog 이 실제로 돌 때, 그것이 조사·확인된 근거로 이어 갈 수 있는 일만 준비됐다고 답한다.

    - AWAIT_WEBHOOK(접수됨, 완료 알림 대기): Webhook 이 오지 않아도 3분부터 getResponse 로 확인한다.
    - RECOVER_UNKNOWN(호출 시작 기록만 있고 접수 불명확): 3분부터 getResponse 로 확인한다(재전송 없음).
    - RECOVER_RESULT_REF(AI 저장 작업이 성공했지만 결과 URL 없음): 접수 3분 뒤부터 같은 task 를 조회해 참조를 저장하고
      POSTPROCESS 를 발행한다.
    - RESUBMIT_PRUNA(확실한 미접수 뒤 재요청 예약): 예약 시각(pruna_retry_at)이 지나면 같은 Submitter 로 새 task 를 보낸다.
      재요청 경로가 조립된 Watchdog(resubmits)일 때만이다.
    그 밖(OPERATOR_REVIEW·인증 오류 보류 등)은 준비되지 않음. 내부 큐 URL 이 로컬 검증을 통과해야 한다(확인 결과를 POSTPROCESS·
    RUN_FALLBACK 으로 이어 가므로). 로그를 남길 수 있다는 것만으로는 준비됐다고 보지 않는다.
    """

    def __init__(self, worker: Callable[[], WatchdogWorker | None] = lambda: _registered):
        self._worker = worker

    def ready_for(self, job: AsyncFittingJob | None, step: NextStep) -> bool:
        if job is None or step not in (
            NextStep.AWAIT_WEBHOOK,
            NextStep.RECOVER_UNKNOWN,
            NextStep.RECOVER_RESULT_REF,
            NextStep.RESUBMIT_PRUNA,
        ):
            return False
        worker = self._worker()
        if worker is None or not worker.alive or job.task_uuid is None:
            return False
        if step is NextStep.RESUBMIT_PRUNA:
            # 예약된 재요청: Watchdog 이 DB 기록(pruna_retry_at)을 읽어 같은 Submitter 로 보낸다(큐를 쓰지 않음).
            return bool(getattr(worker, "resubmits", False)) and job.pruna_retry_at is not None
        try:
            get_queue_url(FittingQueue.AI_FITTING_RESPONSE)
        except SqsConfigError:
            return False
        return True


class AnyFollowUpPaths:
    """여러 실제 경로 중 하나라도 그 일을 이어 갈 수 있으면 준비됨."""

    def __init__(self, *paths: Any) -> None:
        self._paths = paths

    def ready_for(self, job: AsyncFittingJob | None, step: NextStep) -> bool:
        return any(path.ready_for(job, step) for path in self._paths)


class PostprocessPaths:
    """POSTPROCESS(후처리 준비·진행): 결과 참조·후속 작업이 DB 에 있고(next_action=POSTPROCESS, 결과 전), 그 일을 이어 갈
    두 주체가 실제로 돌 때만 준비됐다고 답한다.

    - 후처리 Worker: ai-fitting-response 의 POSTPROCESS 메시지를 소비한다.
    - Watchdog: 후처리 메시지가 유실·발행 실패했거나 진행이 멈추면 DB 기록으로 다시 발행한다(요청 메시지를 지운 뒤에는
      이것이 DB 기록을 다시 큐로 잇는 경로다). Watchdog 판단은 WatchdogPaths 와 같은 기준(실행 중·내부 큐 설정)이다.
    결과 URL 이 없는 AI 저장 작업은 이 단계가 아니다(next_step_for 가 RECOVER_RESULT_REF 로 나눈다).
    """

    def __init__(
        self,
        worker_running: Callable[[], bool] | None = None,
        watchdog: Callable[[], WatchdogWorker | None] = lambda: _registered,
    ):
        self._worker_running = worker_running
        self._watchdog = watchdog

    def ready_for(self, job: AsyncFittingJob | None, step: NextStep) -> bool:
        if job is None or step is not NextStep.POSTPROCESS:
            return False
        if (
            job.next_action is not NextAction.POSTPROCESS
            or job.result_payload is not None
            or job.task_uuid is None
        ):
            return False
        if self._worker_running is None:
            from app.virtual_fitting.postprocess_worker import postprocess_worker_running

            running = postprocess_worker_running()
        else:
            running = self._worker_running()
        if not running:
            return False
        watchdog = self._watchdog()
        if watchdog is None or not watchdog.alive:
            return False
        try:
            get_queue_url(FittingQueue.AI_FITTING_RESPONSE)
        except SqsConfigError:
            return False
        return True


def follow_up_paths() -> AnyFollowUpPaths:
    """요청·후처리 Worker·Webhook 이 쓰는 준비 판단(단계마다 필요한 주체만 본다):
    결과 전송 루프(PUBLISH_RESULT) + Watchdog(AWAIT_WEBHOOK·RECOVER_UNKNOWN·RECOVER_RESULT_REF)
    + 후처리 Worker 와 Watchdog(POSTPROCESS)."""
    from app.virtual_fitting.result_publisher import result_publish_paths

    return AnyFollowUpPaths(result_publish_paths(), WatchdogPaths(), PostprocessPaths())


# --- 조립 ---


def watchdog_settings() -> WatchdogSettings:
    from app.config import settings

    interval = settings.VIRTUAL_FITTING_WATCHDOG_INTERVAL_SECONDS
    batch = settings.VIRTUAL_FITTING_WATCHDOG_BATCH_SIZE
    if not 0 < interval <= 3600:
        raise ValueError(
            "VIRTUAL_FITTING_WATCHDOG_INTERVAL_SECONDS 는 0 초과 3600 이하여야 합니다."
        )
    if not 1 <= batch <= 100:
        raise ValueError("VIRTUAL_FITTING_WATCHDOG_BATCH_SIZE 는 1~100 이어야 합니다.")
    return WatchdogSettings(check_interval_seconds=interval, batch_size=batch)


def create_watchdog(components: Callable[[], dict[str, Any]] = dict) -> WatchdogWorker:
    """실제 의존성으로 조립한다(blocking: DB pool·boto3). DB pool·SQS client 는 Worker 들과 같은 프로세스 전역 자원이다."""
    from app.clients.s3 import S3ImageStorage
    from app.clients.sqs import SqsQueueClient
    from app.config import settings
    from app.config.database import get_connection_pool
    from app.virtual_fitting.providers.runware_async import DEFAULT_SUBMIT_TIMEOUT
    from app.virtual_fitting.providers.runware_status import RunwareTaskStatusClient
    from app.virtual_fitting.repositories.async_job_repository import AsyncJobRepository
    from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository
    from app.virtual_fitting.repositories.pruna_limiter import PrunaLimiter, TaskCostRepository
    from app.virtual_fitting.runware_webhook import RunwareWebhookService

    config = watchdog_settings()
    pool = get_connection_pool()
    jobs, state = AsyncJobRepository(pool), AsyncJobStateRepository(pool)
    sqs = (
        SqsQueueClient()
    )  # 기본 client(total_max_attempts=1). Webhook 의 짧은 timeout client 는 쓰지 않는다
    events = RunwareWebhookService(
        jobs=jobs,
        state=state,
        limiter=PrunaLimiter(
            pool,
            max_in_flight=settings.VIRTUAL_FITTING_PRUNA_MAX_IN_FLIGHT,
            max_concurrent_sends=settings.VIRTUAL_FITTING_PRUNA_MAX_CONCURRENT_SENDS,
            min_send_interval_seconds=settings.VIRTUAL_FITTING_PRUNA_MIN_SEND_INTERVAL_SECONDS,
            send_hold_seconds=DEFAULT_SUBMIT_TIMEOUT + 30,
        ),
        costs=TaskCostRepository(pool),
        queue=sqs,
        source="watchdog",  # 진단: 이 경로의 처리는 HTTP Webhook 이 아니다
    )
    worker: WatchdogWorker

    def publisher_ready() -> bool:
        # 14단계 판단과 같다: 결과 전송 루프가 실제로 돌고 결과 큐 설정이 로컬 검증을 통과한다.
        from app.virtual_fitting import result_publisher

        current = result_publisher._registered
        return current is not None and current.alive and result_publisher._result_queue_configured()

    resubmitter = None
    try:
        from app.virtual_fitting.request_worker import create_pruna_resubmitter

        resubmitter = create_pruna_resubmitter(lambda: worker.stopping)
    except Exception as error:  # noqa: BLE001 — 조립 실패를 Watchdog 전체 중단으로 키우지 않는다
        # 재요청 경로를 만들 수 없다(설정 오류 등). Watchdog 의 다른 점검은 그대로 돌고, 예약된 재요청은 보내지 않고 알린다
        # (RESUBMIT_PRUNA 를 준비됨으로 보지 않아 요청 메시지도 지우지 않는다).
        logger.error("watchdog resubmit path not connected error_type=%s", type(error).__name__)
    watchdog = Watchdog(
        jobs=jobs,
        state=state,
        repo=WatchdogRepository(pool),
        status=StatusQueryGate(
            RunwareTaskStatusClient(),
            min_interval_seconds=settings.VIRTUAL_FITTING_WATCHDOG_STATUS_MIN_INTERVAL_SECONDS,
        ),
        events=events,
        sqs=sqs,
        storage=S3ImageStorage(),
        components=components,
        publisher_ready=publisher_ready,
        config=config,
        should_stop=lambda: worker.stopping,
        resubmit=resubmitter,
    )
    worker = WatchdogWorker(watchdog.run_once, interval_seconds=config.check_interval_seconds)
    worker.resubmits = resubmitter is not None
    return worker
