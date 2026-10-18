"""가상피팅 V2 Watchdog(15단계): 미완료 작업 조사·확인된 근거로 복구·운영 이상 탐지.

FastAPI lifespan 이 시작·종료한다(VIRTUAL_FITTING_WATCHDOG_ENABLED 기본 false, 점검 주기 60초). DB 를 조회해 대상을 찾는다.
**조사할 수 있다 ≠ 자동으로 복구할 수 있다.** 확인된 근거가 있을 때만 기존 경로(9단계 Webhook 처리·POSTPROCESS·RUN_FALLBACK
메시지·5단계 상태 메서드)로 이어 가고, 근거가 없으면 기록을 그대로 두고 운영 알림만 남긴다.

지키는 것: 유료 생성 요청 재호출 없음, 시도 횟수 초기화 없음, 시간 경과·lease 만료만으로 자리 반환·실패 확정·fallback 없음,
RETRY_EXHAUSTED 자동 재발행 없음, 최종 결과·이미지·상태 보존. BE 큐에만 있고 AI DB 에 없는 요청은 대상이 아니다(BE 담당).

작업별 점검(DB 선점: virtual_fitting_watchdog_checks.next_check_at, 작업당 60초에 한 번, updated_at 을 바꾸지 않음):
  - 진행 중 자리가 10분 넘게 남음: 비동기 task 면 getResponse 로 끝났는지 확인해 끝났을 때만 정리(9단계 처리 재사용).
    fallback task 는 조회 근거가 없어(동기) 운영 확인.
  - RECEIVED 접수 불명확(호출 시작 기록 기준) / WAITING_PRUNA(최초 접수 기준): 3분부터 getResponse, 5분부터 지연 알림.
    success → 9단계 성공 처리(비용·자리·POSTPROCESS). timeoutProvider → 9단계 fallback 처리. 그 밖의 오류 코드·조회 실패·
    결과 없음·인증 오류 → 상태를 바꾸지 않고 운영 확인.
  - POSTPROCESS·RUN_FALLBACK 이 유효한 실행 권한 없이 10분 넘게 진전 없음: 이어 갈 근거가 있으면 기존 메시지를 다시 발행
    (작업당 10분에 한 번). S3 확인 2묶음을 다 썼는데 결과가 불명확하면 복구용 단발 HEAD 1번(EXISTS 만 근거로 쓴다).
    fallback 시작 기록만 있으면 고정 Key(pruna-sync.jpg) 단발 HEAD — 이 시도 뒤에 저장된 객체일 때만 기록 복구.
  - PUBLISH_RESULT: 정상 전송은 14단계 결과 전송 루프 몫. RETRY_EXHAUSTED·전송 루프 없음·설정 오류·장기 보류를 알린다.
  - COMPLETED·FAILED: 업무 결과는 끝났어도 남은 자리(위)·비동기 task 비용 누락(호출 시작 후 24시간 안, 접수 확인된 task)을
    getResponse 로 확인해 9단계 처리로 정리한다. 상태·결과·횟수는 바꾸지 않는다. 결론이 없으면 15분 뒤 다시.
전역 점검: 공유 SEND_HOLD(자동 해제 안 함), 켜져 있어야 할 Worker·결과 전송 루프의 중단.
외부 HTTP·S3·SQS 호출 중에는 DB 트랜잭션·잠금을 잡지 않는다. DB 트랜잭션은 5초 잠금 대기 제한을 둔다.
"""

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
from app.virtual_fitting.exceptions import FittingDatabaseError, VirtualFittingError
from app.virtual_fitting.follow_up_queue import publish_follow_up
from app.virtual_fitting.providers.runware_status import TaskStatusKind
from app.virtual_fitting.pruna_submitter import NextStep, auth_hold_active
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncFittingJob,
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import S3_CHECK_LIMIT, ChangeReason
from app.virtual_fitting.runware_webhook import (
    FALLBACK_ERROR_CODES,
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
                        AND j.runware_accepted_at IS NULL AND j.pruna_submit_started_at
                            <= statement_timestamp() - make_interval(secs => %(query)s))
                      OR (j.next_action IN ('POSTPROCESS', 'RUN_FALLBACK')
                        AND (j.worker_lease_token IS NULL OR j.worker_lease_until <= statement_timestamp())
                        AND j.updated_at <= statement_timestamp() - make_interval(secs => %(stale)s))
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
        """DB 시각 기준 경과 초(접수 확인·호출 시작·마지막 변경·진행 중 자리)."""

        def age(column: str) -> str:
            return f"EXTRACT(EPOCH FROM statement_timestamp() - {column})"

        def work(cursor):
            cursor.execute(
                f"""
                SELECT {age("j.runware_accepted_at")}, {age("j.pruna_submit_started_at")},
                       {age("j.updated_at")}, {age("f.acquired_at")}, f.task_uuid::text,
                       EXISTS (SELECT 1 FROM virtual_fitting_task_costs t WHERE t.task_uuid = j.task_uuid)
                FROM virtual_fitting_async_jobs j
                LEFT JOIN virtual_fitting_pruna_in_flight f USING (fitting_job_id)
                WHERE j.fitting_job_id = %s
                """,
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return {}
            keys = ("accepted", "started", "updated", "slot", "slot_task", "has_cost")
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

    def query(self, task_uuid: str):
        from app.virtual_fitting.providers.runware_status import TaskStatus

        now = self._clock()
        if now < self._not_before:
            return TaskStatus(
                TaskStatusKind.RATE_LIMITED, task_uuid
            )  # 대기 안내 중: 조회하지 않는다
        if self._last is not None and now - self._last < self._min:
            self._sleep(self._min - (now - self._last))
        self._last = self._clock()
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
    ) -> None:
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
        elif job.next_action is NextAction.POSTPROCESS and self._stale(job, timing):
            self._postprocess(job, report)
        elif job.next_action is NextAction.RUN_FALLBACK and self._stale(job, timing):
            self._fallback(job, report)
        elif job.next_action is NextAction.PUBLISH_RESULT:
            self._publish(job, timing)
        return report

    @staticmethod
    def _submit_unknown(job: AsyncFittingJob) -> bool:
        return (
            job.status is AsyncJobStatus.RECEIVED
            and job.next_action is NextAction.SUBMIT_PRUNA
            and job.pruna_submit_started_at is not None
            and job.runware_accepted_at is None
        )

    @staticmethod
    def _awaiting_webhook(job: AsyncFittingJob) -> bool:
        return job.status is AsyncJobStatus.WAITING_PRUNA and job.next_action is None

    def _still(self, job_id: int, predicate: Callable[[AsyncFittingJob], bool]) -> bool:
        latest = self._jobs.find_by_fitting_job_id(job_id)
        return latest is not None and predicate(latest)

    def _stale(self, job: AsyncFittingJob, timing: dict) -> bool:
        return (timing.get("updated") or 0.0) >= self._config.stale_seconds and not self._leased(
            job
        )

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
        elif status.kind is TaskStatusKind.FAILED and status.error_code in FALLBACK_ERROR_CODES:
            self._repo.settle_cleanup(job.fitting_job_id)  # 확정 실패, 보고된 비용 없음
            report.actions.append("cleanup_settled_without_cost")
        else:
            self._repo.defer_check(job.fitting_job_id, self._config.cleanup_interval_seconds)
            report.actions.append("cleanup_deferred")

    # 비동기 Pruna 상태 조회 → 확인된 결과만 9단계 처리로
    def _query(self, job: AsyncFittingJob, report: CheckReport, why: str, *, alert: bool = True):
        status = self._status.query(job.task_uuid)
        report.actions.append(f"query_{why}_{status.kind.value}")
        if status.kind is TaskStatusKind.SUCCEEDED:
            self._apply(
                job, WebhookEvent(job.task_uuid, EventKind.SUCCEEDED, cost=status.cost), report
            )
        elif status.kind is TaskStatusKind.FAILED and status.error_code in FALLBACK_ERROR_CODES:
            event = WebhookEvent(
                job.task_uuid, EventKind.FAILED, status.error_code, cost=status.cost
            )
            self._apply(job, event, report)
        elif status.kind is TaskStatusKind.FAILED and alert:
            # 조회 오류인지 생성 실패인지 근거가 부족한 코드. 상태·자리를 바꾸지 않는다.
            self._alert(
                f"pruna_error_{status.error_code or 'unknown'}", "pruna", job.fitting_job_id
            )
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
    def _republish(self, job: AsyncFittingJob, action: NextAction, report: CheckReport) -> None:
        if not self._repo.claim_republish(
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

    def _postprocess(self, job: AsyncFittingJob, report: CheckReport) -> None:
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
        # 생성 요청을 보냈을 수 있는데 이미지 기록이 없다. 동기 결과는 getResponse 근거가 없어 고정 Key 만 확인한다.
        from app.virtual_fitting.fallback_handler import fallback_result_key

        key = fallback_result_key(job.fitting_job_id)
        check = self._storage.head_object(key)
        report.actions.append(f"fallback_head_{check.state.value}")
        if check.state is S3ObjectState.EXISTS and self._fresh(check, job.fallback_started_at):
            # 고정 Key 는 이 작업의 fallback 저장만 쓰고 fallback 은 작업당 1회라, 시작 뒤 저장된 객체는 그 시도의 결과다.
            recovered = self._recover_with_lease(
                job,
                NextAction.RUN_FALLBACK,
                lambda token: self._state.record_fallback_image(job.fitting_job_id, token, key),
            )
            if recovered:
                self._republish(job, NextAction.RUN_FALLBACK, report)
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


# --- 준비 여부(FollowUpPaths) ---

_registered: WatchdogWorker | None = None


def register_watchdog(worker: WatchdogWorker | None) -> None:
    global _registered
    _registered = worker


class WatchdogPaths:
    """Watchdog 이 실제로 돌 때, 그것이 조사·확인된 근거로 이어 갈 수 있는 일만 준비됐다고 답한다.

    - AWAIT_WEBHOOK(접수됨, 완료 알림 대기): Webhook 이 오지 않아도 3분부터 getResponse 로 확인한다.
    - RECOVER_UNKNOWN(호출 시작 기록만 있고 접수 불명확): 3분부터 getResponse 로 확인한다(재전송 없음).
    그 밖(OPERATOR_REVIEW·인증 오류 보류 등)은 준비되지 않음. 내부 큐 URL 이 로컬 검증을 통과해야 한다(확인 결과를 POSTPROCESS·
    RUN_FALLBACK 으로 이어 가므로). 로그를 남길 수 있다는 것만으로는 준비됐다고 보지 않는다.
    """

    def __init__(self, worker: Callable[[], WatchdogWorker | None] = lambda: _registered):
        self._worker = worker

    def ready_for(self, job: AsyncFittingJob | None, step: NextStep) -> bool:
        if job is None or step not in (NextStep.AWAIT_WEBHOOK, NextStep.RECOVER_UNKNOWN):
            return False
        worker = self._worker()
        if worker is None or not worker.alive or job.task_uuid is None:
            return False
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


def follow_up_paths() -> AnyFollowUpPaths:
    """요청·후처리 Worker·Webhook 이 쓰는 준비 판단: 결과 전송 루프(PUBLISH_RESULT) + Watchdog(AWAIT_WEBHOOK·RECOVER_UNKNOWN)."""
    from app.virtual_fitting.result_publisher import result_publish_paths

    return AnyFollowUpPaths(result_publish_paths(), WatchdogPaths())


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
    )
    worker: WatchdogWorker

    def publisher_ready() -> bool:
        # 14단계 판단과 같다: 결과 전송 루프가 실제로 돌고 결과 큐 설정이 로컬 검증을 통과한다.
        from app.virtual_fitting import result_publisher

        current = result_publisher._registered
        return current is not None and current.alive and result_publisher._result_queue_configured()

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
    )
    worker = WatchdogWorker(watchdog.run_once, interval_seconds=config.check_interval_seconds)
    return worker
