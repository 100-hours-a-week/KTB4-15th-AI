"""15단계: Watchdog — 미완료 작업 조사·확인된 근거로 복구·운영 이상 탐지.

실제 Watchdog·WatchdogRepository·AsyncJobStateRepository·RunwareWebhookService(9단계 처리)·PrunaLimiter·TaskCostRepository
(격리 PostgreSQL)를 쓴다. Runware 상태 조회 HTTP·S3·SQS·알림 전송만 가짜다. 시간 경계는 DB 시각을 옮겨 확인한다(실제로 기다리지 않음).
"""

import asyncio
import io
import json
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient

from app.clients.s3 import S3ObjectCheck, S3ObjectState
from app.clients.sqs import FittingQueue, SqsOperation, SqsOutcome, SqsResult
from app.config import settings
from app.virtual_fitting import watchdog as wd
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.providers.runware_status import (
    RunwareTaskStatusClient,
    TaskStatus,
    TaskStatusKind,
    parse_status,
)
from app.virtual_fitting.pruna_submitter import NextStep
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.pruna_limiter import TaskCostRepository
from app.virtual_fitting.runware_webhook import RunwareWebhookService
from app.virtual_fitting.watchdog import (
    StatusQueryGate,
    Watchdog,
    WatchdogPaths,
    WatchdogRepository,
    WatchdogSettings,
    WatchdogWorker,
    register_watchdog,
)
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_pruna_submit import JOB, FakeClock, db, job, limiter  # noqa: F401
from tests.test_virtual_fitting_webhook import submitted

ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_SQL = ROOT / "scripts" / "db" / "virtual_fitting_watchdog.sql"
AI_QUEUE = "https://sqs.ap-northeast-2.amazonaws.com/123456789012/ai-fitting-response"
ASYNC_KEY = f"virtual-fitting/results/{JOB}/pruna-async.jpg"
SYNC_KEY = f"virtual-fitting/results/{JOB}/pruna-sync.jpg"


@pytest.fixture
def wdb(db, monkeypatch):  # noqa: F811
    db["sql"](WATCHDOG_SQL.read_text(encoding="utf-8"))
    monkeypatch.setattr(settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(settings, "AI_FITTING_RESPONSE_QUEUE_URL", AI_QUEUE)
    register_watchdog(None)
    yield db
    register_watchdog(None)


class FakeStatus:
    def __init__(self, *statuses):
        self.statuses, self.calls = list(statuses), []

    def query(self, task_uuid):
        self.calls.append(task_uuid)
        item = self.statuses.pop(0) if self.statuses else TaskStatusKind.PROCESSING
        if isinstance(item, TaskStatus):
            return item
        return TaskStatus(item, task_uuid)


class FakeQueue:
    def __init__(self, *outcomes):
        self.outcomes, self.sent = list(outcomes), []

    def send_json(self, queue, payload):
        self.sent.append((queue, dict(payload)))
        outcome = self.outcomes.pop(0) if self.outcomes else SqsOutcome.SUCCEEDED
        return SqsResult(SqsOperation.SEND, queue, outcome, "ok")


class FakeStorage:
    def __init__(self, checks=None):
        self.checks, self.heads = dict(checks or {}), []

    def head_object(self, key):
        self.heads.append(key)
        return self.checks.get(key, S3ObjectCheck(S3ObjectState.NOT_FOUND, key))


class Sink:
    def __init__(self):
        self.alerts = []

    def emit(self, alert):
        self.alerts.append(alert)


def watchdog(db, status=None, *, queue=None, storage=None, sink=None, **kwargs):  # noqa: F811
    queue = queue or FakeQueue()
    events = RunwareWebhookService(
        jobs=db["jobs"],
        state=db["state"],
        limiter=limiter(db),
        costs=TaskCostRepository(db["pool"]),
        queue=queue,
    )
    w = Watchdog(
        jobs=db["jobs"],
        state=db["state"],
        repo=WatchdogRepository(db["pool"]),
        status=StatusQueryGate(status or FakeStatus(), min_interval_seconds=0),
        events=events,
        sqs=queue,
        storage=storage or FakeStorage(),
        alerts=sink or Sink(),
        **kwargs,
    )
    w.queue, w.sink, w.status_client = queue, w._alerts, status
    return w


def clear_lease(db, job_id=JOB):  # noqa: F811
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_token = NULL, worker_lease_until = NULL "
        "WHERE fitting_job_id = %s",
        (job_id,),
    )


def age(db, column, seconds, job_id=JOB):  # noqa: F811
    db["sql"](
        f"UPDATE virtual_fitting_async_jobs SET {column} = statement_timestamp() - make_interval(secs => %s) "
        "WHERE fitting_job_id = %s",
        (seconds, job_id),
    )


def waiting(db, accepted_ago, job_id=JOB):  # noqa: F811
    task = submitted(db, job_id)
    clear_lease(db, job_id)
    age(db, "runware_accepted_at", accepted_ago, job_id)
    return task


def unknown(db, started_ago, job_id=JOB):  # noqa: F811
    task = submitted(db, job_id, accepted=False)
    clear_lease(db, job_id)
    age(db, "pruna_submit_started_at", started_ago, job_id)
    return task


def in_flight(db):  # noqa: F811
    return db["sql"]("SELECT fitting_job_id, task_uuid::text FROM virtual_fitting_pruna_in_flight")


def costs(db):  # noqa: F811
    return db["sql"]("SELECT task_uuid::text, cost, usage_date FROM virtual_fitting_task_costs")


def actions(w, db):  # noqa: F811
    return [a for report in w.run_once() for a in report.actions]


def reasons(sink):
    return [(a.reason, a.stage, a.fitting_job_id) for a in sink.alerts]


# --- 상태 조회 Provider ---


TASK = "a770f077-f413-47de-9dac-be0b26a35da6"


def body(obj):
    return json.dumps(obj).encode()


def test_status_parsing_matches_the_task_and_never_guesses():
    assert (
        parse_status(body({"data": [{"taskUUID": TASK, "status": "processing"}]}), TASK).kind
        is TaskStatusKind.PROCESSING
    )
    ok = parse_status(
        body(
            {
                "data": [
                    {"taskUUID": TASK, "status": "success", "cost": 0.015, "imageURL": "https://x"}
                ]
            }
        ),
        TASK,
    )
    assert (ok.kind, ok.cost) == (TaskStatusKind.SUCCEEDED, Decimal("0.015"))
    failed = parse_status(
        body({"errors": [{"taskUUID": TASK, "code": "timeoutProvider", "status": "error"}]}), TASK
    )
    assert (failed.kind, failed.error_code) == (TaskStatusKind.FAILED, "timeoutProvider")
    other = str(uuid.uuid4())
    assert (
        parse_status(body({"data": [{"taskUUID": other, "status": "success"}]}), TASK).kind
        is TaskStatusKind.NOT_FOUND
    )
    # 문서 예시처럼 taskUUID 없는 오류는 이 작업의 것인지 알 수 없다.
    assert (
        parse_status(body({"errors": [{"code": "timeoutProvider", "status": "error"}]}), TASK).kind
        is TaskStatusKind.NOT_FOUND
    )
    assert parse_status(b"not json", TASK).kind is TaskStatusKind.UNKNOWN
    bad_cost = parse_status(
        body({"data": [{"taskUUID": TASK, "status": "success", "cost": "x"}]}), TASK
    )
    assert bad_cost.kind is TaskStatusKind.SUCCEEDED and bad_cost.cost is None


class Opener:
    def __init__(self, result):
        self.result, self.requests = result, []

    def __call__(self, request, timeout):
        self.requests.append((json.loads(request.data), timeout))
        if isinstance(self.result, BaseException):
            raise self.result

        class R(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        return R(body(self.result))


@pytest.mark.parametrize(
    ("result", "kind"),
    [
        ({"data": [{"taskUUID": TASK, "status": "processing"}]}, TaskStatusKind.PROCESSING),
        (HTTPError("u", 401, "x", {}, io.BytesIO(b"{}")), TaskStatusKind.AUTH_ERROR),
        (HTTPError("u", 403, "x", {}, io.BytesIO(b"{}")), TaskStatusKind.AUTH_ERROR),
        (HTTPError("u", 500, "x", {}, io.BytesIO(b"{}")), TaskStatusKind.UNKNOWN),
        (TimeoutError(), TaskStatusKind.UNKNOWN),
    ],
)
def test_status_client_sends_one_get_response_and_classifies(result, kind):
    opener = Opener(result)
    status = RunwareTaskStatusClient("k", opener=opener).query(TASK)
    assert status.kind is kind
    ((payload, timeout),) = opener.requests  # 요청 1번, 재시도 없음
    assert payload == [{"taskType": "getResponse", "taskUUID": TASK}] and timeout == 10.0


def test_status_client_respects_retry_after_on_429():
    error = HTTPError("u", 429, "x", {"Retry-After": "30"}, io.BytesIO(b"{}"))
    status = RunwareTaskStatusClient("k", opener=Opener(error)).query(TASK)
    assert (status.kind, status.retry_after) == (TaskStatusKind.RATE_LIMITED, 30.0)


def test_gate_spaces_queries_and_holds_after_rate_limit():
    clock = FakeClock()
    client = FakeStatus(
        TaskStatus(TaskStatusKind.RATE_LIMITED, TASK, retry_after=20), TaskStatusKind.PROCESSING
    )
    gate = StatusQueryGate(client, min_interval_seconds=1, clock=clock, sleep=clock.sleep)
    assert gate.query(TASK).kind is TaskStatusKind.RATE_LIMITED
    assert (
        gate.query(TASK).kind is TaskStatusKind.RATE_LIMITED and len(client.calls) == 1
    )  # 안내 동안 조회 안 함
    clock.sleep(21)
    assert gate.query(TASK).kind is TaskStatusKind.PROCESSING and len(client.calls) == 2


# --- 비동기 지연·접수 불명확 ---


def test_waiting_is_queried_from_three_minutes_after_acceptance(wdb):
    task = waiting(wdb, 179)
    status = FakeStatus()
    w = watchdog(wdb, status)
    assert w.run_once() == [] and status.calls == []
    age(wdb, "runware_accepted_at", 181)
    w.run_once()
    assert status.calls == [task]
    assert job(wdb).status is AsyncJobStatus.WAITING_PRUNA and in_flight(wdb) == [(JOB, task)]
    assert w.sink.alerts == []  # 5분 전에는 지연 알림 없음


def test_delay_alert_from_five_minutes_and_processing_keeps_everything(wdb):
    task = waiting(wdb, 301)
    w = watchdog(wdb, FakeStatus(TaskStatusKind.PROCESSING))
    w.run_once()
    assert reasons(w.sink) == [("pruna_delayed", "pruna", JOB)]
    current = job(wdb)
    assert current.status is AsyncJobStatus.WAITING_PRUNA and in_flight(wdb) == [(JOB, task)]
    assert current.fallback_attempt_count == 0


def test_unknown_submission_uses_the_submit_start_time(wdb):
    task = unknown(wdb, 200)
    status = FakeStatus()
    w = watchdog(wdb, status)
    w.run_once()
    assert status.calls == [task]
    age(wdb, "pruna_submit_started_at", 301)
    wdb["sql"]("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
    w.run_once()
    assert ("pruna_submit_unconfirmed", "pruna", JOB) in reasons(w.sink)
    assert job(wdb).runware_accepted_at is None  # 접수 시각을 만들어 지연 판단을 늦추지 않음


def test_confirmed_success_goes_through_the_existing_success_path(wdb):
    task = waiting(wdb, 200)
    started = job(wdb).pruna_submit_started_at
    w = watchdog(wdb, FakeStatus(TaskStatus(TaskStatusKind.SUCCEEDED, task, Decimal("0.015"))))
    w.run_once()
    current = job(wdb)
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert costs(wdb) == [(task, Decimal("0.015"), started.astimezone(UTC).date())]
    assert in_flight(wdb) == []
    assert [p["action"] for _, p in w.queue.sent] == ["POSTPROCESS"]


def test_confirmed_timeout_failure_goes_to_fallback(wdb):
    waiting(wdb, 200)
    w = watchdog(
        wdb,
        FakeStatus(
            TaskStatus(TaskStatusKind.FAILED, job(wdb).task_uuid, error_code="timeoutProvider")
        ),
    )
    w.run_once()
    assert job(wdb).next_action is NextAction.RUN_FALLBACK and in_flight(wdb) == []
    assert [p["action"] for _, p in w.queue.sent] == ["RUN_FALLBACK"]


@pytest.mark.parametrize(
    "result",
    [
        TaskStatusKind.NOT_FOUND,
        TaskStatusKind.UNKNOWN,
        TaskStatusKind.AUTH_ERROR,
        TaskStatusKind.RATE_LIMITED,
        "someQueryError",
    ],
)
def test_unclear_query_results_change_nothing(wdb, result):
    task = waiting(wdb, 400)
    status = (
        TaskStatus(TaskStatusKind.FAILED, task, error_code=result)
        if isinstance(result, str)
        else result
    )
    w = watchdog(wdb, FakeStatus(status))
    before = job(wdb)
    w.run_once()
    after = job(wdb)
    assert (
        after.status,
        after.next_action,
        after.result_payload,
        after.fallback_attempt_count,
    ) == (before.status, before.next_action, before.result_payload, before.fallback_attempt_count)
    assert in_flight(wdb) == [(JOB, task)] and w.queue.sent == [] and costs(wdb) == []


def test_watchdog_never_sends_generation_requests(wdb):
    waiting(wdb, 400)
    calls = []
    status = FakeStatus(TaskStatusKind.PROCESSING)
    w = watchdog(wdb, status)
    w._events.handle = lambda events: calls.append(events) or []
    w.run_once()
    # 상태 조회만 했고 생성 Provider·fallback 은 부르지 않았다(조립에 생성 Provider 가 없다).
    assert status.calls and calls == []


# --- 다른 인스턴스·Webhook 과 동시에 ---


def test_two_watchdogs_check_a_job_once_per_interval(wdb):
    waiting(wdb, 400)
    status = FakeStatus()
    first, second = watchdog(wdb, status), watchdog(wdb, status)
    results, barrier = [], threading.Barrier(2)

    def go(w):
        barrier.wait()
        results.append(len(w.run_once()))

    threads = [threading.Thread(target=go, args=(w,)) for w in (first, second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert sorted(results) == [0, 1] and len(status.calls) == 1


def test_webhook_and_watchdog_success_together_stay_consistent(wdb):
    from app.virtual_fitting.runware_webhook import parse_webhook
    from tests.test_virtual_fitting_webhook import FakeQueue as WebhookQueue
    from tests.test_virtual_fitting_webhook import success_body

    task = waiting(wdb, 400)
    w = watchdog(wdb, FakeStatus(TaskStatus(TaskStatusKind.SUCCEEDED, task, Decimal("0.015"))))
    hook = RunwareWebhookService(
        jobs=wdb["jobs"],
        state=wdb["state"],
        limiter=limiter(wdb),
        costs=TaskCostRepository(wdb["pool"]),
        queue=WebhookQueue(),
    )
    barrier = threading.Barrier(2)
    errors = []

    def run(fn):
        try:
            barrier.wait()
            fn()
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [
        threading.Thread(target=run, args=(w.run_once,)),
        threading.Thread(
            target=run, args=(lambda: hook.handle(parse_webhook(success_body(task))),)
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert not errors, errors
    assert job(wdb).status is AsyncJobStatus.POSTPROCESSING
    assert len(costs(wdb)) == 1 and in_flight(wdb) == []


def test_candidates_rotate_so_one_job_does_not_starve_others(wdb):
    for job_id in (1, 2):  # 도우미가 작업마다 동시 전송 자리(한도 2)를 쥔다
        waiting(wdb, 400, job_id)
    updated = "SELECT fitting_job_id, updated_at FROM virtual_fitting_async_jobs ORDER BY 1"
    before = wdb["sql"](updated)
    w = watchdog(wdb, FakeStatus(), config=WatchdogSettings(batch_size=1))
    checked = [w.run_once()[0].fitting_job_id for _ in range(2)]
    assert sorted(checked) == [1, 2] and w.run_once() == []
    assert wdb["sql"](updated) == before  # 점검이 지연 기준 시각(updated_at)을 초기화하지 않는다


# --- fallback ---


def fallback_started(db, image=False):  # noqa: F811
    from tests.test_virtual_fitting_fallback_handler import fallback_job

    fallback_job(db)
    fb = str(uuid.uuid4())
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET fallback_task_uuid = %s, fallback_started_at = now() - interval '20 minutes', "
        "fallback_attempt_count = 1, fallback_image_key = %s, worker_lease_token = NULL, worker_lease_until = NULL "
        "WHERE fitting_job_id = %s",
        (fb, SYNC_KEY if image else None, JOB),
    )
    age(db, "updated_at", 700)
    return fb


def test_fallback_object_after_the_attempt_is_recorded_and_resumed(wdb):
    fb = fallback_started(wdb)
    fresh = S3ObjectCheck(
        S3ObjectState.EXISTS, SYNC_KEY, last_modified=datetime.now(UTC) - timedelta(minutes=19)
    )
    status = FakeStatus()
    w = watchdog(wdb, status, storage=FakeStorage({SYNC_KEY: fresh}))
    w.run_once()
    current = job(wdb)
    assert current.fallback_image_key == SYNC_KEY and current.fallback_attempt_count == 1
    assert [p["action"] for _, p in w.queue.sent] == ["RUN_FALLBACK"]
    assert fb not in status.calls  # 동기 fallback 은 getResponse 로 조회하지 않는다
    # 재발행된 메시지를 처리하면 생성 없이 제목·코멘트부터 이어 간다.
    from tests.test_virtual_fitting_fallback_handler import env as fallback_env
    from tests.test_virtual_fitting_fallback_handler import run as run_fallback

    e = fallback_env(wdb)
    run_fallback(wdb, e)
    assert e.opener.requests == [] and job(wdb).result_payload["result_image_key"] == SYNC_KEY


@pytest.mark.parametrize(
    "check",
    [
        S3ObjectCheck(S3ObjectState.NOT_FOUND, SYNC_KEY),
        S3ObjectCheck(S3ObjectState.ACCESS_DENIED, SYNC_KEY),
        S3ObjectCheck(
            S3ObjectState.EXISTS, SYNC_KEY, last_modified=datetime(2020, 1, 1, tzinfo=UTC)
        ),
        S3ObjectCheck(S3ObjectState.EXISTS, SYNC_KEY),  # 시각을 모름
    ],
)
def test_fallback_without_a_matching_object_is_kept_for_operators(wdb, check):
    fallback_started(wdb)
    w = watchdog(wdb, storage=FakeStorage({SYNC_KEY: check}))
    w.run_once()
    current = job(wdb)
    assert current.fallback_image_key is None and current.result_payload is None
    assert current.fallback_attempt_count == 1 and w.queue.sent == []
    assert ("fallback_result_unknown", "fallback", JOB) in reasons(w.sink)


def test_recorded_fallback_image_is_republished_and_throttled(wdb):
    fallback_started(wdb, image=True)
    queue = FakeQueue(SqsOutcome.UNKNOWN)
    w = watchdog(wdb, queue=queue)
    assert "republish_run_fallback_failed" in actions(w, wdb)
    wdb["sql"]("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
    assert "republish_run_fallback_ok" in actions(w, wdb)  # 실패 뒤 다음 점검에서 복구
    wdb["sql"]("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
    assert "republish_run_fallback_throttled" in actions(w, wdb)  # 10분 안에는 다시 보내지 않음
    assert [p for _, p in queue.sent] == [
        {"fitting_job_id": JOB, "task_uuid": job(wdb).task_uuid, "action": "RUN_FALLBACK"}
    ] * 2


def test_fallback_slot_is_not_released_without_evidence(wdb):
    fb = fallback_started(wdb)
    wdb["sql"](
        "INSERT INTO virtual_fitting_pruna_in_flight VALUES (%s, %s, now() - interval '20 minutes')",
        (JOB, fb),
    )
    status = FakeStatus()
    w = watchdog(wdb, status)
    w.run_once()
    assert in_flight(wdb) == [(JOB, fb)] and fb not in status.calls
    assert ("fallback_slot_unconfirmed", "fallback", JOB) in reasons(w.sink)


def test_async_slot_blocking_a_fallback_is_released_only_after_a_confirmed_end(wdb):
    from tests.test_virtual_fitting_postprocess_worker import falling_back

    task = falling_back(wdb)  # 비동기 자리가 남아 fallback 이 OTHER_ATTEMPT_IN_FLIGHT 로 막혀 있다
    wdb["sql"](
        "UPDATE virtual_fitting_pruna_in_flight SET acquired_at = now() - interval '20 minutes'"
    )
    w = watchdog(wdb, FakeStatus(TaskStatusKind.PROCESSING))
    w.run_once()
    assert in_flight(wdb) == [(JOB, task)]  # 진행 중이면 그대로
    wdb["sql"]("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
    w2 = watchdog(
        wdb, FakeStatus(TaskStatus(TaskStatusKind.FAILED, task, error_code="timeoutProvider"))
    )
    w2.run_once()
    assert in_flight(wdb) == []  # 그 시도의 종료가 확인돼 그 자리만 정리
    assert job(wdb).status is AsyncJobStatus.FALLBACK_RUNNING


# --- 중단된 S3 확인 묶음 ---


def stale_postprocess(db, s3_count=2, verified=False):  # noqa: F811
    from tests.test_virtual_fitting_postprocess_worker import postprocessing

    postprocessing(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET s3_check_attempt_count = %s, verified_image_key = %s "
        "WHERE fitting_job_id = %s",
        (s3_count, ASYNC_KEY if verified else None, JOB),
    )
    age(db, "updated_at", 700)


def test_interrupted_second_bundle_recovers_only_with_an_existing_object(wdb):
    stale_postprocess(wdb)
    fresh = S3ObjectCheck(S3ObjectState.EXISTS, ASYNC_KEY, last_modified=datetime.now(UTC))
    w = watchdog(wdb, storage=FakeStorage({ASYNC_KEY: fresh}))
    w.run_once()
    current = job(wdb)
    assert (
        current.verified_image_key == ASYNC_KEY and current.s3_check_attempt_count == 2
    )  # 횟수 그대로
    assert [p["action"] for _, p in w.queue.sent] == ["POSTPROCESS"]


@pytest.mark.parametrize(
    "state", [S3ObjectState.NOT_FOUND, S3ObjectState.ACCESS_DENIED, S3ObjectState.TRANSIENT_ERROR]
)
def test_interrupted_second_bundle_without_an_object_never_falls_back(wdb, state):
    stale_postprocess(wdb)
    w = watchdog(wdb, storage=FakeStorage({ASYNC_KEY: S3ObjectCheck(state, ASYNC_KEY)}))
    w.run_once()
    current = job(wdb)
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert current.s3_check_attempt_count == 2 and current.verified_image_key is None
    assert w.queue.sent == [] and ("s3_check_unconfirmed", "postprocess", JOB) in reasons(w.sink)


def test_verified_image_resumes_llm_without_head(wdb):
    stale_postprocess(wdb, verified=True)
    storage = FakeStorage()
    w = watchdog(wdb, storage=storage)
    w.run_once()
    assert storage.heads == [] and [p["action"] for _, p in w.queue.sent] == ["POSTPROCESS"]


def test_running_worker_is_not_disturbed(wdb):
    stale_postprocess(wdb, verified=True)
    wdb["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60)
    w = watchdog(wdb)
    w.run_once()
    assert w.queue.sent == []


# --- 결과 이후 정리·결과 발행·전역 이상 ---


def test_lingering_slot_after_the_final_result_is_cleaned_without_changing_it(wdb):
    task = waiting(wdb, 400)
    final = {
        "fitting_job_id": JOB,
        "status": "COMPLETED",
        "result_image_key": f"virtual-fitting/results/{JOB}/final.jpg",
        "llm_title": "t",
        "llm_comment": "c",
    }
    wdb["sql"](
        "UPDATE virtual_fitting_async_jobs SET status = 'COMPLETED', next_action = NULL, result_payload = %s::jsonb, "
        "result_message_status = 'PUBLISHED', result_image_key = %s WHERE fitting_job_id = %s",
        (json.dumps(final), final["result_image_key"], JOB),
    )
    wdb["sql"](
        "UPDATE virtual_fitting_pruna_in_flight SET acquired_at = now() - interval '20 minutes'"
    )
    w = watchdog(wdb, FakeStatus(TaskStatus(TaskStatusKind.SUCCEEDED, task, Decimal("0.02"))))
    w.run_once()
    current = job(wdb)
    assert current.status is AsyncJobStatus.COMPLETED and current.result_payload == final
    assert in_flight(wdb) == [] and costs(wdb)[0][1] == Decimal("0.02") and w.queue.sent == []


def test_retry_exhausted_is_alerted_never_republished(wdb):
    from tests.test_virtual_fitting_result_publisher import saved_by_webhook

    saved_by_webhook(wdb)
    wdb["sql"](
        "UPDATE virtual_fitting_async_jobs SET result_message_status = 'RETRY_EXHAUSTED', "
        "result_message_attempt_count = 3 WHERE fitting_job_id = %s",
        (JOB,),
    )
    w = watchdog(wdb)
    w.run_once()
    current = job(wdb)
    assert current.result_message_status is ResultMessageStatus.RETRY_EXHAUSTED
    assert current.result_message_attempt_count == 3 and w.queue.sent == []
    assert ("result_publish_exhausted", "publish", JOB) in reasons(w.sink)


def test_pending_result_without_a_publisher_is_alerted(wdb):
    from tests.test_virtual_fitting_result_publisher import saved_by_webhook

    saved_by_webhook(wdb)
    age(wdb, "updated_at", 700)
    w = watchdog(wdb, publisher_ready=lambda: False)
    w.run_once()
    assert ("result_publisher_unavailable", "publish", JOB) in reasons(w.sink)


def test_send_hold_and_stopped_components_are_alerted_not_changed(wdb):
    limiter(wdb).hold_sends("retry_after_unrepresentable")

    class Stopped:
        class status:
            running, crashed = False, True

    w = watchdog(wdb, components=lambda: {"result_publisher": Stopped(), "request_worker": None})
    w.run_once()
    assert ("send_hold_retry_after_unrepresentable", "pruna_send_hold", None) in reasons(w.sink)
    assert ("component_stopped", "component_result_publisher", None) in reasons(w.sink)
    assert (
        wdb["sql"]("SELECT send_hold_reason FROM virtual_fitting_pruna_limiter")[0][0] is not None
    )


def test_alerts_are_deduplicated_and_realerted_after_the_interval_even_after_restart(wdb):
    waiting(wdb, 400)
    sink = Sink()
    for _ in range(2):
        w = watchdog(wdb, FakeStatus(), sink=sink)  # 매번 새 Watchdog(재시작·다른 인스턴스)
        wdb["sql"]("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
        w.run_once()
    assert reasons(sink) == [("pruna_delayed", "pruna", JOB)]
    seen = wdb["sql"]("SELECT seen_count FROM virtual_fitting_watchdog_alerts")[0][0]
    assert seen == 2
    wdb["sql"](
        "UPDATE virtual_fitting_watchdog_alerts SET last_notified_at = now() - interval '16 minutes'"
    )
    wdb["sql"]("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
    watchdog(wdb, FakeStatus(), sink=sink).run_once()
    assert len(sink.alerts) == 2
    event = sink.alerts[0].as_event()
    assert set(event) == {
        "event",
        "reason",
        "stage",
        "fitting_job_id",
        "first_seen_at",
        "last_seen_at",
        "seen_count",
    }


def test_lock_timeout_and_connection_reuse(wdb):
    waiting(wdb, 400)
    repo = WatchdogRepository(wdb["pool"])
    assert repo.claim_check(JOB, 60)
    connection = wdb["connect"]()
    try:
        connection.execute(
            "SELECT 1 FROM virtual_fitting_watchdog_checks WHERE fitting_job_id = %s FOR UPDATE",
            (JOB,),
        )
        started = time.monotonic()
        with pytest.raises(FittingDatabaseError):
            repo.claim_check(JOB, 60)
        assert time.monotonic() - started < 8
    finally:
        connection.rollback()
        connection.close()
    with wdb["pool"].connection() as c:
        assert c.execute("SHOW lock_timeout").fetchone()[0] == "0"
    assert repo.candidates(WatchdogSettings()) == []  # 다음 점검 시각 전


# --- 준비 여부·Webhook·메시지 삭제 ---


class Alive:
    def __init__(self, alive=True):
        self.alive = alive


def test_watchdog_paths_only_for_its_real_scope(wdb, monkeypatch):
    waiting(wdb, 10)
    accepted = job(wdb)
    ready = WatchdogPaths(lambda: Alive())
    assert ready.ready_for(accepted, NextStep.AWAIT_WEBHOOK)
    assert ready.ready_for(accepted, NextStep.RECOVER_UNKNOWN)
    for step in (
        NextStep.OPERATOR_REVIEW,
        NextStep.PUBLISH_RESULT,
        NextStep.POSTPROCESS,
        NextStep.RUN_FALLBACK,
    ):
        assert not ready.ready_for(accepted, step)
    assert not WatchdogPaths(lambda: None).ready_for(accepted, NextStep.AWAIT_WEBHOOK)
    assert not WatchdogPaths(lambda: Alive(False)).ready_for(accepted, NextStep.AWAIT_WEBHOOK)
    monkeypatch.setattr(settings, "AI_FITTING_RESPONSE_QUEUE_URL", "")
    assert not ready.ready_for(accepted, NextStep.AWAIT_WEBHOOK)


from tests.test_virtual_fitting_webhook import (  # noqa: F401
    api,
    error_body,
    post,
    queue_settings,
)


def test_auth_hold_is_not_solved_by_a_running_watchdog(api, monkeypatch):  # noqa: F811
    api["db"]["sql"](WATCHDOG_SQL.read_text(encoding="utf-8"))
    register_watchdog(Alive())
    try:
        task = submitted(api["db"])
        response = post(api, error_body(task, code="invalidApiKey"))
        assert response.status_code == 503
        assert response.json()["data"]["pending"][0]["next_step"] == "operator_review"
    finally:
        register_watchdog(None)


def test_request_message_is_deleted_and_the_job_is_found_again(wdb):
    from app.virtual_fitting.request_worker import RequestProcessor
    from app.virtual_fitting.watchdog import follow_up_paths
    from tests.test_virtual_fitting_pruna_submit import FixedCheck
    from tests.test_virtual_fitting_request_worker import FakeSqs as RequestSqs
    from tests.test_virtual_fitting_request_worker import valid_message

    task = waiting(wdb, 10)

    def process():
        return RequestProcessor(
            wdb["jobs"],
            wdb["state"],
            FixedCheck(),
            RequestSqs(),
            lease_seconds=60,
            visibility_seconds=120,
            follow_up=follow_up_paths(),
        ).process(valid_message())

    assert process().delete is False  # Watchdog 없음 → 보존
    register_watchdog(Alive())
    assert process().delete is True  # 접수 대기 작업은 Watchdog 이 이어서 확인할 수 있다
    age(wdb, "runware_accepted_at", 200)
    status = FakeStatus(TaskStatus(TaskStatusKind.SUCCEEDED, task, Decimal("0.015")))
    watchdog(wdb, status).run_once()  # 삭제 뒤에도 DB 에서 다시 찾아 확인한다
    assert status.calls == [task] and job(wdb).status is AsyncJobStatus.POSTPROCESSING


# --- 루프·lifespan·실제 조립 ---


def test_worker_runs_stops_backs_off_and_reports_crashes():
    cycles = []

    async def normal():
        worker = WatchdogWorker(lambda: cycles.append(1), interval_seconds=0.01)
        worker.start()
        await asyncio.sleep(0.1)
        assert worker.alive
        await worker.stop(1)
        return worker

    worker = asyncio.run(normal())
    assert cycles and worker.status.stopped and not worker.alive

    def bug():
        raise RuntimeError("bug")

    async def crash():
        worker = WatchdogWorker(bug)
        with pytest.raises(RuntimeError):
            await worker.start()
        return worker

    crashed = asyncio.run(crash())
    assert crashed.status.crashed and not crashed.alive
    assert not WatchdogPaths(lambda: crashed).ready_for(None, NextStep.AWAIT_WEBHOOK)


from tests.test_virtual_fitting_request_worker import app_env  # noqa: F401


def test_lifespan_disabled_and_enabled(app_env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(
        app_env, "create_watchdog", lambda components: pytest.fail("꺼져 있으면 만들지 않는다")
    )
    with TestClient(app_env.app) as client:
        assert client.app.state.watchdog is None
    assert settings.VIRTUAL_FITTING_WATCHDOG_ENABLED is False

    entered, release, order = threading.Event(), threading.Event(), []

    def run_once():
        entered.set()
        release.wait(5)

    worker = WatchdogWorker(run_once, interval_seconds=0.01)
    seen = {}
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_WATCHDOG_ENABLED", True)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_WATCHDOG_SHUTDOWN_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(
        app_env, "create_watchdog", lambda components: seen.update(c=components) or worker
    )
    monkeypatch.setattr(app_env, "close_connection_pool", lambda: order.append("close_pool"))
    try:
        with TestClient(app_env.app) as client:
            assert entered.wait(5)
            started = time.monotonic()
            assert client.get("/health").status_code == 200
            assert time.monotonic() - started < 1  # 점검 중에도 다른 HTTP 요청 처리
            assert wd._registered is worker and worker.alive
            assert set(seen["c"]()) == {"request_worker", "postprocess_worker", "result_publisher"}
            release.set()
        assert wd._registered is None and worker.status.stopped and order == ["close_pool"]
    finally:
        release.set()


class FakeBoto:
    def __init__(self):
        self.bodies = []

    def send_message(self, QueueUrl, MessageBody):
        self.bodies.append(json.loads(MessageBody))
        return {"MessageId": "m"}


def test_real_assembly_finds_and_republishes_through_the_real_sqs_client(wdb, monkeypatch):
    import app.clients.sqs as sqs_module
    from app.config import database

    stale_postprocess(wdb, verified=True)
    boto = FakeBoto()
    monkeypatch.setattr(database, "get_connection_pool", lambda: wdb["pool"])
    monkeypatch.setattr(sqs_module, "get_sqs_client", lambda: boto)
    monkeypatch.setattr(settings, "S3_BUCKET", "test-bucket")
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "test-key")
    worker = wd.create_watchdog()

    async def scenario():
        worker.start()
        deadline = time.monotonic() + 5
        while not boto.bodies:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        await worker.stop(1)

    asyncio.run(scenario())
    assert boto.bodies == [
        {"fitting_job_id": JOB, "task_uuid": job(wdb).task_uuid, "action": "POSTPROCESS"}
    ]
    assert worker.status.cycles >= 1 and FittingQueue.AI_FITTING_RESPONSE
