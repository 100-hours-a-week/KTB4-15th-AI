"""Webhook 의 DB 일시 장애 재시도(최초 1회 + 재시도 3회, 최대 4회)를 장애 주입으로 검증한다.

재시도 단위: (1) 작업 조회, (2) 성공 알림의 [성공 전환(상태·참조) → 필요하면 늦은 참조 보완]. 큐 발행·비용·자리 반환은
재시도 밖이다. 격리 PostgreSQL·가짜 큐만 쓴다(실제 SQS·Runware 호출 없음). 대기(sleep)는 가짜로 기록만 한다.
"""

import json
import logging

import psycopg
import psycopg_pool
import pytest

from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.pruna_submitter import NextStep
from app.virtual_fitting.repositories.async_job_repository import AsyncJobStatus, NextAction
from app.virtual_fitting.repositories.pruna_limiter import TaskCostRepository
from app.virtual_fitting.runware_webhook import (
    DB_ATTEMPTS,
    DB_RETRY_DELAYS,
    EventOutcome,
    RunwareWebhookService,
    WebhookResultRefNotSavedError,
    WebhookUnknownTaskError,
    parse_webhook,
    transient_db_error,
)
from tests.test_virtual_fitting_ai_result_storage import (  # noqa: F401 (autouse fixture)
    IMAGE_UUID,
    URL,
    _follow_up_queue,
    ai_submitted,
    assert_no_secret,
    success,
)
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_postprocess_worker import ReadyPaths
from tests.test_virtual_fitting_pruna_submit import JOB, db, job, limiter  # noqa: F401
from tests.test_virtual_fitting_webhook import (  # noqa: F401 (pytest fixture)
    FakeQueue,
    api,
    in_flight,
    post,
    queue_settings,
    submitted,
    task_costs,
)

OTHER_URL = "https://im.runware.ai/image/ws/2/ii/OTHER.webp"
OTHER_UUID = "7d444840-9dc0-11d1-b245-5ffdce74fad2"


# --- 장애 주입 ---


CAUSES = {
    "connection": lambda: psycopg.OperationalError("server closed the connection unexpectedly"),
    "pool_timeout": lambda: psycopg_pool.PoolTimeout("couldn't get a connection after 30 sec"),
    "serialization": lambda: psycopg.errors.SerializationFailure("could not serialize"),
    "deadlock": lambda: psycopg.errors.DeadlockDetected("deadlock detected"),
    "lock_timeout": lambda: psycopg.errors.LockNotAvailable("lock timeout"),
    "statement_timeout": lambda: psycopg.errors.QueryCanceled("statement timeout"),
    "admin_shutdown": lambda: psycopg.errors.AdminShutdown("terminating connection"),
    "too_many": lambda: psycopg.errors.TooManyConnections("too many clients"),
    "schema": lambda: psycopg.errors.UndefinedColumn("column does not exist"),
    "syntax": lambda: psycopg.errors.SyntaxError("syntax error"),
    "permission": lambda: psycopg.errors.InsufficientPrivilege("permission denied"),
    "check": lambda: psycopg.errors.CheckViolation("violates check constraint"),
    "unique": lambda: psycopg.errors.UniqueViolation("duplicate key"),
    "data": lambda: psycopg.errors.InvalidTextRepresentation("invalid input"),
}
TRANSIENT = [
    "connection",
    "pool_timeout",
    "serialization",
    "deadlock",
    "lock_timeout",
    "statement_timeout",
    "admin_shutdown",
    "too_many",
]
PERMANENT = ["schema", "syntax", "permission", "check", "unique", "data"]


def db_error(kind):
    error = FittingDatabaseError("AI PostgreSQL 비동기 작업 상태 변경에 실패했습니다.")
    error.__cause__ = CAUSES[kind]()
    return error


class Faulty:
    """real 의 method 를 plan 대로 실패시킨다. plan 항목: 'ok' | '<kind>'(실행 전 실패) |
    'commit:<kind>'(실제로 저장한 뒤 응답 유실) | callable(실행 전 부수 효과 후 그 반환값 kind 로 실패)."""

    def __init__(self, real, method, plan):
        self.real, self.method, self.plan, self.calls = real, method, list(plan), 0

    def __getattr__(self, name):
        attr = getattr(self.real, name)
        if name != self.method:
            return attr

        def call(*args, **kwargs):
            self.calls += 1
            step = self.plan.pop(0) if self.plan else "ok"
            if callable(step):
                raise db_error(step())
            if step == "ok":
                return attr(*args, **kwargs)
            if step.startswith("commit:"):
                attr(*args, **kwargs)
                raise db_error(step.split(":", 1)[1])
            raise db_error(step)

        return call


class Counting:
    """limiter·costs 호출 횟수(재시도 밖 부수 효과가 중복되지 않는지)."""

    def __init__(self, real, method):
        self.real, self.method, self.calls = real, method, 0

    def __getattr__(self, name):
        attr = getattr(self.real, name)
        if name != self.method:
            return attr

        def call(*args, **kwargs):
            self.calls += 1
            return attr(*args, **kwargs)

        return call


class Sleeps:
    def __init__(self):
        self.delays = []

    def __call__(self, seconds):
        self.delays.append(seconds)


def hook(db, *, state=None, jobs=None, queue=None, limiter_=None, costs=None, **kwargs):  # noqa: F811
    sleeps = kwargs.pop("sleep", Sleeps())
    service = RunwareWebhookService(
        jobs=jobs or db["jobs"],
        state=state or db["state"],
        limiter=limiter_ or limiter(db, interval=0),
        costs=costs or TaskCostRepository(db["pool"]),
        queue=queue if queue is not None else FakeQueue(),
        follow_up=ReadyPaths(NextStep.RECOVER_RESULT_REF),
        sleep=sleeps,
        **kwargs,
    )
    service.sleeps = sleeps
    return service


def refs(db, job_id=JOB):  # noqa: F811
    current = job(db, job_id)
    return current.runware_image_url, current.runware_image_uuid


def full_success(task):
    return parse_webhook(success(task, imageURL=URL, imageUUID=IMAGE_UUID))


# === 분류: 일시 장애만 재시도 ===


@pytest.mark.parametrize("kind", TRANSIENT)
def test_transient_errors_are_classified_by_driver_type_and_sqlstate(kind):
    assert transient_db_error(db_error(kind))


@pytest.mark.parametrize("kind", PERMANENT)
def test_permanent_errors_are_not_transient(kind):
    assert not transient_db_error(db_error(kind))


def test_errors_without_a_driver_cause_are_not_transient():
    assert not transient_db_error(FittingDatabaseError("결과를 확인하지 못했습니다."))
    assert not transient_db_error(ValueError("bad input"))


# === 1~4. 시도 횟수 ===


def test_first_success_uses_one_db_attempt(db):  # noqa: F811
    task = ai_submitted(db)
    state = Faulty(db["state"], "record_pruna_succeeded", [])
    jobs = Faulty(db["jobs"], "find_by_task_uuid", [])
    service = hook(db, state=state, jobs=jobs)
    (result,) = service.handle(full_success(task))
    assert result.outcome is EventOutcome.APPLIED
    assert (state.calls, jobs.calls, service.sleeps.delays) == (1, 1, [])


@pytest.mark.parametrize("failures", [1, 2, 3])
def test_retry_stops_at_the_first_success_and_the_fourth_attempt_is_allowed(db, failures):  # noqa: F811
    task = ai_submitted(db)
    state = Faulty(db["state"], "record_pruna_succeeded", ["connection"] * failures)
    service = hook(db, state=state)
    (result,) = service.handle(full_success(task))
    assert result.outcome is EventOutcome.APPLIED and state.calls == failures + 1
    assert service.sleeps.delays == list(DB_RETRY_DELAYS[:failures])  # 짧은 대기만
    assert refs(db) == (URL, IMAGE_UUID) and job(db).status is AsyncJobStatus.POSTPROCESSING


def test_all_four_attempts_failing_is_503_for_the_ai_job(db):  # noqa: F811
    task = ai_submitted(db)
    state = Faulty(db["state"], "record_pruna_succeeded", ["deadlock"] * 10)
    queue = FakeQueue()
    with pytest.raises(WebhookResultRefNotSavedError) as error:
        hook(db, state=state, queue=queue).handle(full_success(task))
    assert error.value.status_code == 503 and state.calls == DB_ATTEMPTS == 4
    assert job(db).status is AsyncJobStatus.WAITING_PRUNA and refs(db) == (None, None)
    assert queue.sent == [] and task_costs(db) == [] and in_flight(db) == [(JOB, task)]
    # 재전송(Runware) 때 다시 저장한다. FAILED·fallback 으로 바꾸지 않았다.
    (result,) = hook(db).handle(full_success(task))
    assert result.outcome is EventOutcome.APPLIED and refs(db) == (URL, IMAGE_UUID)
    assert job(db).fallback_attempt_count == 0 and job(db).result_payload is None


def test_direct_upload_job_keeps_its_existing_final_error_after_four_attempts(db, queue_settings):  # noqa: F811
    task = submitted(db)
    state = Faulty(db["state"], "record_pruna_succeeded", ["connection"] * 10)
    with pytest.raises(FittingDatabaseError) as error:
        hook(db, state=state).handle(full_success(task))
    assert not isinstance(error.value, WebhookResultRefNotSavedError)  # 기존 응답(500) 그대로
    assert state.calls == 4 and job(db).status is AsyncJobStatus.WAITING_PRUNA


def test_router_answers_503_after_four_failed_attempts(api, monkeypatch):  # noqa: F811
    from app.virtual_fitting import webhook_router
    from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository

    task = ai_submitted(api["db"])
    original = webhook_router.open_webhook_service
    seen = {}

    from contextlib import contextmanager

    @contextmanager
    def faulty_service():
        with original() as service:
            service._state = Faulty(
                AsyncJobStateRepository(api["db"]["pool"]),
                "record_pruna_succeeded",
                ["serialization"] * 10,
            )
            service._sleep = lambda seconds: None
            seen["state"] = service._state
            yield service

    monkeypatch.setattr(webhook_router, "open_webhook_service", faulty_service)
    response = post(api, success(task, imageURL=URL, imageUUID=IMAGE_UUID))
    assert response.status_code == 503
    assert response.json()["code"] == "WEBHOOK_RESULT_REF_NOT_SAVED"
    assert seen["state"].calls == 4
    assert_no_secret(response.text)


# === 5. 실제 트랜잭션 정리·재조회 (PostgreSQL 이 낸 40001) ===


REAL_SERIALIZATION_FAILURES = """
CREATE SEQUENCE fail_seq;
CREATE FUNCTION fail_serialization() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    -- sequence 는 rollback 되지 않는다: 처음 %(n)s 번의 성공 전환 UPDATE 만 직렬화 실패로 끝낸다.
    IF NEW.status = 'POSTPROCESSING' AND OLD.status <> 'POSTPROCESSING'
       AND nextval('fail_seq') <= %(n)s THEN
        RAISE EXCEPTION 'injected serialization failure' USING ERRCODE = '40001';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER fail_serialization BEFORE UPDATE ON virtual_fitting_async_jobs
    FOR EACH ROW EXECUTE FUNCTION fail_serialization();
"""


def idle_in_transaction(db):  # noqa: F811
    return db["sql"](
        "SELECT count(*) FROM pg_stat_activity "
        "WHERE datname = current_database() AND state LIKE 'idle in transaction%%'"
    )[0][0]


@pytest.mark.parametrize("failures", [3, 4])
def test_real_db_rollbacks_are_cleaned_and_the_retry_rereads_the_row(db, failures):  # noqa: F811
    task = ai_submitted(db)
    accepted = job(db).runware_accepted_at
    db["sql"](REAL_SERIALIZATION_FAILURES.replace("%(n)s", str(failures)))
    queue = FakeQueue()
    service = hook(db, queue=queue)
    if failures < DB_ATTEMPTS:
        (result,) = service.handle(full_success(task))
        assert result.outcome is EventOutcome.APPLIED
        assert refs(db) == (URL, IMAGE_UUID) and job(db).runware_accepted_at == accepted
        assert [p["action"] for _, p in queue.sent] == ["POSTPROCESS"] and len(task_costs(db)) == 1
    else:
        with pytest.raises(WebhookResultRefNotSavedError):
            service.handle(full_success(task))
        assert refs(db) == (None, None) and job(db).status is AsyncJobStatus.WAITING_PRUNA
        assert queue.sent == [] and task_costs(db) == []
    attempts = db["sql"]("SELECT last_value FROM fail_seq")[0][0]
    assert attempts == min(failures + 1, DB_ATTEMPTS)  # 시도마다 새 트랜잭션에서 다시 실행
    assert idle_in_transaction(db) == 0  # 실패한 트랜잭션은 rollback·종료됐다
    assert job(db).task_uuid == task  # pool 연결도 계속 쓸 수 있다


# === 6. commit 응답 유실 뒤 중복 반영 방지 ===


def test_lost_commit_of_the_success_is_not_applied_twice(db):  # noqa: F811
    task = ai_submitted(db)
    accepted = job(db).runware_accepted_at
    state = Faulty(db["state"], "record_pruna_succeeded", ["commit:connection"])
    queue = FakeQueue()
    releases = Counting(limiter(db, interval=0), "release_in_flight")
    costs = Counting(TaskCostRepository(db["pool"]), "record")
    (result,) = hook(db, state=state, queue=queue, limiter_=releases, costs=costs).handle(
        full_success(task)
    )
    # 두 번째 시도는 이미 저장된 상태를 다시 확인했다(다시 적용하지 않음).
    assert state.calls == 2 and result.outcome is EventOutcome.DUPLICATE
    current = job(db)
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert refs(db) == (URL, IMAGE_UUID) and current.runware_accepted_at == accepted
    assert (releases.calls, costs.calls, len(task_costs(db))) == (1, 1, 1)
    assert [p["action"] for _, p in queue.sent] == ["POSTPROCESS"]


def test_lost_commit_of_a_late_fill_is_idempotent(db):  # noqa: F811
    task = ai_submitted(db)
    hook(db).handle(parse_webhook(success(task, imageUUID=IMAGE_UUID)))
    state = Faulty(db["state"], "fill_pruna_image_ref", ["commit:connection"])
    service = hook(db, state=state)
    service.handle(full_success(task))
    assert state.calls == 1  # 두 번째 시도에서는 이미 같은 값이라 보완할 일이 없었다
    assert refs(db) == (URL, IMAGE_UUID)


# === 7. 재시도 사이에 다른 경로가 진행해도 기존 보호 유지 ===


@pytest.mark.parametrize(
    ("change_sql", "expected_status"),
    [
        ("status = 'COMPLETED', next_action = NULL", AsyncJobStatus.COMPLETED),
        (
            "status = 'FALLBACK_RUNNING', next_action = 'RUN_FALLBACK'",
            AsyncJobStatus.FALLBACK_RUNNING,
        ),
    ],
)
def test_completion_or_fallback_between_attempts_is_respected(db, change_sql, expected_status):  # noqa: F811
    task = ai_submitted(db)

    def other_path_progressed():
        db["sql"](
            f"UPDATE virtual_fitting_async_jobs SET {change_sql} WHERE fitting_job_id = %s", (JOB,)
        )
        return "connection"

    state = Faulty(db["state"], "record_pruna_succeeded", [other_path_progressed])
    queue = FakeQueue()
    (result,) = hook(db, state=state, queue=queue).handle(full_success(task))
    assert result.outcome is EventOutcome.IGNORED and state.calls == 2
    assert job(db).status is expected_status and refs(db) == (None, None)
    assert queue.sent == []


def test_conflicting_reference_between_attempts_is_not_mixed(db):  # noqa: F811
    task = ai_submitted(db)

    def other_success_with_other_refs():
        db["sql"](
            "UPDATE virtual_fitting_async_jobs SET status = 'POSTPROCESSING', "
            "next_action = 'POSTPROCESS', runware_image_url = %s, runware_image_uuid = %s "
            "WHERE fitting_job_id = %s",
            (OTHER_URL, OTHER_UUID, JOB),
        )
        return "deadlock"

    state = Faulty(db["state"], "record_pruna_succeeded", [other_success_with_other_refs])
    (result,) = hook(db, state=state).handle(full_success(task))
    assert result.outcome is EventOutcome.DUPLICATE and refs(db) == (OTHER_URL, OTHER_UUID)


# === 8. 영구 오류·인증·입력 오류는 재시도하지 않음 ===


@pytest.mark.parametrize("kind", PERMANENT)
def test_permanent_db_errors_are_not_retried(db, kind):  # noqa: F811
    task = ai_submitted(db)
    state = Faulty(db["state"], "record_pruna_succeeded", [kind] * 10)
    service = hook(db, state=state)
    with pytest.raises(WebhookResultRefNotSavedError):  # AI 작업의 최종 응답은 같다(503)
        service.handle(full_success(task))
    assert state.calls == 1 and service.sleeps.delays == []


def test_auth_unknown_task_and_bad_input_are_not_retried(api, db):  # noqa: F811
    task = ai_submitted(db)
    jobs = Faulty(db["jobs"], "find_by_task_uuid", [])
    # 잘못된 Webhook 인증: DB 에 가지 않는다(401)
    assert post(api, success(task, imageURL=URL), token="wrong").status_code == 401
    # 미등록 task: 조회는 한 번(없음은 오류가 아님)
    service = hook(db, jobs=jobs)
    with pytest.raises(WebhookUnknownTaskError):
        service.handle(parse_webhook(success(OTHER_UUID, imageURL=URL)))
    assert jobs.calls == 1 and service.sleeps.delays == []
    # 허용하지 않는 URL·충돌은 예외가 아니라 저장하지 않는 결과다(재시도·503 아님)
    state = Faulty(db["state"], "fill_pruna_image_ref", [])
    hook(db, state=state).handle(parse_webhook(success(task, imageUUID=IMAGE_UUID)))
    hook(db, state=state).handle(
        parse_webhook(success(task, imageURL="https://evil.example/x.webp", imageUUID=IMAGE_UUID))
    )
    hook(db, state=state).handle(
        parse_webhook(success(task, imageURL=OTHER_URL, imageUUID=OTHER_UUID))
    )
    assert state.calls == 0 and refs(db) == (None, IMAGE_UUID)


# === 단위·중첩·시한 ===


def test_job_lookup_has_its_own_bounded_retry(db):  # noqa: F811
    task = ai_submitted(db)
    jobs = Faulty(db["jobs"], "find_by_task_uuid", ["pool_timeout", "connection"])
    (result,) = hook(db, jobs=jobs).handle(full_success(task))
    assert result.outcome is EventOutcome.APPLIED and jobs.calls == 3
    jobs = Faulty(db["jobs"], "find_by_task_uuid", ["connection"] * 10)
    with pytest.raises(FittingDatabaseError):
        hook(db, jobs=jobs).handle(full_success(task))
    assert jobs.calls == 4


def test_record_and_fill_share_one_budget_not_nested(db, queue_settings):  # noqa: F811
    # 기존 방식 작업: 보완이 매번 실패해도 [성공 전환 확인 + 보완] 단계 전체가 4번이다(4 x 4 가 아님).
    task = submitted(db)
    hook(db).handle(parse_webhook(success(task, imageUUID=IMAGE_UUID)))
    state = Faulty(
        Faulty(db["state"], "fill_pruna_image_ref", ["lock_timeout"] * 10),
        "record_pruna_succeeded",
        [],
    )
    queue = FakeQueue()
    (result,) = hook(db, state=state, queue=queue).handle(full_success(task))
    assert state.calls == 4 and state.real.calls == 4
    assert result.outcome is EventOutcome.DUPLICATE  # 기존 정책: fill_failed 경고 후 처리 계속
    assert refs(db) == (None, IMAGE_UUID) and [p["action"] for _, p in queue.sent] == [
        "POSTPROCESS"
    ]


def test_late_fill_exhausted_for_ai_job_is_503(db):  # noqa: F811
    task = ai_submitted(db)
    hook(db).handle(parse_webhook(success(task, imageUUID=IMAGE_UUID)))
    state = Faulty(db["state"], "fill_pruna_image_ref", ["connection"] * 10)
    with pytest.raises(WebhookResultRefNotSavedError):
        hook(db, state=state).handle(full_success(task))
    assert state.calls == 4 and refs(db) == (None, IMAGE_UUID)


def test_no_new_attempt_after_the_response_deadline(db):  # noqa: F811
    task = ai_submitted(db)
    now = [0.0]
    state = Faulty(db["state"], "record_pruna_succeeded", ["connection"] * 10)
    sleeps = Sleeps()

    def sleep(seconds):
        sleeps(seconds)
        now[0] += 1.0  # 시도·대기마다 1초 흐른다고 본다

    service = hook(db, state=state, sleep=sleep, clock=lambda: now[0], db_deadline_seconds=1.5)
    with pytest.raises(WebhookResultRefNotSavedError):
        service.handle(full_success(task))
    # 0초·1초에 시작한 시도 뒤 대기, 2초 시도 뒤에는 시한(1.5초)을 넘겨 새 시도를 시작하지 않는다.
    assert state.calls == 3 and len(sleeps.delays) == 2


# === 9·11. 부수 효과 1번, 로그 비노출 ===


def test_retries_never_repeat_queue_cost_or_slot_side_effects(db, caplog):  # noqa: F811
    task = ai_submitted(db)
    state = Faulty(
        db["state"], "record_pruna_succeeded", ["connection", "serialization", "commit:deadlock"]
    )
    queue = FakeQueue()
    releases = Counting(limiter(db, interval=0), "release_in_flight")
    costs = Counting(TaskCostRepository(db["pool"]), "record")
    with caplog.at_level(logging.DEBUG):
        hook(db, state=state, queue=queue, limiter_=releases, costs=costs).handle(
            full_success(task)
        )
    assert state.calls == 4
    assert (len(queue.sent), releases.calls, costs.calls, len(task_costs(db))) == (1, 1, 1, 1)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert text.count("runware webhook db retry step=record_result") == 3
    assert "sqlstate=40001" in text and "sqlstate=40P01" in text
    assert_no_secret(text)
    assert_no_secret(json.dumps([payload for _, payload in queue.sent]))
