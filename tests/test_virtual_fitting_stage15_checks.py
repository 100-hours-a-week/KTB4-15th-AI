"""15단계 보완 확인.

1. 실제 준비 판단 조립(`follow_up_paths()`)이 작업 종류별로 판단하는지 — 실제 Router·요청 Worker·후처리 Worker 경로로.
2. 최종 결과 뒤 남은 비용·진행 중 자리를 실제 Watchdog 대상 조회부터 정리하는지(결과·상태·횟수는 그대로).

Runware 상태 조회·SQS·S3 만 가짜, DB 는 격리 PostgreSQL 이다.
"""

from datetime import UTC
from decimal import Decimal

import pytest

from app.config import settings
from app.virtual_fitting.postprocess_worker import PostprocessHandlers
from app.virtual_fitting.providers.runware_status import TaskStatus, TaskStatusKind
from app.virtual_fitting.pruna_submitter import NextStep, deletion_allowed, next_step_for
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.pruna_limiter import TaskCostRepository
from app.virtual_fitting.result_publisher import register_result_publisher
from app.virtual_fitting.runware_webhook import RunwareWebhookService
from app.virtual_fitting.watchdog import (
    StatusQueryGate,
    Watchdog,
    WatchdogRepository,
    follow_up_paths,
    register_watchdog,
)
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_pruna_submit import JOB, db, job, limiter  # noqa: F401
from tests.test_virtual_fitting_result_publisher import (
    FAILED_PAYLOAD,
    QUEUE_URL,
    FakeSqs,
    publisher,
    saved_by_fallback,
    saved_by_postprocess,
    saved_by_request_worker,
    saved_by_webhook,
)
from tests.test_virtual_fitting_watchdog import (
    AI_QUEUE,
    WATCHDOG_SQL,
    FakeQueue,
    FakeStatus,
    FakeStorage,
    Sink,
    age,
    clear_lease,
    costs,
    in_flight,
    unknown,
    waiting,
)
from tests.test_virtual_fitting_webhook import (  # noqa: F401 (pytest fixture)
    Flaky,
    api,
    error_body,
    post,
    queue_settings,
    recorded_total,
    submitted,
)

BAD_BE_QUEUE = "https://sqs.us-east-1.amazonaws.com/123456789012/be-fitting-response"  # 리전 불일치


class Alive:
    def __init__(self, alive=True):
        self.alive = alive


@pytest.fixture(autouse=True)
def queues(monkeypatch):
    monkeypatch.setattr(settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", QUEUE_URL)
    monkeypatch.setattr(settings, "AI_FITTING_RESPONSE_QUEUE_URL", AI_QUEUE)
    register_result_publisher(None)
    register_watchdog(None)
    yield
    register_result_publisher(None)
    register_watchdog(None)


@pytest.fixture
def wdb(db):  # noqa: F811
    db["sql"](WATCHDOG_SQL.read_text(encoding="utf-8"))
    return db


def run(publisher_=False, watchdog_=False):
    register_result_publisher(Alive() if publisher_ else None)
    register_watchdog(Alive() if watchdog_ else None)


def request_message_deleted(db):  # noqa: F811
    from app.virtual_fitting.request_worker import RequestProcessor
    from tests.test_virtual_fitting_pruna_submit import FixedCheck
    from tests.test_virtual_fitting_request_worker import FakeSqs as RequestSqs
    from tests.test_virtual_fitting_request_worker import valid_message

    return (
        RequestProcessor(
            db["jobs"],
            db["state"],
            FixedCheck(),
            RequestSqs(),
            lease_seconds=60,
            visibility_seconds=120,
            follow_up=follow_up_paths(),  # 요청 Worker 실제 조립(create_request_worker)과 같은 판단
        )
        .process(valid_message())
        .delete
    )


def postprocess_message_outcome(db):  # noqa: F811
    from tests.test_virtual_fitting_postprocess_worker import Handler, item
    from tests.test_virtual_fitting_postprocess_worker import processor as postprocess_processor

    handler = Handler()
    outcome = postprocess_processor(
        db, PostprocessHandlers(handler, handler), follow_up=follow_up_paths()
    ).process(item(JOB, job(db).task_uuid))
    assert handler.calls == []  # 결과가 저장된 작업은 다시 실행하지 않는다
    return outcome.action, outcome.delete


# --- 1. 작업 종류별 준비 판단 ---


def test_watchdog_alone_does_not_make_publish_result_ready(wdb):
    run(watchdog_=True)
    saved_by_request_worker(wdb)
    assert not follow_up_paths().ready_for(job(wdb), NextStep.PUBLISH_RESULT)
    assert request_message_deleted(wdb) is False  # 결과가 저장된 요청 메시지는 지우지 않는다


def test_watchdog_alone_keeps_the_saved_postprocess_message(wdb):
    saved_by_postprocess(wdb)
    run(watchdog_=True)
    assert not deletion_allowed(job(wdb), follow_up_paths())
    assert postprocess_message_outcome(wdb) == ("state_mismatch", False)
    run(publisher_=True)  # 대조: 결과 전송 루프가 있으면 기존처럼 지운다
    assert postprocess_message_outcome(wdb) == ("state_mismatch", True)


def test_watchdog_alone_keeps_the_webhook_permanent_failure_503(api):  # noqa: F811
    run(watchdog_=True)
    task = submitted(api["db"])
    response = post(api, error_body(task, code="someNewCode"))
    assert response.status_code == 503
    assert response.json()["data"]["pending"][0]["next_step"] == "publish_result"
    assert job(api["db"]).result_payload == FAILED_PAYLOAD  # 결과는 저장, 발행 경로만 없음


def test_watchdog_alone_handles_its_own_steps(wdb):
    run(watchdog_=True)
    waiting(wdb, 10)
    assert follow_up_paths().ready_for(job(wdb), NextStep.AWAIT_WEBHOOK)
    assert request_message_deleted(wdb) is True  # 대조: Watchdog 이 이어 볼 수 있는 일은 지운다


def test_publisher_alone_is_ready_only_for_publish_result(wdb):
    run(publisher_=True)
    saved_by_webhook(wdb)
    paths = follow_up_paths()
    pending = job(wdb)
    assert paths.ready_for(pending, NextStep.PUBLISH_RESULT)
    for step in NextStep:
        if step is not NextStep.PUBLISH_RESULT:
            assert not paths.ready_for(pending, step), step


def test_publisher_alone_does_not_cover_waiting_or_unknown(wdb):
    run(publisher_=True)
    waiting(wdb, 10)
    assert not follow_up_paths().ready_for(job(wdb), NextStep.AWAIT_WEBHOOK)
    assert request_message_deleted(wdb) is False
    unknown(wdb, 10, job_id=2)
    other = wdb["jobs"].find_by_fitting_job_id(2)
    assert not follow_up_paths().ready_for(other, NextStep.RECOVER_UNKNOWN)


def test_both_running_judge_each_step_by_its_own_settings(wdb, monkeypatch):
    run(publisher_=True, watchdog_=True)
    saved_by_webhook(wdb)
    unknown(wdb, 10, job_id=2)
    pending, unclear = job(wdb), wdb["jobs"].find_by_fitting_job_id(2)
    paths = follow_up_paths()
    assert paths.ready_for(pending, NextStep.PUBLISH_RESULT)
    assert paths.ready_for(unclear, NextStep.RECOVER_UNKNOWN)
    # 잘못된 결과 큐: Watchdog 이 돌아도 결과 발행 준비를 대신 통과시키지 않는다
    monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", BAD_BE_QUEUE)
    assert not paths.ready_for(pending, NextStep.PUBLISH_RESULT)
    assert paths.ready_for(unclear, NextStep.RECOVER_UNKNOWN)
    # 잘못된 내부 큐: Watchdog 일만 준비 안 됨
    monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", QUEUE_URL)
    monkeypatch.setattr(settings, "AI_FITTING_RESPONSE_QUEUE_URL", "")
    assert paths.ready_for(pending, NextStep.PUBLISH_RESULT)
    assert not paths.ready_for(unclear, NextStep.RECOVER_UNKNOWN)


def test_both_running_with_a_bad_result_queue_keep_the_webhook_503(api, monkeypatch):  # noqa: F811
    run(publisher_=True, watchdog_=True)
    monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", BAD_BE_QUEUE)
    task = submitted(api["db"])
    assert post(api, error_body(task, code="someNewCode")).status_code == 503
    monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", QUEUE_URL)
    assert post(api, error_body(task, code="someNewCode")).status_code == 200


def test_operator_review_is_never_ready_even_with_both_running(wdb):
    run(publisher_=True, watchdog_=True)
    waiting(wdb, 10)
    assert not follow_up_paths().ready_for(job(wdb), NextStep.OPERATOR_REVIEW)
    unknown(wdb, 10, job_id=2)
    other = wdb["jobs"].find_by_fitting_job_id(2)
    assert not follow_up_paths().ready_for(other, NextStep.OPERATOR_REVIEW)


def test_invalid_api_key_hold_stays_503_with_both_running(api):  # noqa: F811
    run(publisher_=True, watchdog_=True)
    task = submitted(api["db"])
    response = post(api, error_body(task, code="invalidApiKey"))
    assert response.status_code == 503
    assert response.json()["data"]["pending"][0]["next_step"] == "operator_review"
    # operator_review 는 next_action 이 아니라 DB 보류 기록(auth_hold_*)에서 계산한 논리 단계다(next_action 은 그대로 NULL).
    held = job(api["db"])
    assert (held.status, held.next_action, held.result_payload) == (
        AsyncJobStatus.WAITING_PRUNA,
        None,
        None,
    )
    assert held.auth_hold_task_uuid == task
    assert next_step_for(held) is NextStep.OPERATOR_REVIEW


def test_published_duplicate_webhook_is_still_200_without_any_loop(api):  # noqa: F811
    run(publisher_=True)
    task = submitted(api["db"])
    post(api, error_body(task, code="someNewCode"))
    clear_lease(api["db"])  # 요청 Worker 의 lease 반납(submitted 도우미는 쥔 채 끝난다)
    assert publisher(api["db"], FakeSqs()).publish(JOB).value == "published"
    assert job(api["db"]).result_message_status is ResultMessageStatus.PUBLISHED
    final = job(api["db"])
    run()  # 루프가 모두 꺼져도 발행 완료 작업의 중복 알림은 안전하게 200
    response = post(api, error_body(task, code="someNewCode"))
    assert response.status_code == 200 and job(api["db"]) == final


# --- 2. 최종 결과 뒤 누락 비용·자리 정리 ---


def snapshot(db, job_id=JOB):  # noqa: F811
    current = db["jobs"].find_by_fitting_job_id(job_id)
    return (
        current.status,
        current.result_payload,
        current.result_image_key,
        current.result_message_status,
        current.next_action,
        current.s3_check_attempt_count,
        current.llm_generation_attempt_count,
        current.fallback_attempt_count,
        current.result_message_attempt_count,
    )


def published_completed(db):  # noqa: F811
    """POSTPROCESS 로 완료·BE 발행까지 끝난 작업. 성공은 상태 메서드로만 기록돼 비용 없음·비동기 자리 남음."""
    saved_by_postprocess(db)
    assert publisher(db, FakeSqs()).publish(JOB).value == "published"
    current = job(db)
    assert current.result_message_status is ResultMessageStatus.PUBLISHED
    assert in_flight(db) == [(JOB, current.task_uuid)] and costs(db) == []
    return current.task_uuid


def watchdog(db, status, *, limiter_=None, sink=None):  # noqa: F811
    queue = FakeQueue()
    events = RunwareWebhookService(
        jobs=db["jobs"],
        state=db["state"],
        limiter=limiter_ or limiter(db),
        costs=TaskCostRepository(db["pool"]),
        queue=queue,
    )
    w = Watchdog(
        jobs=db["jobs"],
        state=db["state"],
        repo=WatchdogRepository(db["pool"]),
        status=StatusQueryGate(status, min_interval_seconds=0),
        events=events,
        sqs=queue,
        storage=FakeStorage(),
        alerts=sink or Sink(),
    )
    w.queue = queue
    return w


def due(db):  # noqa: F811
    db["sql"]("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")


def start_day(db):  # noqa: F811
    return job(db).pruna_submit_started_at.astimezone(UTC).date()


def test_lingering_slot_of_that_task_is_released_and_the_result_kept(wdb):
    task = published_completed(wdb)
    TaskCostRepository(wdb["pool"]).record(task, JOB, Decimal("0.015"), start_day(wdb))
    age(wdb, "updated_at", 200)
    wdb["sql"](
        "UPDATE virtual_fitting_pruna_in_flight SET acquired_at = now() - interval '11 minutes'"
    )
    waiting(wdb, 10, job_id=2)  # 다른 작업의 진행 중 자리
    before = snapshot(wdb)
    status = FakeStatus(TaskStatus(TaskStatusKind.SUCCEEDED, task, Decimal("0.015")))
    w = watchdog(wdb, status)
    w.run_once()
    assert status.calls == [task]  # 실제 대상 조회로 찾았다
    assert in_flight(wdb) == [(2, job(wdb, 2).task_uuid)]  # 그 task 자리만 반환
    assert snapshot(wdb) == before and w.queue.sent == []  # 결과·상태·횟수 그대로, 후속 발행 없음
    assert len(costs(wdb)) == 1


def test_fallback_slot_on_a_finished_job_is_protected(wdb):
    saved_by_fallback(wdb)
    assert publisher(wdb, FakeSqs()).publish(JOB).value == "published"
    fb = job(wdb).fallback_task_uuid
    wdb["sql"](
        "INSERT INTO virtual_fitting_pruna_in_flight VALUES (%s, %s, now() - interval '11 minutes')",
        (JOB, fb),
    )
    before = snapshot(wdb)
    status = FakeStatus(
        TaskStatus(TaskStatusKind.FAILED, job(wdb).task_uuid, error_code="timeoutProvider")
    )
    sink = Sink()
    watchdog(wdb, status, sink=sink).run_once()
    assert (
        in_flight(wdb) == [(JOB, fb)] and fb not in status.calls
    )  # 동기 fallback 은 조회하지 않음
    assert ("fallback_slot_unconfirmed", "fallback", JOB) in [
        (a.reason, a.stage, a.fitting_job_id) for a in sink.alerts
    ]
    assert snapshot(wdb) == before


def test_missing_cost_is_recorded_once_on_the_start_day(wdb):
    task = published_completed(wdb)
    limiter(wdb).release_in_flight(JOB, task_uuid=task)  # 자리는 이미 정리됨, 비용만 누락
    status = FakeStatus(TaskStatus(TaskStatusKind.SUCCEEDED, task, Decimal("0.015")))
    w = watchdog(wdb, status)
    w.run_once()
    assert status.calls == []  # 결과 직후에는 Webhook 에 시간을 준다(3분)
    age(wdb, "updated_at", 200)
    before = snapshot(wdb)
    w.run_once()
    day = start_day(wdb)
    assert costs(wdb) == [(task, Decimal("0.015"), day)]
    assert recorded_total(wdb, day) == Decimal("0.015")
    for _ in range(2):  # 중복 점검에도 추가 합산 없음(기록되면 대상에서 빠진다)
        due(wdb)
        w.run_once()
    assert status.calls == [task]
    assert recorded_total(wdb, day) == Decimal("0.015") and len(costs(wdb)) == 1
    assert snapshot(wdb) == before and w.queue.sent == []


def test_partial_cleanup_failure_is_finished_on_the_next_check(wdb):
    task = published_completed(wdb)
    age(wdb, "updated_at", 200)
    wdb["sql"](
        "UPDATE virtual_fitting_pruna_in_flight SET acquired_at = now() - interval '11 minutes'"
    )
    before = snapshot(wdb)
    flaky = Flaky(limiter(wdb), "release_in_flight")
    status = FakeStatus(*[TaskStatus(TaskStatusKind.SUCCEEDED, task, Decimal("0.015"))] * 2)
    w = watchdog(wdb, status, limiter_=flaky)
    (report,) = w.run_once()
    assert report.actions == ["database_error"]
    assert len(costs(wdb)) == 1 and in_flight(wdb) == [(JOB, task)]  # 비용만 됐고 자리 반환은 실패
    due(wdb)
    w.run_once()
    assert in_flight(wdb) == [] and len(costs(wdb)) == 1  # 빠진 단계만 이어서, 비용은 한 번
    assert recorded_total(wdb, start_day(wdb)) == Decimal("0.015")
    assert snapshot(wdb) == before


@pytest.mark.parametrize(
    "result",
    [
        TaskStatusKind.PROCESSING,
        TaskStatusKind.UNKNOWN,
        TaskStatusKind.NOT_FOUND,
        TaskStatusKind.RATE_LIMITED,
        "someQueryError",
    ],
)
def test_unclear_results_release_nothing_and_never_record_zero(wdb, result):
    task = published_completed(wdb)
    age(wdb, "updated_at", 200)
    wdb["sql"](
        "UPDATE virtual_fitting_pruna_in_flight SET acquired_at = now() - interval '11 minutes'"
    )
    before = snapshot(wdb)
    status = (
        TaskStatus(TaskStatusKind.FAILED, task, error_code=result)
        if isinstance(result, str)
        else result
    )
    w = watchdog(wdb, FakeStatus(status))
    (report,) = w.run_once()
    assert "cleanup_deferred" in report.actions
    assert in_flight(wdb) == [(JOB, task)] and costs(wdb) == []
    assert recorded_total(wdb, start_day(wdb)) in (None, Decimal(0))
    assert snapshot(wdb) == before
    # 결론이 없으면 매 주기가 아니라 정리 간격(15분) 뒤에 다시 본다
    (later,) = wdb["sql"](
        "SELECT next_check_at > now() + interval '14 minutes' FROM virtual_fitting_watchdog_checks"
    )
    assert later[0] is True


def test_success_without_a_cost_is_not_recorded_as_zero_and_stops_checking(wdb):
    task = published_completed(wdb)
    limiter(wdb).release_in_flight(JOB, task_uuid=task)
    age(wdb, "updated_at", 200)
    status = FakeStatus(TaskStatus(TaskStatusKind.SUCCEEDED, task, None))
    sink = Sink()
    w = watchdog(wdb, status, sink=sink)
    w.run_once()
    assert costs(wdb) == [] and [a.reason for a in sink.alerts] == ["cost_missing"]
    due(wdb)
    w.run_once()
    assert status.calls == [task]  # 점검 완료 기록 뒤에는 대상에서 빠진다


def test_cleanup_scope_is_bounded(wdb):
    task = published_completed(wdb)
    limiter(wdb).release_in_flight(JOB, task_uuid=task)
    age(wdb, "updated_at", 200)
    age(wdb, "pruna_submit_started_at", 25 * 3600)  # 기간(24시간) 밖의 과거 종료 작업
    status = FakeStatus()
    watchdog(wdb, status).run_once()
    assert status.calls == []


def test_fallback_task_cost_is_not_queried(wdb):
    saved_by_fallback(wdb)
    assert publisher(wdb, FakeSqs()).publish(JOB).value == "published"
    age(wdb, "updated_at", 200)
    current = job(wdb)
    status = FakeStatus(
        TaskStatus(TaskStatusKind.FAILED, current.task_uuid, error_code="timeoutProvider")
    )
    w = watchdog(wdb, status)
    (report,) = w.run_once()
    # 비동기 task(확정 실패)만 조회하고 점검을 끝낸다. 동기 fallback task 비용은 원격 조회 근거가 없어 다루지 않는다.
    assert status.calls == [current.task_uuid] and current.fallback_task_uuid not in status.calls
    assert "cleanup_settled_without_cost" in report.actions
    due(wdb)
    w.run_once()
    assert len(status.calls) == 1
