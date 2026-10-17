"""10단계: AI 후처리 Worker (ai-fitting-response).

실제 AWS·Runware·OpenAI 는 부르지 않는다. DB 는 격리 PostgreSQL(TEST_DATABASE_URL, 서로 다른 연결로 선점 경쟁 확인),
SQS 는 가짜 객체(또는 실제 SqsQueueClient + 가짜 boto3 client)다. 핸들러(11·13단계)는 테스트에서만 가짜를 주입한다.
"""

import asyncio
import itertools
import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.clients.sqs import (
    FittingQueue,
    ReceivedMessage,
    SqsOperation,
    SqsOutcome,
    SqsResult,
)
from app.config import settings
from app.virtual_fitting import postprocess_worker as pw
from app.virtual_fitting.follow_up_queue import (
    FollowUpErrorKind,
    FollowUpMessage,
    InvalidFollowUp,
    InvalidFollowUpMessage,
    ValidFollowUp,
    parse_follow_up_message,
    receive_follow_ups,
)
from app.virtual_fitting.postprocess_worker import (
    FollowUpProcessor,
    HandlerResult,
    HandlerStatus,
    PostprocessHandlers,
)
from app.virtual_fitting.pruna_submitter import NextStep
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    NextAction,
)
from app.virtual_fitting.request_worker import RequestWorker
from app.virtual_fitting.runware_webhook import follow_up_message
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_pruna_submit import JOB, db, job  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_request_worker import FakeSqs, app_env  # noqa: F401 (fixture)
from tests.test_virtual_fitting_webhook import submitted

OTHER_TASK = "a770f077-f413-47de-9dac-999999999999"
RESULT_KEY = f"virtual-fitting/results/{JOB}/final.jpg"


# --- 준비 ---


def _request_worker_done(db, job_id):  # noqa: F811
    # 요청 Worker 는 전송을 마치면 자기 lease 를 반납한다(submitted 도우미는 쥔 채 끝나므로 그 반납을 흉내 낸다).
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_token = NULL, worker_lease_until = NULL "
        "WHERE fitting_job_id = %s",
        (job_id,),
    )


def postprocessing(db, job_id=JOB):  # noqa: F811
    """Webhook 성공까지 받은 POSTPROCESSING / POSTPROCESS 작업."""
    task = submitted(db, job_id)
    assert db["state"].record_pruna_succeeded(task).applied
    _request_worker_done(db, job_id)
    return task


def falling_back(db, job_id=JOB):  # noqa: F811
    task = submitted(db, job_id)
    assert db["state"].record_pruna_failed(task, recoverable=True).applied
    _request_worker_done(db, job_id)
    return task


def item(job_id, task, action=NextAction.POSTPROCESS, message_id="m-1", receive_count=1):
    return ValidFollowUp(
        message_id=message_id,
        receipt_handle=f"receipt-{message_id}",
        approximate_receive_count=receive_count,
        message=FollowUpMessage(job_id, task, action),
    )


DONE = HandlerResult(HandlerStatus.DONE)


class Handler:
    """가짜 핸들러. 받은 (작업, 토큰) 을 기록하고 result 를 돌려준다. work 로 DB 작업을 흉내 낸다."""

    def __init__(self, result=DONE, work=None, gate=None):
        self.result, self.work, self.gate = result, work, gate
        self.calls, self.entered = [], threading.Event()

    def __call__(self, job, token, context):
        self.calls.append((job.fitting_job_id, token))
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        if self.work is not None:
            return self.work(job, token, context)
        return self.result


class ReadyPaths:
    def __init__(self, *steps):
        self.steps = set(steps)

    def ready_for(self, job, step):
        return step in self.steps


def processor(db, handlers, sqs=None, **kwargs):  # noqa: F811
    kwargs.setdefault("lease_seconds", 60)
    kwargs.setdefault("visibility_seconds", 300)
    kwargs.setdefault("heartbeat_interval_seconds", 0)
    kwargs.setdefault("keepalive_seconds", 5)
    return FollowUpProcessor(db["jobs"], db["state"], sqs or FakeSqs(), handlers, **kwargs)


def save_success(db):  # noqa: F811
    def work(job, token, context):
        assert context.may_continue()
        payload = {
            "fitting_job_id": job.fitting_job_id,
            "status": "COMPLETED",
            "result_image_key": RESULT_KEY,
            "llm_title": "t",
            "llm_comment": "c",
        }
        assert db["state"].save_final_result(job.fitting_job_id, token, payload, RESULT_KEY).applied
        return HandlerResult(HandlerStatus.DONE)

    return work


def counts(db, job_id=JOB):  # noqa: F811
    current = job(db, job_id)
    return (
        current.s3_check_attempt_count,
        current.llm_generation_attempt_count,
        current.fallback_attempt_count,
        current.result_message_attempt_count,
    )


# --- 메시지 계약 ---


def test_stage9_message_is_read_by_the_stage10_model(db):  # noqa: F811
    task = postprocessing(db)
    for action in (NextAction.POSTPROCESS, NextAction.RUN_FALLBACK):
        payload = json.loads(json.dumps(follow_up_message(job(db), action)))  # 실제 큐 본문처럼
        assert parse_follow_up_message(payload) == FollowUpMessage(JOB, task, action)


@pytest.mark.parametrize(
    ("payload", "kind", "field"),
    [
        ([], "not_object", None),
        ({"task_uuid": OTHER_TASK, "action": "POSTPROCESS"}, "missing_field", "fitting_job_id"),
        ({"fitting_job_id": 1, "action": "POSTPROCESS"}, "missing_field", "task_uuid"),
        ({"fitting_job_id": 1, "task_uuid": OTHER_TASK}, "missing_field", "action"),
        (
            {
                "fitting_job_id": 1,
                "task_uuid": OTHER_TASK,
                "action": "POSTPROCESS",
                "image_url": "https://x",
            },
            "unexpected_field",
            "image_url",
        ),
        (
            {"fitting_job_id": True, "task_uuid": OTHER_TASK, "action": "POSTPROCESS"},
            "invalid_field",
            "fitting_job_id",
        ),
        (
            {"fitting_job_id": 0, "task_uuid": OTHER_TASK, "action": "POSTPROCESS"},
            "invalid_field",
            "fitting_job_id",
        ),
        (
            {"fitting_job_id": 2**63, "task_uuid": OTHER_TASK, "action": "POSTPROCESS"},
            "invalid_field",
            "fitting_job_id",
        ),
        (
            {"fitting_job_id": "1", "task_uuid": OTHER_TASK, "action": "POSTPROCESS"},
            "invalid_field",
            "fitting_job_id",
        ),
        (
            {"fitting_job_id": 1.0, "task_uuid": OTHER_TASK, "action": "POSTPROCESS"},
            "invalid_field",
            "fitting_job_id",
        ),
        (
            {"fitting_job_id": 1, "task_uuid": "nope", "action": "POSTPROCESS"},
            "invalid_field",
            "task_uuid",
        ),
        (
            {"fitting_job_id": 1, "task_uuid": None, "action": "POSTPROCESS"},
            "invalid_field",
            "task_uuid",
        ),
        (
            {"fitting_job_id": 1, "task_uuid": 7, "action": "POSTPROCESS"},
            "invalid_field",
            "task_uuid",
        ),
        (
            {"fitting_job_id": 1, "task_uuid": OTHER_TASK, "action": "PUBLISH_RESULT"},
            "invalid_field",
            "action",
        ),
        (
            {"fitting_job_id": 1, "task_uuid": OTHER_TASK, "action": "postprocess"},
            "invalid_field",
            "action",
        ),
        (
            {"fitting_job_id": 1, "task_uuid": OTHER_TASK, "action": "SUBMIT_PRUNA"},
            "invalid_field",
            "action",
        ),
    ],
)
def test_invalid_messages_are_rejected(payload, kind, field):
    with pytest.raises(InvalidFollowUpMessage) as error:
        parse_follow_up_message(payload)
    assert (error.value.kind.value, error.value.field) == (kind, field)


def test_uuid_is_normalized():
    message = parse_follow_up_message(
        {"fitting_job_id": 1, "task_uuid": OTHER_TASK.upper(), "action": "RUN_FALLBACK"}
    )
    assert message.task_uuid == OTHER_TASK and message.action is NextAction.RUN_FALLBACK


class FakeReceiveSqs:
    def __init__(self, messages=(), outcome=SqsOutcome.SUCCEEDED):
        self.messages, self.outcome, self.calls = messages, outcome, []

    def receive(self, queue, **kwargs):
        self.calls.append((queue, kwargs))
        return SqsResult(SqsOperation.RECEIVE, queue, self.outcome, messages=tuple(self.messages))


def test_receive_uses_the_internal_queue_and_classifies_messages():
    good = {"fitting_job_id": 1, "task_uuid": OTHER_TASK, "action": "POSTPROCESS"}
    sqs = FakeReceiveSqs(
        [
            ReceivedMessage("a", "SECRET-RECEIPT-A", json.dumps(good), good, None, 2),
            ReceivedMessage("b", "SECRET-RECEIPT-B", "[1]", None, "not_object", 1),
            ReceivedMessage("c", "SECRET-RECEIPT-C", "{}", {"fitting_job_id": 5}, None, 4),
        ]
    )
    result = receive_follow_ups(sqs, visibility_timeout=300)
    assert sqs.calls == [
        (
            FittingQueue.AI_FITTING_RESPONSE,
            {"max_messages": 1, "wait_time_seconds": 20, "visibility_timeout": 300},
        )
    ]
    a, b, c = result.messages
    assert isinstance(a, ValidFollowUp) and a.approximate_receive_count == 2
    assert isinstance(b, InvalidFollowUp) and b.error.kind.value == "not_object"
    assert isinstance(c, InvalidFollowUp) and c.fitting_job_id == 5
    assert all("SECRET-RECEIPT" not in repr(m) for m in result.messages)


def test_receive_failure_is_not_a_message():
    result = receive_follow_ups(FakeReceiveSqs(outcome=SqsOutcome.UNKNOWN))
    assert not result.received and result.messages == ()


# --- 처리와 삭제 판단 ---


def test_invalid_message_is_kept_without_touching_the_db(db):  # noqa: F811
    postprocessing(db)
    bad = InvalidFollowUp(
        message_id="x",
        receipt_handle="r",
        approximate_receive_count=3,
        error=InvalidFollowUpMessage(FollowUpErrorKind.MISSING_FIELD, "action"),
        fitting_job_id=JOB,
    )
    handler = Handler()
    result = processor(db, PostprocessHandlers(handler, handler)).process(bad)
    assert (result.action, result.delete) == ("invalid_kept", False)
    assert handler.calls == [] and job(db).worker_lease_token is None


def test_handler_done_without_a_db_record_is_not_deleted(db):  # noqa: F811
    task = postprocessing(db)
    handler = Handler()
    result = processor(db, PostprocessHandlers(handler)).process(item(JOB, task))
    assert (result.action, result.delete) == ("handler_done_not_recorded_kept", False)
    assert len(handler.calls) == 1
    assert job(db).worker_lease_token is None  # 자기 토큰은 반납


def test_saved_result_without_a_publish_path_is_kept(db):  # noqa: F811
    task = postprocessing(db)
    handler = Handler(work=save_success(db))
    result = processor(db, PostprocessHandlers(handler)).process(item(JOB, task))
    assert (result.action, result.delete) == (
        "handler_done",
        False,
    )  # PUBLISH_RESULT 경로(14단계) 없음
    assert job(db).next_action is NextAction.PUBLISH_RESULT


def test_saved_result_with_a_real_publish_path_is_deleted(db):  # noqa: F811
    task = postprocessing(db)
    handler = Handler(work=save_success(db))
    result = processor(
        db, PostprocessHandlers(handler), follow_up=ReadyPaths(NextStep.PUBLISH_RESULT)
    ).process(item(JOB, task))
    assert (result.action, result.delete) == ("handler_done", True)


@pytest.mark.parametrize("status", [HandlerStatus.DEFERRED, HandlerStatus.FAILED])
def test_deferred_or_failed_handlers_keep_the_message(db, status):  # noqa: F811
    task = postprocessing(db)
    handler = Handler(HandlerResult(status, "x"))
    result = processor(db, PostprocessHandlers(handler), follow_up=ReadyPaths(*NextStep)).process(
        item(JOB, task)
    )
    assert result.delete is False and result.action == f"handler_{status.value}_kept"


def test_handler_returning_nothing_is_not_completion(db):  # noqa: F811
    task = postprocessing(db)
    handler = Handler(result=None)
    result = processor(db, PostprocessHandlers(handler), follow_up=ReadyPaths(*NextStep)).process(
        item(JOB, task)
    )
    assert (result.action, result.delete) == ("handler_invalid_result_kept", False)


def test_crashing_handler_keeps_the_message(db):  # noqa: F811
    task = postprocessing(db)

    def boom(job, token, context):
        raise RuntimeError("bug")

    result = processor(db, PostprocessHandlers(boom)).process(item(JOB, task))
    assert (result.action, result.delete) == ("handler_error_kept", False)
    assert job(db).worker_lease_token is None


def test_unconnected_handler_does_not_run_or_delete(db):  # noqa: F811
    task = falling_back(db)
    result = processor(db, PostprocessHandlers(postprocess=Handler())).process(
        item(JOB, task, NextAction.RUN_FALLBACK)
    )
    assert (result.action, result.delete) == ("handler_not_connected_kept", False)
    assert job(db).worker_lease_token is None and counts(db) == (0, 0, 0, 0)


def test_run_fallback_uses_its_own_handler(db):  # noqa: F811
    task = falling_back(db)
    postprocess, fallback = Handler(), Handler()
    processor(db, PostprocessHandlers(postprocess, fallback)).process(
        item(JOB, task, NextAction.RUN_FALLBACK)
    )
    assert postprocess.calls == [] and len(fallback.calls) == 1
    assert job(db).fallback_attempt_count == 0  # Worker 는 횟수를 늘리지 않는다(13단계 책임)


def test_unknown_job_and_task_mismatch_are_kept_without_changes(db):  # noqa: F811
    task = postprocessing(db)
    before = job(db)
    handler = Handler()
    run = processor(db, PostprocessHandlers(handler, handler), follow_up=ReadyPaths(*NextStep))
    assert run.process(item(999, task)).action == "unknown_job_kept"
    mismatch = run.process(item(JOB, OTHER_TASK))
    assert (mismatch.action, mismatch.delete) == ("task_mismatch_kept", False)
    assert handler.calls == [] and job(db) == before


def test_stale_postprocess_message_does_not_disturb_a_fallback(db):  # noqa: F811
    task = falling_back(db)
    before = job(db)
    handler = Handler()
    result = processor(db, PostprocessHandlers(handler, handler)).process(item(JOB, task))
    assert (result.action, result.delete) == ("state_mismatch", False)
    assert handler.calls == [] and job(db) == before


def test_published_duplicate_is_deleted_without_running(db):  # noqa: F811
    task = postprocessing(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET status = 'COMPLETED', next_action = NULL, "
        "result_message_status = 'PUBLISHED' WHERE fitting_job_id = %s",
        (JOB,),
    )
    handler = Handler()
    result = processor(db, PostprocessHandlers(handler, handler)).process(item(JOB, task))
    assert (result.action, result.delete) == ("already_published_deleted", True)
    assert handler.calls == []


def test_another_workers_valid_lease_blocks_execution(db):  # noqa: F811
    task = postprocessing(db)
    other = db["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60)
    handler = Handler()
    result = processor(db, PostprocessHandlers(handler)).process(item(JOB, task))
    assert (result.action, result.delete) == ("lease_lease_held", False)
    assert handler.calls == [] and job(db).worker_lease_token == other.token


def test_concurrent_deliveries_run_the_handler_once(db):  # noqa: F811
    task = postprocessing(db)
    gate = threading.Event()
    handler = Handler(gate=gate)
    handlers = PostprocessHandlers(handler)
    results, errors = [], []
    barrier = threading.Barrier(5)

    def run(n):
        try:
            barrier.wait()
            results.append(processor(db, handlers).process(item(JOB, task, message_id=f"m-{n}")))
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run, args=(n,)) for n in range(5)]
    for thread in threads:
        thread.start()
    assert handler.entered.wait(10)
    deadline = time.monotonic() + 10
    while len(results) < 4:  # 나머지 4개는 선점에 실패하고 바로 끝난다
        assert time.monotonic() < deadline
        time.sleep(0.02)
    gate.set()
    for thread in threads:
        thread.join(10)
    assert not errors, errors
    assert len(handler.calls) == 1
    assert sorted(r.action for r in results).count("lease_lease_held") == 4
    assert not any(r.delete for r in results)


def test_a_new_worker_keeps_counts_and_results(db):  # noqa: F811
    task = postprocessing(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET s3_check_attempt_count = 2, "
        "llm_generation_attempt_count = 1, verified_image_key = %s WHERE fitting_job_id = %s",
        (f"virtual-fitting/results/{JOB}/pruna-async.jpg", JOB),
    )
    old = db["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second' "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    seen = []

    def work(job_, token, context):
        seen.append(
            (
                job_.s3_check_attempt_count,
                job_.llm_generation_attempt_count,
                job_.verified_image_key,
            )
        )
        # S3 횟수를 다 썼어도 확인된 이미지가 있으면 남은 LLM 시도는 막히지 않는다.
        assert db["state"].claim_llm_generation_attempt(JOB, token).applied
        return HandlerResult(HandlerStatus.DEFERRED)

    processor(db, PostprocessHandlers(Handler(work=work))).process(item(JOB, task, receive_count=9))
    assert seen == [(2, 1, f"virtual-fitting/results/{JOB}/pruna-async.jpg")]
    assert counts(db) == (2, 2, 0, 0)  # 핸들러가 늘린 1회만, Worker·수신 횟수(9)는 반영하지 않음
    # 이전 Worker 의 늦은 저장은 거절된다.
    late = db["state"].claim_llm_generation_attempt(JOB, old.token)
    assert not late.applied and late.reason.value == "lease_invalid"


# --- 연장·중단 ---


def test_lease_and_visibility_are_extended_during_a_long_handler(db):  # noqa: F811
    task = postprocessing(db)
    sqs = FakeSqs()
    leases = []

    def work(job_, token, context):
        for _ in range(3):
            time.sleep(0.08)
            leases.append(job(db).worker_lease_until)
        return HandlerResult(HandlerStatus.DEFERRED)

    processor(db, PostprocessHandlers(Handler(work=work)), sqs=sqs, keepalive_seconds=0.03).process(
        item(JOB, task)
    )
    assert len(sqs.visibility_calls) >= 3
    assert {call[0] for call in sqs.visibility_calls} == {FittingQueue.AI_FITTING_RESPONSE}
    assert all(call[2] == 300 for call in sqs.visibility_calls)
    assert leases == sorted(leases) and leases[0] < leases[-1]  # lease 도 계속 늘었다


def test_extension_failure_stops_new_side_effects_and_keeps_the_message(db):  # noqa: F811
    task = postprocessing(db)
    sqs = FakeSqs()
    started = []

    def work(job_, token, context):
        sqs.visibility_ok = False
        deadline = time.monotonic() + 5
        while context.may_continue():  # keepalive 가 실패를 표시할 때까지
            assert time.monotonic() < deadline
            time.sleep(0.02)
        started.append("no new side effect")
        return HandlerResult(HandlerStatus.DONE)

    result = processor(
        db,
        PostprocessHandlers(Handler(work=work)),
        sqs=sqs,
        keepalive_seconds=0.02,
        follow_up=ReadyPaths(*NextStep),
    ).process(item(JOB, task))
    assert (result.action, result.delete) == ("keepalive_failed_kept", False)


def test_expired_lease_blocks_late_saves(db):  # noqa: F811
    task = postprocessing(db)

    def work(job_, token, context):
        db["sql"](
            "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second' "
            "WHERE fitting_job_id = %s",
            (JOB,),
        )
        assert not context.may_continue()  # 연장(renew)이 실패한다
        late = db["state"].claim_s3_check_attempt(JOB, token)
        assert not late.applied and late.reason.value == "lease_invalid"
        return HandlerResult(HandlerStatus.DEFERRED)

    result = processor(db, PostprocessHandlers(Handler(work=work))).process(item(JOB, task))
    assert result.delete is False and counts(db) == (0, 0, 0, 0)


def test_stop_before_and_during_processing_keeps_messages(db):  # noqa: F811
    task = postprocessing(db)
    handler = Handler()
    stopped = processor(db, PostprocessHandlers(handler), should_stop=lambda: True).process(
        item(JOB, task)
    )
    assert (stopped.action, stopped.delete) == ("stopping_kept", False) and handler.calls == []

    stop = {"now": False}

    def work(job_, token, context):
        stop["now"] = True  # 처리 중 종료 요청
        return HandlerResult(
            HandlerStatus.DONE if context.may_continue() else HandlerStatus.DEFERRED, "stopping"
        )

    result = processor(
        db,
        PostprocessHandlers(Handler(work=work)),
        should_stop=lambda: stop["now"],
        follow_up=ReadyPaths(*NextStep),
    ).process(item(JOB, task))
    assert (result.action, result.delete) == ("handler_deferred_kept", False)
    assert job(db).worker_lease_token is None  # 자기 실행 권한만 반납


# --- 반복 수신 루프(name=postprocess) ---


def follow_up_batch(*messages, outcome=SqsOutcome.SUCCEEDED, reason="ok"):
    from app.virtual_fitting.follow_up_queue import FollowUpBatch

    return FollowUpBatch(
        SqsResult(SqsOperation.RECEIVE, FittingQueue.AI_FITTING_RESPONSE, outcome, reason=reason),
        tuple(messages),
    )


def test_receive_errors_back_off_and_crash_is_visible(caplog):
    calls = []

    def receive():
        calls.append(time.monotonic())
        if len(calls) == 1:
            raise RuntimeError("config")
        return follow_up_batch(outcome=SqsOutcome.UNKNOWN, reason="timeout")

    async def scenario():
        worker = RequestWorker(
            receive,
            lambda m: None,
            lambda r: None,
            backoff_initial_seconds=0.05,
            name="postprocess",
        )
        worker.start()
        await asyncio.sleep(0.4)
        await worker.stop(1)
        return worker

    with caplog.at_level(logging.WARNING):
        worker = asyncio.run(scenario())
    assert 2 <= len(calls) <= 5  # 빠른 무한 반복 없음
    assert all(b - a >= 0.04 for a, b in itertools.pairwise(calls))
    assert "postprocess worker receive crashed" in caplog.text
    assert worker.status.consecutive_errors >= 2 and worker.status.stopped

    def process(item):
        raise RuntimeError("bug")

    async def crash():
        worker = RequestWorker(
            lambda: follow_up_batch(item(1, OTHER_TASK)),
            process,
            lambda r: None,
            name="postprocess",
        )
        with pytest.raises(RuntimeError):
            await worker.start()
        return worker

    crashed = asyncio.run(crash())
    assert crashed.status.crashed and crashed.status.last_error == "RuntimeError"


def test_loop_keeps_unprocessed_messages_on_stop():
    processed, deleted = [], []
    holder = {}

    def process(message):
        processed.append(message.message_id)
        holder["worker"]._stop.set()
        return SimpleNamespace(action="x", delete=False)

    async def scenario():
        worker = RequestWorker(
            lambda: follow_up_batch(
                item(1, OTHER_TASK, message_id="a"), item(1, OTHER_TASK, message_id="b")
            ),
            process,
            lambda receipt: deleted.append(receipt),
            name="postprocess",
        )
        holder["worker"] = worker
        await asyncio.wait_for(worker.start(), 2)

    asyncio.run(scenario())
    assert processed == ["a"] and deleted == []


# --- 설정·조립 ---


def test_defaults_are_disabled_and_match_the_queue_timeout():
    assert settings.VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED is False
    assert settings.VIRTUAL_FITTING_REQUEST_WORKER_ENABLED is False
    config = pw.postprocess_settings()
    assert (config.lease_seconds, config.visibility_seconds) == (300.0, 300)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VIRTUAL_FITTING_POSTPROCESS_LEASE_SECONDS", 0),
        ("VIRTUAL_FITTING_POSTPROCESS_LEASE_SECONDS", 3601),
        ("VIRTUAL_FITTING_POSTPROCESS_VISIBILITY_SECONDS", 0),
        ("VIRTUAL_FITTING_POSTPROCESS_VISIBILITY_SECONDS", 43201),
        ("VIRTUAL_FITTING_POSTPROCESS_SHUTDOWN_TIMEOUT_SECONDS", 0),
    ],
)
def test_settings_are_range_checked(monkeypatch, name, value):
    monkeypatch.setattr(settings, name, value)
    with pytest.raises(ValueError):
        pw.postprocess_settings()


def test_real_assembly_has_no_handlers_and_refuses_to_build():
    handlers = pw.connected_postprocess_handlers()
    assert handlers.missing() == ["postprocess", "run_fallback"]
    with pytest.raises(ValueError):
        pw.create_postprocess_worker(handlers)


class FakeBoto:
    """실제 SqsQueueClient 아래의 boto3 자리. receive 는 한 번만 메시지를 준다."""

    def __init__(self, bodies):
        self.bodies, self.deleted, self.visibility, self.receives = list(bodies), [], [], []

    def receive_message(self, **params):
        self.receives.append(params)
        if not self.bodies:
            time.sleep(0.01)
            return {}
        body = self.bodies.pop(0)
        return {
            "Messages": [
                {
                    "MessageId": "real-1",
                    "ReceiptHandle": "SECRET-RECEIPT",
                    "Body": body,
                    "Attributes": {"ApproximateReceiveCount": "1"},
                }
            ]
        }

    def delete_message(self, **params):
        self.deleted.append(params["ReceiptHandle"])
        return {}

    def change_message_visibility(self, **params):
        self.visibility.append(params["VisibilityTimeout"])
        return {}


def test_real_assembly_with_injected_handlers_processes_through_real_clients(db, monkeypatch):  # noqa: F811
    postprocessing(db)
    body = json.dumps(follow_up_message(job(db), NextAction.POSTPROCESS))
    boto = FakeBoto([body])
    queue_url = "https://sqs.ap-northeast-2.amazonaws.com/123456789012/ai-fitting-response"
    monkeypatch.setattr(settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(settings, "AI_FITTING_RESPONSE_QUEUE_URL", queue_url)
    import app.clients.sqs as sqs_module
    from app.config import database

    monkeypatch.setattr(database, "get_connection_pool", lambda: db["pool"])
    monkeypatch.setattr(sqs_module, "get_sqs_client", lambda: boto)
    monkeypatch.setattr(
        sqs_module,
        "get_webhook_sqs_client",
        lambda: pytest.fail("long polling 에 Webhook client 를 쓰지 않는다"),
    )
    handler = Handler(work=save_success(db))
    worker = pw.create_postprocess_worker(PostprocessHandlers(handler, Handler()))

    async def scenario():
        worker.start()
        deadline = time.monotonic() + 5
        while not handler.calls:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)
        await worker.stop(1)

    asyncio.run(scenario())
    assert len(handler.calls) == 1 and job(db).next_action is NextAction.PUBLISH_RESULT
    assert boto.deleted == []  # 결과는 저장했지만 결과 발행 경로(14단계)가 없어 지우지 않는다
    assert boto.receives[0]["QueueUrl"] == queue_url
    assert (
        boto.receives[0]["WaitTimeSeconds"] == 20 and boto.receives[0]["VisibilityTimeout"] == 300
    )
    assert worker.status.outcomes.get("handler_done") == 1


# --- FastAPI lifespan ---


def test_disabled_postprocess_worker_makes_no_sqs(app_env, monkeypatch):  # noqa: F811
    def forbidden(*args, **kwargs):
        raise AssertionError("꺼져 있으면 Worker·SQS 를 만들지 않는다")

    monkeypatch.setattr(app_env, "create_postprocess_worker", forbidden)
    monkeypatch.setattr("app.clients.sqs.get_sqs_client", forbidden)
    with TestClient(app_env.app) as client:
        assert client.get("/health").status_code == 200
        assert client.app.state.postprocess_worker is None
        assert client.app.state.postprocess_worker_status == "disabled"


def test_enabled_without_handlers_does_not_start_receiving(app_env, monkeypatch, caplog):  # noqa: F811
    def forbidden(*args, **kwargs):
        raise AssertionError("핸들러가 없으면 수신을 시작하지 않는다")

    monkeypatch.setattr(settings, "VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED", True)
    monkeypatch.setattr(app_env, "create_postprocess_worker", forbidden)
    monkeypatch.setattr("app.clients.sqs.get_sqs_client", forbidden)
    with caplog.at_level(logging.WARNING), TestClient(app_env.app) as client:
        assert client.app.state.postprocess_worker is None
        assert client.app.state.postprocess_worker_status == "not_ready:postprocess,run_fallback"
        assert client.get("/health").status_code == 200
    assert "handlers not connected" in caplog.text


def test_enabled_with_injected_handlers_runs_and_stops_with_fastapi(app_env, monkeypatch):  # noqa: F811
    receiving, release, calls = threading.Event(), threading.Event(), []

    def receive():
        calls.append(1)
        receiving.set()
        release.wait(10)  # long polling 중
        return follow_up_batch()

    worker = RequestWorker(
        receive, lambda m: None, lambda r: None, idle_seconds=0.01, name="postprocess"
    )
    fake = PostprocessHandlers(Handler(), Handler())
    built = []
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED", True)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_POSTPROCESS_SHUTDOWN_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(app_env, "connected_postprocess_handlers", lambda: fake)
    monkeypatch.setattr(app_env, "create_postprocess_worker", lambda h: built.append(h) or worker)
    try:
        with TestClient(app_env.app) as client:
            assert receiving.wait(5)
            started = time.monotonic()
            assert client.get("/health").status_code == 200
            assert time.monotonic() - started < 1  # long polling 중에도 HTTP 요청을 처리한다
            assert client.app.state.postprocess_worker_status == "running"
            assert built == [fake] and worker.status.running
        assert worker.stopping and not worker.status.running
    finally:
        release.set()
    assert len(calls) == 1


def test_both_workers_stop_before_the_shared_pool_closes(app_env, monkeypatch):  # noqa: F811
    order = []

    class Recorder:
        def __init__(self, name):
            self.name = name
            self.stopping = False

        def start(self):
            order.append(f"start:{self.name}")

        async def stop(self, timeout):
            await asyncio.sleep(0.05)
            order.append(f"stop:{self.name}")

    fake = PostprocessHandlers(Handler(), Handler())
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_REQUEST_WORKER_ENABLED", True)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED", True)
    monkeypatch.setattr(app_env, "create_request_worker", lambda: Recorder("request"))
    monkeypatch.setattr(app_env, "connected_postprocess_handlers", lambda: fake)
    monkeypatch.setattr(app_env, "create_postprocess_worker", lambda h: Recorder("postprocess"))
    monkeypatch.setattr(app_env, "close_connection_pool", lambda: order.append("close_pool"))
    with TestClient(app_env.app):
        pass
    assert order[:2] == ["start:request", "start:postprocess"]
    assert sorted(order[2:4]) == ["stop:postprocess", "stop:request"]
    assert order[4] == "close_pool"


def test_postprocess_assembly_calls_no_models_storage_or_results(db, monkeypatch):  # noqa: F811
    import app.virtual_fitting.providers.runware as v1
    import app.virtual_fitting.providers.runware_comment as comment
    from app.clients import s3

    def no(*args, **kwargs):
        raise AssertionError("10단계는 S3·V1·LLM 을 만들지 않는다")

    monkeypatch.setattr(s3.S3ImageStorage, "__init__", no)
    monkeypatch.setattr(v1.RunwarePrunaProvider, "__init__", no)
    monkeypatch.setattr(comment.RunwareCommentProvider, "__init__", no)
    sent = []
    sqs = FakeSqs()
    sqs.send_json = lambda queue, payload: sent.append(queue)
    task = postprocessing(db)
    processor(db, PostprocessHandlers(Handler(), Handler()), sqs=sqs).process(item(JOB, task))
    assert sent == []  # 결과 큐(be-fitting-response)·후처리 큐 어느 곳에도 보내지 않는다


def test_handler_receives_the_latest_job_and_its_own_token(db):  # noqa: F811
    task = postprocessing(db)
    handler = Handler(HandlerResult(HandlerStatus.DEFERRED))
    processor(db, PostprocessHandlers(handler)).process(item(JOB, task))
    ((job_id, token),) = handler.calls
    assert job_id == JOB and token
    assert job(db).status is AsyncJobStatus.POSTPROCESSING  # 핸들러 호출만으로 상태를 바꾸지 않는다
