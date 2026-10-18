"""13단계 보완: (1) 8단계 제출 거절 → RUN_FALLBACK 내부 큐 발행·재수신 복구, (2) 상태 변경 트랜잭션의 잠금 대기 제한.

실제 RequestProcessor·PrunaSubmitter·FollowUpProcessor·RunFallbackHandler·AsyncJobStateRepository·PrunaLimiter
(격리 PostgreSQL)를 쓴다. SQS·Runware HTTP·S3·LLM 은 가짜다. 잠금은 별도 DB 연결로 실제 행 잠금을 잡아 확인한다.
"""

import threading
import time
from contextlib import contextmanager
from decimal import Decimal

import pytest

from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.providers.runware_async import SubmissionOutcome
from app.virtual_fitting.pruna_submitter import PrunaSubmitter
from app.virtual_fitting.repositories.async_job_repository import AsyncJobStatus, NextAction
from app.virtual_fitting.repositories.async_job_state import (
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    AsyncJobStateRepository,
    lock_timeout_setting,
)
from app.virtual_fitting.repositories.pruna_limiter import TaskCostRepository
from app.virtual_fitting.request_worker import RequestProcessor
from tests.test_virtual_fitting_async_job_state import lock_waiters, pg  # noqa: F401 (fixture)
from tests.test_virtual_fitting_fallback_handler import completed
from tests.test_virtual_fitting_fallback_handler import env as fallback_env
from tests.test_virtual_fitting_fallback_handler import run as run_fallback
from tests.test_virtual_fitting_postprocess_worker import (
    Handler,
    SendingSqs,
    item,
    postprocessing,
    processor,
)
from tests.test_virtual_fitting_pruna_submit import (  # noqa: F401 (fixture)
    JOB,
    WEBHOOK,
    FakeProvider,
    FakeStorage,
    FixedBalance,
    FixedCheck,
    db,
    job,
    limiter,
)
from tests.test_virtual_fitting_request_worker import FakeBudget, valid_message

# --- 1. 8단계 제출 거절 → RUN_FALLBACK 발행 ---


def build(db, provider, queue, *, follow_up_queue=True, check=None):  # noqa: F811
    from app.virtual_fitting.providers.runware_balance import MinimumBalanceCheck

    target = queue if follow_up_queue else None
    submitter = PrunaSubmitter(
        jobs=db["jobs"],
        state=db["state"],
        limiter=limiter(db, interval=0),
        provider=provider,
        storage=FakeStorage(),
        budget=FakeBudget(),
        balance=MinimumBalanceCheck(FixedBalance("10"), Decimal("1.00")),
        webhook_url=WEBHOOK,
        keepalive_seconds=5,
        follow_up_queue=target,
    )
    return RequestProcessor(
        db["jobs"],
        db["state"],
        check or FixedCheck(),
        queue,
        lease_seconds=60,
        visibility_seconds=120,
        heartbeat_interval_seconds=0,
        submit=submitter,
        follow_up_queue=target,
    )


def retryable_reject():
    # 8단계 기존 정책: REJECTED + retryable(재시도 가능 코드) → V1 fallback 전환.
    return FakeProvider(
        SubmissionOutcome.REJECTED,
        reason="error_response",
        http_status=200,
        retryable=True,
        error_codes=("timeoutProvider",),
    )


def published(queue):
    return [payload for _, payload in queue.sent]


def test_retryable_rejection_moves_to_fallback_and_publishes_run_fallback(db):  # noqa: F811
    queue = SendingSqs()
    provider = retryable_reject()
    outcome = build(db, provider, queue).process(valid_message())
    current = job(db)
    assert (current.status, current.next_action) == (
        AsyncJobStatus.FALLBACK_RUNNING,
        NextAction.RUN_FALLBACK,
    )
    assert published(queue) == [
        {"fitting_job_id": JOB, "task_uuid": current.task_uuid, "action": "RUN_FALLBACK"}
    ]
    assert (
        current.fallback_task_uuid is None
    )  # 비동기 task_uuid 를 썼다(fallback UUID 와 혼동 없음)
    assert outcome.delete is True and len(provider.calls) == 1  # 등록이 확인돼 요청 메시지를 지운다
    # 실제 RUN_FALLBACK 핸들러가 그 메시지를 처리한다.
    e = fallback_env(db)
    run_fallback(db, e)
    assert len(e.opener.requests) == 1 and job(db).result_payload == completed()


@pytest.mark.parametrize("failure", [SubmissionOutcome.UNKNOWN, "failed"])
def test_publish_failure_keeps_the_request_and_redelivery_recovers(db, failure):  # noqa: F811
    from app.clients.sqs import SqsOutcome

    outcome_value = (
        SqsOutcome.UNKNOWN if failure is SubmissionOutcome.UNKNOWN else SqsOutcome.FAILED
    )
    queue = SendingSqs(outcome_value)
    provider = retryable_reject()
    first = build(db, provider, queue).process(valid_message())
    assert first.delete is False  # 발행이 확인되지 않아 요청 메시지를 남긴다
    assert job(db).next_action is NextAction.RUN_FALLBACK
    again = build(db, provider, queue).process(valid_message())  # 같은 요청 메시지 재수신
    assert (again.action, again.delete) == ("run_fallback_published", True)
    assert len(provider.calls) == 1  # 비동기 생성 요청을 다시 보내지 않았다
    assert [p["action"] for p in published(queue)] == ["RUN_FALLBACK", "RUN_FALLBACK"]


def test_stopped_between_transition_and_publish_is_recovered_on_redelivery(db):  # noqa: F811
    queue = SendingSqs()
    provider = retryable_reject()
    # 전환은 됐지만 발행 경로가 없는 상태(발행 직전 중단과 같음).
    first = build(db, provider, queue, follow_up_queue=False).process(valid_message())
    assert first.delete is False and queue.sent == []
    again = build(db, provider, queue).process(valid_message())
    assert (again.action, again.delete) == ("run_fallback_published", True)
    assert len(provider.calls) == 1


def test_duplicate_requests_and_messages_generate_the_fallback_once(db):  # noqa: F811
    queue = SendingSqs()
    provider = retryable_reject()
    for _ in range(3):
        build(db, provider, queue).process(valid_message())
    assert len(provider.calls) == 1 and len(published(queue)) == 3  # 중복 발행은 허용
    e = fallback_env(db)
    for _ in published(queue):
        run_fallback(db, e)
    assert len(e.opener.requests) == 1  # fallback 유료 생성 최대 1회


def test_redelivery_after_the_fallback_result_is_saved_does_not_republish(db):  # noqa: F811
    queue = SendingSqs()
    provider = retryable_reject()
    build(db, provider, queue).process(valid_message())
    run_fallback(db, fallback_env(db))
    assert job(db).next_action is NextAction.PUBLISH_RESULT
    again = build(db, provider, queue).process(valid_message())
    assert again.delete is False and len(published(queue)) == 1  # 결과 발행 경로(14단계) 없음


@pytest.mark.parametrize(
    ("provider", "expected_next"),
    [
        (
            FakeProvider(SubmissionOutcome.REJECTED, reason="balance_exhausted", http_status=402),
            NextAction.PUBLISH_RESULT,
        ),
        (
            FakeProvider(SubmissionOutcome.REJECTED, reason="client_error", http_status=401),
            NextAction.SUBMIT_PRUNA,
        ),
        (
            FakeProvider(SubmissionOutcome.REJECTED, reason="client_error", http_status=403),
            NextAction.SUBMIT_PRUNA,
        ),
        (
            FakeProvider(SubmissionOutcome.REJECTED, reason="client_error", http_status=400),
            NextAction.PUBLISH_RESULT,
        ),
        (
            FakeProvider(SubmissionOutcome.UNKNOWN, reason="timeout"),
            NextAction.SUBMIT_PRUNA,
        ),
        (
            FakeProvider(SubmissionOutcome.UNKNOWN, reason="rate_limited", http_status=429),
            NextAction.SUBMIT_PRUNA,
        ),
    ],
)
def test_non_fallback_outcomes_are_not_turned_into_fallback(db, provider, expected_next):  # noqa: F811
    queue = SendingSqs()
    build(db, provider, queue).process(valid_message())
    current = job(db)
    assert current.status is not AsyncJobStatus.FALLBACK_RUNNING
    assert current.next_action is expected_next and queue.sent == []


def test_budget_shortage_at_precheck_is_not_fallback(db):  # noqa: F811
    from app.virtual_fitting.request_worker import PrecheckOutcome, PrecheckStatus

    queue = SendingSqs()
    check = FixedCheck(
        PrecheckOutcome(PrecheckStatus.FAILED, "FITTING_DAILY_BUDGET_EXCEEDED", "budget")
    )
    provider = FakeProvider()
    build(db, provider, queue, check=check).process(valid_message())
    assert job(db).result_payload["error_code"] == "FITTING_DAILY_BUDGET_EXCEEDED"
    assert job(db).status is AsyncJobStatus.RECEIVED and queue.sent == [] and provider.calls == []


# --- 2. 잠금 대기 제한 ---


@contextmanager
def job_row_locked(db, job_id=JOB):  # noqa: F811
    connection = db["connect"]()
    try:
        connection.execute(
            "SELECT 1 FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s FOR UPDATE",
            (job_id,),
        )
        yield
    finally:
        connection.rollback()
        connection.close()


def short_state(db):  # noqa: F811
    # 테스트에서만 짧은 제한(0.3초)을 쓴다. 실제 기본값은 아래 테스트가 확인한다.
    return AsyncJobStateRepository(db["pool"], lock_timeout_seconds=0.3)


def snapshot(db):  # noqa: F811
    current = job(db)
    return (
        current.status,
        current.next_action,
        current.worker_lease_token,
        current.s3_check_attempt_count,
        current.llm_generation_attempt_count,
        current.fallback_attempt_count,
        current.fallback_started_at,
        current.result_payload,
    )


def lock_timeout_shown(db):  # noqa: F811
    with db["pool"].connection() as connection:
        return connection.execute("SHOW lock_timeout").fetchone()[0]


def test_default_lock_timeout_is_five_seconds_and_bad_values_are_rejected(db):  # noqa: F811
    assert DEFAULT_LOCK_TIMEOUT_SECONDS == 5.0
    assert lock_timeout_setting(DEFAULT_LOCK_TIMEOUT_SECONDS) == "5000ms"
    assert AsyncJobStateRepository(db["pool"])._lock_timeout == "5000ms"
    for bad in (0, -1, 61, True, "5"):
        with pytest.raises(ValueError):
            lock_timeout_setting(bad)


def test_held_row_lock_times_out_without_changing_anything(db):  # noqa: F811
    postprocessing(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_token = NULL, worker_lease_until = NULL "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    state = short_state(db)
    before = snapshot(db)
    with job_row_locked(db):
        started = time.monotonic()
        with pytest.raises(FittingDatabaseError):
            state.acquire_lease(JOB, NextAction.POSTPROCESS, 60)
        assert time.monotonic() - started < 3  # 무한 대기 없이 제한 안에서 끝났다
    assert snapshot(db) == before
    assert lock_timeout_shown(db) == "0"  # 트랜잭션 설정이 pool 연결에 남지 않는다
    grant = state.acquire_lease(
        JOB, NextAction.POSTPROCESS, 60
    )  # 잠금 해제 뒤 같은 pool 로 정상 처리
    assert grant.acquired
    with job_row_locked(db), pytest.raises(FittingDatabaseError):
        state.claim_s3_check_attempt(JOB, grant.token)
    assert job(db).s3_check_attempt_count == 0  # 횟수가 늘지 않았다
    assert state.claim_s3_check_attempt(JOB, grant.token).applied


def test_fallback_slot_and_count_are_all_or_nothing_under_a_lock_timeout(db):  # noqa: F811
    from tests.test_virtual_fitting_fallback_handler import fallback_job, in_flight

    fallback_job(db)
    token = db["state"].acquire_lease(JOB, NextAction.RUN_FALLBACK, 60).token
    assert db["state"].prepare_fallback(JOB, token, "a770f077-f413-47de-9dac-0000000000fa").applied
    shared = limiter(db, interval=0)
    before = snapshot(db)
    sends_before = db["sql"]("SELECT fitting_job_id FROM virtual_fitting_pruna_sends ORDER BY 1")
    with job_row_locked(db):
        started = time.monotonic()
        with pytest.raises(FittingDatabaseError):
            shared.begin_fallback(JOB, token, "a770f077-f413-47de-9dac-0000000000fa")
        assert time.monotonic() - started < 8  # limiter 의 5초 제한
    assert (
        snapshot(db) == before and in_flight(db) == []
    )  # 자리·횟수·시작 기록 중 일부만 남지 않았다
    sends_after = db["sql"]("SELECT fitting_job_id FROM virtual_fitting_pruna_sends ORDER BY 1")
    assert sends_after == sends_before  # 동시 전송 자리도 새로 생기지 않았다
    next_at = db["sql"]("SELECT next_send_at FROM virtual_fitting_pruna_limiter")
    assert db["sql"]("SELECT next_send_at FROM virtual_fitting_pruna_limiter") == next_at
    assert shared.begin_fallback(JOB, token, "a770f077-f413-47de-9dac-0000000000fa").granted


def test_cost_record_times_out_atomically(db):  # noqa: F811
    from datetime import date

    from app.virtual_fitting.repositories.usage_repository import UsageRepository

    day = date(2026, 10, 18)
    UsageRepository(db["pool"]).add(day, Decimal("1.00"))
    costs = TaskCostRepository(db["pool"])
    connection = db["connect"]()
    try:
        connection.execute("SELECT 1 FROM usage_amount WHERE usage_date = %s FOR UPDATE", (day,))
        with pytest.raises(FittingDatabaseError):
            costs.record("a770f077-f413-47de-9dac-0000000000c1", JOB, Decimal("0.02"), day)
    finally:
        connection.rollback()
        connection.close()
    assert (
        db["sql"]("SELECT count(*) FROM virtual_fitting_task_costs")[0][0] == 0
    )  # 비용 행도 되돌려졌다
    assert costs.record("a770f077-f413-47de-9dac-0000000000c1", JOB, Decimal("0.02"), day)
    assert UsageRepository(db["pool"]).get(day).amount == Decimal("1.02")


def test_worker_keeps_the_message_and_calls_nothing_on_a_lock_timeout(db):  # noqa: F811
    task = postprocessing(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_token = NULL, worker_lease_until = NULL "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    handler = Handler()
    from app.virtual_fitting.postprocess_worker import PostprocessHandlers

    run = processor(db, PostprocessHandlers(handler, handler))
    run._state = short_state(db)
    with job_row_locked(db):
        outcome = run.process(item(JOB, task))
    assert (outcome.action, outcome.delete) == ("error_kept", False)
    assert handler.calls == [] and job(db).worker_lease_token is None
    assert run.process(item(JOB, task)).action != "error_kept"  # 해제 뒤 다음 수신은 정상 처리


def test_fallback_generation_is_not_called_when_the_lease_step_times_out(db):  # noqa: F811
    from tests.test_virtual_fitting_fallback_handler import fallback_job

    fallback_job(db)
    e = fallback_env(db)
    e.handler._resolve()
    real = e.handler._deps
    import dataclasses

    e.handler._deps = dataclasses.replace(real, state=short_state(db))
    original = e.storage.head_object
    holder = {}

    def head_then_lock(key):
        # 사전 확인 중 다른 연결이 작업 행을 잠근다(fallback UUID 저장에서 잠금 대기).
        holder["lock"] = job_row_locked(db)
        holder["lock"].__enter__()
        return original(key)

    e.storage.head_object = head_then_lock
    try:
        outcome = run_fallback(db, e)
    finally:
        holder["lock"].__exit__(None, None, None)
    assert outcome.delete is False and e.opener.requests == []
    assert job(db).fallback_attempt_count == 0 and job(db).fallback_task_uuid is None


def test_concurrent_lease_contention_is_unchanged(db):  # noqa: F811
    postprocessing(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_token = NULL, worker_lease_until = NULL "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    results, barrier = [], threading.Barrier(6)

    def go():
        barrier.wait()
        results.append(db["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60).acquired)

    threads = [threading.Thread(target=go) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
    assert sum(results) == 1 and len(results) == 6
    assert lock_waiters(db) == 0
