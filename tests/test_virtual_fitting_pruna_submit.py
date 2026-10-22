"""가상피팅 V2 8단계: 잔액 사전 확인, 공유 Pruna 제한, 비용 기록, Pruna 제출 연결.

실제 Runware·AWS 는 부르지 않는다. 잔액 응답·Provider·S3 서명·SQS 는 가짜를 쓰고, 작업·제한·비용 기록은 격리된
PostgreSQL(TEST_DATABASE_URL)에서 실제 SQL 로 확인한다(없으면 그 테스트는 건너뜀). 대기는 가짜 시계로 한다.
"""

import io
import json
import logging
import threading
import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from app.virtual_fitting.budget import FittingBudget
from app.virtual_fitting.exceptions import (
    FittingBalanceExhaustedError,
    FittingBalanceUnknownError,
    FittingDailyBudgetExceededError,
)
from app.virtual_fitting.providers.runware_async import AsyncSubmission, SubmissionOutcome
from app.virtual_fitting.providers.runware_balance import (
    BalanceReading,
    MinimumBalanceCheck,
    RunwareBalanceClient,
    parse_balance,
)
from app.virtual_fitting.pruna_submitter import (
    NextStep,
    PrunaSubmitter,
    deletion_allowed,
    keepalive,
    next_step_for,
)
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    NextAction,
)
from app.virtual_fitting.repositories.pruna_limiter import (
    PrunaLimiter,
    SubmissionGrant,
    SubmissionReason,
    TaskCostRepository,
)
from app.virtual_fitting.repositories.usage_repository import UsageRepository
from app.virtual_fitting.request_worker import (
    PrecheckOutcome,
    PrecheckStatus,
    RequestPrecheck,
    RequestProcessor,
)
from tests.test_virtual_fitting_async_job_state import (  # noqa: F401 (pytest fixture)
    lock_waiters,
    pg,
)
from tests.test_virtual_fitting_request_worker import (
    BOTTOM,
    IMAGE_KEY,
    TOP,
    FakeBudget,
    FakeHead,
    FakeProducts,
    FakeSqs,
    make_request,
    valid_message,
)

ROOT = Path(__file__).resolve().parents[1]
LIMITS_SQL = ROOT / "scripts" / "db" / "virtual_fitting_pruna_limits.sql"
USAGE_SPLIT_SQL = ROOT / "scripts" / "db" / "usage_amount_split_totals.sql"
USAGE_SQL = """
CREATE TABLE IF NOT EXISTS usage_amount (
    usage_date date PRIMARY KEY, amount numeric NOT NULL DEFAULT 0, reconciled_at timestamptz
)
"""
JOB = 123
WEBHOOK = "https://ai.example.com/api/v2/webhooks/runware?token=WEBHOOKTOKEN"
PASSED = PrecheckOutcome(PrecheckStatus.PASSED, products=(TOP, BOTTOM))


# --- 잔액 응답 해석 ---


def details(amount=12.5, free=3.0, currency="USD", extra=None):
    item = {
        "taskType": "accountManagement",
        "operation": "getDetails",
        "organizationName": "Secret Org",
        "balance": {"amount": amount, "freeBalance": free, "currency": currency},
        "team": [{"email": "owner@example.com"}],
        "apiKeys": [{"apiKey": "SECRETAPIKEY****"}],
        **(extra or {}),
    }
    return json.dumps({"data": [item]}).encode()


def test_balance_uses_amount_only_without_adding_free_balance():
    reading = parse_balance(details(amount=12.5, free=3.0))
    assert reading.amount == Decimal("12.5") and reading.currency == "USD"
    assert reading.checked_at.tzinfo is not None


def test_balance_is_parsed_as_exact_decimal():
    assert parse_balance(
        b'{"data":[{"balance":{"amount":1.01,"currency":"USD"}}]}'
    ).amount == Decimal("1.01")


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"{}",
        b'{"data": []}',
        b'{"data": [{"balance": {"currency": "USD"}}]}',
        b'{"data": [{"balance": {"amount": "12.5", "currency": "USD"}}]}',
        b'{"data": [{"balance": {"amount": true, "currency": "USD"}}]}',
        details(currency="EUR"),
        details(currency=None),
        b'{"errors": [{"code": "unauthorized"}]}',
    ],
)
def test_unclear_balance_is_unknown_not_zero(body):
    with pytest.raises(FittingBalanceUnknownError):
        parse_balance(body)


class FakeOpener:
    def __init__(self, body=None, error=None):
        self.body, self.error, self.requests = body, error, []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error:
            raise self.error
        body = self.body

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return body

        return Response()


def test_balance_request_uses_its_own_lookup_uuid_and_does_not_log_the_response(caplog):
    opener = FakeOpener(details())
    client = RunwareBalanceClient("rw-key", opener=opener, timeout=7)
    with caplog.at_level(logging.DEBUG):
        client.read()
        client.read()
    (first, timeout), (second, _) = opener.requests
    a, b = json.loads(first.data)[0], json.loads(second.data)[0]
    assert (a["taskType"], a["operation"]) == ("accountManagement", "getDetails")
    assert a["taskUUID"] != b["taskUUID"] and timeout == 7
    for secret in ("Secret Org", "owner@example.com", "SECRETAPIKEY", "rw-key"):
        assert secret not in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("u", 401, "x", {}, io.BytesIO(b"{}")),
        HTTPError("u", 500, "x", {}, io.BytesIO(b"{}")),
        HTTPError("u", 402, "x", {}, io.BytesIO(b"{}")),
        URLError(TimeoutError("timed out")),
        TimeoutError("timed out"),
    ],
)
def test_balance_lookup_failure_is_unknown(error):
    opener = FakeOpener(error=error)
    with pytest.raises(FittingBalanceUnknownError):
        RunwareBalanceClient("k", opener=opener).read()
    assert len(opener.requests) == 1  # 재시도 없음


class FixedBalance:
    def __init__(self, amount=None, error=None):
        self.amount, self.error, self.calls = amount, error, 0

    def read(self):
        self.calls += 1
        if self.error:
            raise self.error
        return BalanceReading(Decimal(self.amount), "USD", datetime.now(UTC))


@pytest.mark.parametrize("amount", ["0", "0.99", "1.00", "-5"])
def test_balance_at_or_below_one_dollar_blocks(amount):
    with pytest.raises(FittingBalanceExhaustedError):
        MinimumBalanceCheck(FixedBalance(amount), Decimal("1.00")).ensure_available()


def test_balance_above_one_dollar_passes():
    MinimumBalanceCheck(FixedBalance("1.01"), Decimal("1.00")).ensure_available()


@pytest.mark.parametrize("bad", [Decimal(-1), Decimal("NaN"), Decimal("Infinity"), 1.0])
def test_minimum_balance_must_be_a_finite_non_negative_decimal(bad):
    with pytest.raises(ValueError):
        MinimumBalanceCheck(FixedBalance("10"), bad)


class FakeUsage:
    def __init__(self, amount):
        from app.virtual_fitting.repositories.usage_repository import DailyUsage

        self.usage = DailyUsage(Decimal(amount), datetime.now(UTC))

    def get(self, day):
        return self.usage

    def reconcile(self, day, amount):
        raise AssertionError("최근 보정이면 Runware 를 다시 부르지 않는다")


def budget_with(amount):
    return FittingBudget(
        FakeUsage(amount), None, daily_limit=Decimal(5), reconcile_interval=timedelta(seconds=600)
    )


@pytest.mark.parametrize(
    ("spent", "balance", "status", "code"),
    [
        ("4.99", "10", PrecheckStatus.PASSED, None),
        ("5.00", "10", PrecheckStatus.FAILED, "FITTING_DAILY_BUDGET_EXCEEDED"),
        ("4.99", "1.00", PrecheckStatus.FAILED, "FITTING_BALANCE_EXHAUSTED"),
        ("4.99", "1.01", PrecheckStatus.PASSED, None),
    ],
)
def test_daily_budget_and_balance_are_separate_checks(spent, balance, status, code):
    precheck = RequestPrecheck(
        FakeProducts((TOP, BOTTOM)),
        FakeHead(),
        budget_with(spent),
        MinimumBalanceCheck(FixedBalance(balance), Decimal("1.00")),
    )
    outcome = precheck.run(make_request())
    assert (outcome.status, outcome.error_code) == (status, code)


def test_balance_lookup_failure_postpones_instead_of_failing():
    precheck = RequestPrecheck(
        FakeProducts((TOP, BOTTOM)),
        FakeHead(),
        budget_with("0"),
        MinimumBalanceCheck(FixedBalance(error=FittingBalanceUnknownError("x")), Decimal("1.00")),
    )
    outcome = precheck.run(make_request())
    assert (outcome.status, outcome.reason) == (PrecheckStatus.RETRY, "FITTING_BALANCE_UNKNOWN")


# --- keepalive ---


def test_keepalive_beats_during_a_long_call_and_reports_failure():
    beats = []

    class Beat:
        def beat(self):
            beats.append(time.monotonic())
            return len(beats) < 3

    with keepalive(Beat(), 0.02) as state:
        time.sleep(0.2)
    assert len(beats) == 3 and state.failed  # 세 번째 연장이 실패하자 멈췄다


# --- 공유 제한·비용 기록 (격리 PostgreSQL) ---


@pytest.fixture
def db(pg):  # noqa: F811
    pg["sql"](LIMITS_SQL.read_text(encoding="utf-8"))
    pg["sql"](USAGE_SQL)
    pg["sql"](USAGE_SPLIT_SQL.read_text(encoding="utf-8"))
    return pg


def ready_job(db, job_id=JOB, task=None):
    """RECEIVED / SUBMIT_PRUNA, lease 확보, task_uuid·async_image_key 준비까지 마친 작업."""
    db["jobs"].create(_new_job(job_id))
    token = db["state"].acquire_lease(job_id, NextAction.SUBMIT_PRUNA, 60).token
    task = task or f"a770f077-f413-47de-9dac-{job_id:012d}"
    assert (
        db["state"]
        .prepare_pruna_submission(
            job_id, token, task, f"virtual-fitting/results/{job_id}/pruna-async.jpg"
        )
        .applied
    )
    return token, task


def _new_job(job_id):
    from app.virtual_fitting.repositories.async_job_repository import NewAsyncFittingJob
    from app.virtual_fitting.request_worker import request_fingerprint

    codes = ("123456", "789012")
    return NewAsyncFittingJob(
        job_id,
        request_fingerprint(IMAGE_KEY, codes),
        IMAGE_KEY,
        codes,
        datetime(2026, 10, 17, 3, 0, tzinfo=UTC),
    )


def limiter(db, in_flight=3, sends=2, interval=10.0, hold=60.0):
    return PrunaLimiter(
        db["pool"],
        max_in_flight=in_flight,
        max_concurrent_sends=sends,
        min_send_interval_seconds=interval,
        send_hold_seconds=hold,
    )


def counts(db):
    return (
        db["sql"]("SELECT count(*) FROM virtual_fitting_pruna_in_flight")[0][0],
        db["sql"]("SELECT count(*) FROM virtual_fitting_pruna_sends")[0][0],
    )


def test_grant_records_the_start_and_takes_both_slots(db):
    token, task = ready_job(db)
    grant = limiter(db).begin_submission(JOB, token, task)

    assert grant.granted and grant.started_at is not None
    job = db["jobs"].find_by_fitting_job_id(JOB)
    assert job.pruna_submit_started_at == grant.started_at.astimezone(UTC)
    assert job.result_image_key is None
    assert counts(db) == (1, 1)
    again = limiter(db).begin_submission(JOB, token, task)
    assert (again.granted, again.reason) == (False, SubmissionReason.ALREADY_STARTED)


def test_min_interval_is_shared_and_counts_even_failed_sends(db):
    first_token, first_task = ready_job(db, 1)
    second_token, second_task = ready_job(db, 2)
    shared = limiter(db, interval=10)
    assert shared.begin_submission(1, first_token, first_task).granted
    shared.finish_send(1)  # 전송은 실패했다고 해도 간격은 그대로다

    wait = shared.begin_submission(2, second_token, second_task)
    assert wait.reason is SubmissionReason.INTERVAL_WAIT and 9 < wait.wait_seconds <= 10
    db["sql"]("UPDATE virtual_fitting_pruna_limiter SET next_send_at = now() - interval '1 second'")
    assert shared.begin_submission(2, second_token, second_task).granted


def test_retry_after_defers_sends_but_never_brings_them_forward(db):
    shared = limiter(db, interval=0)
    shared.defer_sends(120)
    first = db["sql"]("SELECT next_send_at - now() FROM virtual_fitting_pruna_limiter")[0][0]
    shared.defer_sends(5)
    second = db["sql"]("SELECT next_send_at - now() FROM virtual_fitting_pruna_limiter")[0][0]
    assert first > timedelta(seconds=110) and second > timedelta(seconds=100)


def test_in_flight_limit_three(db):
    shared = limiter(db, in_flight=3, sends=10, interval=0)
    jobs = [ready_job(db, i) for i in range(1, 5)]
    results = [shared.begin_submission(i, *jobs[i - 1]) for i in range(1, 5)]
    assert [r.granted for r in results] == [True, True, True, False]
    assert results[3].reason is SubmissionReason.IN_FLIGHT_FULL
    assert db["jobs"].find_by_fitting_job_id(4).pruna_submit_started_at is None

    assert shared.release_in_flight(1) is True
    assert shared.release_in_flight(1) is False  # 중복 종료 이벤트로 두 번 반환되지 않는다
    assert shared.begin_submission(4, *jobs[3]).granted


def test_concurrent_sends_limit_two_and_expired_holds(db):
    shared = limiter(db, in_flight=10, sends=2, interval=0)
    jobs = [ready_job(db, i) for i in range(1, 4)]
    assert shared.begin_submission(1, *jobs[0]).granted
    assert shared.begin_submission(2, *jobs[1]).granted
    assert shared.begin_submission(3, *jobs[2]).reason is SubmissionReason.SENDS_FULL
    shared.finish_send(1)
    assert counts(db) == (2, 1)  # 전송 자리만 놓였다. 진행 중 자리는 그대로
    assert shared.begin_submission(3, *jobs[2]).granted


def test_dead_sender_hold_expires(db):
    shared = limiter(db, in_flight=10, sends=1, interval=0, hold=60)
    first, second = ready_job(db, 1), ready_job(db, 2)
    assert shared.begin_submission(1, *first).granted
    assert shared.begin_submission(2, *second).reason is SubmissionReason.SENDS_FULL
    db["sql"]("UPDATE virtual_fitting_pruna_sends SET held_until = now() - interval '1 second'")
    assert shared.begin_submission(2, *second).granted


def test_expired_lease_or_wrong_state_is_refused(db):
    token, task = ready_job(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second'"
    )
    grant = limiter(db).begin_submission(JOB, token, task)
    assert grant.reason is SubmissionReason.LEASE_INVALID
    assert counts(db) == (0, 0)
    assert db["jobs"].find_by_fitting_job_id(JOB).pruna_submit_started_at is None


def test_cancel_unsent_only_before_acceptance(db):
    token, task = ready_job(db)
    shared = limiter(db, interval=0)
    assert shared.begin_submission(JOB, token, task).granted
    assert shared.cancel_unsent(JOB, token, task) is True
    assert counts(db) == (0, 0)
    assert db["jobs"].find_by_fitting_job_id(JOB).pruna_submit_started_at is None

    assert shared.begin_submission(JOB, token, task).granted
    assert db["state"].record_submission_accepted(task).applied
    assert shared.cancel_unsent(JOB, token, task) is False  # 접수된 뒤에는 되돌리지 않는다
    assert counts(db)[0] == 1


def limiter_row_lock(db):
    from contextlib import contextmanager

    @contextmanager
    def hold():
        connection = db["connect"]()
        try:
            connection.execute(
                "SELECT 1 FROM virtual_fitting_pruna_limiter WHERE id = 1 FOR UPDATE"
            )
            yield
        finally:
            connection.rollback()
            connection.close()

    return hold


def run_contended(db, calls, hold):
    from tests.test_virtual_fitting_async_job_state import run_contended as run

    return run(db, calls, hold)


@pytest.mark.parametrize(("sends", "in_flight", "expected"), [(2, 3, 2), (5, 3, 3)])
def test_concurrent_workers_never_exceed_the_shared_limits(db, sends, in_flight, expected):
    shared = limiter(db, in_flight=in_flight, sends=sends, interval=0)
    jobs = [ready_job(db, i) for i in range(1, 7)]
    calls = [lambda i=i: shared.begin_submission(i, *jobs[i - 1]) for i in range(1, 7)]

    results = run_contended(db, calls, limiter_row_lock(db))

    assert sum(r.granted for r in results) == expected
    started = db["sql"](
        "SELECT count(*) FROM virtual_fitting_async_jobs WHERE pruna_submit_started_at IS NOT NULL"
    )[0][0]
    assert started == expected


def test_same_job_concurrently_is_started_once(db):
    token, task = ready_job(db)
    shared = limiter(db, interval=0)
    calls = [lambda: shared.begin_submission(JOB, token, task) for _ in range(5)]
    results = run_contended(db, calls, limiter_row_lock(db))
    assert sum(r.granted for r in results) == 1
    assert {r.reason for r in results if not r.granted} == {SubmissionReason.ALREADY_STARTED}


def test_cost_is_recorded_once_per_task_and_reconcile_does_not_double_count(db):
    costs, usage = TaskCostRepository(db["pool"]), UsageRepository(db["pool"])
    day = date(2026, 10, 17)
    task = "a770f077-f413-47de-9dac-be0b26a35da6"
    assert costs.record(task, JOB, Decimal("0.0132"), day) is True
    assert costs.record(task, JOB, Decimal("0.0132"), day) is False  # 같은 task 의 비용은 한 번만
    assert usage.get(day).amount == Decimal("0.0132")
    # Runware 집계가 같은 금액을 알려 줘도 더하지 않고 큰 쪽만 남긴다.
    assert usage.reconcile(day, Decimal("0.0132")) == Decimal("0.0132")


@pytest.mark.parametrize("bad", [None, Decimal("-0.01"), Decimal("NaN"), 0.01])
def test_missing_or_invalid_cost_is_never_recorded_as_zero(db, bad):
    with pytest.raises(ValueError):
        TaskCostRepository(db["pool"]).record(
            "a770f077-f413-47de-9dac-be0b26a35da6", JOB, bad, date(2026, 10, 17)
        )
    assert db["sql"]("SELECT count(*) FROM virtual_fitting_task_costs")[0][0] == 0


# --- 제출 연결 (격리 PostgreSQL + 가짜 Provider·S3·SQS) ---


class FakeStorage:
    def __init__(self):
        self.calls = []

    def presigned_get_url(self, key, *, expires_in):
        self.calls.append(("GET", key))
        return f"https://bucket.s3.ap-northeast-2.amazonaws.com/{key}?X-Amz-Signature=GETSIG"

    def presigned_put_url(self, key, *, expires_in, content_type=None):
        self.calls.append(("PUT", key))
        return f"https://bucket.s3.ap-northeast-2.amazonaws.com/{key}?X-Amz-Signature=PUTSIG"


class FakeProvider:
    def __init__(self, outcome=SubmissionOutcome.ACCEPTED, during=None, **fields):
        self.outcome, self.during, self.fields, self.calls = outcome, during, fields, []

    def submit(self, fitting_input, *, task_uuid, webhook_url, upload_endpoint):
        self.calls.append(
            {
                "input": fitting_input,
                "task": task_uuid,
                "webhook": webhook_url,
                "upload": upload_endpoint,
            }
        )
        if self.during:
            self.during(task_uuid)
        reason = {SubmissionOutcome.ACCEPTED: "accepted"}.get(self.outcome, "x")
        fields = {"reason": reason, **self.fields}
        return AsyncSubmission(self.outcome, task_uuid, **fields)


class Ready:
    """테스트용 이어 갈 경로. 정해 둔 단계만 '있음'이라고 답한다."""

    def __init__(self, *steps):
        self.steps, self.asked = set(steps), []

    def ready_for(self, job, step):
        self.asked.append(step)
        return step in self.steps


class FixedCheck:
    def __init__(self, outcome=PASSED):
        self.outcome = outcome

    def run(self, request, before_step=lambda: True):
        return (
            self.outcome
            if before_step()
            else PrecheckOutcome(PrecheckStatus.RETRY, reason="stopped")
        )


def build(db, provider=None, follow_up=None, limiter_=None, sqs=None, **kwargs):
    storage = FakeStorage()
    budget = kwargs.pop("budget", FakeBudget())
    balance = kwargs.pop("balance", MinimumBalanceCheck(FixedBalance("10"), Decimal("1.00")))
    submitter = PrunaSubmitter(
        jobs=db["jobs"],
        state=db["state"],
        limiter=limiter_ or limiter(db, interval=0),
        provider=provider or FakeProvider(),
        storage=storage,
        budget=budget,
        balance=balance,
        webhook_url=kwargs.pop("webhook_url", WEBHOOK),
        follow_up=follow_up,
        keepalive_seconds=kwargs.pop("keepalive_seconds", 5),
        **kwargs,
    )
    processor = RequestProcessor(
        db["jobs"],
        db["state"],
        FixedCheck(),
        sqs or FakeSqs(),
        lease_seconds=60,
        visibility_seconds=120,
        heartbeat_interval_seconds=0,
        submit=submitter,
        follow_up=follow_up,
    )
    return processor, storage


def job(db, job_id=JOB):
    return db["jobs"].find_by_fitting_job_id(job_id)


def test_accepted_submission_order_and_records(db):
    seen = {}

    def during(task):
        current = job(db)
        seen.update(
            task=current.task_uuid,
            key=current.async_image_key,
            started=current.pruna_submit_started_at,
            result=current.result_image_key,
            in_flight=counts(db),
        )

    provider = FakeProvider(during=during)
    processor, storage = build(db, provider)

    result = processor.process(valid_message())

    assert result.action == "submitted" and result.delete is False  # 이어 갈 경로가 아직 없다
    (call,) = provider.calls
    # task_uuid·Key·호출 시작 기록은 전송 전에 DB 에 있었다.
    assert seen["task"] == call["task"] and seen["started"] is not None and seen["result"] is None
    assert seen["key"] == f"virtual-fitting/results/{JOB}/pruna-async.jpg"
    assert seen["in_flight"] == (1, 1)
    assert call["webhook"] == WEBHOOK and "PUTSIG" in call["upload"]
    assert "GETSIG" in call["input"].person_image_url
    assert call["input"].garment_image_urls == [TOP.image_url, BOTTOM.image_url]
    assert storage.calls == [("GET", IMAGE_KEY), ("PUT", seen["key"])]
    after = job(db)
    assert (after.status, after.next_action) == (AsyncJobStatus.WAITING_PRUNA, None)
    assert after.runware_accepted_at is not None and after.result_image_key is None
    assert counts(db) == (1, 0)  # 생성은 아직 끝나지 않았다: 진행 중 자리 유지, 전송 자리 반환
    assert after.worker_lease_token is None


def test_accepted_is_deleted_only_when_the_webhook_path_exists(db):
    follow_up = Ready(NextStep.AWAIT_WEBHOOK)
    processor, _ = build(db, follow_up=follow_up)
    assert processor.process(valid_message()).delete is True
    assert follow_up.asked == [NextStep.AWAIT_WEBHOOK]


def test_balance_exhausted_rejection_saves_failure_without_fallback(db):
    provider = FakeProvider(SubmissionOutcome.REJECTED, reason="balance_exhausted", http_status=402)
    processor, _ = build(db, provider)
    result = processor.process(valid_message())
    after = job(db)
    assert result.delete is False
    assert after.result_payload["error_code"] == "FITTING_BALANCE_EXHAUSTED"
    assert after.next_action is NextAction.PUBLISH_RESULT  # RUN_FALLBACK 아님
    assert counts(db) == (0, 0)


def test_retryable_rejection_moves_to_fallback_and_frees_the_slot(db):
    provider = FakeProvider(
        SubmissionOutcome.REJECTED, reason="error_response", retryable=True, http_status=200
    )
    processor, _ = build(db, provider, follow_up=Ready(NextStep.RUN_FALLBACK))
    assert processor.process(valid_message()).delete is True
    after = job(db)
    assert (after.status, after.next_action) == (
        AsyncJobStatus.FALLBACK_RUNNING,
        NextAction.RUN_FALLBACK,
    )
    assert counts(db) == (0, 0)


def test_plain_rejection_is_a_model_failure(db):
    provider = FakeProvider(SubmissionOutcome.REJECTED, reason="client_error", http_status=400)
    processor, _ = build(db, provider)
    processor.process(valid_message())
    assert job(db).result_payload["error_code"] == "FITTING_MODEL_FAILED"


def test_auth_rejection_is_not_a_user_failure_and_is_not_resent(db):
    provider = FakeProvider(SubmissionOutcome.REJECTED, reason="client_error", http_status=401)
    processor, _ = build(db, provider)
    processor.process(valid_message())
    after = job(db)
    assert after.result_payload is None and after.pruna_submit_started_at is not None
    assert counts(db) == (0, 0)
    processor.process(valid_message(message_id="m-2"))
    assert len(provider.calls) == 1


def test_unknown_keeps_everything_and_redelivery_never_resends(db):
    provider = FakeProvider(SubmissionOutcome.UNKNOWN, reason="timeout")
    processor, _ = build(db, provider)
    first = processor.process(valid_message())
    after = job(db)
    assert first.delete is False
    assert after.status is AsyncJobStatus.RECEIVED and after.pruna_submit_started_at is not None
    assert counts(db) == (1, 0)  # UNKNOWN 이면 진행 중 자리를 놓지 않는다

    # Worker 가 바뀌고(lease 만료) 메시지가 다시 와도 다시 보내지 않는다.
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second'"
    )
    other, _ = build(db, provider, follow_up=Ready(NextStep.RECOVER_UNKNOWN))
    again = other.process(valid_message(message_id="m-2", receive_count=2))
    assert again.action == "submit_already_started"
    assert again.delete is True  # 복구 경로가 있다고 주입했을 때만
    assert len(provider.calls) == 1 and counts(db) == (1, 0)


def test_rate_limited_unknown_defers_the_next_send(db):
    provider = FakeProvider(
        SubmissionOutcome.UNKNOWN, reason="rate_limited", http_status=429, retry_after=90
    )
    processor, _ = build(db, provider)
    processor.process(valid_message())
    left = db["sql"]("SELECT next_send_at - now() FROM virtual_fitting_pruna_limiter")[0][0]
    assert left > timedelta(seconds=80)
    assert counts(db) == (1, 0)


def test_accepted_but_db_record_failed_keeps_slot_and_message(db, monkeypatch):
    from app.virtual_fitting.repositories.async_job_state import ChangeReason, StateChange

    monkeypatch.setattr(
        db["state"],
        "record_submission_accepted",
        lambda task: StateChange(False, ChangeReason.STATE_MISMATCH, None),
    )
    processor, _ = build(db, follow_up=Ready(*NextStep))
    result = processor.process(valid_message())
    assert (
        result.delete is False
    )  # RECOVER_UNKNOWN 경로가 있다고 해도 기록 실패 직후에는 지우지 않는다
    assert counts(db) == (1, 0) and job(db).runware_accepted_at is None


def test_webhook_before_the_ack_is_not_rolled_back(db):
    provider = FakeProvider(during=lambda task: db["state"].record_pruna_succeeded(task))
    processor, _ = build(db, provider)
    processor.process(valid_message())
    after = job(db)
    assert (after.status, after.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )


def test_two_workers_on_the_same_job_send_once(db):
    gate = threading.Event()
    provider = FakeProvider(during=lambda task: gate.wait(5))
    results = {}
    db["jobs"].create(_new_job(JOB))

    def work(name):
        processor, _ = build(db, provider)
        results[name] = processor.process(valid_message(message_id=name))

    first = threading.Thread(target=work, args=("a",))
    first.start()
    deadline = time.monotonic() + 10
    while not provider.calls:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    second = threading.Thread(target=work, args=("b",))
    second.start()
    second.join(10)
    gate.set()
    first.join(10)
    assert len(provider.calls) == 1
    assert results["b"].action.startswith("lease_")


def test_crash_after_invoking_keeps_the_start_record(db):
    class Crashing(FakeProvider):
        def submit(self, *args, **kwargs):
            super().submit(*args, **kwargs)
            raise RuntimeError("connection reset after send")

    provider = Crashing()
    processor, _ = build(db, provider)
    assert processor.process(valid_message()).delete is False
    assert job(db).pruna_submit_started_at is not None and counts(db) == (1, 0)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second'"
    )
    processor.process(valid_message(message_id="m-2"))
    assert len(provider.calls) == 1


def test_input_error_before_sending_is_undone(db):
    class Invalid(FakeProvider):
        def submit(self, *args, **kwargs):
            raise ValueError("bad input, nothing sent")

    processor, _ = build(db, Invalid())
    processor.process(valid_message())
    assert job(db).pruna_submit_started_at is None and counts(db) == (0, 0)


def test_missing_webhook_url_starts_nothing(db):
    provider = FakeProvider()
    processor, _ = build(db, provider, webhook_url="")
    processor.process(valid_message())
    assert provider.calls == [] and job(db).pruna_submit_started_at is None


class FakeClock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedLimiter:
    """앞의 몇 번은 기다리라고 답하고, 그 뒤 실제 limiter 로 넘긴다."""

    def __init__(self, real, waits):
        self.real, self.waits, self.calls = real, list(waits), 0

    def begin_submission(self, *args):
        self.calls += 1
        if self.waits:
            return SubmissionGrant(False, self.waits.pop(0), wait_seconds=7.5)
        return self.real.begin_submission(*args)

    def __getattr__(self, name):
        return getattr(self.real, name)


def test_waiting_for_a_slot_uses_the_clock_and_extends_each_round(db):
    clock, sqs = FakeClock(), FakeSqs()
    scripted = ScriptedLimiter(
        limiter(db, interval=0), [SubmissionReason.INTERVAL_WAIT, SubmissionReason.SENDS_FULL]
    )
    provider = FakeProvider()
    processor, _ = build(
        db, provider, limiter_=scripted, sqs=sqs, clock=clock, sleep=clock.sleep, poll_seconds=2
    )
    processor.process(valid_message())
    assert clock.sleeps == [2, 2] and scripted.calls == 3
    assert len(provider.calls) == 1
    assert len(sqs.visibility_calls) >= 3  # 대기하는 동안 단계마다 연장


def test_waiting_too_long_postpones_without_a_start_record(db):
    clock = FakeClock()
    scripted = ScriptedLimiter(limiter(db, interval=0), [SubmissionReason.IN_FLIGHT_FULL] * 100)
    provider = FakeProvider()
    processor, _ = build(
        db,
        provider,
        limiter_=scripted,
        clock=clock,
        sleep=clock.sleep,
        max_wait_seconds=10,
        poll_seconds=2,
    )
    result = processor.process(valid_message())
    assert result.delete is False and provider.calls == []
    assert job(db).pruna_submit_started_at is None and sum(clock.sleeps) <= 10


def test_stale_checks_are_redone_before_sending(db):
    clock = FakeClock()
    scripted = ScriptedLimiter(limiter(db, interval=0), [SubmissionReason.INTERVAL_WAIT] * 20)
    balance = MinimumBalanceCheck(FixedBalance("0.50"), Decimal("1.00"))
    provider = FakeProvider()
    processor, _ = build(
        db,
        provider,
        limiter_=scripted,
        clock=clock,
        sleep=clock.sleep,
        balance=balance,
        recheck_after_seconds=5,
        poll_seconds=2,
    )
    processor.process(valid_message())
    after = job(db)
    assert provider.calls == []
    assert after.result_payload["error_code"] == "FITTING_BALANCE_EXHAUSTED"
    assert after.next_action is NextAction.PUBLISH_RESULT and after.pruna_submit_started_at is None


def test_balance_lookup_failure_during_wait_postpones(db):
    clock = FakeClock()
    scripted = ScriptedLimiter(limiter(db, interval=0), [SubmissionReason.INTERVAL_WAIT] * 20)
    processor, _ = build(
        db,
        FakeProvider(),
        limiter_=scripted,
        clock=clock,
        sleep=clock.sleep,
        balance=MinimumBalanceCheck(
            FixedBalance(error=FittingBalanceUnknownError("x")), Decimal(1)
        ),
        recheck_after_seconds=3,
    )
    processor.process(valid_message())
    after = job(db)
    assert after.result_payload is None and after.pruna_submit_started_at is None


def test_daily_budget_exceeded_while_waiting_is_a_failure_without_fallback(db):
    clock = FakeClock()
    scripted = ScriptedLimiter(limiter(db, interval=0), [SubmissionReason.INTERVAL_WAIT] * 20)
    processor, _ = build(
        db,
        FakeProvider(),
        limiter_=scripted,
        clock=clock,
        sleep=clock.sleep,
        budget=FakeBudget(FittingDailyBudgetExceededError("over")),
        recheck_after_seconds=3,
    )
    processor.process(valid_message())
    after = job(db)
    assert after.result_payload["error_code"] == "FITTING_DAILY_BUDGET_EXCEEDED"
    assert after.next_action is NextAction.PUBLISH_RESULT


def test_extension_failure_stops_before_a_new_send(db):
    provider = FakeProvider()
    processor, _ = build(db, provider, sqs=FakeSqs(visibility_ok=False))
    processor.process(valid_message())
    assert provider.calls == [] and job(db).pruna_submit_started_at is None


def test_keepalive_runs_during_the_send_and_failure_blocks_deletion(db):
    beats = threading.Event()

    class Slow(FakeProvider):
        def submit(self, *args, **kwargs):
            beats.wait(2)
            return super().submit(*args, **kwargs)

    class CountingSqs(FakeSqs):
        def change_visibility(self, queue, receipt_handle, seconds):
            # 전송 전 연장은 3번(사전 검증 단계, submit 직전, 자리 대기). 그다음 keepalive 연장부터 실패한다.
            if len(self.visibility_calls) >= 3:
                self.visibility_ok = False
            result = super().change_visibility(queue, receipt_handle, seconds)
            if not result.succeeded:
                beats.set()
            return result

    sqs = CountingSqs()
    processor, _ = build(db, Slow(), sqs=sqs, follow_up=Ready(*NextStep), keepalive_seconds=0.02)
    result = processor.process(valid_message())
    assert job(db).status is AsyncJobStatus.WAITING_PRUNA  # 결과는 기록한다
    assert result.delete is False  # 연장이 실패한 receipt 로는 지우지 않는다


def test_stop_request_prevents_a_new_send(db):
    provider = FakeProvider()
    processor, _ = build(db, provider, should_stop=lambda: True)
    processor.process(valid_message())
    assert provider.calls == [] and job(db).pruna_submit_started_at is None


def test_presigned_urls_and_webhook_token_never_reach_logs(db, caplog):
    processor, _ = build(db, FakeProvider())
    with caplog.at_level(logging.DEBUG):
        processor.process(valid_message())
    for secret in ("GETSIG", "PUTSIG", "WEBHOOKTOKEN", "X-Amz-Signature"):
        assert secret not in caplog.text


# --- 삭제 판단 ---


def test_next_step_and_deletion_need_a_record_and_a_real_path(db):
    token, task = ready_job(db)
    current = job(db)
    assert next_step_for(current) is None  # RECEIVED 준비만으로는 다음 일이 없다
    assert deletion_allowed(current, Ready(*NextStep)) is False

    assert limiter(db, interval=0).begin_submission(JOB, token, task).granted
    started = job(db)
    assert next_step_for(started) is NextStep.RECOVER_UNKNOWN
    assert deletion_allowed(started, Ready()) is False
    assert deletion_allowed(started, Ready(NextStep.RECOVER_UNKNOWN)) is True


def test_default_follow_up_never_allows_deletion_of_unfinished_work(db):
    processor, _ = build(db)  # follow_up 기본값
    assert processor.process(valid_message()).delete is False
    assert processor.process(valid_message(message_id="m-2")).delete is False  # WAITING_PRUNA 중복


def test_published_duplicate_is_still_deleted(db):
    db["jobs"].create(_new_job(JOB))
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET status = 'COMPLETED', result_message_status = 'PUBLISHED', "
        "next_action = NULL"
    )
    processor, _ = build(db)
    assert processor.process(valid_message()).delete is True
