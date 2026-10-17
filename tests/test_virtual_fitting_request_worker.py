"""가상피팅 V2 AI 요청 Worker(7단계).

실제 SQS·S3·Runware·OpenAI 는 부르지 않는다. 작업 등록·실행 권한은 격리된 PostgreSQL(TEST_DATABASE_URL)에서
실제 Repository 로 확인하고, 없으면 그 테스트는 건너뛴다. 큐·사전 검증은 가짜 의존성을 주입한다.
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.clients.s3 import S3ObjectState
from app.clients.sqs import FittingQueue, SqsOperation, SqsOutcome, SqsResult
from app.config import settings
from app.virtual_fitting import request_worker as worker_module
from app.virtual_fitting.async_request import (
    FittingRequest,
    InvalidFittingRequest,
    InvalidRequestMessage,
    RequestBatch,
    RequestErrorKind,
    ValidRequestMessage,
)
from app.virtual_fitting.exceptions import (
    FittingBalanceExhaustedError,
    FittingDailyBudgetExceededError,
    FittingDatabaseError,
)
from app.virtual_fitting.models import FittingProduct
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.request_worker import (
    PrecheckOutcome,
    PrecheckStatus,
    RequestPrecheck,
    RequestProcessor,
    RequestWorker,
    failure_payload,
    request_fingerprint,
)
from tests.test_virtual_fitting_async_job_state import (  # noqa: F401 (pytest fixture)
    lock_waiters,
    pg,
)

JOB = 123
IMAGE_KEY = "users/10/body.png"
REQUESTED_AT = datetime(2026, 10, 17, 3, 0, tzinfo=UTC)
TOP = FittingProduct("123456", "https://img/top.jpg", "상의", "반소매 티셔츠", "티셔츠")
BOTTOM = FittingProduct("789012", "https://img/bottom.jpg", "하의", "슬림 팬츠", "팬츠")


def make_request(
    job_id=JOB, codes=("123456", "789012"), image_key=IMAGE_KEY, requested_at=REQUESTED_AT
):
    return FittingRequest(job_id, requested_at, image_key, tuple(codes))


def valid_message(request=None, message_id="m-1", receive_count=1):
    return ValidRequestMessage(
        message_id=message_id,
        receipt_handle=f"receipt-{message_id}",
        approximate_receive_count=receive_count,
        request=request or make_request(),
    )


def ok(operation=SqsOperation.CHANGE_VISIBILITY):
    return SqsResult(operation, FittingQueue.BE_FITTING_REQUEST, SqsOutcome.SUCCEEDED)


class FakeSqs:
    def __init__(self, visibility_ok=True):
        self.visibility_ok = visibility_ok
        self.visibility_calls = []
        self.deletes = []

    def change_visibility(self, queue, receipt_handle, seconds):
        self.visibility_calls.append((queue, receipt_handle, seconds))
        if self.visibility_ok:
            return ok()
        return SqsResult(
            SqsOperation.CHANGE_VISIBILITY,
            queue,
            SqsOutcome.FAILED,
            reason="receipt_handle_invalid",
        )

    def delete(self, queue, receipt_handle):
        self.deletes.append(receipt_handle)
        return ok(SqsOperation.DELETE)


class FakePrecheck:
    def __init__(self, outcome=None, steps=1, gate=None):
        self.outcome = outcome or PrecheckOutcome(PrecheckStatus.PASSED, products=(TOP, BOTTOM))
        self.steps = steps
        self.gate = gate
        self.calls = []

    def run(self, request, before_step=lambda: True):
        self.calls.append(request.fitting_job_id)
        if self.gate is not None:
            self.gate.wait(10)
        for _ in range(self.steps):
            if not before_step():
                return PrecheckOutcome(PrecheckStatus.RETRY, reason="stopped")
        return self.outcome


def processor(pg, precheck=None, sqs=None, **kwargs):  # noqa: F811
    options = {"lease_seconds": 60, "visibility_seconds": 120, "heartbeat_interval_seconds": 0}
    options.update(kwargs)
    return RequestProcessor(
        pg["jobs"], pg["state"], precheck or FakePrecheck(), sqs or FakeSqs(), **options
    )


def row(pg, job_id=JOB):  # noqa: F811
    return pg["jobs"].find_by_fitting_job_id(job_id)


# --- fingerprint ---


def test_fingerprint_rule_is_explicit_sha256_over_canonical_json():
    canonical = json.dumps(
        {"product_codes": ["123456", "789012"], "user_image_key": IMAGE_KEY},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    expected = "v1:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert request_fingerprint(IMAGE_KEY, ["789012", "123456"]) == expected
    # 실행할 때마다 달라지지 않는 고정값(파이썬 hash() 를 쓰지 않는다)
    assert request_fingerprint(IMAGE_KEY, ["123456"]) == request_fingerprint(IMAGE_KEY, ("123456",))


def test_fingerprint_ignores_product_order_but_keeps_zeros_duplicates_and_key():
    base = request_fingerprint(IMAGE_KEY, ["123456", "789012"])
    assert request_fingerprint(IMAGE_KEY, ["789012", "123456"]) == base
    assert request_fingerprint(IMAGE_KEY, ["0123456", "789012"]) != base  # 앞자리 0 보존
    assert request_fingerprint(IMAGE_KEY, ["123456", "123456"]) != request_fingerprint(
        IMAGE_KEY, ["123456"]
    )  # 중복 개수 반영
    assert request_fingerprint("users/11/body.png", ["123456", "789012"]) != base


def test_fingerprint_excludes_requested_at_and_delivery_info():
    a = make_request(requested_at=REQUESTED_AT)
    b = make_request(requested_at=datetime(2030, 1, 1, tzinfo=UTC))
    assert request_fingerprint(a.user_image_key, a.product_codes) == request_fingerprint(
        b.user_image_key, b.product_codes
    )


# --- 사전 검증 (V1 검증 재사용) ---


class FakeProducts:
    def __init__(self, products):
        self.products = {int(p.product_code): p for p in products}
        self.calls = 0

    def find_by_codes(self, codes):
        self.calls += 1
        return [self.products[int(c)] for c in set(codes) if int(c) in self.products]


class FakeHead:
    def __init__(self, state=S3ObjectState.EXISTS):
        self.state = state
        self.keys = []

    def head_object(self, key):
        self.keys.append(key)
        return SimpleNamespace(state=self.state)


class FakeBudget:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def ensure_available(self):
        self.calls += 1
        if self.error:
            raise self.error

    def record(self, cost):
        raise AssertionError("사전 검증은 사용액을 기록하지 않는다")


def precheck(products=(TOP, BOTTOM), head=None, budget=None, balance=None):
    return RequestPrecheck(
        FakeProducts(products), head or FakeHead(), budget or FakeBudget(), balance
    )


def test_precheck_passes_and_orders_top_then_bottom():
    head, budget = FakeHead(), FakeBudget()
    outcome = precheck(head=head, budget=budget).run(make_request(codes=("789012", "123456")))
    assert outcome.status is PrecheckStatus.PASSED
    assert [p.product_code for p in outcome.products] == ["123456", "789012"]
    assert head.keys == [IMAGE_KEY] and budget.calls == 1


@pytest.mark.parametrize(
    ("codes", "products", "code"),
    [
        (("999999",), (TOP,), "PRODUCT_NOT_FOUND"),
        (("abc123",), (TOP,), "PRODUCT_NOT_FOUND"),
        (("123456", "123456"), (TOP,), "INVALID_FITTING_COMBINATION"),  # 같은 상품 코드 2개
        (
            ("123456", "555555"),
            (TOP, FittingProduct("555555", "https://img/t2.jpg", "상의", "후디", "")),
            "INVALID_FITTING_COMBINATION",
        ),
        (
            ("777777",),
            (FittingProduct("777777", "https://x", "신발", "스니커즈", ""),),
            "INVALID_FITTING_COMBINATION",
        ),
        (
            ("123456",),
            (FittingProduct("123456", "", "상의", "반소매 티셔츠", ""),),
            "PRODUCT_IMAGE_MISSING",
        ),
        (
            ("123456",),
            (FittingProduct("123456", "https://x", "상의", "없는 분류", ""),),
            "UNSUPPORTED_SUB_CATEGORY",
        ),
    ],
)
def test_business_failures_are_final_without_touching_image_or_budget(codes, products, code):
    head, budget = FakeHead(), FakeBudget()
    outcome = precheck(products, head, budget).run(make_request(codes=codes))
    assert (outcome.status, outcome.error_code) == (PrecheckStatus.FAILED, code)
    assert outcome.error_message
    assert head.keys == [] and budget.calls == 0


def test_missing_user_image_is_a_final_failure():
    outcome = precheck(head=FakeHead(S3ObjectState.NOT_FOUND)).run(make_request())
    assert (outcome.status, outcome.error_code) == (PrecheckStatus.FAILED, "INVALID_REQUEST")


@pytest.mark.parametrize(
    "state", [S3ObjectState.ACCESS_DENIED, S3ObjectState.TRANSIENT_ERROR, S3ObjectState.ERROR]
)
def test_s3_access_or_transient_problems_are_not_final(state):
    budget = FakeBudget()
    outcome = precheck(head=FakeHead(state), budget=budget).run(make_request())
    assert outcome.status is PrecheckStatus.RETRY and budget.calls == 0


def test_budget_and_balance_results():
    exceeded = precheck(budget=FakeBudget(FittingDailyBudgetExceededError("over"))).run(
        make_request()
    )
    assert (exceeded.status, exceeded.error_code) == (
        PrecheckStatus.FAILED,
        "FITTING_DAILY_BUDGET_EXCEEDED",
    )
    db_down = precheck(budget=FakeBudget(FittingDatabaseError("db"))).run(make_request())
    assert (db_down.status, db_down.reason) == (PrecheckStatus.RETRY, "DATABASE_ERROR")

    class EmptyBalance:
        def ensure_available(self):
            raise FittingBalanceExhaustedError("no balance")

    no_balance = precheck(balance=EmptyBalance()).run(make_request())
    assert (no_balance.status, no_balance.error_code) == (
        PrecheckStatus.FAILED,
        "FITTING_BALANCE_EXHAUSTED",
    )


def test_unexpected_or_db_errors_are_not_final():
    class Broken:
        def find_by_codes(self, codes):
            raise RuntimeError("bug")

    outcome = RequestPrecheck(Broken(), FakeHead(), FakeBudget()).run(make_request())
    assert outcome.status is PrecheckStatus.RETRY


def test_precheck_stops_before_the_next_external_step():
    head, budget = FakeHead(), FakeBudget()
    answers = iter([True, False])
    outcome = precheck(head=head, budget=budget).run(
        make_request(), before_step=lambda: next(answers)
    )
    assert outcome.status is PrecheckStatus.RETRY
    assert head.keys == [] and budget.calls == 0


# --- 메시지 처리 (격리 PostgreSQL) ---


def test_first_request_is_registered_and_not_deleted(pg):  # noqa: F811
    sqs, check = FakeSqs(), FakePrecheck()
    request = make_request(codes=("789012", "001234"))

    result = processor(pg, check, sqs).process(valid_message(request))

    assert (result.action, result.delete) == ("prechecked_awaiting_submit", False)
    job = row(pg)
    assert job.status is AsyncJobStatus.RECEIVED and job.next_action is NextAction.SUBMIT_PRUNA
    assert job.requested_at == REQUESTED_AT
    assert job.products == ("789012", "001234")  # 요청 순서·앞자리 0 그대로
    assert job.request_fingerprint == request_fingerprint(IMAGE_KEY, ["789012", "001234"])
    assert (job.worker_lease_token, job.worker_lease_until) == (None, None)  # 자기 lease 반납
    assert job.task_uuid is None and job.pruna_submit_started_at is None  # Pruna 준비는 8단계
    assert sqs.deletes == [] and check.calls == [JOB]


def test_same_input_again_keeps_the_job_and_times(pg):  # noqa: F811
    processor(pg).process(valid_message())
    pg["sql"](
        "UPDATE virtual_fitting_async_jobs SET s3_check_attempt_count = 1, "
        "llm_generation_attempt_count = 2 WHERE fitting_job_id = %s",
        (JOB,),
    )
    before = row(pg)

    later = make_request(
        codes=("789012", "123456"), requested_at=datetime(2026, 10, 18, tzinfo=UTC)
    )
    result = processor(pg).process(valid_message(later, "m-2", receive_count=3))

    after = row(pg)
    assert result.delete is False
    assert after.requested_at == before.requested_at and after.created_at == before.created_at
    assert (after.s3_check_attempt_count, after.llm_generation_attempt_count) == (1, 2)
    assert after.status is AsyncJobStatus.RECEIVED and after.products == before.products


def test_different_input_with_the_same_job_id_is_a_kept_conflict(pg):  # noqa: F811
    processor(pg).process(valid_message())
    before = row(pg)
    alerts, check = [], FakePrecheck()

    result = processor(
        pg, check, on_conflict=lambda job, req, mid: alerts.append((job.fitting_job_id, mid))
    ).process(valid_message(make_request(image_key="users/99/other.png"), "m-conflict"))

    assert (result.action, result.delete) == ("input_conflict_kept", False)
    assert alerts == [(JOB, "m-conflict")]
    assert row(pg) == before  # 덮어쓰기·실패 전환 없음
    assert check.calls == []


def test_other_worker_holding_the_lease_blocks_processing(pg):  # noqa: F811
    processor(pg).process(valid_message())
    other = pg["state"].acquire_lease(JOB, NextAction.SUBMIT_PRUNA, 60)
    assert other.acquired
    check = FakePrecheck()

    result = processor(pg, check).process(valid_message(message_id="m-2"))

    assert (result.action, result.delete) == ("lease_lease_held", False)
    assert check.calls == []
    assert row(pg).worker_lease_token == other.token  # 남의 lease 는 건드리지 않는다


def test_business_failure_saves_a_v2_result_without_deleting(pg):  # noqa: F811
    failed = PrecheckOutcome(
        PrecheckStatus.FAILED, "INVALID_FITTING_COMBINATION", "가상 피팅할 수 없는 상품 조합입니다."
    )
    sqs = FakeSqs()

    result = processor(pg, FakePrecheck(failed), sqs).process(valid_message())

    assert (result.action, result.delete) == ("failure_saved_INVALID_FITTING_COMBINATION", False)
    job = row(pg)
    assert job.result_payload == failure_payload(
        JOB, "INVALID_FITTING_COMBINATION", "가상 피팅할 수 없는 상품 조합입니다."
    )
    assert job.next_action is NextAction.PUBLISH_RESULT
    assert job.result_message_status is ResultMessageStatus.PENDING
    assert job.status is AsyncJobStatus.RECEIVED  # 발행 성공 전에는 FAILED 로 확정하지 않는다
    assert sqs.deletes == []

    # 같은 메시지가 다시 와도 결과를 다시 만들지 않는다.
    check = FakePrecheck()
    again = processor(pg, check).process(valid_message(message_id="m-2"))
    assert (again.action, again.delete) == ("waiting_publish_result", False)
    assert check.calls == [] and row(pg).result_payload == job.result_payload


def test_temporary_problem_is_not_saved_as_a_failure(pg):  # noqa: F811
    retry = PrecheckOutcome(PrecheckStatus.RETRY, reason="DATABASE_ERROR")
    result = processor(pg, FakePrecheck(retry)).process(valid_message())
    assert (result.action, result.delete) == ("retry_DATABASE_ERROR", False)
    job = row(pg)
    assert job.result_payload is None and job.next_action is NextAction.SUBMIT_PRUNA


def test_db_outage_keeps_the_message(pg):  # noqa: F811
    class DownJobs:
        def create(self, job):
            raise FittingDatabaseError("down")

    result = RequestProcessor(
        DownJobs(), pg["state"], FakePrecheck(), FakeSqs(), lease_seconds=60, visibility_seconds=120
    ).process(valid_message())
    assert (result.action, result.delete) == ("error", False)


def test_invalid_message_never_touches_the_db():
    class NoDb:
        def __getattr__(self, name):
            raise AssertionError("형식 오류 메시지로 DB 를 쓰지 않는다")

    message = InvalidRequestMessage(
        message_id="m-bad",
        receipt_handle="receipt",
        approximate_receive_count=2,
        error=InvalidFittingRequest(RequestErrorKind.INVALID_FIELD, "products", "bad"),
        fitting_job_id=JOB,
    )
    result = RequestProcessor(
        NoDb(), NoDb(), FakePrecheck(), FakeSqs(), lease_seconds=60, visibility_seconds=120
    ).process(message)
    assert (result.action, result.delete, result.fitting_job_id) == ("invalid_kept", False, JOB)


def test_invalid_message_does_not_damage_an_existing_job(pg):  # noqa: F811
    processor(pg).process(valid_message())
    before = row(pg)
    message = InvalidRequestMessage(
        "m-bad",
        "receipt",
        1,
        InvalidFittingRequest(RequestErrorKind.MISSING_FIELD, "requested_at", "x"),
        JOB,
    )
    assert processor(pg).process(message).delete is False
    assert row(pg) == before


def test_only_a_published_job_duplicate_is_deleted(pg):  # noqa: F811
    processor(pg).process(valid_message())
    pg["sql"](
        "UPDATE virtual_fitting_async_jobs SET status = 'COMPLETED', result_message_status = 'PUBLISHED', "
        "next_action = NULL WHERE fitting_job_id = %s",
        (JOB,),
    )
    result = processor(pg).process(valid_message(message_id="m-dup"))
    assert (result.action, result.delete) == ("already_published_deleted", True)


def test_saved_but_unpublished_terminal_result_is_kept(pg):  # noqa: F811
    processor(pg).process(valid_message())
    pg["sql"](
        "UPDATE virtual_fitting_async_jobs SET result_message_status = 'RETRY_EXHAUSTED', "
        "next_action = 'PUBLISH_RESULT' WHERE fitting_job_id = %s",
        (JOB,),
    )
    assert processor(pg).process(valid_message(message_id="m-dup")).delete is False


def test_heartbeat_extends_lease_and_visibility_separately(pg):  # noqa: F811
    sqs = FakeSqs()
    processor(pg, FakePrecheck(steps=2), sqs, visibility_seconds=150).process(valid_message())
    assert sqs.visibility_calls == [(FittingQueue.BE_FITTING_REQUEST, "receipt-m-1", 150)] * 2


def test_visibility_extension_failure_stops_further_work(pg):  # noqa: F811
    failed_result = PrecheckOutcome(PrecheckStatus.FAILED, "PRODUCT_NOT_FOUND", "없음")
    result = processor(pg, FakePrecheck(failed_result), FakeSqs(visibility_ok=False)).process(
        valid_message()
    )
    assert result.action == "retry_stopped" and result.delete is False
    assert row(pg).result_payload is None  # 실패 결과도 저장하지 않는다


def test_lost_lease_stops_further_work(pg):  # noqa: F811
    class ExpiringPrecheck(FakePrecheck):
        def run(self, request, before_step=lambda: True):
            pg["sql"](
                "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second' "
                "WHERE fitting_job_id = %s",
                (JOB,),
            )
            return super().run(request, before_step)

    failed = PrecheckOutcome(PrecheckStatus.FAILED, "PRODUCT_NOT_FOUND", "없음")
    sqs = FakeSqs()
    result = processor(pg, ExpiringPrecheck(failed), sqs).process(valid_message())
    assert result.action == "retry_stopped"
    assert sqs.visibility_calls == []  # lease 연장이 먼저 실패해 Visibility 도 늘리지 않았다
    assert row(pg).result_payload is None


def test_two_workers_on_the_same_job_run_the_precheck_once(pg):  # noqa: F811
    processor(pg).process(valid_message())  # 등록
    gate = threading.Event()
    check = FakePrecheck(gate=gate)
    results = {}

    def work(name):
        results[name] = processor(pg, check).process(valid_message(message_id=name))

    first = threading.Thread(target=work, args=("first",))
    first.start()
    deadline = time.monotonic() + 10
    while not check.calls:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    second = threading.Thread(target=work, args=("second",))
    second.start()
    second.join(10)  # 첫 Worker 가 lease 를 쥐고 사전 검증 중일 때 두 번째가 들어온다
    gate.set()
    first.join(10)

    assert check.calls == [JOB]
    assert results["second"].action == "lease_lease_held"
    assert results["first"].action == "prechecked_awaiting_submit"


def test_stop_request_prevents_new_lease(pg):  # noqa: F811
    check = FakePrecheck()
    result = processor(pg, check, should_stop=lambda: True).process(valid_message())
    assert result.action == "stopping_kept" and check.calls == []
    assert row(pg).worker_lease_token is None


def test_submit_handler_is_the_stage_8_connection_point(pg):  # noqa: F811
    calls = []

    def submit(job, token, request, check, heartbeat):
        calls.append((job.fitting_job_id, bool(token), request.fitting_job_id, check.status))
        return False  # 8단계가 복구 기록을 남긴 뒤에만 True 를 돌려준다

    result = processor(pg, submit=submit).process(valid_message())
    assert calls == [(JOB, True, JOB, PrecheckStatus.PASSED)]
    assert (result.action, result.delete) == ("submitted", False)


# --- 반복 수신 Worker ---


def batch(*messages, ok_=True, reason="ok"):
    outcome = SqsOutcome.SUCCEEDED if ok_ else SqsOutcome.UNKNOWN
    return RequestBatch(
        SqsResult(SqsOperation.RECEIVE, FittingQueue.BE_FITTING_REQUEST, outcome, reason=reason),
        tuple(messages),
    )


def run(coro):
    return asyncio.run(coro)


def test_worker_processes_each_message_and_deletes_only_when_allowed():
    batches = iter([batch(valid_message(message_id="a"), valid_message(message_id="b"))])
    deleted = []

    def receive():
        return next(batches, batch())

    def process(item):
        return worker_module.RequestOutcome(
            item.message_id, JOB, "x", delete=item.message_id == "b"
        )

    async def scenario():
        worker = RequestWorker(
            receive,
            process,
            lambda receipt: deleted.append(receipt) or ok(SqsOperation.DELETE),
            idle_seconds=0.01,
        )
        worker.start()
        await asyncio.sleep(0.2)
        await worker.stop(timeout=2)
        return worker

    worker = run(scenario())
    assert deleted == ["receipt-b"]
    assert (worker.status.received, worker.status.deleted, worker.status.crashed) == (2, 1, False)


def test_receive_failures_back_off_instead_of_spinning():
    calls = []

    def receive():
        calls.append(time.monotonic())
        return batch(ok_=False, reason="access_denied")

    async def scenario():
        worker = RequestWorker(
            receive,
            lambda item: None,
            lambda r: ok(),
            backoff_initial_seconds=0.1,
            backoff_max_seconds=0.4,
        )
        worker.start()
        await asyncio.sleep(0.9)
        await worker.stop(timeout=2)
        return worker

    worker = run(scenario())
    assert 2 <= len(calls) <= 5  # 0.1 → 0.2 → 0.4 … 로 물러난다(빠른 무한 반복 없음)
    assert worker.status.consecutive_errors == len(calls)
    assert worker.status.last_error == "access_denied"


def test_receive_exceptions_are_logged_and_backed_off(caplog):
    calls = []

    def receive():
        calls.append(1)
        raise RuntimeError("SQS 설정 오류")

    async def scenario():
        worker = RequestWorker(
            receive,
            lambda item: None,
            lambda r: ok(),
            backoff_initial_seconds=0.1,
            backoff_max_seconds=0.2,
        )
        worker.start()
        await asyncio.sleep(0.5)
        await worker.stop(timeout=2)
        return worker

    with caplog.at_level(logging.ERROR, logger=worker_module.__name__):
        worker = run(scenario())
    assert 2 <= len(calls) <= 5
    assert not worker.status.crashed and worker.status.stopped
    assert "request worker receive crashed" in caplog.text


def test_unexpected_worker_crash_is_visible(caplog):
    def process(item):
        raise RuntimeError("bug in processor")

    async def scenario():
        worker = RequestWorker(lambda: batch(valid_message()), process, lambda r: ok())
        task = worker.start()
        with pytest.raises(RuntimeError):
            await task
        return worker

    with caplog.at_level(logging.ERROR, logger=worker_module.__name__):
        worker = run(scenario())
    assert worker.status.crashed and worker.status.last_error == "RuntimeError"
    assert "request worker crashed" in caplog.text


def test_stop_abandons_a_blocked_receive_and_keeps_its_messages():
    release = threading.Event()
    processed = []

    def receive():
        release.wait(10)  # long polling 중
        return batch(valid_message())

    async def scenario():
        worker = RequestWorker(
            receive,
            lambda item: processed.append(item) or worker_module.RequestOutcome("m", JOB, "x"),
            lambda r: ok(),
        )
        worker.start()
        await asyncio.sleep(0.1)
        started = time.monotonic()
        await worker.stop(timeout=0.3)
        return time.monotonic() - started

    try:
        elapsed = run(scenario())
    finally:
        release.set()
    assert elapsed < 2
    assert processed == []  # 종료 뒤에 받은 메시지는 처리·삭제하지 않는다


def test_stop_between_messages_leaves_the_rest_unprocessed():
    processed = []
    holder = {}

    def process(item):
        processed.append(item.message_id)
        holder["worker"]._stop.set()  # 첫 메시지 처리 중 종료 요청
        return worker_module.RequestOutcome(item.message_id, JOB, "x", delete=True)

    deleted = []

    async def scenario():
        worker = RequestWorker(
            lambda: batch(valid_message(message_id="a"), valid_message(message_id="b")),
            process,
            lambda receipt: deleted.append(receipt) or ok(SqsOperation.DELETE),
        )
        holder["worker"] = worker
        await asyncio.wait_for(worker.start(), 2)  # 첫 메시지 처리 중 스스로 종료 요청을 받는다

    run(scenario())
    assert processed == ["a"] and deleted == ["receipt-a"]


# --- FastAPI 시작·종료 ---


@pytest.fixture
def app_env(monkeypatch):
    monkeypatch.setattr(settings, "CHECKPOINT_DSN", "")
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "test-vton-key")
    monkeypatch.setenv("RUNWARE_LLM_API_KEY", "test-llm-key")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from app import main

    return main


def test_disabled_worker_starts_nothing_and_never_touches_sqs(app_env, monkeypatch):
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_REQUEST_WORKER_ENABLED", False)

    def forbidden(*args, **kwargs):
        raise AssertionError("꺼져 있으면 Worker·SQS 를 만들지 않는다")

    monkeypatch.setattr(app_env, "create_request_worker", forbidden)
    monkeypatch.setattr("app.clients.sqs.get_sqs_client", forbidden)

    with TestClient(app_env.app) as client:
        assert client.get("/health").status_code == 200
        assert client.app.state.request_worker is None


def test_default_setting_is_disabled():
    assert settings.VIRTUAL_FITTING_REQUEST_WORKER_ENABLED is False


def test_enabled_worker_runs_with_fastapi_without_blocking_http(app_env, monkeypatch):
    receiving = threading.Event()
    release = threading.Event()
    calls = []

    def receive():
        calls.append(1)
        receiving.set()
        release.wait(10)  # 동기 long polling 이 thread 에서 막혀 있다
        return batch()

    worker = RequestWorker(receive, lambda item: None, lambda r: ok(), idle_seconds=0.01)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_REQUEST_WORKER_ENABLED", True)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_REQUEST_SHUTDOWN_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(app_env, "create_request_worker", lambda: worker)

    try:
        with TestClient(app_env.app) as client:
            assert receiving.wait(5)
            started = time.monotonic()
            assert client.get("/health").status_code == 200
            assert time.monotonic() - started < 1  # 수신 대기 중에도 HTTP 요청을 처리한다
            assert client.app.state.request_worker is worker and worker.status.running
        assert worker.stopping and not worker.status.running  # 종료 때 함께 멈췄다
    finally:
        release.set()
    assert len(calls) == 1  # 종료 뒤 새 수신 없음
