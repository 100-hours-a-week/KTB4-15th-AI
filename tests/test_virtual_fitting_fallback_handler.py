"""13단계: RUN_FALLBACK 핸들러 (V1 동기 생성 1회 → pruna-sync.jpg → 제목·코멘트 → V2 결과).

실제 10단계 FollowUpProcessor + RunFallbackHandler + 실제 AsyncJobStateRepository·PrunaLimiter·TaskCostRepository
(격리 PostgreSQL)로 돌린다. 실제 RunwarePrunaProvider 를 쓰고 HTTP(opener)만 가짜로 바꿔 유료 생성 요청 수를 센다.
S3·LLM·상품 조회·예산·잔액·SQS 는 가짜다. 시간은 가짜 시계다.
"""

import io
import json
import threading
import uuid
from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from app.clients.s3 import ImageStorageError, S3ObjectCheck, S3ObjectState
from app.clients.sqs import FittingQueue, SqsOutcome
from app.virtual_fitting.exceptions import (
    FittingBalanceExhaustedError,
    FittingBalanceUnknownError,
    FittingDailyBudgetExceededError,
    FittingPostprocessError,
)
from app.virtual_fitting.fallback_handler import RunFallbackHandler, fallback_result_key
from app.virtual_fitting.postprocess_handler import PostprocessHandler
from app.virtual_fitting.postprocess_worker import PostprocessHandlers
from app.virtual_fitting.providers.runware import (
    RUNWARE_ENDPOINT,
    RunwarePrunaProvider,
    parse_result,
)
from app.virtual_fitting.pruna_submitter import NextStep
from app.virtual_fitting.repositories.async_job_repository import AsyncJobStatus, NextAction
from app.virtual_fitting.repositories.pruna_limiter import TaskCostRepository
from app.virtual_fitting.service import FALLBACK_COMMENT, FALLBACK_TITLE
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_postprocess_handler import FakeComments
from tests.test_virtual_fitting_postprocess_worker import (
    Handler,
    ReadyPaths,
    SendingSqs,
    counts,
    falling_back,
    item,
    postprocessing,
    processor,
)
from tests.test_virtual_fitting_pruna_submit import JOB, FakeClock, db, job, limiter  # noqa: F401
from tests.virtual_fitting_fixtures import make_bottom, make_top

KEY = fallback_result_key(JOB)
ASYNC_KEY = f"virtual-fitting/results/{JOB}/pruna-async.jpg"
CDN = "https://im.runware.ai/image/os/a14d18/result.jpg"
TOP = make_top("123456", sub_category="후디", image_url="https://img/top.jpg")
BOTTOM = make_bottom("789012", image_url="https://img/bottom.jpg")


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class CountingOpener:
    """Runware 동기 생성 HTTP 자리. 요청 본문을 모으고 scripted 응답(bytes·dict·Exception)을 돌려준다."""

    def __init__(self, *responses, during=None):
        self.responses, self.requests, self.during = list(responses), [], during

    def __call__(self, request, timeout):
        self.requests.append(json.loads(request.data))
        if self.during is not None:
            self.during()
        response = self.responses.pop(0) if self.responses else success()
        if isinstance(response, BaseException):
            raise response
        return Response(response if isinstance(response, bytes) else json.dumps(response).encode())


def success(cost=0.015):
    item_ = {"taskType": "imageInference", "imageURL": CDN}
    if cost is not None:
        item_["cost"] = cost
    return {"data": [item_]}


def http(code):
    return HTTPError(RUNWARE_ENDPOINT, code, "x", {}, io.BytesIO(b"{}"))


class FakeStorage:
    def __init__(self, user_state=S3ObjectState.EXISTS, store_error=False, presign_error=False):
        self.user_state, self.store_error, self.presign_error = (
            user_state,
            store_error,
            presign_error,
        )
        self.presigned, self.stored, self.heads = [], [], []

    def presigned_get_url(self, key, *, expires_in):
        if self.presign_error:
            raise ImageStorageError("signing failed")
        self.presigned.append(key)
        return f"https://bucket.s3.ap-northeast-2.amazonaws.com/{key}?X-Amz-Signature=SIG{len(self.presigned)}"

    def head_object(self, key):
        self.heads.append(key)
        return S3ObjectCheck(self.user_state, key)

    def store_remote_image_at(self, url, key):
        if self.store_error:
            raise ImageStorageError("download failed")
        self.stored.append((url, key))
        return key


class FakeProducts:
    def __init__(self, *products):
        self.products = products or (TOP, BOTTOM)

    def find_by_codes(self, codes):
        wanted = {str(int(code)) for code in codes}
        return [p for p in self.products if str(int(p.product_code)) in wanted]


class Check:
    def __init__(self, *errors):
        self.errors, self.calls = list(errors), 0

    def ensure_available(self):
        self.calls += 1
        error = self.errors.pop(0) if self.errors else None
        if error is not None:
            raise error


def env(db, *responses, **kwargs):  # noqa: F811
    clock = FakeClock()
    e = SimpleNamespace(
        clock=clock,
        opener=CountingOpener(*responses, during=kwargs.pop("during", None)),
        storage=kwargs.pop("storage", FakeStorage()),
        comments=kwargs.pop("comments", FakeComments()),
        budget=kwargs.pop("budget", Check()),
        balance=kwargs.pop("balance", Check()),
        products=kwargs.pop("products", FakeProducts()),
        limiter=kwargs.pop("limiter_", limiter(db, interval=0)),
    )
    e.handler = RunFallbackHandler(
        jobs=db["jobs"],
        state=db["state"],
        products=e.products,
        storage=e.storage,
        comments=e.comments,
        provider=RunwarePrunaProvider("k", opener=e.opener),
        limiter=e.limiter,
        costs=TaskCostRepository(db["pool"]),
        budget=e.budget,
        balance=e.balance,
        clock=clock,
        sleep=clock.sleep,
        max_wait_seconds=kwargs.pop("max_wait_seconds", 60),
        result_url_expires_seconds=600,
    )
    return e


def fallback_job(db):  # noqa: F811
    """Webhook 의 확인된 실패(timeoutProvider)로 FALLBACK_RUNNING / RUN_FALLBACK 이 된 작업. 비동기 자리는 반환됐다."""
    task = falling_back(db)
    limiter(db).release_in_flight(JOB, task_uuid=task)
    return task


def run(db, e, task=None, **kwargs):  # noqa: F811
    task = task or job(db).task_uuid
    return processor(
        db, PostprocessHandlers(postprocess=Handler(), run_fallback=e.handler), **kwargs
    ).process(item(JOB, task, NextAction.RUN_FALLBACK))


def in_flight(db):  # noqa: F811
    return db["sql"]("SELECT fitting_job_id, task_uuid::text FROM virtual_fitting_pruna_in_flight")


def task_costs(db):  # noqa: F811
    return db["sql"]("SELECT task_uuid::text, cost, usage_date FROM virtual_fitting_task_costs")


def completed(title="차분한 데일리룩", comment="좋은 코디입니다.\n잘 어울립니다."):
    return {
        "fitting_job_id": JOB,
        "status": "COMPLETED",
        "result_image_key": KEY,
        "llm_title": title,
        "llm_comment": comment,
    }


def failed(code, message):
    return {"fitting_job_id": JOB, "status": "FAILED", "error_code": code, "error_message": message}


# --- 정상 ---


def test_fallback_generates_once_stores_fixed_key_and_saves_the_result(db):  # noqa: F811
    async_task = fallback_job(db)
    e = env(db, success(0.015))
    outcome = run(db, e)
    current = job(db)

    (request,) = e.opener.requests  # 유료 생성 정확히 1번
    fallback_task = current.fallback_task_uuid
    assert request[0]["taskUUID"] == fallback_task and fallback_task != async_task
    assert request[0]["deliveryMethod"] == "sync"
    person = request[0]["inputs"]["referenceImages"][0]
    assert person["role"] == "person" and person["image"].startswith(
        "https://bucket.s3.ap-northeast-2.amazonaws.com/users/"
    )
    assert (
        current.task_uuid == async_task and current.async_image_key == ASYNC_KEY
    )  # 비동기 기록 유지
    assert e.storage.stored == [(CDN, KEY)]
    assert current.fallback_image_key == KEY and current.fallback_started_at is not None
    assert task_costs(db) == [(fallback_task, Decimal("0.015"), current.fallback_started_at.date())]
    assert in_flight(db) == []
    assert current.result_payload == completed() and current.result_image_key == KEY
    assert current.next_action is NextAction.PUBLISH_RESULT
    assert current.status is AsyncJobStatus.FALLBACK_RUNNING  # 최종 상태 확정은 14단계
    assert counts(db) == (0, 1, 1, 0)
    # LLM 은 저장한 fallback 이미지의 새 URL 로 만든다(원본 사진·비동기 이미지가 아님).
    ((llm_url, _),) = e.comments.comment_calls
    assert f"/{KEY}?" in llm_url and e.storage.presigned[-1] == KEY
    assert (outcome.action, outcome.delete) == (
        "handler_done",
        False,
    )  # 결과 발행 경로(14단계) 없음


def test_saved_result_with_a_ready_publish_path_deletes_the_message(db):  # noqa: F811
    fallback_job(db)
    outcome = run(db, env(db), follow_up=ReadyPaths(NextStep.PUBLISH_RESULT))
    assert (outcome.action, outcome.delete) == ("handler_done", True)


# --- 중복·동시 ---


def test_repeated_messages_generate_once(db):  # noqa: F811
    fallback_job(db)
    e = env(db)
    run(db, e)
    saved = job(db)
    for _ in range(3):
        again = run(db, e)
        assert again.action == "state_mismatch" and not again.delete
    assert len(e.opener.requests) == 1 and job(db) == saved


def test_concurrent_deliveries_generate_once(db):  # noqa: F811
    fallback_job(db)
    gate = threading.Event()
    e = env(db, during=lambda: gate.wait(5))
    results, errors = [], []
    barrier = threading.Barrier(5)

    def go():
        try:
            barrier.wait()
            results.append(run(db, e))
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=go) for _ in range(5)]
    for thread in threads:
        thread.start()
    import time

    deadline = time.monotonic() + 10
    while len(results) < 4:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    gate.set()
    for thread in threads:
        thread.join(10)
    assert not errors, errors
    assert len(e.opener.requests) == 1 and counts(db)[2] == 1
    assert job(db).result_payload == completed()


def test_fallback_uuid_is_saved_before_the_call_and_reused(db):  # noqa: F811
    async_task = fallback_job(db)
    others = [str(uuid.uuid4()) for _ in range(3)]
    for n, other in enumerate(others, start=500):  # 다른 작업 3건이 진행 중 자리를 모두 쥐고 있다
        db["sql"]("INSERT INTO virtual_fitting_pruna_in_flight VALUES (%s, %s, now())", (n, other))
    e = env(db, max_wait_seconds=6)
    first = run(db, e)
    saved = job(db)
    assert first.action == "handler_deferred_kept" and e.opener.requests == []
    assert saved.fallback_task_uuid not in (None, async_task)
    assert (
        saved.fallback_attempt_count == 0 and saved.fallback_started_at is None
    )  # 기다리기만 했다
    assert e.clock.sleeps and all(s <= 2 for s in e.clock.sleeps)
    db["sql"]("DELETE FROM virtual_fitting_pruna_in_flight WHERE fitting_job_id >= 500")
    run(db, env(db))
    assert job(db).fallback_task_uuid == saved.fallback_task_uuid  # 재수신 때 재사용
    assert job(db).result_payload == completed()


# --- 공유 제한 ---


def test_fallback_and_async_share_the_interval_and_hold(db):  # noqa: F811
    fallback_job(db)
    shared = limiter(db, interval=10)
    shared.defer_sends(30)  # 비동기 생성이 남긴 공유 대기
    e = env(db, limiter_=shared, max_wait_seconds=6)
    outcome = run(db, e)
    assert outcome.action == "handler_deferred_kept" and e.opener.requests == []
    shared.hold_sends("retry_after_unrepresentable")
    db["sql"]("UPDATE virtual_fitting_pruna_limiter SET next_send_at = 'epoch'")
    held = env(db, limiter_=shared)
    assert run(db, held).action == "handler_deferred_kept" and held.opener.requests == []
    assert counts(db)[2] == 0


def test_fallback_counts_against_the_in_flight_limit_for_async_submissions(db):  # noqa: F811
    from tests.test_virtual_fitting_pruna_submit import ready_job

    fallback_job(db)
    e = env(db, http(504))  # 결과 불명확 → fallback 자리를 유지한다
    run(db, e)
    fallback_task = job(db).fallback_task_uuid
    assert in_flight(db) == [(JOB, fallback_task)]
    shared = limiter(db, in_flight=1, interval=0)
    token, task = ready_job(db, 7)
    assert shared.begin_submission(7, token, task).reason.value == "in_flight_full"


def test_other_attempt_holding_the_jobs_slot_blocks_the_fallback(db):  # noqa: F811
    async_task = falling_back(db)  # 비동기 시도가 아직 자리를 쥐고 있다(반환 근거 없음)
    e = env(db)
    outcome = run(db, e)
    assert outcome.action == "handler_deferred_kept" and e.opener.requests == []
    assert in_flight(db) == [(JOB, async_task)] and counts(db)[2] == 0


# --- 입력·예산·잔액 (생성 전 실패는 횟수를 쓰지 않는다) ---


@pytest.mark.parametrize(
    ("budget", "balance", "expected"),
    [
        (
            Check(FittingDailyBudgetExceededError("x")),
            Check(),
            failed(FittingDailyBudgetExceededError.code, FittingDailyBudgetExceededError.message),
        ),
        (
            Check(),
            Check(FittingBalanceExhaustedError("x")),
            failed(FittingBalanceExhaustedError.code, FittingBalanceExhaustedError.message),
        ),
    ],
)
def test_budget_or_balance_shortage_fails_without_generation(db, budget, balance, expected):  # noqa: F811
    fallback_job(db)
    e = env(db, budget=budget, balance=balance)
    run(db, e)
    assert e.opener.requests == [] and counts(db)[2] == 0
    assert job(db).result_payload == expected and job(db).next_action is NextAction.PUBLISH_RESULT


def test_balance_unknown_is_held_not_failed(db):  # noqa: F811
    fallback_job(db)
    e = env(db, balance=Check(FittingBalanceUnknownError("x")))
    assert run(db, e).action == "handler_deferred_kept"
    assert e.opener.requests == [] and job(db).result_payload is None and counts(db)[2] == 0


def test_budget_is_rechecked_after_a_long_slot_wait(db):  # noqa: F811
    fallback_job(db)
    shared = limiter(db, interval=0)
    shared.defer_sends(40)
    e = env(db, limiter_=shared, budget=Check(None, FittingDailyBudgetExceededError("x")))

    real_sleep = e.clock.sleep

    def sleep(seconds):
        real_sleep(seconds)
        db["sql"](
            "UPDATE virtual_fitting_pruna_limiter SET next_send_at = next_send_at - make_interval(secs => %s)",
            (seconds,),
        )

    e.handler._sleep = sleep
    run(db, e)
    assert e.budget.calls == 2 and e.opener.requests == []
    assert job(db).result_payload["error_code"] == "FITTING_DAILY_BUDGET_EXCEEDED"


def test_missing_user_image_and_products_fail_before_generation(db):  # noqa: F811
    fallback_job(db)
    e = env(db, storage=FakeStorage(user_state=S3ObjectState.NOT_FOUND))
    run(db, e)
    assert job(db).result_payload["error_code"] == "INVALID_REQUEST" and e.opener.requests == []


def test_missing_product_fails_and_access_denied_user_image_is_held(db):  # noqa: F811
    fallback_job(db)
    e = env(db, products=FakeProducts(TOP))
    run(db, e)
    assert job(db).result_payload["error_code"] == "PRODUCT_NOT_FOUND" and e.opener.requests == []


@pytest.mark.parametrize("state", [S3ObjectState.ACCESS_DENIED, S3ObjectState.TRANSIENT_ERROR])
def test_user_image_access_problems_are_held(db, state):  # noqa: F811
    fallback_job(db)
    e = env(db, storage=FakeStorage(user_state=state))
    assert run(db, e).action == "handler_deferred_kept"
    assert e.opener.requests == [] and job(db).result_payload is None and counts(db)[2] == 0


def test_presign_failure_uses_no_fallback_attempt(db):  # noqa: F811
    fallback_job(db)
    e = env(db, storage=FakeStorage(presign_error=True))
    assert run(db, e).action == "handler_deferred_kept"
    assert e.opener.requests == [] and counts(db)[2] == 0


# --- 생성 오류 ---


def test_clear_rejection_saves_model_failure_and_releases_the_slot(db):  # noqa: F811
    fallback_job(db)
    e = env(db, http(400))
    run(db, e)
    assert len(e.opener.requests) == 1 and in_flight(db) == []
    assert job(db).result_payload == failed(
        "FITTING_MODEL_FAILED", "가상 피팅 이미지 생성에 실패했습니다."
    )
    assert counts(db)[2] == 1


def test_402_saves_balance_exhausted_without_more_fallback(db):  # noqa: F811
    fallback_job(db)
    e = env(db, http(402))
    run(db, e)
    assert job(db).result_payload["error_code"] == "FITTING_BALANCE_EXHAUSTED"
    assert len(e.opener.requests) == 1 and in_flight(db) == []


@pytest.mark.parametrize("code", [401, 403])
def test_auth_rejection_is_held_for_operators(db, code):  # noqa: F811
    fallback_job(db)
    e = env(db, http(code))
    assert run(db, e).action == "handler_deferred_kept"
    assert job(db).result_payload is None and in_flight(db) == [] and len(e.opener.requests) == 1


@pytest.mark.parametrize(
    "error",
    [
        URLError(TimeoutError("timed out")),
        TimeoutError(),
        http(504),
        http(500),
        http(429),
        ConnectionResetError(),
        b"not json",
        {"data": []},
        {"errors": [{"message": "x"}]},
    ],
)
def test_unclear_results_keep_the_slot_and_are_never_resent(db, error):  # noqa: F811
    fallback_job(db)
    e = env(db, error)
    assert run(db, e).action == "handler_deferred_kept"
    fallback_task = job(db).fallback_task_uuid
    assert in_flight(db) == [(JOB, fallback_task)]  # 원격 생성이 끝났다고 가정하지 않는다
    assert job(db).result_payload is None and counts(db)[2] == 1
    again = env(db)  # 재수신·다른 Worker
    assert run(db, again).action == "handler_deferred_kept" and again.opener.requests == []
    assert len(e.opener.requests) == 1


def test_lost_lease_before_the_slot_means_no_call(db):  # noqa: F811
    fallback_job(db)

    class ExpiringStorage(FakeStorage):
        def presigned_get_url(self, key, *, expires_in):
            if len(self.presigned) == 1:  # 자리 확보 직전 발급에서 lease 를 잃는다
                db["sql"](
                    "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second' "
                    "WHERE fitting_job_id = %s",
                    (JOB,),
                )
            return super().presigned_get_url(key, expires_in=expires_in)

    e = env(db, storage=ExpiringStorage())
    outcome = run(db, e)
    assert outcome.delete is False and e.opener.requests == [] and counts(db)[2] == 0


# --- 비용·이미지·재개 ---


def test_bad_cost_keeps_the_image_and_does_not_record_a_cost(db):  # noqa: F811
    fallback_job(db)
    e = env(db, {"data": [{"imageURL": CDN, "cost": "not-a-number"}]})
    run(db, e)
    assert task_costs(db) == [] and job(db).result_payload == completed()
    with pytest.raises(ArithmeticError):  # V1 기본(strict) 동작은 그대로
        parse_result(json.dumps({"data": [{"imageURL": CDN, "cost": "not-a-number"}]}).encode())
    assert (
        parse_result(
            json.dumps({"data": [{"imageURL": CDN, "cost": "x"}]}).encode(), strict_cost=False
        ).cost
        is None
    )


def test_missing_cost_is_not_recorded_as_zero(db):  # noqa: F811
    fallback_job(db)
    run(db, env(db, success(None)))
    assert task_costs(db) == [] and job(db).result_payload == completed()


def test_image_store_failure_keeps_the_cost_and_never_regenerates(db):  # noqa: F811
    fallback_job(db)
    e = env(db, success(0.02), storage=FakeStorage(store_error=True))
    assert run(db, e).action == "handler_deferred_kept"
    current = job(db)
    assert task_costs(db)[0][1] == Decimal("0.02")  # 이미지 저장 실패로 비용을 잃지 않는다
    assert current.fallback_image_key is None and current.result_payload is None
    again = env(db)
    assert run(db, again).action == "handler_deferred_kept" and again.opener.requests == []


def test_recorded_fallback_image_resumes_from_llm_without_generation(db):  # noqa: F811
    fallback_job(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET fallback_task_uuid = %s, fallback_started_at = now(), "
        "fallback_attempt_count = 1, fallback_image_key = %s, llm_generation_attempt_count = 2 "
        "WHERE fitting_job_id = %s",
        (str(uuid.uuid4()), KEY, JOB),
    )
    comments = FakeComments(comments=[FittingPostprocessError("x")] * 5)
    e = env(db, comments=comments)
    run(db, e)
    assert e.opener.requests == [] and e.storage.heads == []
    assert len(comments.comment_calls) == 1  # 남은 1회만(POSTPROCESS 에서 쓴 2회 유지)
    assert job(db).result_payload == completed(FALLBACK_TITLE, FALLBACK_COMMENT)
    assert counts(db) == (0, 3, 1, 0)


def test_llm_three_failures_use_default_text_and_keep_the_image(db):  # noqa: F811
    fallback_job(db)
    comments = FakeComments(comments=["코멘트"] * 4, titles=[FittingPostprocessError("x")] * 4)
    e = env(db, comments=comments)
    run(db, e)
    assert len(comments.comment_calls) == 3 and len(comments.title_calls) == 3
    assert job(db).result_payload == completed(FALLBACK_TITLE, FALLBACK_COMMENT)
    assert len(e.opener.requests) == 1  # LLM 실패로 다시 생성하지 않는다


def test_cost_date_is_the_fallback_start_date(db):  # noqa: F811
    fallback_job(db)
    run(db, env(db))
    current = job(db)
    ((_, _, usage_date),) = task_costs(db)
    assert usage_date == current.fallback_started_at.date() and isinstance(usage_date, date)


# --- 늦은 비동기 Webhook ---


def test_late_async_webhooks_do_not_touch_the_fallback(db, monkeypatch):  # noqa: F811
    from app.config import settings
    from app.virtual_fitting.runware_webhook import RunwareWebhookService, parse_webhook
    from tests.test_virtual_fitting_webhook import FakeQueue, error_body, success_body

    monkeypatch.setattr(settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(
        settings,
        "AI_FITTING_RESPONSE_QUEUE_URL",
        "https://sqs.ap-northeast-2.amazonaws.com/123456789012/ai-fitting-response",
    )
    async_task = fallback_job(db)
    run(db, env(db, http(504)))  # fallback 결과 불명확: fallback 자리 유지
    before = job(db)
    service = RunwareWebhookService(
        jobs=db["jobs"],
        state=db["state"],
        limiter=limiter(db),
        costs=TaskCostRepository(db["pool"]),
        queue=FakeQueue(),
    )
    for body in (success_body(async_task), error_body(async_task)):
        service.handle(parse_webhook(body))
    after = job(db)
    assert (after.status, after.next_action, after.fallback_task_uuid) == (
        before.status,
        before.next_action,
        before.fallback_task_uuid,
    )
    assert in_flight(db) == [
        (JOB, before.fallback_task_uuid)
    ]  # 비동기 알림이 fallback 자리를 놓지 않는다


# --- 11단계 S3 실패 → RUN_FALLBACK 후속 큐 ---


def _s3_failure_handler(db):  # noqa: F811
    from tests.test_virtual_fitting_postprocess_handler import env as postprocess_env

    return postprocess_env(db, [S3ObjectState.NOT_FOUND] * 6).handler


def test_s3_failure_moves_to_fallback_and_publishes_run_fallback(db):  # noqa: F811
    task = postprocessing(db)
    sqs = SendingSqs()
    handlers = PostprocessHandlers(_s3_failure_handler(db), env(db).handler)
    outcome = processor(db, handlers, sqs=sqs).process(item(JOB, task))
    assert (outcome.action, outcome.delete) == ("handed_off_deleted", True)
    assert sqs.sent == [
        (
            FittingQueue.AI_FITTING_RESPONSE,
            {"fitting_job_id": JOB, "task_uuid": task, "action": "RUN_FALLBACK"},
        )
    ]
    assert job(db).status is AsyncJobStatus.FALLBACK_RUNNING and counts(db)[2] == 0


def test_hand_off_publish_failure_is_recovered_on_redelivery(db):  # noqa: F811
    task = postprocessing(db)
    limiter(db).release_in_flight(JOB, task_uuid=task)  # 성공 Webhook 이 비동기 자리를 반환했다
    sqs = SendingSqs(SqsOutcome.UNKNOWN)
    handlers = PostprocessHandlers(_s3_failure_handler(db), env(db).handler)
    first = processor(db, handlers, sqs=sqs).process(item(JOB, task))
    assert (first.action, first.delete) == ("handed_off_publish_failed_kept", False)
    again = processor(db, handlers, sqs=sqs).process(
        item(JOB, task)
    )  # 같은 POSTPROCESS 메시지 재수신
    assert (again.action, again.delete) == ("republished_deleted", True)
    assert [p["action"] for _, p in sqs.sent] == ["RUN_FALLBACK", "RUN_FALLBACK"]
    # 이제 RUN_FALLBACK 메시지를 처리한다(같은 비동기 task_uuid 로 검증).
    e = env(db)
    run(db, e, task)
    assert len(e.opener.requests) == 1 and job(db).result_payload == completed()


# --- 실제 Worker 조립 ---


def test_connected_handlers_build_without_io(monkeypatch):
    from app.config import database
    from app.virtual_fitting import postprocess_worker as pw

    monkeypatch.setattr(
        database, "get_connection_pool", lambda: pytest.fail("연결만으로 DB 를 열지 않는다")
    )
    handlers = pw.connected_postprocess_handlers()
    assert isinstance(handlers.postprocess, PostprocessHandler)
    assert isinstance(handlers.run_fallback, RunFallbackHandler) and handlers.missing() == []


def test_real_worker_runs_run_fallback_through_real_clients(db, monkeypatch):  # noqa: F811
    import asyncio
    import time

    from app.config import settings
    from app.virtual_fitting import postprocess_worker as pw
    from app.virtual_fitting.runware_webhook import follow_up_message
    from tests.test_virtual_fitting_postprocess_worker import FakeBoto

    task = fallback_job(db)
    boto = FakeBoto([json.dumps(follow_up_message(job(db), NextAction.RUN_FALLBACK))])
    monkeypatch.setattr(settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(
        settings,
        "AI_FITTING_RESPONSE_QUEUE_URL",
        "https://sqs.ap-northeast-2.amazonaws.com/123456789012/ai-fitting-response",
    )
    import app.clients.sqs as sqs_module
    from app.config import database

    monkeypatch.setattr(database, "get_connection_pool", lambda: db["pool"])
    monkeypatch.setattr(sqs_module, "get_sqs_client", lambda: boto)
    e = env(db)
    worker = pw.create_postprocess_worker(PostprocessHandlers(Handler(), e.handler))

    async def scenario():
        worker.start()
        deadline = time.monotonic() + 5
        while job(db).result_payload is None:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.05)
        await worker.stop(1)

    asyncio.run(scenario())
    assert len(e.opener.requests) == 1 and job(db).result_payload == completed()
    assert job(db).task_uuid == task
    assert boto.deleted == []  # 결과 발행 경로(14단계)가 없어 지우지 않는다
    assert worker.status.outcomes.get("handler_done") == 1
