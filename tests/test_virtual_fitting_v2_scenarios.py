"""16단계: 실제 조립 객체(create_* 로 만든 요청·후처리 Worker·결과 전송·Watchdog)와 실제 Router 를 연결한 분기·중단·경합 검증.

루프 Task 대신 루프가 한 번에 하는 일을 테스트가 차례로 부른다(tests/v2_integration_harness.Stack). 재시작은 새 DB pool 과
새로 조립한 객체로 가정한다(DB·가짜 SQS/S3/Runware 상태는 유지). 시간은 가짜 시계·DB 시각 조정으로만 옮긴다.
외부 경계만 가짜다: SQS·S3 boto3 client, Runware HTTP(urlopen). 실제 AWS·Runware·OpenAI 는 부르지 않는다.
"""

import os
import threading
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.fallback_handler import RunFallbackHandler
from app.virtual_fitting.postprocess_handler import PostprocessHandler
from app.virtual_fitting.postprocess_worker import PostprocessHandlers
from app.virtual_fitting.pruna_submitter import NextStep, next_step_for
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository
from tests.v2_integration_harness import (
    COMMENT,
    TITLE,
    Stack,
    close_env,
    make_env,
    post_webhook,
)

JOB = 9101
ASYNC_KEY = f"virtual-fitting/results/{JOB}/pruna-async.jpg"
SYNC_KEY = f"virtual-fitting/results/{JOB}/pruna-sync.jpg"


@pytest.fixture
def env(monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL 이 없다")
    try:
        admin = psycopg.connect(url, autocommit=True, connect_timeout=3)
    except psycopg.Error as error:
        pytest.skip(f"테스트 DB 에 연결할 수 없다: {error}")
    schema = f"vf_v2scn_{uuid.uuid4().hex[:12]}"
    admin.execute(f"CREATE SCHEMA {schema}")
    e = make_env(monkeypatch, url, schema)
    try:
        yield e
    finally:
        assert e.network_attempts == []
        close_env(e)
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()


class Clock:
    """가짜 시계. sleep 마다 시간이 흐르고, at(t) 에 걸어 둔 일을 그 시각에 한 번 한다(예: 늦은 PUT)."""

    def __init__(self):
        self.now, self.events = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        for event in list(self.events):
            if self.now >= event[0]:
                self.events.remove(event)
                event[1]()

    def at(self, t, action):
        self.events.append((t, action))


def clocked_handlers(clock):
    return PostprocessHandlers(
        postprocess=PostprocessHandler(clock=clock, sleep=clock.sleep),
        run_fallback=RunFallbackHandler(clock=clock, sleep=clock.sleep),
    )


def client():
    from app.main import app

    return TestClient(app)  # lifespan 없이 Router 만(Webhook)


def row(env, job_id=JOB):
    return env.job(job_id)


def snapshot(env, job_id=JOB):
    current = row(env, job_id)
    return (
        current.status,
        current.next_action,
        current.result_payload,
        current.result_image_key,
        current.result_message_status,
        current.task_uuid,
        current.fallback_task_uuid,
        current.s3_check_attempt_count,
        current.llm_generation_attempt_count,
        current.fallback_attempt_count,
        current.result_message_attempt_count,
    )


def costs(env):
    return env.sql("SELECT task_uuid::text, cost FROM virtual_fitting_task_costs ORDER BY 1")


def in_flight(env):
    return env.sql("SELECT fitting_job_id, task_uuid::text FROM virtual_fitting_pruna_in_flight")


def expire_lease(env, job_id=JOB):
    env.sql(
        "UPDATE virtual_fitting_async_jobs SET worker_lease_until = statement_timestamp() "
        "- interval '1 second' WHERE fitting_job_id = %s AND worker_lease_token IS NOT NULL",
        (job_id,),
    )


def age(env, column, seconds, job_id=JOB):
    env.sql(
        f"UPDATE virtual_fitting_async_jobs SET {column} = statement_timestamp() "
        "- make_interval(secs => %s) WHERE fitting_job_id = %s",
        (seconds, job_id),
    )


def restart(env, handlers=None, running=True):
    """프로세스 재시작 가정: 새 pool·새 조립 객체."""
    env.new_process_pool()
    stack = Stack(env, handlers)
    return stack.mark_running() if running else stack


def submitted(env, stack, job_id=JOB):
    env.send_request(job_id)
    stack.request_once()
    current = row(env, job_id)
    assert current.task_uuid is not None
    return current.task_uuid


def finish(env, stack):
    """남은 후처리·결과 전송을 끝까지(테스트에서 한 단계씩)."""
    for _ in range(3):
        stack.postprocess_once()
    expire_lease(env)
    stack.publish_once()


# --- B. fallback 진입점 ---


def test_fallback_from_a_retryable_submit_rejection(env):
    env.runware.async_modes.append(
        ("errors", [{"code": "providerRateLimitExceeded", "message": "x"}])
    )
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    (outcome,) = stack.request_once()
    assert outcome.delete is True  # RUN_FALLBACK 발행 확인 뒤에만 지운다
    assert [b["action"] for b in env.sqs.bodies("ai-fitting-response")] == ["RUN_FALLBACK"]
    finish(env, stack)
    final = row(env)
    assert final.result_image_key == SYNC_KEY and final.status is AsyncJobStatus.COMPLETED
    assert final.fallback_task_uuid != final.task_uuid
    assert env.runware.count("sync_generate") == 1
    assert env.sqs.bodies("be-fitting-response")[0]["result_image_key"] == SYNC_KEY


def test_fallback_after_both_s3_check_bundles_find_nothing(env):
    clock = Clock()
    stack = Stack(env, clocked_handlers(clock)).mark_running()
    task = submitted(env, stack)
    # Webhook 은 성공이지만 결과 객체가 끝내 나타나지 않는다(PUT 없음).
    assert post_webhook(client(), env.runware.complete(task, put=False)).status_code == 200
    stack.postprocess_once()  # 한 번의 처리에서 1·2묶음(각 0·2·7초) → 둘 다 404 → RUN_FALLBACK 발행
    current = row(env)
    assert current.s3_check_attempt_count == 2 and current.next_action is NextAction.RUN_FALLBACK
    assert env.s3.heads.count(ASYNC_KEY) == 6
    finish(env, stack)
    assert row(env).result_image_key == SYNC_KEY and env.runware.count("sync_generate") == 1


# --- C. PUT 보다 Webhook 이 먼저 ---


def test_object_appearing_after_the_webhook_succeeds_inside_the_check_window(env):
    clock = Clock()
    stack = Stack(env, clocked_handlers(clock)).mark_running()
    task = submitted(env, stack)
    body = env.runware.complete(task, put=False)
    clock.at(2.0, lambda: env.s3.store(ASYNC_KEY))  # 2초 확인 직전에 PUT 이 끝난다
    assert post_webhook(client(), body).status_code == 200
    stack.postprocess_once()
    current = row(env)
    assert current.verified_image_key == ASYNC_KEY and current.s3_check_attempt_count == 1
    assert current.fallback_attempt_count == 0 and current.result_payload["status"] == "COMPLETED"
    assert env.runware.count("sync_generate") == 0


# --- D. 접수 응답보다 Webhook 이 먼저 ---


@pytest.mark.parametrize("code", [None, "timeoutProvider", "someNewCode"])
def test_webhook_before_the_accept_response(env, code):
    stack = Stack(env).mark_running()
    seen = {}

    def deliver_first(submission):
        seen["status"] = row(env).status
        body = (
            env.runware.complete(submission.task_uuid)
            if code is None
            else env.runware.failure(submission.task_uuid, code)
        )
        seen["response"] = post_webhook(client(), body).status_code
        return "accept"  # 그 뒤 접수 응답

    env.runware.async_modes.append(deliver_first)
    env.send_request(JOB)
    stack.request_once()
    current = row(env)
    assert seen == {"status": AsyncJobStatus.RECEIVED, "response": 200}
    expected = {
        None: (AsyncJobStatus.POSTPROCESSING, NextAction.POSTPROCESS),
        "timeoutProvider": (AsyncJobStatus.FALLBACK_RUNNING, NextAction.RUN_FALLBACK),
        "someNewCode": (AsyncJobStatus.RECEIVED, NextAction.PUBLISH_RESULT),
    }[code]
    assert (current.status, current.next_action) == expected  # 늦은 접수 응답이 되돌리지 않음
    assert current.runware_accepted_at is not None
    if code == "someNewCode":
        assert current.result_payload["error_code"] == "FITTING_MODEL_FAILED"
    finish(env, stack)
    assert row(env).result_message_status is ResultMessageStatus.PUBLISHED


# --- E. LLM 실패 ---


def test_comment_failing_twice_uses_default_text_without_a_title_call(env):
    for _ in range(5):
        env.runware.comment_modes.append(("http", 500))
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    post_webhook(client(), env.runware.complete(task))
    stack.postprocess_once()
    current = row(env)
    assert current.llm_generation_attempt_count == 2
    assert env.runware.count("llm_comment") == 2 and env.runware.count("llm_title") == 0
    assert current.result_payload["status"] == "COMPLETED"  # 이미지 성공 유지(기본 문구)
    assert current.result_payload["result_image_key"] == ASYNC_KEY
    presigned = len(env.s3.presigned)
    env.sqs.expire_visibility("ai-fitting-response")
    stack.postprocess_once()  # 결과 저장 뒤 재수신: URL 발급·LLM 호출 없음
    assert len(env.s3.presigned) == presigned and env.runware.count("llm_comment") == 2


def test_title_only_failures_make_one_comment_and_two_title_calls(env):
    for _ in range(5):
        env.runware.title_modes.append(("http", 500))
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    post_webhook(client(), env.runware.complete(task))
    stack.postprocess_once()
    payload = row(env).result_payload
    assert env.runware.count("llm_comment") == 1 and env.runware.count("llm_title") == 2
    assert payload["status"] == "COMPLETED" and payload["llm_title"] != TITLE
    assert payload["llm_comment"] == COMMENT  # 생성 코멘트 유지 + 기본 제목


# --- F. 업무 실패 ---


def _today():
    return datetime.now(UTC).date()


@pytest.mark.parametrize("case", ["budget", "balance", "unknown_product"])
def test_business_failures_are_published_without_generation(env, case):
    if case == "budget":
        env.sql(
            "INSERT INTO usage_amount (usage_date, amount, legacy_amount) VALUES (%s, 5, 5)",
            (_today(),),
        )
        expected = "FITTING_DAILY_BUDGET_EXCEEDED"
    elif case == "balance":
        env.runware.balance = "0.50"
        expected = "FITTING_BALANCE_EXHAUSTED"
    else:
        expected = "PRODUCT_NOT_FOUND"
    stack = Stack(env).mark_running()
    env.send_request(JOB, **({"codes": ("123456", "999999")} if case == "unknown_product" else {}))
    stack.request_once()
    current = row(env)
    assert current.result_payload["status"] == "FAILED"
    assert current.result_payload["error_code"] == expected
    stack.publish_once()
    (payload,) = env.sqs.bodies("be-fitting-response")
    assert payload == current.result_payload
    assert env.runware.count("async_submit") == 0 and env.runware.count("sync_generate") == 0
    assert row(env).result_message_status is ResultMessageStatus.PUBLISHED


# --- G. 제출 응답 401·403 인증 보류(모든 경로 '실행 중') ---


@pytest.mark.parametrize("status", [401, 403])
def test_submit_auth_rejection_is_held_across_redelivery_and_restart(env, status):
    env.runware.async_modes.append(("http", status))
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    (outcome,) = stack.request_once()
    assert outcome.delete is False
    held = row(env)
    assert (held.auth_hold_task_uuid, held.auth_hold_reason) == (held.task_uuid, f"http_{status}")
    assert next_step_for(held) is NextStep.OPERATOR_REVIEW
    stack = restart(env)
    env.sqs.expire_visibility("be-fitting-request")
    (again,) = stack.request_once()
    assert again.delete is False and env.sqs.live("be-fitting-request")
    age(env, "pruna_submit_started_at", 400)  # 접수 불명확 조회·알림 시각이 지나도
    stack.watchdog_once()  # 운영 알림만(조회·해제·정상 대기 전환 없음)
    alerts = env.sql("SELECT reason FROM virtual_fitting_watchdog_alerts")
    assert alerts == [(f"pruna_auth_hold_http_{status}",)]
    assert next_step_for(row(env)) is NextStep.OPERATOR_REVIEW
    assert env.runware.count("async_submit") == 1 and env.runware.count("get_response") == 0
    assert row(env).result_payload is None


# --- H. 결과 전송 ---


def saved_failure(env, stack):
    task = submitted(env, stack)
    post_webhook(client(), env.runware.failure(task, "someNewCode"))
    expire_lease(env)
    return task


def test_publish_retries_the_same_payload_up_to_three_times(env):
    stack = Stack(env).mark_running()
    saved_failure(env, stack)
    env.sqs.fault("be-fitting-response", "send", "throttled", "stored_timeout")
    for _ in range(3):
        stack.publish_once()
        expire_lease(env)
    current = row(env)
    assert current.result_message_status is ResultMessageStatus.PUBLISHED
    assert current.result_message_attempt_count == 3
    # FAILED(저장 안 됨) → UNKNOWN(실제로는 저장됨) → 성공: 같은 payload 가 두 번 도착(중복 가능, BE 멱등 처리 필요)
    assert env.sqs.bodies("be-fitting-response") == [current.result_payload] * 2


def test_retry_exhausted_is_not_resent_by_the_watchdog(env):
    stack = Stack(env).mark_running()
    saved_failure(env, stack)
    env.sqs.fault("be-fitting-response", "send", "throttled", "throttled", "throttled")
    for _ in range(4):
        stack.publish_once()
        expire_lease(env)
    assert row(env).result_message_status is ResultMessageStatus.RETRY_EXHAUSTED
    for _ in range(2):
        env.sql("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
        stack.watchdog_once()
        stack.publish_once()
    current = row(env)
    assert current.result_message_attempt_count == 3 and env.sqs.bodies("be-fitting-response") == []
    alerts = env.sql("SELECT reason, seen_count FROM virtual_fitting_watchdog_alerts")
    assert alerts == [("result_publish_exhausted", 2)]


def test_sqs_success_then_db_record_failure_resends_the_same_payload(env, monkeypatch):
    stack = Stack(env).mark_running()
    saved_failure(env, stack)
    real = AsyncJobStateRepository.mark_result_published
    calls = []

    def fail_once(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise FittingDatabaseError("injected after send")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(AsyncJobStateRepository, "mark_result_published", fail_once)
    with pytest.raises(FittingDatabaseError):
        stack.publish_once()  # 루프는 이 오류로 backoff 한다
    assert row(env).result_message_status is ResultMessageStatus.PENDING
    expire_lease(env)
    stack.publish_once()
    current = row(env)
    assert current.result_message_status is ResultMessageStatus.PUBLISHED
    assert current.result_message_attempt_count == 2
    assert env.sqs.bodies("be-fitting-response") == [current.result_payload] * 2


def test_after_publish_duplicates_and_late_webhooks_change_nothing(env):
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    post_webhook(client(), env.runware.complete(task))
    finish(env, stack)
    final = snapshot(env)
    assert final[4] is ResultMessageStatus.PUBLISHED
    env.send_request(JOB)  # 중복 요청
    (outcome,) = stack.request_once()
    assert outcome.delete is True
    for body in (env.runware.complete(task), env.runware.failure(task, "timeoutProvider")):
        assert post_webhook(client(), body).status_code == 200
    stack.postprocess_once()
    stack.publish_once()
    assert snapshot(env) == final
    assert env.runware.count("async_submit") == 1 and env.runware.count("llm_comment") == 1
    assert len(env.sqs.bodies("be-fitting-response")) == 1
    assert costs(env) == [(task, Decimal("0.023"))]


# --- 5. 중단·재시작·부분 실패 ---


def test_submit_started_and_response_unknown_is_recovered_by_get_response(env):
    env.runware.async_modes.append("timeout")
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    stack.request_once()
    unknown = row(env)
    task = unknown.task_uuid
    assert unknown.pruna_submit_started_at is not None and unknown.runware_accepted_at is None
    assert next_step_for(unknown) is NextStep.RECOVER_UNKNOWN
    assert in_flight(env) == [(JOB, task)]  # 근거 없이 자리를 놓지 않음
    stack = restart(env)
    env.sqs.expire_visibility("be-fitting-request")
    stack.request_once()  # 재수신: 다시 보내지 않음
    assert env.runware.count("async_submit") == 1
    # 3분 전에는 조회하지 않는다(실패 확정 시간이 아님).
    age(env, "pruna_submit_started_at", 170)
    stack.watchdog_once()
    assert env.runware.count("get_response") == 0
    age(env, "pruna_submit_started_at", 200)
    env.sql("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
    env.runware.complete(task)  # Runware 쪽에서는 끝났고 결과도 PUT 됨
    env.runware.success_status(task, cost=0.023)
    stack.watchdog_once()
    current = row(env)
    assert current.next_action is NextAction.POSTPROCESS and current.task_uuid == task
    finish(env, stack)
    assert row(env).result_message_status is ResultMessageStatus.PUBLISHED
    assert env.runware.count("async_submit") == 1 and costs(env) == [(task, Decimal("0.023"))]


def test_state_changed_but_internal_publish_failed_is_completed_on_redelivery(env):
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    env.sqs.fault("ai-fitting-response", "send", "throttled")
    body = env.runware.complete(task)
    first = post_webhook(client(), body)
    assert first.status_code == 503 and first.json()["code"] == "WEBHOOK_FOLLOW_UP_FAILED"
    assert (
        row(env).next_action is NextAction.POSTPROCESS
        and env.sqs.bodies("ai-fitting-response") == []
    )
    assert post_webhook(client(), body).status_code == 200  # Runware 재전송
    assert len(env.sqs.bodies("ai-fitting-response")) == 1 and costs(env) == [
        (task, Decimal("0.023"))
    ]


def test_internal_publish_unknown_creates_a_duplicate_that_runs_once(env):
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    env.sqs.fault("ai-fitting-response", "send", "stored_timeout")
    body = env.runware.complete(task)
    assert post_webhook(client(), body).status_code == 503  # UNKNOWN: 실제로는 들어감
    assert post_webhook(client(), body).status_code == 200
    assert len(env.sqs.bodies("ai-fitting-response")) == 2
    stack = restart(env)
    for _ in range(2):
        stack.postprocess_once()
    assert env.runware.count("llm_comment") == 1 and env.runware.count("llm_title") == 1
    assert env.sqs.live("ai-fitting-response") == []  # 중복 메시지는 재실행 없이 지운다
    assert costs(env) == [(task, Decimal("0.023"))]


def test_fallback_started_without_a_result_is_kept_for_operators(env):
    env.runware.sync_modes.append("timeout")
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    post_webhook(client(), env.runware.failure(task, "timeoutProvider"))
    stack.postprocess_once()
    started = row(env)
    assert started.fallback_started_at is not None and started.fallback_image_key is None
    fb = started.fallback_task_uuid
    assert in_flight(env) == [(JOB, fb)]  # 결과 불명확: 자리 유지
    stack = restart(env)
    env.sqs.expire_visibility("ai-fitting-response")
    expire_lease(env)
    stack.postprocess_once()
    assert env.runware.count("sync_generate") == 1  # 재생성 없음
    age(env, "updated_at", 700)
    env.sql(
        "UPDATE virtual_fitting_pruna_in_flight SET acquired_at = now() - interval '11 minutes'"
    )
    expire_lease(env)
    stack.watchdog_once()
    alerts = {r for (r,) in env.sql("SELECT reason FROM virtual_fitting_watchdog_alerts")}
    assert alerts == {"fallback_slot_unconfirmed", "fallback_result_unknown"}
    assert fb not in [
        d for k, d in env.runware.calls if k == "get_response"
    ]  # 동기 fallback 은 조회하지 않음
    current = row(env)
    assert current.result_payload is None and current.fallback_attempt_count == 1
    assert in_flight(env) == [(JOB, fb)] and current.fallback_task_uuid == fb


def test_s3_saved_before_the_db_image_record_is_recovered_without_regeneration(env, monkeypatch):
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    post_webhook(client(), env.runware.failure(task, "timeoutProvider"))
    real = AsyncJobStateRepository.record_fallback_image
    calls = []

    def fail_once(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise FittingDatabaseError("injected after S3 put")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(AsyncJobStateRepository, "record_fallback_image", fail_once)
    stack.postprocess_once()
    assert SYNC_KEY in env.s3.objects and row(env).fallback_image_key is None
    stack = restart(env)
    age(env, "updated_at", 700)
    expire_lease(env)
    stack.watchdog_once()  # 고정 Key HEAD(이 시도 뒤 저장) → 기록 → RUN_FALLBACK 재발행
    assert row(env).fallback_image_key == SYNC_KEY
    env.sqs.expire_visibility("ai-fitting-response")
    finish(env, stack)
    final = row(env)
    assert (
        final.result_image_key == SYNC_KEY
        and final.result_message_status is ResultMessageStatus.PUBLISHED
    )
    assert env.runware.count("sync_generate") == 1


def test_result_saved_before_publish_is_published_after_restart(env):
    stack = Stack(env).mark_running(publisher=False)  # 결과 전송 루프 없음
    task = submitted(env, stack)
    post_webhook(client(), env.runware.complete(task))
    (outcome,) = stack.postprocess_once()
    assert outcome.delete is False and row(env).next_action is NextAction.PUBLISH_RESULT
    stack = restart(env)
    stack.publish_once()
    assert env.sqs.bodies("be-fitting-response")[0]["status"] == "COMPLETED"
    env.sqs.expire_visibility("ai-fitting-response")
    (again,) = stack.postprocess_once()
    assert again.delete is True and env.runware.count("llm_comment") == 1


# --- 6. 중복·경합(여러 DB 연결·독립 조립 객체, 같은 프로세스 안의 thread) ---


def run_together(*calls):
    barrier, errors = threading.Barrier(len(calls)), []

    def run(fn):
        try:
            barrier.wait()
            fn()
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run, args=(fn,)) for fn in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not any(t.is_alive() for t in threads)
    return errors


def test_duplicate_request_messages_on_two_independent_stacks_send_once(env):
    first = Stack(env).mark_running()
    env.new_process_pool()
    second = Stack(env)
    env.send_request(JOB)
    env.send_request(JOB)
    assert run_together(first.request_once, second.request_once) == []
    assert env.runware.count("async_submit") == 1


def test_webhook_and_watchdog_on_the_same_task(env):
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    age(env, "runware_accepted_at", 200)
    body = env.runware.complete(task)
    env.runware.success_status(task)
    responses = []
    errors = run_together(
        lambda: responses.append(post_webhook(client(), body)), stack.watchdog_once
    )
    assert errors == [] and responses[0].status_code == 200
    assert costs(env) == [(task, Decimal("0.023"))] and in_flight(env) == []
    finish(env, stack)
    assert env.runware.count("llm_comment") == 1
    assert len(env.sqs.bodies("be-fitting-response")) == 1


def test_two_publishers_send_once(env):
    stack = Stack(env).mark_running()
    saved_failure(env, stack)
    env.new_process_pool()
    other = Stack(env)
    assert run_together(stack.publish_once, other.publish_once) == []
    assert len(env.sqs.bodies("be-fitting-response")) == 1
    assert row(env).result_message_attempt_count == 1


def test_shared_pruna_limit_across_jobs_and_stacks(env, monkeypatch):
    # 실제 기본 제한(진행 중 3건). 네 번째 작업은 자리를 기다리다(가짜 시계) 보내지 않고 미룬다.
    first = Stack(env).mark_running()
    env.new_process_pool()
    second = Stack(env).mark_running()
    clock = Clock()
    for stack in (first, second):
        submitter = stack.request._process.__self__._submit
        submitter._clock, submitter._sleep = clock, clock.sleep
    for job_id in (1, 2, 3, 4):
        env.send_request(job_id)
    for _ in range(2):
        assert run_together(first.request_once, second.request_once) == []
    assert env.runware.count("async_submit") == 3
    assert len(in_flight(env)) == 3
    waiting = [j for j in (1, 2, 3, 4) if row(env, j).pruna_submit_started_at is None]
    assert len(waiting) == 1 and len(env.sqs.live("be-fitting-request")) == 1


# --- 7. 준비 여부·삭제 ---


def test_received_record_alone_never_deletes_the_request_message(env):
    stack = Stack(env).mark_running()
    env.s3.head_faults.append("server_error")  # 사용자 사진 HEAD 일시 오류 → 다음 수신 때 다시
    env.send_request(JOB)
    (outcome,) = stack.request_once()
    current = row(env)
    assert outcome.delete is False and current.status is AsyncJobStatus.RECEIVED
    assert current.pruna_submit_started_at is None and env.runware.count("async_submit") == 0
    assert env.sqs.live("be-fitting-request")


def test_bad_result_queue_keeps_the_webhook_503_and_the_postprocess_message(env, monkeypatch):
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    monkeypatch.setattr(
        settings,
        "BE_FITTING_RESPONSE_QUEUE_URL",
        "https://sqs.us-east-1.amazonaws.com/1/be-fitting-response",
    )
    response = post_webhook(client(), env.runware.failure(task, "someNewCode"))
    assert response.status_code == 503
    assert response.json()["data"]["pending"][0]["next_step"] == "publish_result"
    assert row(env).result_payload["status"] == "FAILED"
    stack.publish_once()
    assert env.sqs.bodies("be-fitting-response") == []
    assert row(env).result_message_attempt_count == 0  # 설정 오류는 전송 횟수를 쓰지 않음
