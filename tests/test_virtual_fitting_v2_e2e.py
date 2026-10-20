"""16단계: 실제 FastAPI lifespan 이 돌리는 루프(요청 Worker·후처리 Worker·결과 전송 루프·Watchdog)로 전체 흐름 검증.

외부 경계만 가짜다(tests/v2_integration_harness.py): SQS·S3 boto3 client, Runware HTTP(urlopen), 네트워크 차단.
DB 는 격리 PostgreSQL schema. 실제 AWS·Runware·OpenAI·BE 는 부르지 않는다(BE 결과 큐 도착까지만 확인).
"""

import os
import threading
import uuid
from decimal import Decimal

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    ResultMessageStatus,
)
from tests.v2_integration_harness import (
    COMMENT,
    TITLE,
    USER_IMAGE_KEY,
    WEBHOOK_TOKEN,
    close_env,
    make_env,
    post_webhook,
    wait_until,
)

JOB = 9001


@pytest.fixture
def env(monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL 이 없다")
    try:
        admin = psycopg.connect(url, autocommit=True, connect_timeout=3)
    except psycopg.Error as error:
        pytest.skip(f"테스트 DB 에 연결할 수 없다: {error}")
    schema = f"vf_v2e2e_{uuid.uuid4().hex[:12]}"
    admin.execute(f"CREATE SCHEMA {schema}")
    e = make_env(monkeypatch, url, schema)
    try:
        yield e
    finally:
        close_env(e)
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()


def enable_all(monkeypatch):
    for name in (
        "VIRTUAL_FITTING_REQUEST_WORKER_ENABLED",
        "VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED",
        "VIRTUAL_FITTING_RESULT_PUBLISHER_ENABLED",
        "VIRTUAL_FITTING_WATCHDOG_ENABLED",
    ):
        monkeypatch.setattr(settings, name, True)


def app():
    from app.main import app as fastapi_app

    return fastapi_app


def row(env, job_id=JOB):
    return env.job(job_id)


def counts(env, job_id=JOB):
    return env.sql(
        "SELECT s3_check_attempt_count, llm_generation_attempt_count, fallback_attempt_count, "
        "result_message_attempt_count, worker_lease_token FROM virtual_fitting_async_jobs "
        "WHERE fitting_job_id = %s",
        (job_id,),
    )[0]


def costs(env):
    return env.sql("SELECT task_uuid::text, fitting_job_id, cost FROM virtual_fitting_task_costs")


def in_flight(env):
    return env.sql("SELECT fitting_job_id, task_uuid::text FROM virtual_fitting_pruna_in_flight")


def test_defaults_keep_every_v2_loop_disabled():
    for name in (
        "VIRTUAL_FITTING_REQUEST_WORKER_ENABLED",
        "VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED",
        "VIRTUAL_FITTING_RESULT_PUBLISHER_ENABLED",
        "VIRTUAL_FITTING_WATCHDOG_ENABLED",
    ):
        assert getattr(settings, name) is False, name


def test_normal_flow_through_the_real_lifespan_loops(env, monkeypatch):
    enable_all(monkeypatch)
    env.send_request(JOB)
    with TestClient(app()) as client:
        state = client.app.state
        assert state.request_worker and state.postprocess_worker and state.result_publisher
        assert state.watchdog and state.postprocess_worker_status == "running"

        # 요청 Worker: 수신·등록·사전 확인·식별자 선저장·공유 제한·Pruna 비동기 전송
        task = wait_until(lambda: next(iter(env.runware.submissions), None), what="async submit")
        submission = env.runware.submissions[task]
        accepted = wait_until(
            lambda: (r := row(env)) and r.runware_accepted_at and r, what="accept"
        )
        assert accepted.task_uuid == task and accepted.status is AsyncJobStatus.WAITING_PRUNA
        assert submission.person_key == USER_IMAGE_KEY  # 사람 이미지 = 사용자 원본(Presigned GET)
        assert (
            submission.upload_key
            == accepted.async_image_key
            == (f"virtual-fitting/results/{JOB}/pruna-async.jpg")
        )
        assert submission.webhook_url.endswith(f"token={WEBHOOK_TOKEN}")
        # Watchdog 이 실제로 돌고 있어 접수 대기(AWAIT_WEBHOOK)를 이어 볼 수 있으므로 요청 메시지를 지운다.
        wait_until(lambda: not env.sqs.live("be-fitting-request"), what="request deleted")

        # Runware 가 결과를 PUT 하고 완료 Webhook 을 보낸다(실제 Router).
        response = post_webhook(client, env.runware.complete(task, cost=0.023))
        assert response.status_code == 200, response.text

        # 후처리 Worker(이미지 확인 → 코멘트 → 제목 → 저장) → 결과 전송 루프(BE 결과 큐)
        wait_until(lambda: env.sqs.bodies("be-fitting-response"), what="BE result")
        final = wait_until(
            lambda: (r := row(env)).result_message_status is ResultMessageStatus.PUBLISHED and r,
            what="published",
        )
        wait_until(lambda: not env.sqs.live("ai-fitting-response"), what="internal deleted")

    (payload,) = env.sqs.bodies("be-fitting-response")
    assert payload == {  # 합의한 평평한 JSON(code/data/message 로 감싸지 않음)
        "fitting_job_id": JOB,
        "status": "COMPLETED",
        "result_image_key": f"virtual-fitting/results/{JOB}/pruna-async.jpg",
        "llm_title": TITLE,
        "llm_comment": COMMENT,
    }
    assert final.status is AsyncJobStatus.COMPLETED and final.next_action is None
    assert final.user_image_key == USER_IMAGE_KEY  # 원본·비동기·최종 Key 구분 유지
    assert final.result_image_key == final.async_image_key == final.verified_image_key
    # LLM: 코멘트에는 생성된 결과 이미지(Presigned GET)만, 제목에는 그 코멘트만
    comment_request, title_request = env.runware.llm_requests
    image_url = comment_request["messages"][1]["content"][1]["image_url"]["url"]
    assert f"/virtual-fitting/results/{JOB}/pruna-async.jpg?" in image_url
    assert USER_IMAGE_KEY not in image_url
    assert title_request["messages"][1]["content"] == f"코디 설명: {COMMENT}"
    # 비용·횟수·자리·실행 권한
    assert costs(env) == [(task, JOB, Decimal("0.023"))]
    assert counts(env) == (1, 2, 0, 1, None)
    assert in_flight(env) == []
    assert env.runware.count("async_submit") == 1 and env.runware.count("sync_generate") == 0
    assert env.network_attempts == []  # 실제 네트워크로 나간 시도 없음
    assert env.app_pools[0].closed  # 종료 때 루프들을 멈춘 뒤 pool 을 닫았다


def test_restart_between_accept_and_webhook_continues_from_db(env, monkeypatch):
    enable_all(monkeypatch)
    env.send_request(JOB)
    with TestClient(app()):
        task = wait_until(lambda: (r := row(env)) and r.runware_accepted_at and r.task_uuid)
        wait_until(lambda: not env.sqs.live("be-fitting-request"))
    first_pool = env.app_pools[-1]
    assert first_pool.closed
    # 재시작: 새 pool·새 Worker 객체. DB·가짜 S3·SQS·Runware 상태는 그대로.
    env.new_process_pool()
    with TestClient(app()) as client:
        assert post_webhook(client, env.runware.complete(task)).status_code == 200
        wait_until(lambda: row(env).result_message_status is ResultMessageStatus.PUBLISHED)
    assert env.runware.count("async_submit") == 1  # 재시작해도 다시 생성하지 않음
    assert len(env.sqs.bodies("be-fitting-response")) == 1
    assert row(env).task_uuid == task and counts(env) == (1, 2, 0, 1, None)


def test_fallback_flow_through_the_real_loops_and_a_late_async_webhook(env, monkeypatch):
    enable_all(monkeypatch)
    env.send_request(JOB)
    with TestClient(app()) as client:
        task = wait_until(lambda: (r := row(env)) and r.runware_accepted_at and r.task_uuid)
        # 확인된 실패(timeoutProvider) → RUN_FALLBACK → 실제 RunFallbackHandler(V1 동기 생성) → 결과 전송
        response = post_webhook(client, env.runware.failure(task, "timeoutProvider"))
        assert response.status_code == 200, response.text
        final = wait_until(
            lambda: (r := row(env)).result_message_status is ResultMessageStatus.PUBLISHED and r
        )
        # 늦은 비동기 성공 알림: 결과를 바꾸지 않는다(비용은 그 task 로 1회 기록)
        late = post_webhook(client, env.runware.complete(task, cost=0.02))
        assert late.status_code == 200
        assert row(env) == final
    (payload,) = env.sqs.bodies("be-fitting-response")
    assert payload["status"] == "COMPLETED"
    assert payload["result_image_key"] == f"virtual-fitting/results/{JOB}/pruna-sync.jpg"
    assert final.fallback_task_uuid not in (None, task)  # 비동기 UUID 와 fallback UUID 분리
    assert env.runware.count("sync_generate") == 1 and env.runware.count("async_submit") == 1
    recorded = {t: c for t, _, c in costs(env)}
    assert recorded == {final.fallback_task_uuid: Decimal("0.015"), task: Decimal("0.02")}
    assert in_flight(env) == []


def test_auth_hold_with_every_loop_running_survives_redelivery_and_restart(env, monkeypatch):
    enable_all(monkeypatch)
    # 요청 Worker 의 Visibility 를 짧게 두어 재전달을 실제 루프가 다시 받게 한다(테스트 설정).
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_REQUEST_VISIBILITY_SECONDS", 1)
    env.send_request(JOB)
    with TestClient(app()) as client:
        task = wait_until(lambda: (r := row(env)) and r.runware_accepted_at and r.task_uuid)
        wait_until(lambda: not env.sqs.live("be-fitting-request"))
        response = post_webhook(client, env.runware.failure(task, "invalidApiKey"))
        assert response.status_code == 503
        assert response.json()["data"]["pending"][0]["next_step"] == "operator_review"
        # 같은 요청이 다시 온다(BE 재전송·삭제 실패 후 재수신과 같은 상황).
        env.send_request(JOB)
        wait_until(
            lambda: any(m.receive_count >= 2 for m in env.sqs.live("be-fitting-request")),
            what="redelivered",
        )
        assert env.sqs.live("be-fitting-request")  # 지우지 않는다
    env.new_process_pool()
    with TestClient(app()):
        wait_until(
            lambda: any(m.receive_count >= 3 for m in env.sqs.live("be-fitting-request")),
            what="redelivered after restart",
        )
        assert env.sqs.live("be-fitting-request")
    held = row(env)
    assert (held.auth_hold_task_uuid, held.auth_hold_reason) == (task, "invalidApiKey")
    assert held.result_payload is None and held.status is AsyncJobStatus.WAITING_PRUNA
    assert env.runware.count("async_submit") == 1 and env.runware.count("sync_generate") == 0
    assert env.sqs.bodies("be-fitting-response") == []


def test_shutdown_waits_for_an_in_progress_submit_and_closes_the_pool_last(env, monkeypatch):
    enable_all(monkeypatch)
    env.runware.gate = threading.Event()
    env.send_request(JOB)
    client = TestClient(app())
    client.__enter__()
    try:
        assert env.runware.entered.wait(10)  # 생성 요청 HTTP 가 진행 중
        timer = threading.Timer(0.5, env.runware.gate.set)
        timer.start()
    finally:
        client.__exit__(None, None, None)  # 종료: 진행 중 호출이 끝나고 기록한 뒤 pool 을 닫는다
    timer.join()
    current = env.sql(
        "SELECT runware_accepted_at IS NOT NULL, status FROM virtual_fitting_async_jobs "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    assert current == [(True, "WAITING_PRUNA")]
    assert env.app_pools[0].closed and env.runware.count("async_submit") == 1
    # 재시작 뒤 같은 작업을 다시 보내지 않는다.
    env.runware.gate = None
    env.new_process_pool()
    env.sqs.expire_visibility("be-fitting-request")
    with TestClient(app()):
        wait_until(lambda: not env.sqs.live("be-fitting-request"), what="kept message handled")
    assert env.runware.count("async_submit") == 1


def test_startup_does_not_alert_running_components_as_stopped(env, monkeypatch):
    # 16단계에서 재현한 결함: Watchdog 이 먼저 시작해, 막 start() 한(아직 Task 가 돌기 전) Worker 를 "멈춤"으로 알렸다.
    enable_all(monkeypatch)
    with TestClient(app()) as client:
        watchdog = client.app.state.watchdog
        wait_until(lambda: watchdog.status.cycles >= 3, what="watchdog cycles")
        workers = (client.app.state.request_worker, client.app.state.postprocess_worker)
        assert all(worker.status.running for worker in workers)
    alerts = env.sql("SELECT reason, stage FROM virtual_fitting_watchdog_alerts")
    assert [a for a in alerts if a[0] == "component_stopped"] == []


def test_a_crashed_component_is_still_alerted(env, monkeypatch):
    enable_all(monkeypatch)
    with TestClient(app()) as client:
        worker = client.app.state.postprocess_worker
        wait_until(lambda: worker.status.running, what="postprocess running")
        worker.status.crashed = True  # 비정상 종료 상태(테스트에서 표시)
        wait_until(
            lambda: env.sql(
                "SELECT 1 FROM virtual_fitting_watchdog_alerts WHERE reason = 'component_stopped' "
                "AND stage = 'component_postprocess_worker'"
            ),
            what="crash alert",
        )
