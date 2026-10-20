import pytest

from app.config import settings
from app.virtual_fitting import diagnostics as diag
from app.virtual_fitting import webhook_router
from app.virtual_fitting.exceptions import FittingDatabaseError
from tests.test_virtual_fitting_diagnostic_logs import (  # noqa: F401 — fixture 재사용
    env,
    events,
    rendered,
)
from tests.test_virtual_fitting_v2_scenarios import (
    JOB,
    age,
    client,
    finish,
    run_together,
    submitted,
)
from tests.v2_integration_harness import WEBHOOK_TOKEN, Stack, post_webhook

SENSITIVE = (WEBHOOK_TOKEN, "im.runware.ai", "imageURL", "fake external message", "body-images/")


def chain(buffer, request_id):
    return [e for e in events(buffer) if e.get("request_id") == request_id]


def no_secrets(buffer):
    text = buffer.getvalue()
    for value in SENSITIVE:
        assert value not in text, value


def follow_up_ids(stack_env):
    return {
        mid
        for op, queue, mid in stack_env.sqs.log
        if op == "send" and queue == "ai-fitting-response"
    }


# 1. 정상 Webhook: 수신 → 검증 → 처리 → 응답이 request_id 하나로 이어진다


def test_normal_webhook_links_received_validation_processing_and_response(env, rendered):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack, JOB)
    response = post_webhook(client(), env.runware.complete(task))
    assert response.status_code == 200 and response.json()["code"] == "WEBHOOK_PROCESSED"

    (received,) = events(rendered, "webhook_http_received")
    request_id = received["request_id"]
    assert received["source"] == "webhook" and received["run_id"] == diag.RUN_ID
    names = [e["event"] for e in chain(rendered, request_id)]
    assert names == [
        "webhook_http_received",
        "webhook_validation",
        "runware_result_processed",
        "webhook_http_completed",
    ]
    (validation,) = events(rendered, "webhook_validation")
    assert (validation["stage"], validation["result"], validation["reason"]) == (
        "body",
        "passed",
        "ok",
    )
    assert validation["task_uuid"] == task and validation["event_count"] == "1"
    (processed,) = events(rendered, "runware_result_processed")
    assert processed["source"] == "webhook" and processed["fitting_job_id"] == str(JOB)
    assert (processed["outcome"], processed["reason"]) == ("applied", "state_recorded")
    assert processed["state_change"] == "applied"
    assert processed["follow_up"] == "postprocess_succeeded"
    assert processed["follow_up_message_id"] in follow_up_ids(env)
    # applied 는 이 알림의 처리 결과일 뿐이다: 작업은 아직 후처리가 남았다
    assert processed["job_status"] == "POSTPROCESSING" and processed["next_action"] == "POSTPROCESS"
    assert int(processed["elapsed_ms"]) >= 0
    (completed,) = events(rendered, "webhook_http_completed")
    assert completed["http_status"] == "200" and completed["code"] == "WEBHOOK_PROCESSED"
    assert int(completed["elapsed_ms"]) >= int(processed["elapsed_ms"])
    # 실제 formatter 출력 형식(KST 시각·logger 이름·vf_diag)
    assert (
        " INFO app.virtual_fitting.diagnostics - vf_diag event=webhook_http_received "
        in received["_line"]
    )
    no_secrets(rendered)


# 2. 토큰·본문 검증 실패와 처리 오류: 기존 HTTP 응답 그대로, 수신 로그는 항상 남는다


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ({}, "token_missing"),
        ({"token": "wrong"}, "token_mismatch"),
        ([("token", WEBHOOK_TOKEN), ("token", WEBHOOK_TOKEN)], "token_multiple"),
    ],
)
def test_token_rejection_is_logged_with_unchanged_401(env, rendered, params, reason):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack, JOB)
    response = client().post(
        "/api/v2/webhooks/runware", params=params, content=env.runware.complete(task)
    )
    assert response.status_code == 401 and response.json()["code"] == "UNAUTHORIZED"
    (received,) = events(rendered, "webhook_http_received")
    names = [e["event"] for e in chain(rendered, received["request_id"])]
    assert names == ["webhook_http_received", "webhook_validation", "webhook_http_completed"]
    (validation,) = events(rendered, "webhook_validation")
    assert (validation["stage"], validation["result"], validation["reason"]) == (
        "token",
        "rejected",
        reason,
    )
    assert "task_uuid" not in validation  # 검증 전에는 본문에서 아무것도 꺼내지 않는다
    (completed,) = events(rendered, "webhook_http_completed")
    assert (completed["http_status"], completed["code"]) == ("401", "UNAUTHORIZED")
    assert events(rendered, "runware_result_processed") == []
    no_secrets(rendered)


def test_missing_token_config_body_errors_and_processing_errors_keep_their_responses(
    env,  # noqa: F811
    rendered,  # noqa: F811
    monkeypatch,
):
    stack = Stack(env).mark_running()
    task = submitted(env, stack, JOB)
    cases = []

    bad = post_webhook(client(), b'{"not json')
    cases.append((bad, "body", "invalid_json"))
    unknown = post_webhook(
        client(), env.runware.complete(task).replace(task.encode(), b"0" * 8 + task[8:].encode())
    )
    cases.append((unknown, "body", "ok"))
    monkeypatch.setattr(settings, "WEBHOOK_TOKEN", "")
    unconfigured = post_webhook(client(), env.runware.complete(task))
    cases.append((unconfigured, "config", "token_not_configured"))
    monkeypatch.setattr(settings, "WEBHOOK_TOKEN", WEBHOOK_TOKEN)

    def broken():
        raise FittingDatabaseError("down postgresql://user:SECRETPW@db.internal/x")

    monkeypatch.setattr(webhook_router, "get_connection_pool", broken)
    db_down = post_webhook(client(), env.runware.complete(task))
    cases.append((db_down, "body", "ok"))

    assert [(r.status_code, r.json()["code"]) for r, _, _ in cases] == [
        (400, "INVALID_REQUEST"),
        (400, "WEBHOOK_UNKNOWN_TASK"),
        (500, "WEBHOOK_NOT_CONFIGURED"),
        (500, "DATABASE_ERROR"),
    ]
    received = events(rendered, "webhook_http_received")
    assert len(received) == 4 and len({e["request_id"] for e in received}) == 4
    for e, (response, stage, reason) in zip(received, cases, strict=True):
        line = chain(rendered, e["request_id"])
        (validation,) = [x for x in line if x["event"] == "webhook_validation"]
        assert (validation["stage"], validation["reason"]) == (stage, reason)
        (completed,) = [x for x in line if x["event"] == "webhook_http_completed"]
        assert completed["http_status"] == str(response.status_code)
        assert completed["code"] == response.json()["code"]
    # 등록되지 않은 task: 처리 결과는 failed(부작용 없음), HTTP 200 이 아님
    (processed,) = events(rendered, "runware_result_processed")
    assert (processed["outcome"], processed["reason"], processed["state_change"]) == (
        "failed",
        "unknown_task",
        "none",
    )
    assert processed["request_id"] == received[1]["request_id"]
    no_secrets(rendered)
    assert "SECRETPW" not in rendered.getvalue()


# 3. Watchdog 경로: HTTP 수신 로그가 없고, 조회와 처리에 source=watchdog


def test_watchdog_path_never_logs_http_receipt(env, rendered):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack, JOB)
    age(env, "runware_accepted_at", 200)
    env.runware.complete(task)
    env.runware.success_status(task)
    stack.watchdog_once()

    assert events(rendered, "webhook_http_received") == []
    assert events(rendered, "webhook_http_completed") == []
    (checked,) = events(rendered, "runware_status_checked")
    assert checked["source"] == "watchdog" and checked["task_uuid"] == task
    assert checked["fitting_job_id"] == str(JOB) and checked["check_reason"] == "waiting"
    assert (checked["result"], checked["called"], checked["http_status"]) == (
        "succeeded",
        "true",
        "200",
    )
    assert int(checked["elapsed_ms"]) >= 0
    (processed,) = events(rendered, "runware_result_processed")
    assert processed["source"] == "watchdog" and "request_id" not in processed
    assert processed["outcome"] == "applied" and processed["next_action"] == "POSTPROCESS"
    no_secrets(rendered)


# 4. 중복·늦은 결과는 ignored(HTTP 200 이어도 applied 가 아님)


def test_duplicate_and_late_results_are_ignored_not_applied(env, rendered):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack, JOB)
    body = env.runware.complete(task)
    assert post_webhook(client(), body).status_code == 200
    assert post_webhook(client(), body).status_code == 200  # Runware 재전송
    finish(env, stack)
    assert env.job(JOB).status.value == "COMPLETED"
    assert post_webhook(client(), body).status_code == 200  # 완료 뒤 늦게 온 재전송

    first, duplicate, late = events(rendered, "runware_result_processed")
    assert first["outcome"] == "applied"
    assert (duplicate["outcome"], duplicate["reason"]) == ("ignored", "duplicate")
    # 기존 정책: 상태가 이미 바뀌었어도 후속 작업을 기다리면 다시 발행한다(재전송 복구). ignored 와 별개로 보인다.
    assert duplicate["follow_up"] == "postprocess_succeeded"
    assert late["follow_up"] == "not_needed"
    assert (late["outcome"], late["reason"]) == ("ignored", "already_progressed")
    assert late["job_status"] == "COMPLETED" and late["next_action"] == "none"
    assert len({e["request_id"] for e in (first, duplicate, late)}) == 3


# 5. Webhook·Watchdog 경합: 식별자와 출처가 섞이지 않는다


def test_webhook_and_watchdog_race_keep_ids_and_sources_apart(env, rendered):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack, JOB)
    age(env, "runware_accepted_at", 200)
    body = env.runware.complete(task)
    env.runware.success_status(task)
    responses = []
    errors = run_together(
        lambda: responses.append(post_webhook(client(), body)), stack.watchdog_once
    )
    assert errors == [] and responses[0].status_code == 200

    (received,) = events(rendered, "webhook_http_received")
    processed = events(rendered, "runware_result_processed")
    by_source = {}
    for e in processed:
        by_source.setdefault(e["source"], []).append(e)
    (from_webhook,) = by_source["webhook"]
    assert from_webhook["request_id"] == received["request_id"]
    for e in by_source.get("watchdog", []):
        assert "request_id" not in e
    for e in events(rendered, "runware_status_checked"):
        assert e["source"] == "watchdog" and "request_id" not in e
    # 상태 전환은 한 번(어느 경로가 먼저였든)
    assert [e["outcome"] for e in processed].count("applied") == 1
    assert {e["source"] for e in processed} <= {"webhook", "watchdog"}


# 6. DB 반영 뒤 후속 큐 발행 실패: failed 이지만 state_change=applied 로 부분 성공을 구분


@pytest.mark.parametrize(
    ("fault", "follow_up"),
    [("throttled", "postprocess_failed"), ("stored_timeout", "postprocess_unknown")],
)
def test_recorded_state_but_failed_follow_up_is_distinguished(env, rendered, fault, follow_up):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack, JOB)
    env.sqs.fault("ai-fitting-response", "send", fault)
    response = post_webhook(client(), env.runware.complete(task))
    assert response.status_code == 503 and response.json()["code"] == "WEBHOOK_FOLLOW_UP_FAILED"
    (processed,) = events(rendered, "runware_result_processed")
    assert (processed["outcome"], processed["reason"]) == ("failed", "follow_up_publish_failed")
    assert processed["state_change"] == "applied" and processed["follow_up"] == follow_up
    assert processed["next_action"] == "POSTPROCESS"
    assert "WARNING app.virtual_fitting.diagnostics" in processed["_line"]
    (completed,) = events(rendered, "webhook_http_completed")
    assert (completed["http_status"], completed["code"]) == ("503", "WEBHOOK_FOLLOW_UP_FAILED")


# 기록은 했지만 남은 일(결과 발행)을 이어 갈 경로가 없음: deferred


def test_recorded_but_follow_up_path_not_ready_is_deferred(env, rendered):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack, JOB)
    stack.mark_running(publisher=False, watchdog=False)  # 결과 전송·복구 경로 없음
    response = post_webhook(client(), env.runware.failure(task, "someNewCode"))
    assert response.status_code == 503 and response.json()["code"] == "WEBHOOK_FOLLOW_UP_NOT_READY"
    (processed,) = events(rendered, "runware_result_processed")
    assert processed["outcome"] == "deferred"
    assert processed["reason"] == "follow_up_not_ready_publish_result"
    assert processed["state_change"] == "applied" and processed["next_action"] == "PUBLISH_RESULT"


# 7. 로그 출력이 실패해도 업무 처리는 그대로


def test_logging_failure_does_not_change_the_webhook_result(env, rendered, monkeypatch):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack, JOB)

    def explode(*args, **kwargs):
        raise RuntimeError("log sink down")

    before = rendered.getvalue().count(" vf_diag ")
    monkeypatch.setattr(diag.logger, "log", explode)
    response = post_webhook(client(), env.runware.complete(task))
    assert response.status_code == 200
    assert response.json()["data"]["results"][0]["outcome"] == "applied"
    assert env.job(JOB).next_action.value == "POSTPROCESS"
    assert rendered.getvalue().count(" vf_diag ") == before  # 출력은 실패했고 처리 결과는 같다


# 8. 처리 시간 초과·예기치 않은 처리 오류도 수신 → 검증 → 응답이 같은 request_id 로 이어진다


def _any_success_body() -> bytes:
    import json
    import uuid

    return json.dumps(
        {
            "taskType": "imageInference",
            "taskUUID": str(uuid.uuid4()),
            "imageUUID": str(uuid.uuid4()),
            "imageURL": "https://im.runware.ai/image/fake/result.jpg",
        }
    ).encode()


def test_timeout_and_unhandled_error_are_linked_to_the_received_request(
    env,  # noqa: F811
    rendered,  # noqa: F811
    monkeypatch,
):
    import threading

    from fastapi.testclient import TestClient

    from app.main import app

    release = threading.Event()

    def slow(events_):
        release.wait(5)
        return []

    monkeypatch.setattr(webhook_router, "WEBHOOK_DEADLINE_SECONDS", 0.2)
    monkeypatch.setattr(webhook_router, "_process", slow)
    timed_out = post_webhook(client(), _any_success_body())
    release.set()

    def broken(events_):
        raise RuntimeError("unexpected token=SHOULD_NOT_APPEAR")

    monkeypatch.setattr(webhook_router, "_process", broken)
    crashed = post_webhook(TestClient(app, raise_server_exceptions=False), _any_success_body())

    assert (timed_out.status_code, timed_out.json()["code"]) == (503, "WEBHOOK_TIMEOUT")
    assert crashed.status_code == 500
    first, second = events(rendered, "webhook_http_received")
    for received, (status, code) in (
        (first, ("503", "WEBHOOK_TIMEOUT")),
        (second, ("500", "unhandled_exception")),
    ):
        names = [e["event"] for e in chain(rendered, received["request_id"])]
        assert names == ["webhook_http_received", "webhook_validation", "webhook_http_completed"]
        (completed,) = [
            e
            for e in events(rendered, "webhook_http_completed")
            if e["request_id"] == received["request_id"]
        ]
        assert (completed["http_status"], completed["code"]) == (status, code)
    no_secrets(rendered)
    assert "SHOULD_NOT_APPEAR" not in rendered.getvalue()
