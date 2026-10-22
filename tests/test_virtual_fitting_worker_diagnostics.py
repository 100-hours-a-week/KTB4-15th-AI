"""SQS 소비 Worker(요청·후처리)가 메시지를 처리 대상으로 고른 시점의 진단 로그(vf_diag message_selected).

실제 lifespan 으로 네 루프를 켜고(설정 True), 실제 RequestWorker._run 루프가 가짜 SQS 에서 메시지를 받게 한다.
외부 경계는 tests/v2_integration_harness 의 가짜(SQS·S3·Runware), DB 는 격리 PostgreSQL schema 다. 네트워크는 막는다.
"""

import os
import subprocess
import sys
import time

from fastapi.testclient import TestClient

from app.config import settings
from app.virtual_fitting import diagnostics as diag
from tests.test_virtual_fitting_diagnostic_logs import (  # noqa: F401 — fixture 재사용
    FLAGS,
    JOB,
    env,
    events,
    fastapi_app,
    rendered,
)
from tests.v2_integration_harness import (
    QUEUE_URLS,
    USER_IMAGE_KEY,
    WEBHOOK_TOKEN,
    post_webhook,
    wait_until,
)

DECISIONS = ("message_processing_result", "message_delete_result")


def enable_all(monkeypatch):
    for name in FLAGS:
        monkeypatch.setattr(settings, name, True)


def decision_for(buffer, worker, message_id):
    return [
        e
        for e in events(buffer)
        if e["event"] in DECISIONS
        and e.get("worker") == worker
        and e.get("message_id") == message_id
    ]


def receipts(stack_env):
    return [m.receipt for messages in stack_env.sqs.queues.values() for m in messages if m.receipt]


def assert_no_sensitive(stack_env, buffer):
    lines = "\n".join(e["_line"] for e in events(buffer))
    for value in (WEBHOOK_TOKEN, USER_IMAGE_KEY, "user_image_key", '"products"', "requested_at"):
        assert value not in lines, value
    for value in ("image.example.test", "im.runware.ai", "X-Amz-"):
        assert value not in lines, value
    for receipt in receipts(stack_env):
        assert receipt not in buffer.getvalue()


# 1. 요청 Worker·후처리 Worker: 선택 시점 로그(프로세스·Worker·메시지 식별)와 처리 결과·삭제/유지가 이어진다


def test_request_and_postprocess_workers_log_selection_linked_to_the_decision(
    env,  # noqa: F811
    rendered,  # noqa: F811
    monkeypatch,
):
    enable_all(monkeypatch)
    with TestClient(fastapi_app) as client:
        request_message = env.send_request(JOB)
        wait_until(
            lambda: decision_for(rendered, "request", request_message), what="request decision"
        )
        task = env.job(JOB).task_uuid
        assert post_webhook(client, env.runware.complete(task)).status_code == 200
        (follow_up_message,) = [
            mid for op, queue, mid in env.sqs.log if op == "send" and queue == "ai-fitting-response"
        ]
        wait_until(
            lambda: decision_for(rendered, "postprocess", follow_up_message),
            what="postprocess decision",
        )

    selected = {e["worker"]: e for e in events(rendered, "message_selected")}
    assert set(selected) == {"request", "postprocess"}
    for worker, message_id in (("request", request_message), ("postprocess", follow_up_message)):
        queue = "be-fitting-request" if worker == "request" else "ai-fitting-response"
        received = [
            e for e in events(rendered, "message_received") if e.get("message_id") == message_id
        ]
        validated = [
            e for e in events(rendered, "message_validated") if e.get("message_id") == message_id
        ]
        assert len(received) == len(validated) == 1
        assert received[0]["queue"] == validated[0]["queue"] == queue
        assert validated[0]["message_valid"] == "true"
        e = selected[worker]
        assert e["component"] == f"{worker}_worker" and e["message_id"] == message_id
        assert e["queue"] == queue
        assert e["run_id"] == diag.RUN_ID and e["pid"] == str(os.getpid())
        assert e["hostname"] == diag.hostname() and e["receive_count"] == "1"
        assert e["message_valid"] == "true" and e["fitting_job_id"] == str(JOB)
        # 같은 프로세스·Worker·메시지의 처리 판단과 실제 삭제 결과
        stages = decision_for(rendered, worker, message_id)
        (processing,) = [stage for stage in stages if stage["event"] == "message_processing_result"]
        assert processing["run_id"] == e["run_id"] and processing["component"] == e["component"]
        assert processing["fitting_job_id"] == str(JOB)
        if worker == "request":
            assert int(processing["processing_elapsed_ms"]) >= 0
        assert processing["delete_decided"] == "true" and processing["action"] != "none"
        (deleted,) = [stage for stage in stages if stage["event"] == "message_delete_result"]
        assert deleted["run_id"] == e["run_id"] and deleted["message_id"] == e["message_id"]
        assert deleted["sqs_result"] == "succeeded"
    # 후처리 메시지는 검증된 task_uuid 를 이미 갖고 있다. 요청 메시지에는 task 가 아직 없다.
    assert selected["postprocess"]["task_uuid"] == task and "task_uuid" not in selected["request"]
    # 선택은 처리 결과보다 먼저 찍힌다
    names = [e["event"] for e in events(rendered) if e.get("message_id") == request_message]
    assert names.index("message_received") < names.index("message_validated")
    assert names.index("message_validated") < names.index("message_selected")
    assert names.index("message_selected") < names.index("message_processing_result")
    assert names.index("message_processing_result") < names.index("message_delete_result")
    assert_no_sensitive(env, rendered)


# 2. 형식 오류 메시지: 신뢰할 수 없는 값은 꺼내지 않는다(message_valid=false, 작업 식별자 없음)


def test_invalid_message_selection_has_no_untrusted_identifiers(env, rendered, monkeypatch):  # noqa: F811
    enable_all(monkeypatch)
    payload = env.request(JOB)
    payload["products"] = "not-a-list"  # fitting_job_id 는 읽을 수 있지만 메시지는 형식 오류
    message_id = env.sqs.put("be-fitting-request", payload)
    with TestClient(fastapi_app):
        wait_until(lambda: decision_for(rendered, "request", message_id), what="invalid decision")
    (selected,) = events(rendered, "message_selected")
    assert selected["message_id"] == message_id and selected["message_valid"] == "false"
    assert "fitting_job_id" not in selected and "task_uuid" not in selected
    (validation,) = [
        e for e in events(rendered, "message_validation_failed") if e["message_id"] == message_id
    ]
    assert validation["message_valid"] == "false"
    assert "fitting_job_id" not in validation and "task_uuid" not in validation
    (decision,) = decision_for(rendered, "request", message_id)
    assert (
        decision["event"] == "message_processing_result"
    )  # 기존 정책: 형식 오류는 DLQ 로 가도록 남겨 둔다
    assert decision["delete_decided"] == "false"
    names = [e["event"] for e in events(rendered) if e.get("message_id") == message_id]
    assert names.index("message_received") < names.index("message_validation_failed")
    assert names.index("message_validation_failed") < names.index("message_selected")
    assert names.index("message_selected") < names.index("message_processing_result")
    assert_no_sensitive(env, rendered)


# 3. 진단 출력이 실패해도 Worker 처리 결과는 같다


def test_logging_failure_does_not_change_worker_processing(env, rendered, monkeypatch):  # noqa: F811
    enable_all(monkeypatch)

    def explode(*args, **kwargs):
        raise RuntimeError("log sink down")

    monkeypatch.setattr(diag.logger, "log", explode)
    with TestClient(fastapi_app):
        env.send_request(JOB)
        wait_until(
            lambda: env.sqs.live("be-fitting-request") == [] and env.job(JOB) is not None,
            what="request processed and deleted",
        )
        time.sleep(0.1)
    current = env.job(JOB)
    assert current.task_uuid is not None and env.runware.count("async_submit") == 1
    assert " vf_diag " not in rendered.getvalue()
    assert QUEUE_URLS["be-fitting-request"] not in rendered.getvalue()


# 4. 로깅 설정: 운영과 같은 순서(uvicorn 설정 → app.main import)에서 vf_diag 가 실제 stderr 로 나간다


def test_diagnostics_reach_the_application_log_output():
    code = (
        "from uvicorn.config import Config\n"
        "Config('app.main:app')  # 컨테이너 CMD(uvicorn app.main:app)가 먼저 하는 로깅 설정\n"
        "import app.main\n"
        "from app.virtual_fitting import diagnostics as diag\n"
        "diag.event('message_selected', component='request_worker', worker='request', "
        "**diag.process_fields(), message_id='m-1')\n"
    )
    child_env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=child_env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert out.returncode == 0, out.stderr[-2000:]
    (line,) = [x for x in out.stderr.splitlines() if " vf_diag " in x]
    assert " INFO app.virtual_fitting.diagnostics - vf_diag event=message_selected run_id=" in line
    assert (
        "+09:00 " in line
        and " worker=request " in line
        and " pid=" in line
        and " hostname=" in line
    )
