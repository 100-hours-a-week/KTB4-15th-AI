"""예상된 deadline 504 는 traceback 없는 WARNING, 그 밖의 5xx 는 기존처럼 ERROR + traceback.

응답(status·code·message)은 바꾸지 않는다. 아래 표의 응답 열은 변경 전 계약 그대로다.
"""

import asyncio
import logging
import time

import httpx
import pytest

from app.body_image_validation.deadline_middleware import BodyImageDeadlineMiddleware
from app.body_image_validation.exceptions import SYSTEM_FAILURE_MESSAGE
from app.config import body_image_validation as config
from tests.test_body_image_deadline import started_worker, worker_client, worker_post
from tests.test_body_image_remote import (
    SCALE,
    TIMEOUT,
    UNAVAILABLE,
    FakeWorker,
    common,
    jpeg,
    post,
    remote_service,
    use,
)
from tests.test_body_image_remote import client as remote_client  # noqa: F401 (pytest fixture)
from tests.test_body_image_request_deadline import asgi_call, bounded, statuses
from tests.test_body_image_worker import worker_env  # noqa: F401 (pytest fixture)

APP = "app.body_image_validation"
ROUTER = "app.body_image_validation.router"
FAILED = "body image validation failed: BODY_IMAGE_WORKER_TIMEOUT"
WORKER_TIMEOUT_BODY = common("BODY_IMAGE_WORKER_TIMEOUT", SYSTEM_FAILURE_MESSAGE)
SYSTEM_FAILURE = {"code": "BACKGROUND_REMOVAL_FAILED", "data": None, "message": SYSTEM_FAILURE_MESSAGE}


def app_records(caplog):
    return [r for r in caplog.records if r.name.startswith(APP)]


def router_record(caplog):
    [record] = [r for r in caplog.records if r.name == ROUTER]
    return record


def errors(caplog):
    return [r.getMessage() for r in app_records(caplog) if r.levelno >= logging.ERROR]


# --- AI Server(remote 모드): 예상된 504 와 그 밖의 실패를 나눈다 ---


def deadline_exceeded():
    # AI Server 의 deadline(50) 이 worker 응답(60)보다 먼저 끝난다.
    fake = FakeWorker(steps=[(60 * SCALE, 200, {}, b"")])
    return remote_service(fake, timeout=50 * SCALE, deadline=50 * SCALE)


@pytest.mark.parametrize(
    ("make_service", "status", "body", "level", "traceback"),
    [
        # 예상된 deadline 보호: AI Server 가 끊었거나 worker 가 deadline header 를 보고 멈췄다.
        (deadline_exceeded, 504, TIMEOUT, logging.WARNING, False),
        (lambda: remote_service(FakeWorker(status=504, body=WORKER_TIMEOUT_BODY)), 504, TIMEOUT, logging.WARNING, False),
        # 같은 504 라도 deadline 보호가 아니면 기존 그대로다(네트워크 timeout·Modal gateway 504).
        (lambda: remote_service(FakeWorker(error=httpx.ConnectTimeout("timed out"))), 504, TIMEOUT, logging.ERROR, True),
        (lambda: remote_service(FakeWorker(status=504, body=b"gateway timeout")), 504, TIMEOUT, logging.ERROR, True),
        # 다른 5xx 도 기존 그대로다.
        (lambda: remote_service(FakeWorker(status=500, body=common("BACKGROUND_REMOVAL_FAILED",
                                                                    SYSTEM_FAILURE_MESSAGE))),
         500, SYSTEM_FAILURE, logging.ERROR, True),
        (lambda: remote_service(FakeWorker(error=httpx.ConnectError("refused"))), 503, UNAVAILABLE, logging.ERROR, True),
    ],
    ids=["ai-deadline", "worker-deadline", "network-timeout", "modal-gateway-504", "relayed-500", "unreachable"],
)
def test_ai_server_logs_only_the_deadline_stops_as_warnings(
    remote_client, monkeypatch, caplog, make_service, status, body, level, traceback  # noqa: F811
):
    use(monkeypatch, make_service())

    with caplog.at_level(logging.INFO, logger=APP):
        response = post(remote_client, jpeg())

    assert (response.status_code, response.json()) == (status, body)
    record = router_record(caplog)
    assert (record.levelno, record.exc_info is not None) == (level, traceback)
    if level == logging.WARNING:
        assert record.getMessage() == FAILED  # 기존 오류 로그와 같은 문구라 같은 검색어로 찾는다
        assert errors(caplog) == []


def test_worker_deadline_relay_leaves_the_status_in_the_summary(remote_client, monkeypatch, caplog):  # noqa: F811
    use(monkeypatch, remote_service(FakeWorker(status=504, body=WORKER_TIMEOUT_BODY)))

    with caplog.at_level(logging.INFO, logger=APP):
        post(remote_client, jpeg())

    [summary] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("body image request:")]
    assert "outcome=response" in summary and "status=504" in summary and "code=BODY_IMAGE_WORKER_TIMEOUT" in summary
    assert "gateway timed out" not in caplog.text  # worker 가 멈춘 것이지 Modal gateway timeout 이 아니다


def test_modal_gateway_504_still_logs_its_own_error(remote_client, monkeypatch, caplog):  # noqa: F811
    use(monkeypatch, remote_service(FakeWorker(status=504, body=b"gateway timeout")))

    with caplog.at_level(logging.INFO, logger=APP):
        post(remote_client, jpeg())

    assert "body image worker gateway timed out: status=504 code=None" in errors(caplog)


def test_deadline_before_the_service_is_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(config, "VALIDATION_MODE", "remote")
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 0.02)

    async def app(request, receive, send):
        await asyncio.sleep(5)

    with caplog.at_level(logging.INFO, logger=APP):
        bounded(asgi_call(BodyImageDeadlineMiddleware(app), sent=(sent := [])))

    assert statuses(sent) == [504] and b'"code":"BODY_IMAGE_WORKER_TIMEOUT"' in sent[1]["body"]
    [record] = [r for r in caplog.records if r.getMessage() == FAILED]
    assert (record.levelno, record.exc_info) == (logging.WARNING, None)
    assert errors(caplog) == []


# --- Modal worker(local 모드): RequestDeadlineExceeded 만 WARNING ---


@pytest.fixture
def keep_caplog(monkeypatch):
    """worker.create_app 의 dictConfig 가 테스트 중에 caplog handler 를 root 에서 떼지 않게 한다."""
    from app.config import logging_config

    monkeypatch.setattr(logging_config, "configure_logging", lambda: None)


def test_worker_deadline_stop_is_a_warning_without_traceback(worker_env, keep_caplog, caplog):  # noqa: F811
    started_worker(worker_env)

    with caplog.at_level(logging.INFO, logger=APP):
        response = worker_post(worker_client(), f"{time.time() - 5:.3f}")

    assert (response.status_code, response.json()) == (504, TIMEOUT)
    record = router_record(caplog)
    assert (record.levelno, record.exc_info, record.getMessage()) == (logging.WARNING, None, FAILED)
    assert any("stopped at before_processing" in r.getMessage() for r in caplog.records)  # 멈춘 단계는 그대로 남는다
    assert errors(caplog) == []


def test_other_worker_5xx_keeps_error_and_traceback(worker_env, keep_caplog, monkeypatch, caplog):  # noqa: F811
    started_worker(worker_env)
    remover = worker_env["manager"].runtime.background_remover

    def broken(image):
        raise RuntimeError("model failed")

    monkeypatch.setattr(remover, "remove", broken)

    with caplog.at_level(logging.INFO, logger=APP):
        response = worker_post(worker_client(), f"{time.time() + 60:.3f}")

    assert (response.status_code, response.json()) == (500, SYSTEM_FAILURE)
    record = router_record(caplog)
    assert (record.levelno, record.exc_info is not None) == (logging.ERROR, True)
    assert record.getMessage() == "body image validation failed: BACKGROUND_REMOVAL_FAILED"
