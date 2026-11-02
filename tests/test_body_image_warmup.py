"""AI Server → Modal worker GET /health warm-up 호출. 실제 Modal 은 부르지 않고 httpx MockTransport 를 쓴다."""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from app.body_image_validation import warmup
from app.body_image_validation.remote import RemoteWorkerConfigError
from app.body_image_validation.warmup import ModalWarmupClient, WarmupOutcome, create_warmup_client
from app.config import body_image_validation as config
from app.config import settings

WORKER = "https://example--lookddak-body-image-staging.modal.run"
PROXY_KEY = "wk-warmup-test"
PROXY_SECRET = "ws-warmup-test"
API_KEY = "worker-bearer-key"
ROOT = Path(__file__).resolve().parents[1]


class FakeWorker:
    """Modal worker 자리의 httpx MockTransport. 받은 요청을 기록하고 정해 둔 응답(또는 예외)을 낸다.

    지연은 asyncio.sleep 이라 호출 전체 timeout(asyncio.timeout)이 실제로 취소할 수 있다.
    """

    def __init__(self, status=200, body=b'{"status":"ok"}', headers=None, error=None, delay=0.0):
        self.status, self.body, self.headers = status, body, headers or {}
        self.error, self.delay = error, delay
        self.calls = []
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request):
        await request.aread()
        self.calls.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return httpx.Response(self.status, headers=self.headers, content=self.body)


def client_for(fake, base_url=WORKER, timeout=5.0, clock=None):
    kwargs = {"clock": clock} if clock else {}
    return ModalWarmupClient(
        base_url,
        proxy_key=PROXY_KEY,
        proxy_secret=PROXY_SECRET,
        timeout=timeout,
        transport=fake.transport,
        **kwargs,
    )


def run(fake, **kwargs):
    return asyncio.run(client_for(fake, **kwargs).warm_up())


# --- 요청 구성 ---


def test_health_request_carries_only_the_proxy_auth_and_no_body():
    fake = FakeWorker()

    run(fake)

    [request] = fake.calls
    assert request.method == "GET"
    assert str(request.url) == WORKER + "/health"
    assert request.headers["Modal-Key"] == PROXY_KEY
    assert request.headers["Modal-Secret"] == PROXY_SECRET
    assert "authorization" not in request.headers
    assert request.content == b""


@pytest.mark.parametrize("base_url", [WORKER + "/", WORKER + "//"])
def test_trailing_slash_in_the_base_url_is_ignored(base_url):
    fake = FakeWorker()

    run(fake, base_url=base_url)

    assert str(fake.calls[0].url) == WORKER + "/health"


def test_settings_are_read_from_the_existing_remote_configuration(monkeypatch):
    monkeypatch.setattr(config, "REMOTE_URL", WORKER)
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_KEY", PROXY_KEY)
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_SECRET", PROXY_SECRET)
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_API_KEY", API_KEY)
    fake = FakeWorker()

    client = create_warmup_client(transport=fake.transport)
    result = asyncio.run(client.warm_up())

    assert result.succeeded
    assert client.endpoint == WORKER + "/health"
    assert client.timeout == config.WARMUP_TIMEOUT_SECONDS == 90.0
    assert API_KEY not in fake.calls[0].headers.values()


def test_warmup_timeout_is_shorter_than_modal_redirect():
    from app.body_image_validation.remote import MODAL_REDIRECT_AFTER_SECONDS

    assert 0 < config.WARMUP_TIMEOUT_SECONDS < MODAL_REDIRECT_AFTER_SECONDS


# --- 성공·실패 분류 ---


def test_200_with_status_ok_is_success():
    result = run(FakeWorker())

    assert result.succeeded
    assert (result.outcome, result.status_code) == (WarmupOutcome.SUCCESS, 200)


def test_elapsed_comes_from_the_injected_clock():
    times = iter([10.0, 12.5])

    result = run(FakeWorker(), clock=lambda: next(times))

    assert result.elapsed == 2.5


@pytest.mark.parametrize(
    "body", [b'{"status":"starting"}', b'{"status":null}', b"{}", b'["ok"]', b'"ok"']
)
def test_200_without_status_ok_is_unexpected_body(body):
    result = run(FakeWorker(body=body))

    assert not result.succeeded
    assert (result.outcome, result.status_code) == (WarmupOutcome.UNEXPECTED_BODY, 200)


@pytest.mark.parametrize("body", [b"ok", b"<html>", b"", b'{"status": "ok"', b"\xff\xfe"])
def test_200_with_invalid_json_is_invalid_json(body):
    result = run(FakeWorker(body=body))

    assert (result.outcome, result.status_code) == (WarmupOutcome.INVALID_JSON, 200)


@pytest.mark.parametrize(
    "status, outcome",
    [
        (401, WarmupOutcome.AUTH_FAILED),
        (403, WarmupOutcome.AUTH_FAILED),
        (404, WarmupOutcome.NOT_FOUND),
        (500, WarmupOutcome.SERVER_ERROR),
        (502, WarmupOutcome.SERVER_ERROR),
        (503, WarmupOutcome.SERVER_ERROR),
        (504, WarmupOutcome.SERVER_ERROR),
        (400, WarmupOutcome.UNEXPECTED_STATUS),
        (204, WarmupOutcome.UNEXPECTED_STATUS),
    ],
)
def test_http_failures_are_classified(status, outcome):
    result = run(FakeWorker(status=status, body=b'{"status":"ok"}'))

    assert not result.succeeded
    assert (result.outcome, result.status_code) == (outcome, status)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirects_are_not_followed(status):
    fake = FakeWorker(
        status=status, headers={"location": WORKER + "/health?__modal_function_call_id=fc-1"}
    )

    result = run(fake)

    assert (result.outcome, result.status_code) == (WarmupOutcome.REDIRECT, status)
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("connection refused"),
        httpx.RemoteProtocolError("closed"),
        httpx.ReadError("reset"),
    ],
)
def test_connection_failures_are_connection_failed(error):
    result = run(FakeWorker(error=error))

    assert (result.outcome, result.status_code) == (WarmupOutcome.CONNECTION_FAILED, None)


@pytest.mark.parametrize(
    "error", [httpx.ConnectTimeout("connect"), httpx.ReadTimeout("read"), httpx.PoolTimeout("pool")]
)
def test_httpx_timeouts_are_network_timeout(error):
    result = run(FakeWorker(error=error))

    assert (result.outcome, result.status_code) == (WarmupOutcome.NETWORK_TIMEOUT, None)


def test_whole_call_is_bounded_by_the_timeout():
    result = run(FakeWorker(delay=5.0), timeout=0.05)

    assert (result.outcome, result.status_code) == (WarmupOutcome.DEADLINE_EXCEEDED, None)
    assert result.elapsed < 1.0


def test_cancellation_is_not_swallowed():
    async def scenario():
        task = asyncio.create_task(client_for(FakeWorker(delay=5.0)).warm_up())
        await asyncio.sleep(0.01)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())


# --- 설정 ---


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_url": ""},
        {"proxy_key": ""},
        {"proxy_secret": ""},
        {"base_url": "http://example.modal.run"},
        {"base_url": "example.modal.run"},
        {"base_url": "https://"},
        {"base_url": WORKER + "?token=x"},
        {"base_url": WORKER + "#frag"},
        {"timeout": 0},
        {"timeout": -1.0},
    ],
)
def test_missing_or_unsafe_settings_are_rejected(overrides):
    kwargs = {
        "base_url": WORKER,
        "proxy_key": PROXY_KEY,
        "proxy_secret": PROXY_SECRET,
        "timeout": 5.0,
    } | overrides
    base_url = kwargs.pop("base_url")

    with pytest.raises(RemoteWorkerConfigError):
        ModalWarmupClient(base_url, **kwargs)


def test_missing_settings_from_the_environment_are_a_config_error(monkeypatch):
    monkeypatch.setattr(config, "REMOTE_URL", WORKER)
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_KEY", "")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_SECRET", PROXY_SECRET)

    with pytest.raises(RemoteWorkerConfigError) as error:
        create_warmup_client()

    assert "BODY_IMAGE_REMOTE_PROXY_KEY" in str(error.value)
    assert PROXY_SECRET not in str(error.value)


def test_api_key_is_not_required_for_warmup(monkeypatch):
    monkeypatch.setattr(config, "REMOTE_URL", WORKER)
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_KEY", PROXY_KEY)
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_SECRET", PROXY_SECRET)
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_API_KEY", "")

    assert create_warmup_client().endpoint == WORKER + "/health"


# --- 로그 ---


@pytest.mark.parametrize(
    "fake",
    [
        FakeWorker(),
        FakeWorker(status=401),
        FakeWorker(
            status=303, headers={"location": WORKER + "/health?__modal_function_call_id=fc-1"}
        ),
        FakeWorker(error=httpx.ConnectError(f"cannot connect to {WORKER} with {PROXY_SECRET}")),
        FakeWorker(error=httpx.ReadTimeout(f"timed out reading {WORKER}")),
    ],
)
def test_logs_name_the_outcome_without_secrets_or_urls(fake, caplog):
    caplog.set_level(logging.DEBUG, logger=warmup.__name__)

    result = run(fake)

    records = [r for r in caplog.records if r.name == warmup.__name__]
    assert len(records) == 1
    text = records[0].getMessage()
    assert f"outcome={result.outcome}" in text
    for secret in (
        PROXY_KEY,
        PROXY_SECRET,
        API_KEY,
        "modal.run",
        "Modal-Key",
        "Modal-Secret",
        "Bearer",
    ):
        assert secret not in text
    assert records[0].exc_info is None
    assert records[0].levelno == (logging.INFO if result.succeeded else logging.WARNING)


# --- remote 모드 import 제약 ---


def test_warmup_module_imports_no_model_libraries():
    code = (
        "import sys, json\n"
        "import app.body_image_validation.warmup\n"
        "print(json.dumps(sorted(m for m in ('mediapipe', 'rembg', 'onnxruntime', 'torch', 'cv2')"
        " if m in sys.modules)))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "[]"
