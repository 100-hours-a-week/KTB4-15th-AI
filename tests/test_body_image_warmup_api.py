"""POST /api/v1/validation/body-image/warmup 과 AI Server lifespan 연결. 실제 Modal 은 부르지 않는다."""

import asyncio
import logging
import time
from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.body_image_validation import router as validation_router
from app.body_image_validation import warmup, warmup_manager, warmup_router, worker
from app.body_image_validation.models import BodyImageValidationResult
from app.body_image_validation.remote import RemoteWorkerConfigError
from app.body_image_validation.runtime import BodyImageRuntimeManager, RemoteBodyImageRuntime
from app.body_image_validation.warmup import WarmupOutcome, WarmupResult
from app.body_image_validation.warmup_manager import (
    BodyImageWarmupManager,
    WarmupAcceptance,
    WarmupStatus,
)
from app.config import body_image_validation as config
from app.config import settings

URL = "/api/v1/validation/body-image/warmup"
VALIDATE_URL = "/api/v1/validation/body-image"
KEY = "ai-server-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
WORKER = "https://example--lookddak-body-image-staging.modal.run"
PROXY_SECRET = "ws-api-secret"
OK = WarmupResult(WarmupOutcome.SUCCESS, 200, 0.1)
FAILED = WarmupResult(WarmupOutcome.CONNECTION_FAILED, None, 0.1)
STARTED = WarmupAcceptance(WarmupStatus.STARTED)

ACCEPTED_BODY = {
    "code": "BODY_IMAGE_WARMUP_ACCEPTED",
    "data": None,
    "message": "전신 이미지 검증 사전 활성화 요청을 접수했습니다.",
}
RATE_LIMITED_BODY = {
    "code": "BODY_IMAGE_WARMUP_RATE_LIMITED",
    "data": None,
    "message": "사전 활성화 요청이 너무 많습니다. 잠시 후 다시 시도해주세요.",
}
UNAVAILABLE_BODY = {
    "code": "BODY_IMAGE_WARMUP_UNAVAILABLE",
    "data": None,
    "message": "현재 사전 활성화 요청을 처리할 수 없습니다.",
}


class StubManager:
    """정해 둔 접수 결과를 돌려주는 manager 자리. request·aclose 호출을 기록한다."""

    def __init__(self, acceptance=STARTED, close_error=None):
        self.acceptance = acceptance
        self.close_error = close_error
        self.requests = 0
        self.closed = 0

    async def request(self):
        self.requests += 1
        return self.acceptance

    async def aclose(self):
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


class FakeClient:
    """ModalWarmupClient 자리. block 이면 취소될 때까지 붙잡는다."""

    def __init__(self, result=OK, block=False):
        self.result = result
        self.block = block
        self.calls = 0
        self.cancelled = 0

    async def warm_up(self):
        self.calls += 1
        if self.block:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return self.result


class StubService:
    def validate(self, body):
        return BodyImageValidationResult(s3_key="body-images/stub.png")


@pytest.fixture
def runtime(monkeypatch):
    """lifespan 이 쓸 새 runtime manager. 기본은 local 이고 S3 설정이 없어 모델을 올리지 않는다."""
    monkeypatch.setattr(settings, "CHECKPOINT_DSN", "")
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", KEY)
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)
    monkeypatch.setattr(settings, "S3_BUCKET", "")
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "test-vton-key")
    monkeypatch.setenv("RUNWARE_LLM_API_KEY", "test-llm-key")
    monkeypatch.setattr(config, "VALIDATION_MODE", "local")
    manager = BodyImageRuntimeManager()
    monkeypatch.setattr(main, "runtime_manager", manager)
    return manager


@pytest.fixture
def remote(runtime, monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_MODE", "remote")
    monkeypatch.setattr(config, "REMOTE_URL", WORKER)
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_KEY", "wk-test")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_SECRET", PROXY_SECRET)
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_API_KEY", "worker-key")
    return runtime


def use_client(monkeypatch, fake):
    """lifespan 이 만드는 manager 가 실제 Modal 대신 fake client 를 쓰게 한다(나머지는 코드 상수 그대로)."""
    monkeypatch.setattr(
        warmup_router,
        "create_warmup_manager",
        lambda: warmup_manager.create_warmup_manager(fake),
    )


def wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not met in time")
        time.sleep(0.005)


def warm(client, headers=AUTH, **kwargs):
    return client.post(URL, headers=headers, **kwargs)


def validate(client):
    return client.post(
        VALIDATE_URL, files={"image": ("body.jpg", b"jpeg", "image/jpeg")}, headers=AUTH
    )


@pytest.fixture
def stub_client(runtime):
    """lifespan 을 띄운 뒤 app.state 의 manager 를 StubManager 로 바꿔 끼운다."""
    with TestClient(main.app, raise_server_exceptions=False) as client:

        def install(manager):
            client.app.state.body_image_warmup = manager
            return manager

        yield client, install


# --- 인증 ---


def test_internal_key_is_accepted(stub_client):
    client, install = stub_client
    manager = install(StubManager())

    response = warm(client)

    assert response.status_code == 202
    assert response.json() == ACCEPTED_BODY
    assert manager.requests == 1


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer wrong-key"}, {"Authorization": KEY}]
)
def test_missing_or_wrong_key_is_401_without_touching_the_manager(stub_client, headers):
    client, install = stub_client
    manager = install(StubManager())

    response = warm(client, headers=headers)

    assert response.status_code == 401
    assert response.json() == {
        "code": "UNAUTHORIZED",
        "data": None,
        "message": "인증에 실패했습니다.",
    }
    assert manager.requests == 0


def test_request_body_is_not_required_and_is_ignored(stub_client):
    client, install = stub_client
    install(StubManager())

    assert warm(client, json={"user_id": 1}).status_code == 202
    assert warm(client, content=b"not json").status_code == 202


# --- 상태별 응답 ---


@pytest.mark.parametrize(
    "status", [WarmupStatus.STARTED, WarmupStatus.IN_PROGRESS, WarmupStatus.COOLDOWN]
)
def test_accepted_statuses_are_202(stub_client, status):
    client, install = stub_client
    install(StubManager(WarmupAcceptance(status, 3.0 if status is WarmupStatus.COOLDOWN else None)))

    response = warm(client)

    assert response.status_code == 202
    assert response.json() == ACCEPTED_BODY
    assert "retry-after" not in response.headers


def test_rate_limited_is_429_with_retry_after(stub_client):
    client, install = stub_client
    install(StubManager(WarmupAcceptance(WarmupStatus.RATE_LIMITED, 12.0)))

    response = warm(client)

    assert response.status_code == 429
    assert response.json() == RATE_LIMITED_BODY
    assert response.headers["retry-after"] == "12"


def test_manager_unavailable_is_503(stub_client):
    client, install = stub_client
    install(StubManager(WarmupAcceptance(WarmupStatus.UNAVAILABLE)))

    response = warm(client)

    assert response.status_code == 503
    assert response.json() == UNAVAILABLE_BODY


@pytest.mark.parametrize(
    "seconds, header", [(12.0, "12"), (0.2, "1"), (5.0001, "6"), (29.9, "30"), (1, "1")]
)
def test_retry_after_is_rounded_up(stub_client, seconds, header):
    client, install = stub_client
    install(StubManager(WarmupAcceptance(WarmupStatus.RATE_LIMITED, seconds)))

    assert warm(client).headers["retry-after"] == header


@pytest.mark.parametrize("seconds", [None, 0, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_retry_after_adds_no_header(stub_client, seconds):
    client, install = stub_client
    install(StubManager(WarmupAcceptance(WarmupStatus.RATE_LIMITED, seconds)))

    response = warm(client)

    assert response.status_code == 429
    assert response.json() == RATE_LIMITED_BODY
    assert "retry-after" not in response.headers


# --- lifespan: manager 생성 ---


def test_local_mode_creates_no_manager_and_answers_503(runtime):
    with TestClient(main.app, raise_server_exceptions=False) as client:
        assert client.app.state.body_image_warmup is None
        response = warm(client)

    assert response.status_code == 503
    assert response.json() == UNAVAILABLE_BODY


def test_remote_mode_creates_a_manager_with_the_code_constants(remote):
    with TestClient(main.app, raise_server_exceptions=False) as client:
        manager = client.app.state.body_image_warmup
        assert isinstance(remote.runtime, RemoteBodyImageRuntime)
        assert isinstance(manager, BodyImageWarmupManager)
        assert manager._client.endpoint == WORKER + "/health"
        assert manager._cooldown == config.WARMUP_COOLDOWN_SECONDS

    assert client.app.state.body_image_warmup is None


def test_warmup_config_error_disables_only_warmup(remote, monkeypatch, caplog):
    def broken():
        raise RemoteWorkerConfigError("warm-up 설정이 없습니다: ['BODY_IMAGE_REMOTE_PROXY_KEY']")

    monkeypatch.setattr(warmup_router, "create_warmup_manager", broken)
    monkeypatch.setattr(validation_router, "get_service", lambda: StubService())

    with (
        caplog.at_level(logging.INFO),
        TestClient(main.app, raise_server_exceptions=False) as client,
    ):
        warmup_response = warm(client)
        validation_response = validate(client)

    assert warmup_response.status_code == 503
    assert warmup_response.json() == UNAVAILABLE_BODY
    assert validation_response.status_code == 200  # 검증 API 는 그대로
    messages = [r.getMessage() for r in caplog.records if r.name == warmup_router.__name__]
    assert any("BODY_IMAGE_REMOTE_PROXY_KEY" in m for m in messages)
    assert all(PROXY_SECRET not in m and WORKER not in m for m in messages)


def test_remote_runtime_failure_creates_no_manager(remote, monkeypatch, caplog):
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_API_KEY", "")  # 검증 runtime 만 실패하는 설정

    with (
        caplog.at_level(logging.INFO),
        TestClient(main.app, raise_server_exceptions=False) as client,
    ):
        assert client.app.state.body_image_warmup is None
        response = warm(client)

    assert response.status_code == 503
    assert any("remote runtime is unavailable" in r.getMessage() for r in caplog.records)


def test_unexpected_creation_errors_are_not_hidden(remote, monkeypatch):
    def broken():
        raise RuntimeError("bug")

    monkeypatch.setattr(warmup_router, "create_warmup_manager", broken)

    with pytest.raises(RuntimeError), TestClient(main.app):
        pass


# --- 비동기 동작 ---


def test_response_does_not_wait_for_the_modal_call(remote, monkeypatch):
    fake = FakeClient(block=True)
    use_client(monkeypatch, fake)

    with TestClient(main.app, raise_server_exceptions=False) as client:
        first = warm(client)
        wait_until(lambda: fake.calls == 1)
        manager = client.app.state.body_image_warmup
        assert manager.in_progress  # 호출이 끝나지 않았는데 이미 응답했다
        second = warm(client)

    assert first.status_code == second.status_code == 202
    assert fake.calls == 1  # 두 번째는 진행 중인 호출을 재사용
    assert fake.cancelled == 1  # 종료가 진행 중인 호출을 취소하고 기다렸다
    assert not manager.in_progress


def test_failed_call_rate_limits_warmup_but_not_validation(remote, monkeypatch):
    fake = FakeClient(FAILED)
    use_client(monkeypatch, fake)
    monkeypatch.setattr(validation_router, "get_service", lambda: StubService())

    with TestClient(main.app, raise_server_exceptions=False) as client:
        assert warm(client).status_code == 202
        manager = client.app.state.body_image_warmup
        wait_until(lambda: manager.last_result is not None)
        retried = warm(client)
        validation = validate(client)

    assert retried.status_code == 429
    assert retried.json() == RATE_LIMITED_BODY
    assert 1 <= int(retried.headers["retry-after"]) <= config.WARMUP_MIN_RETRY_INTERVAL_SECONDS
    assert fake.calls == 1  # 429 는 새 호출을 만들지 않는다
    assert validation.status_code == 200
    assert validation.json()["data"] == {"s3_key": "body-images/stub.png"}


def test_validation_runs_while_warmup_is_in_progress_and_shares_no_limiter(remote, monkeypatch):
    fake = FakeClient(block=True)
    use_client(monkeypatch, fake)
    monkeypatch.setattr(validation_router, "get_service", lambda: StubService())

    with TestClient(main.app, raise_server_exceptions=False) as client:
        warm(client)
        wait_until(lambda: fake.calls == 1)
        limiter = remote.runtime.service.limiter
        slots = (limiter.running, limiter.waiting)
        validation = validate(client)

    assert slots == (0, 0)  # warm-up 은 검증 대기열 슬롯을 쓰지 않는다
    assert validation.status_code == 200


# --- lifespan: 종료 ---


def test_shutdown_closes_warmup_before_the_runtime(remote, monkeypatch):
    order = []
    stub = StubManager()
    stub.aclose = lambda: _record(order, "warmup")  # type: ignore[method-assign]
    monkeypatch.setattr(warmup_router, "create_warmup_manager", lambda: stub)
    close = remote.close
    monkeypatch.setattr(remote, "close", lambda: (order.append("runtime"), close()))

    with TestClient(main.app):
        pass

    assert order == ["warmup", "runtime"]


async def _record(order, name):
    order.append(name)


def test_close_failure_does_not_skip_the_rest_of_shutdown(remote, monkeypatch, caplog):
    stub = StubManager(close_error=RuntimeError(f"secret {PROXY_SECRET}"))
    monkeypatch.setattr(warmup_router, "create_warmup_manager", lambda: stub)
    closed = []
    monkeypatch.setattr(remote, "close", lambda: closed.append(True))

    with caplog.at_level(logging.ERROR), TestClient(main.app) as client:
        pass

    assert stub.closed == 1
    assert closed == [True]
    assert client.app.state.body_image_warmup is None
    messages = [r.getMessage() for r in caplog.records if r.name == warmup_router.__name__]
    assert messages == ["body image warm-up close failed: error=RuntimeError"]


def test_restart_starts_from_a_fresh_manager(remote, monkeypatch):
    fake = FakeClient(OK)
    use_client(monkeypatch, fake)

    with TestClient(main.app) as client:
        assert warm(client).status_code == 202
        first = client.app.state.body_image_warmup
        wait_until(lambda: first.last_result is not None)
        assert warm(client).status_code == 202  # COOLDOWN

    with TestClient(main.app) as client:
        second = client.app.state.body_image_warmup
        assert second is not first
        assert second.last_result is None and not second._starts
        assert warm(client).status_code == 202
        wait_until(lambda: second.last_result is not None)

    assert fake.calls == 2  # 재시작 뒤에는 이전 cooldown 없이 새로 부른다
    assert first._closed  # 이전 manager 는 첫 종료 때 닫혔다


# --- router 분리 ---


def test_ai_server_exposes_the_warmup_endpoint():
    paths = main.app.openapi()["paths"]

    assert sorted(paths[URL]) == ["post"]
    assert sorted(paths[VALIDATE_URL]) == ["post"]


def test_modal_worker_does_not_expose_the_warmup_endpoint():
    paths = worker.create_app().openapi()["paths"]

    assert URL not in paths
    assert set(paths) == {VALIDATE_URL, "/health"}


def test_shared_validation_router_has_no_warmup_route():
    assert [route.path for route in validation_router.router.routes] == [VALIDATE_URL]


# --- 4단계: HTTP 계층의 동시 요청 ---


def test_concurrent_api_requests_make_one_modal_call(runtime):
    fake = FakeClient(block=True)

    async def scenario():
        manager = warmup_manager.create_warmup_manager(fake)
        main.app.state.body_image_warmup = manager
        try:
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://ai") as client:
                responses = await asyncio.gather(
                    *(client.post(URL, headers=AUTH) for _ in range(20))
                )
                unauthorized = await client.post(URL)
            for _ in range(100):
                if fake.calls:
                    break
                await asyncio.sleep(0)
            in_progress = manager.in_progress
        finally:
            main.app.state.body_image_warmup = None
            await manager.aclose()
        return responses, unauthorized, in_progress

    responses, unauthorized, in_progress = asyncio.run(scenario())

    assert [r.status_code for r in responses] == [202] * 20
    assert all(r.json() == ACCEPTED_BODY for r in responses)
    assert unauthorized.status_code == 401
    assert in_progress
    assert fake.calls == 1 and fake.cancelled == 1


# --- 4단계: lifespan 시작 오류 정리 ---


def test_unexpected_startup_error_still_releases_started_resources(remote, monkeypatch):
    def broken():
        raise RuntimeError("bug")

    cleaned = []

    async def close_async_pool():
        cleaned.append("async_pool")

    monkeypatch.setattr(warmup_router, "create_warmup_manager", broken)
    monkeypatch.setattr(remote, "close", lambda: cleaned.append("runtime"))
    monkeypatch.setattr(main, "close_connection_pool", lambda: cleaned.append("sync_pool"))
    monkeypatch.setattr(main, "close_async_pool", close_async_pool)

    with pytest.raises(RuntimeError), TestClient(main.app):
        pass

    assert cleaned == ["runtime", "sync_pool", "async_pool"]
    assert main.app.state.body_image_warmup is None


# --- 4단계: warm-up 실패와 실제 remote 검증 경로의 독립성 ---

WORKER_SUCCESS = (
    b'{"code":"BODY_IMAGE_UPLOAD_SUCCESS","data":{"s3_key":"body-images/real.png"},"message":"ok"}'
)


def jpeg(size=(600, 800)):
    output = BytesIO()
    Image.new("RGB", size, (180, 170, 160)).save(output, format="JPEG")
    return output.getvalue()


@pytest.fixture
def real_validation(remote, monkeypatch):
    """검증 API 가 실제 RemoteBodyImageValidationService(limiter·deadline 포함)를 거쳐 가짜 worker 를 부르게 한다."""
    monkeypatch.setattr(validation_router, "runtime_manager", remote)
    calls = []

    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(200, content=WORKER_SUCCESS)

    def install():
        monkeypatch.setattr(remote.runtime.service, "_transport", httpx.MockTransport(handle))

    return install, calls


def use_modal(monkeypatch, handle):
    """lifespan 이 만드는 manager 가 실제 ModalWarmupClient 로 가짜 Modal /health 를 부르게 한다."""
    monkeypatch.setattr(
        warmup_router,
        "create_warmup_manager",
        lambda: warmup_manager.create_warmup_manager(
            warmup.create_warmup_client(httpx.MockTransport(handle))
        ),
    )


def validate_jpeg(client):
    return client.post(
        VALIDATE_URL, files={"image": ("body.jpg", jpeg(), "image/jpeg")}, headers=AUTH
    )


async def _connect_error(request):
    raise httpx.ConnectError("refused")


async def _read_timeout(request):
    raise httpx.ReadTimeout("slow")


async def _hang(request):
    await asyncio.sleep(5)
    return httpx.Response(200, json={"status": "ok"})


def _status(code):
    async def handle(request):
        return httpx.Response(code, content=b"modal says no")

    return handle


@pytest.mark.parametrize(
    "handle, outcome",
    [
        (_connect_error, WarmupOutcome.CONNECTION_FAILED),
        (_read_timeout, WarmupOutcome.NETWORK_TIMEOUT),
        (_hang, WarmupOutcome.DEADLINE_EXCEEDED),
        (_status(401), WarmupOutcome.AUTH_FAILED),
        (_status(403), WarmupOutcome.AUTH_FAILED),
        (_status(503), WarmupOutcome.SERVER_ERROR),
    ],
)
def test_validation_is_unaffected_by_each_warmup_failure(
    real_validation, monkeypatch, handle, outcome
):
    install, worker_calls = real_validation
    monkeypatch.setattr(config, "WARMUP_TIMEOUT_SECONDS", 0.05)  # 전체 시간 초과를 빨리 낸다
    use_modal(monkeypatch, handle)

    with TestClient(main.app, raise_server_exceptions=False) as client:
        install()
        manager = client.app.state.body_image_warmup
        accepted = warm(client)
        wait_until(lambda: manager.last_result is not None)
        limited = warm(client)
        validation = validate_jpeg(client)
        limiter = validation_router.runtime_manager.runtime.service.limiter
        slots = (limiter.running, limiter.waiting)

    assert accepted.status_code == 202
    assert manager.last_result.outcome is outcome
    assert limited.status_code == 429 and limited.json() == RATE_LIMITED_BODY
    assert validation.status_code == 200
    assert validation.json()["data"] == {"s3_key": "body-images/real.png"}
    assert worker_calls == [VALIDATE_URL]  # warm-up 은 검증 endpoint 를 부르지 않는다
    assert slots == (0, 0)


def test_validation_works_after_warmup_is_closed_or_cancelled(real_validation, monkeypatch):
    install, worker_calls = real_validation
    fake = FakeClient(block=True)
    use_client(monkeypatch, fake)

    with TestClient(main.app, raise_server_exceptions=False) as client:
        install()
        manager = client.app.state.body_image_warmup
        assert warm(client).status_code == 202
        wait_until(lambda: fake.calls == 1)
        client.portal.call(manager.aclose)  # 종료된 manager(상태에는 남아 있음)
        closed = warm(client)
        client.portal.call(warmup_router.close_warmup, client.app)  # manager 없음
        removed = warm(client)
        validation = validate_jpeg(client)

    assert fake.cancelled == 1 and manager.last_result is None
    assert closed.status_code == removed.status_code == 503
    assert closed.json() == removed.json() == UNAVAILABLE_BODY
    assert validation.status_code == 200
    assert worker_calls == [VALIDATE_URL]


# --- 최종 리뷰: 종료 중 취소 ---


def test_cancelled_shutdown_still_releases_existing_resources(remote, monkeypatch):
    entered = asyncio.Event()

    class HangingManager(StubManager):
        async def aclose(self):
            self.closed += 1
            entered.set()
            await asyncio.sleep(3600)  # lifespan 이 이 지점에서 취소된다

    hanging = HangingManager()
    cleaned = []

    async def close_async_pool():
        cleaned.append("async_pool")

    monkeypatch.setattr(warmup_router, "create_warmup_manager", lambda: hanging)
    monkeypatch.setattr(remote, "close", lambda: cleaned.append("runtime"))
    monkeypatch.setattr(main, "close_connection_pool", lambda: cleaned.append("sync_pool"))
    monkeypatch.setattr(main, "close_async_pool", close_async_pool)

    async def scenario():
        async def run_lifespan():
            async with main.lifespan(main.app):
                pass

        task = asyncio.create_task(run_lifespan())
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task  # 취소는 삼키지 않고 그대로 전한다

    asyncio.run(scenario())

    assert hanging.closed == 1
    assert cleaned == ["runtime", "sync_pool", "async_pool"]
    assert main.app.state.body_image_warmup is None
