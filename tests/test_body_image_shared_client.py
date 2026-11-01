import asyncio
import logging
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.body_image_validation import router as validation_router
from app.body_image_validation import warmup_router
from app.body_image_validation.async_concurrency import AsyncBodyValidationLimiter
from app.body_image_validation.remote import RemoteBodyImageValidationService
from app.body_image_validation.runtime import (
    BodyImageRuntimeError,
    BodyImageRuntimeManager,
    RemoteBodyImageRuntime,
)
from app.config import body_image_validation as config
from app.config import settings
from tests.test_body_image_async_concurrency import state, wait_for_queue, wait_until
from tests.test_body_image_concurrent_api import HeldWorker
from tests.test_body_image_remote import (
    POLL,
    SUCCESS,
    FakeWorker,
    jpeg,
    local_worker,  # noqa: F401 (pytest fixture)
    post,
    redirect,
)
from tests.test_body_image_remote import client as remote_client  # noqa: F401 (pytest fixture)
from tests.test_body_image_remote_async import (
    KEY,
    OK,
    TIMED_OUT,
    UNAVAILABLE,
    opened,
    rejected,
    run,
    service,
    tcp_service,
)
from tests.test_body_image_warmup_api import (  # noqa: F401 (pytest fixture)
    VALIDATE_URL,
    StubManager,
    real_validation,
    remote,
    runtime,
    validate_jpeg,
)


class ClientSpy:
    """httpx.AsyncClient 생성을 기록한다. 만든 client·send 횟수·event loop 를 본다."""

    def __init__(self, monkeypatch):
        spy = self
        self.clients = []
        self.sends = {}
        self.loops = {"created": [], "closed": []}

        class Spy(httpx.AsyncClient):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                spy.clients.append(self)
                spy.sends[id(self)] = 0
                try:
                    spy.loops["created"].append(asyncio.get_running_loop())
                except RuntimeError:
                    spy.loops["created"].append(None)

            async def send(self, *args, **kwargs):
                spy.sends[id(self)] += 1
                return await super().send(*args, **kwargs)

            async def aclose(self):
                spy.loops["closed"].append(asyncio.get_running_loop())
                await super().aclose()

        monkeypatch.setattr(httpx, "AsyncClient", Spy)


@pytest.fixture
def clients(monkeypatch):
    return ClientSpy(monkeypatch)


# --- 재사용 ---


def test_requests_share_one_client_and_it_stays_open(clients):
    fake = FakeWorker()

    async def scenario():
        svc = await opened(service(fake))
        client = svc._shared_client

        for _ in range(3):
            assert (await svc.avalidate(jpeg())).s3_key == KEY

        assert clients.clients == [client]  # 요청마다 만들지 않았다
        assert svc._shared_client is client
        assert not client.is_closed
        assert clients.sends[id(client)] == 3

    run(scenario)


def test_tcp_connection_is_reused_across_requests(local_worker):  # noqa: F811
    async def scenario():
        svc = await opened(tcp_service(local_worker))
        for _ in range(6):
            assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)

    assert (local_worker.requests, local_worker.connections) == (6, 1)


def test_server_that_refuses_keep_alive_gets_a_new_connection_each_time(local_worker):  # noqa: F811
    local_worker.keep_alive = False

    async def scenario():
        svc = await opened(tcp_service(local_worker))
        for _ in range(4):
            assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)

    assert (local_worker.requests, local_worker.connections) == (4, 4)


def test_idle_connection_closed_by_the_server_is_replaced(local_worker):  # noqa: F811
    local_worker.close_silently = True

    async def scenario():
        svc = await opened(tcp_service(local_worker))
        await svc.avalidate(jpeg())
        await asyncio.sleep(0.05)  # 서버가 닫은 것을 client 가 볼 시간

        assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)

    assert (local_worker.requests, local_worker.connections) == (2, 2)


def test_concurrent_requests_use_separate_connections_and_keep_them(local_worker):  # noqa: F811
    local_worker.delays = [0.05, 0.05]

    async def scenario():
        svc = await opened(tcp_service(local_worker, limiter=AsyncBodyValidationLimiter(2, 0)))
        await asyncio.gather(svc.avalidate(jpeg()), svc.avalidate(jpeg()))
        await svc.avalidate(jpeg())

    run(scenario)

    assert (local_worker.requests, local_worker.connections) == (3, 2)


# --- 열림·닫힘 ---


def test_avalidate_without_an_open_client_is_503_and_sends_nothing(caplog):
    fake = FakeWorker()

    async def scenario():
        svc = service(fake)
        assert await rejected(svc.avalidate(jpeg())) == UNAVAILABLE
        assert state(svc.limiter) == (0, 0)

    with caplog.at_level(logging.ERROR, logger="app.body_image_validation.remote"):
        run(scenario)

    assert "client is not open" in caplog.text
    assert fake.calls == []


def test_closed_client_is_never_used_again(clients):
    fake = FakeWorker()

    async def scenario():
        svc = service(fake)
        await svc.aopen()
        client = svc._shared_client
        await svc.aclose()

        assert client.is_closed
        assert await rejected(svc.avalidate(jpeg())) == UNAVAILABLE
        assert clients.sends[id(client)] == 0
        assert state(svc.limiter) == (0, 0)

    run(scenario)

    assert fake.calls == []


def test_open_and_close_can_be_repeated(clients):
    async def scenario():
        svc = service(FakeWorker())
        await svc.aopen()
        first = svc._shared_client
        await svc.aopen()
        assert svc._shared_client is first  # 새로 만들지 않는다

        await svc.aclose()
        await svc.aclose()

        assert len(clients.loops["closed"]) == 1
        await svc.aclose()

    run(scenario)


def test_failing_client_close_still_leaves_the_service_closed(monkeypatch):
    fake = FakeWorker()

    async def scenario():
        svc = service(fake)
        await svc.aopen()

        async def broken_close():
            raise OSError("close failed")

        monkeypatch.setattr(svc._shared_client, "aclose", broken_close)
        with pytest.raises(OSError):
            await svc.aclose()

        assert svc._shared_client is None
        assert await rejected(svc.avalidate(jpeg())) == UNAVAILABLE

    run(scenario)

    assert fake.calls == []


# --- 진행 중인 호출과 종료 ---


def test_close_waits_for_the_running_call_and_turns_new_ones_away():
    async def scenario():
        worker = HeldWorker()
        svc = await opened(service(worker, limiter=AsyncBodyValidationLimiter(2, 0)))
        client = svc._shared_client
        running = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(worker.entered.is_set)

        closing = asyncio.create_task(svc.aclose())
        await asyncio.sleep(0.01)

        assert not closing.done() and not client.is_closed  # 취소하지 않고 기다린다
        assert await rejected(svc.avalidate(jpeg())) == UNAVAILABLE
        assert worker.arrivals == 1
        worker.released.set()
        assert (await running).s3_key == KEY
        await closing
        assert client.is_closed

    run(scenario)


def test_call_in_flight_over_a_real_connection_finishes_before_the_close(local_worker):  # noqa: F811
    local_worker.delays = [0.1]

    async def scenario():
        svc = await opened(tcp_service(local_worker))
        running = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(lambda: local_worker.requests == 1)

        await svc.aclose()

        assert running.done()
        assert running.result().s3_key == KEY

    run(scenario)


def test_request_waiting_in_the_queue_gets_503_after_the_close():
    async def scenario():
        worker = HeldWorker()
        svc = await opened(service(worker))
        running = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(worker.entered.is_set)
        queued = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_for_queue(svc.limiter, 1)
        closing = asyncio.create_task(svc.aclose())
        await asyncio.sleep(0)

        worker.released.set()

        assert (await running).s3_key == KEY
        assert await rejected(queued) == UNAVAILABLE
        await closing
        assert worker.arrivals == 1
        assert state(svc.limiter) == (0, 0)

    run(scenario)


def test_failed_and_cancelled_calls_do_not_hold_the_close_back():
    async def scenario():
        fake = FakeWorker(steps=[httpx.ConnectError("refused"), (5.0, 200, {}, SUCCESS)])
        svc = await opened(service(fake))
        assert await rejected(svc.avalidate(jpeg())) == UNAVAILABLE
        task = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(lambda: len(fake.calls) == 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        async with asyncio.timeout(1.0):
            await svc.aclose()  # 호출 수가 어긋나면 여기서 멈춘다

    run(scenario)


def test_cancelled_close_still_closes_the_client_and_the_running_call_gets_503(local_worker):  # noqa: F811
    local_worker.delays = [0.3]

    async def scenario():
        svc = await opened(tcp_service(local_worker))
        client = svc._shared_client
        running = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(lambda: local_worker.requests == 1)
        closing = asyncio.create_task(svc.aclose())
        await asyncio.sleep(0.01)

        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing

        assert client.is_closed
        assert await rejected(running) == UNAVAILABLE
        assert state(svc.limiter) == (0, 0)

    run(scenario)


# --- 이벤트 루프 ---


def test_client_is_created_used_and_closed_on_one_event_loop(clients):
    used = []

    class LoopRecordingWorker(FakeWorker):
        async def _handle(self, request):
            used.append(asyncio.get_running_loop())
            return await super()._handle(request)

    async def scenario():
        svc = service(LoopRecordingWorker())
        await svc.aopen()
        await svc.avalidate(jpeg())
        await svc.aclose()
        loop = asyncio.get_running_loop()

        assert clients.loops == {"created": [loop], "closed": [loop]}
        assert used == [loop]

    run(scenario)


def test_client_is_not_created_at_construction(clients):
    service(FakeWorker())

    assert clients.clients == []


def test_avalidate_on_another_loop_is_refused_and_the_client_survives():
    fake = FakeWorker()
    svc = service(fake)

    with asyncio.Runner() as owner, asyncio.Runner() as other:
        owner.run(svc.aopen())
        client = svc._shared_client

        with pytest.raises(RuntimeError, match="event loop"):
            other.run(svc.avalidate(jpeg()))

        assert fake.calls == []
        assert state(svc.limiter) == (0, 0)
        assert svc._shared_client is client and not client.is_closed
        owner.run(svc.aclose())

    assert client.is_closed


@pytest.mark.parametrize("method", ["aopen", "aclose"])
def test_opening_or_closing_on_another_loop_is_refused(method):
    svc = service(FakeWorker())

    with asyncio.Runner() as owner, asyncio.Runner() as other:
        owner.run(svc.aopen())
        client = svc._shared_client

        with pytest.raises(RuntimeError, match="event loop"):
            other.run(getattr(svc, method)())

        assert svc._shared_client is client and not client.is_closed
        owner.run(svc.aclose())


# --- Cookie ---


def test_cookie_from_one_request_does_not_reach_the_next():
    fake = FakeWorker(
        steps=[(0, 200, {"set-cookie": "sid=1; Path=/"}, SUCCESS), OK]
    )

    async def scenario():
        svc = await opened(service(fake))
        await svc.avalidate(jpeg())
        await svc.avalidate(jpeg())

        assert len(svc._shared_client.cookies.jar) == 0

    run(scenario)

    assert "cookie" not in fake.calls[1].headers


def test_cookie_from_a_redirect_stays_in_its_own_request():
    fake = FakeWorker(
        steps=[
            (0, 303, {"location": POLL, "set-cookie": "sid=1; Path=/"}, b""),
            OK,
            redirect(POLL),  # 다음 요청도 303 을 받는다. cookie 가 남아 있으면 이 GET 에 실린다
            OK,
        ]
    )

    async def scenario():
        svc = await opened(service(fake))
        await svc.avalidate(jpeg())
        await svc.avalidate(jpeg())

        assert len(svc._shared_client.cookies.jar) == 0

    run(scenario)

    _, first_get, second_post, second_get = fake.calls
    assert first_get.headers["cookie"] == "sid=1"  # 같은 요청의 303 에는 그대로 간다
    assert "cookie" not in second_post.headers
    assert "cookie" not in second_get.headers


def test_cookies_of_concurrent_requests_do_not_mix():
    arrived = []
    seen = {}

    async def handle(request):
        await request.aread()
        if request.method == "POST":
            number = len(arrived)
            arrived.append(number)
            while len(arrived) < 2:  # 두 요청이 모두 진행 중일 때 응답한다
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.002 * (2 - number))  # 늦게 온 요청이 먼저 303 을 받는다
            return httpx.Response(
                303,
                headers={"location": f"{POLL}&n={number}", "set-cookie": f"c{number}=1; Path=/"},
            )
        seen[request.url.params["n"]] = request.headers.get("cookie")
        await asyncio.sleep(0.002)
        return httpx.Response(200, content=SUCCESS)

    worker = FakeWorker()
    worker.transport = httpx.MockTransport(handle)

    async def scenario():
        svc = await opened(service(worker, limiter=AsyncBodyValidationLimiter(2, 0)))
        await asyncio.gather(svc.avalidate(jpeg()), svc.avalidate(jpeg()))

    run(scenario)

    assert seen == {"0": "c0=1", "1": "c1=1"}  # 다른 요청의 cookie 가 함께 실리지 않는다


COOKIE_CASES = {
    "plain": (["sid=1"], [None, "sid=1"]),
    "path-root": (["sid=1; Path=/"], [None, "sid=1"]),
    "other-path": (["sid=1; Path=/other"], [None, None]),
    "expired": (["sid=1; Max-Age=0"], [None, None]),
    "other-domain": (["sid=1; Domain=other.example.com"], [None, None]),
    "secure": (["sid=1; Secure"], [None, "sid=1"]),
    "accumulates": (["a=1; Path=/", "b=2; Path=/"], [None, "a=1", "a=1; b=2"]),
    "overwrites": (["a=1; Path=/", "a=2; Path=/"], [None, "a=1", "a=2"]),
}


@pytest.mark.parametrize(("set_cookies", "expected"), COOKIE_CASES.values(), ids=COOKIE_CASES)
def test_redirect_cookies_keep_the_existing_behaviour(set_cookies, expected):
    # expected 는 A 방식(요청마다 client) 코드에서 확인한 값이다.
    steps = [(0, 303, {"location": POLL, "set-cookie": value}, b"") for value in set_cookies] + [OK]
    worker = FakeWorker(steps=steps)

    async def scenario():
        await (await opened(service(worker))).avalidate(jpeg())

    run(scenario)

    assert [call.headers.get("cookie") for call in worker.calls] == expected


def test_cookies_stay_isolated_on_a_reused_real_connection(local_worker):  # noqa: F811
    local_worker.set_cookie = True

    async def scenario():
        svc = await opened(tcp_service(local_worker))
        for _ in range(3):
            await svc.avalidate(jpeg())

    run(scenario)

    assert local_worker.seen_cookies == [None, None, None]
    assert local_worker.connections == 1


# --- 인증 헤더 ---


def test_auth_and_deadline_headers_go_on_each_request_and_not_on_the_client():
    fake = FakeWorker(steps=[redirect(POLL), OK, OK])
    now = [1_000_000.0]

    async def scenario():
        svc = await opened(service(fake, deadline=50.0, wall_clock=lambda: now[0]))
        await svc.avalidate(jpeg())
        now[0] += 100.0
        await svc.avalidate(jpeg())

        client_headers = {name.lower() for name in svc._shared_client.headers}
        assert not client_headers & {"modal-key", "modal-secret", "authorization"}

    run(scenario)

    for call in fake.calls:
        assert call.headers["modal-key"] == "wk-test"
        assert call.headers["modal-secret"] == "ws-test"
        assert call.headers["authorization"] == "Bearer worker-key"
    first_post, first_get, second_post = fake.calls
    header = "x-body-image-deadline"
    assert first_get.headers[header] == first_post.headers[header]
    assert float(second_post.headers[header]) == pytest.approx(float(first_post.headers[header]) + 100.0, abs=0.1)


# --- 실패 뒤 재사용 ---


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (httpx.ConnectError("refused"), UNAVAILABLE),
        (httpx.RemoteProtocolError("closed"), UNAVAILABLE),
        (httpx.ReadTimeout("timed out"), TIMED_OUT),
    ],
    ids=["refused", "closed", "timeout"],
)
def test_shared_client_is_reused_after_a_failure(clients, failure, expected):
    fake = FakeWorker(steps=[failure, OK])

    async def scenario():
        svc = await opened(service(fake))
        client = svc._shared_client

        assert await rejected(svc.avalidate(jpeg())) == expected

        assert (await svc.avalidate(jpeg())).s3_key == KEY
        assert svc._shared_client is client and not client.is_closed
        assert len(clients.clients) == 1

    run(scenario)

    assert len(fake.calls) == 2  # 재시도 없음


def test_connection_dropped_mid_request_is_503_without_a_retry(local_worker):  # noqa: F811
    async def scenario():
        svc = await opened(tcp_service(local_worker))
        await svc.avalidate(jpeg())
        local_worker.drop_next = True  # 재사용 중인 연결에서 요청을 받고 끊는다

        assert await rejected(svc.avalidate(jpeg())) == UNAVAILABLE

        assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)

    assert local_worker.requests == 3  # 끊긴 요청을 다시 보내지 않았다
    assert local_worker.connections == 2


def test_connection_lost_while_receiving_the_response_is_503_without_a_retry(local_worker):  # noqa: F811
    async def scenario():
        svc = await opened(tcp_service(local_worker))
        await svc.avalidate(jpeg())
        local_worker.truncate_next = True

        assert await rejected(svc.avalidate(jpeg())) == UNAVAILABLE

        assert (await svc.avalidate(jpeg())).s3_key == KEY
        assert state(svc.limiter) == (0, 0)

    run(scenario)

    assert local_worker.requests == 3
    assert local_worker.connections == 2


def test_reused_connection_closing_as_the_request_arrives_is_503_without_a_resend(local_worker, caplog):  # noqa: F811
    # 서버는 응답 뒤 0.2초 있다가 연결을 닫고, 그 사이에 도착한 요청은 읽지 않는다. 그 요청은 처리되지 않는다.
    local_worker.close_silently = True
    local_worker.close_after = 0.2

    async def scenario():
        svc = await opened(tcp_service(local_worker))
        await svc.avalidate(jpeg())

        assert await rejected(svc.avalidate(jpeg())) == UNAVAILABLE

        assert state(svc.limiter) == (0, 0)
        assert (await svc.avalidate(jpeg())).s3_key == KEY  # 다음 요청은 새 연결로 간다

    with caplog.at_level(logging.INFO, logger="app.body_image_validation.remote"):
        run(scenario)

    assert local_worker.requests == 2  # 버려진 요청을 다시 보내지 않았다
    assert local_worker.connections == 2
    assert re.search(r"outcome=connection_failed error=(read_error|remote_protocol_error) ", caplog.text)


def test_server_closing_during_the_upload_is_503_without_a_retry(local_worker, caplog):  # noqa: F811
    local_worker.reject_upload = True
    upload = jpeg() + b"\0" * (5 * 1024 * 1024)  # socket buffer 를 넘는 크기

    async def scenario():
        svc = await opened(tcp_service(local_worker))
        assert await rejected(svc.avalidate(upload)) == UNAVAILABLE
        assert state(svc.limiter) == (0, 0)

    with caplog.at_level(logging.INFO, logger="app.body_image_validation.remote"):
        run(scenario)

    assert local_worker.rejected == 1
    assert re.search(r"error=(write_error|read_error|remote_protocol_error) ", caplog.text)


def test_deadline_over_a_reused_connection_is_504_and_the_next_request_works(local_worker):  # noqa: F811
    async def scenario():
        svc = await opened(tcp_service(local_worker))
        await svc.avalidate(jpeg())
        local_worker.delays = [1.0]
        svc.total_deadline, svc.timeout = 0.4, httpx.Timeout(0.4)  # 요청이 서버에 닿은 뒤 끊기도록 넉넉히 둔다

        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT

        svc.total_deadline, svc.timeout = 5.0, httpx.Timeout(5.0)
        assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)

    assert local_worker.requests == 3


def test_cancelled_call_over_a_reused_connection_leaves_the_client_usable(local_worker):  # noqa: F811
    async def scenario():
        svc = await opened(tcp_service(local_worker))
        await svc.avalidate(jpeg())
        local_worker.delays = [0.3]
        task = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(lambda: local_worker.requests == 2)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert (await svc.avalidate(jpeg())).s3_key == KEY
        assert state(svc.limiter) == (0, 0)

    run(scenario)


# --- lifespan ---


@pytest.fixture
def lifespan_manager(remote, monkeypatch):  # noqa: F811
    """lifespan 이 쓸 remote manager. 공유 client 가 가짜 worker 를 부르도록 transport 를 미리 넘긴다."""

    def build(fake):
        manager = BodyImageRuntimeManager(remote_transport=fake.transport)
        monkeypatch.setattr(main, "runtime_manager", manager)
        return manager

    return build


def test_lifespan_owns_the_shared_client(lifespan_manager, monkeypatch, clients):
    fake = FakeWorker()
    manager = lifespan_manager(fake)

    with TestClient(main.app) as client:
        service_ = manager.get_service()
        shared = service_._shared_client
        assert not shared.is_closed
        result = client.portal.call(service_.avalidate, jpeg())
        loop = client.portal.call(asyncio.get_running_loop)

    assert result.s3_key == KEY
    assert len(fake.calls) == 1  # 주입한 transport 로 나갔다
    assert shared.is_closed
    assert manager.runtime is None
    assert clients.loops == {"created": [loop], "closed": [loop]}


def test_restart_uses_a_new_client_on_the_new_event_loop(lifespan_manager, monkeypatch, clients):
    fake = FakeWorker()
    manager = lifespan_manager(fake)
    seen = []

    for _ in range(2):
        with TestClient(main.app) as client:
            service_ = manager.get_service()
            client.portal.call(service_.avalidate, jpeg())
            seen.append((service_._shared_client, client.portal.call(asyncio.get_running_loop)))

    (first, first_loop), (second, second_loop) = seen
    assert second is not first
    assert first_loop is not second_loop
    assert first.is_closed and second.is_closed
    assert len(fake.calls) == 2


def test_local_mode_creates_no_client(runtime, clients):  # noqa: F811
    with TestClient(main.app):
        pass

    assert clients.clients == []


def test_failed_remote_runtime_creates_no_client(lifespan_manager, monkeypatch, clients):
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_API_KEY", "")
    manager = lifespan_manager(FakeWorker())

    with TestClient(main.app):
        assert manager.runtime is None

    assert clients.clients == []


def test_client_creation_failure_stops_startup_and_releases_resources(remote, monkeypatch):  # noqa: F811
    def broken_build(self):
        raise OSError("bad CA bundle")

    cleaned = []

    async def close_async_pool():
        cleaned.append("async_pool")

    monkeypatch.setattr(RemoteBodyImageValidationService, "_build_client", broken_build)
    close = remote.close
    monkeypatch.setattr(remote, "close", lambda: (cleaned.append("runtime"), close()))
    monkeypatch.setattr(main, "close_connection_pool", lambda: cleaned.append("sync_pool"))
    monkeypatch.setattr(main, "close_async_pool", close_async_pool)

    with pytest.raises(OSError), TestClient(main.app):
        pass

    assert cleaned == ["runtime", "sync_pool", "async_pool"]
    assert remote.runtime is None
    assert main.app.state.body_image_warmup is None


def test_error_after_the_client_opened_still_closes_it(lifespan_manager, monkeypatch, clients):
    manager = lifespan_manager(FakeWorker())

    def broken():
        raise RuntimeError("bug")

    monkeypatch.setattr(warmup_router, "create_warmup_manager", broken)

    with pytest.raises(RuntimeError), TestClient(main.app):
        pass

    assert [client.is_closed for client in clients.clients] == [True]
    assert manager.runtime is None


def test_shutdown_order_is_warmup_then_client_then_runtime(lifespan_manager, monkeypatch):
    manager = lifespan_manager(FakeWorker())
    order = []

    async def record_warmup():
        order.append("warmup")

    stub = StubManager()
    stub.aclose = record_warmup
    monkeypatch.setattr(warmup_router, "create_warmup_manager", lambda: stub)
    real_aclose = RemoteBodyImageRuntime.aclose

    async def record_client(self):
        order.append("client")
        await real_aclose(self)

    monkeypatch.setattr(RemoteBodyImageRuntime, "aclose", record_client)
    close = manager.close
    monkeypatch.setattr(manager, "close", lambda: (order.append("runtime"), close()))

    with TestClient(main.app):
        pass

    assert order == ["warmup", "client", "runtime"]


def test_client_close_failure_does_not_skip_the_rest_of_shutdown(lifespan_manager, monkeypatch, caplog):
    manager = lifespan_manager(FakeWorker())
    cleaned = []

    async def broken_close(self):
        raise RuntimeError("secret ws-test")

    async def close_async_pool():
        cleaned.append("async_pool")

    monkeypatch.setattr(RemoteBodyImageValidationService, "aclose", broken_close)
    close = manager.close
    monkeypatch.setattr(manager, "close", lambda: (cleaned.append("runtime"), close()))
    monkeypatch.setattr(main, "close_connection_pool", lambda: cleaned.append("sync_pool"))
    monkeypatch.setattr(main, "close_async_pool", close_async_pool)

    with caplog.at_level(logging.ERROR), TestClient(main.app):
        pass

    assert cleaned == ["runtime", "sync_pool", "async_pool"]
    assert manager.runtime is None
    assert "body image runtime close failed: error=RuntimeError" in caplog.text
    assert "ws-test" not in caplog.text


def test_cancelled_shutdown_while_draining_still_closes_everything(lifespan_manager, monkeypatch, clients):
    worker = HeldWorker()
    manager = lifespan_manager(worker)
    cleaned = []

    async def close_async_pool():
        cleaned.append("async_pool")

    close = manager.close
    monkeypatch.setattr(manager, "close", lambda: (cleaned.append("runtime"), close()))
    monkeypatch.setattr(main, "close_connection_pool", lambda: cleaned.append("sync_pool"))
    monkeypatch.setattr(main, "close_async_pool", close_async_pool)

    async def scenario():
        running = []

        async def run_lifespan():
            async with main.lifespan(main.app):
                running.append(asyncio.create_task(manager.get_service().avalidate(jpeg())))
                await wait_until(worker.entered.is_set)

        task = asyncio.create_task(run_lifespan())
        await wait_until(lambda: running and worker.entered.is_set())
        await asyncio.sleep(0.01)
        assert not task.done()  # 진행 중인 호출을 기다리고 있다

        task.cancel()
        with pytest.raises(asyncio.CancelledError):  # 취소는 삼키지 않는다
            await task

        assert cleaned == ["runtime", "sync_pool", "async_pool"]
        assert clients.clients[0].is_closed
        worker.released.set()
        await asyncio.gather(*running, return_exceptions=True)

    run(scenario)


def test_runtime_manager_close_stays_for_the_modal_worker():
    closed = []

    class LocalRuntime:
        def close(self):
            closed.append(True)

    manager = BodyImageRuntimeManager()
    manager._runtime = LocalRuntime()

    asyncio.run(manager.aopen())
    asyncio.run(manager.aclose())

    assert closed == [True]  # local 은 기존 동기 close 그대로
    assert manager.runtime is None


def test_remote_transport_reaches_the_service(monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_MODE", "remote")
    monkeypatch.setattr(config, "REMOTE_URL", "https://example.modal.run")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_KEY", "k")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_SECRET", "s")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_API_KEY", "a")
    fake = FakeWorker()
    manager = BodyImageRuntimeManager(remote_transport=fake.transport)

    manager.start()

    assert manager.get_service()._transport is fake.transport


# --- API 는 공유 client 를 쓴다 ---


def test_api_requests_share_the_client_opened_by_the_lifespan(real_validation, clients):  # noqa: F811
    worker_calls = real_validation

    with TestClient(main.app, raise_server_exceptions=False) as client:
        responses = [validate_jpeg(client) for _ in range(3)]
        shared = main.runtime_manager.get_service()._shared_client
        sends = clients.sends[id(shared)]

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert worker_calls == [VALIDATE_URL] * 3
    assert clients.clients == [shared]  # 요청마다 만들지 않는다
    assert sends == 3
    assert shared.is_closed


def test_remote_runtime_uses_the_async_limiter_and_each_lifespan_gets_a_new_one(lifespan_manager):
    manager = lifespan_manager(FakeWorker())
    limiters = []

    for _ in range(2):
        with TestClient(main.app):
            limiters.append(manager.get_service().limiter)

    assert all(isinstance(limiter, AsyncBodyValidationLimiter) for limiter in limiters)
    assert limiters[0] is not limiters[1]


def test_remote_runtime_is_not_built_outside_the_lifespan(remote, monkeypatch, clients):  # noqa: F811
    monkeypatch.setattr(validation_router, "runtime_manager", remote)

    with pytest.raises(BodyImageRuntimeError):
        validation_router.get_service()

    assert remote.runtime is None
    assert clients.clients == []


def test_api_without_a_lifespan_is_runtime_unavailable(remote, monkeypatch, remote_client):  # noqa: F811
    monkeypatch.setattr(validation_router, "runtime_manager", remote)

    response = post(remote_client, jpeg())

    assert response.status_code == 500
    assert response.json()["code"] == "BODY_IMAGE_RUNTIME_UNAVAILABLE"
    assert remote.runtime is None


def test_runtime_whose_client_was_not_opened_answers_503(
    remote, monkeypatch, remote_client, caplog  # noqa: F811
):
    monkeypatch.setattr(validation_router, "runtime_manager", remote)
    remote.start()

    with caplog.at_level(logging.ERROR, logger="app.body_image_validation.remote"):
        response = post(remote_client, jpeg())

    assert response.status_code == 503
    assert response.json()["code"] == "BODY_IMAGE_WORKER_UNAVAILABLE"
    assert "client is not open" in caplog.text
