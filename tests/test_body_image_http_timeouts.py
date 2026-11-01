import asyncio
import logging
import socket
import time
from types import SimpleNamespace

import httpx
import pytest

from app.body_image_validation import deadline as worker_deadline
from app.body_image_validation import runtime as runtime_module
from app.body_image_validation.async_concurrency import AsyncBodyValidationLimiter
from app.body_image_validation.deadline import RequestDeadline
from app.body_image_validation.remote import ENDPOINT_PATH, RemoteWorkerConfigError
from app.body_image_validation.warmup import create_warmup_client
from app.config import body_image_validation as config
from app.config import settings
from tests.test_body_image_remote import POLL, SUCCESS, TIMEOUT, FakeWorker, jpeg, redirect
from tests.test_body_image_remote_async import KEY, OK, TIMED_OUT, opened, rejected, run, service
from tests.test_body_image_request_deadline import (  # noqa: F401 (pytest fixture)
    deadline_app,
    live,
    spent_limit,
    summaries,
)
from tests.test_body_image_warmup_api import remote, runtime  # noqa: F401 (pytest fixture)

LONG = 5.0
SHORT = 0.05
LARGE_BODY = 9_900_000  # 업로드 상한(10MB) 근처. socket buffer 를 넘겨 쓰기가 막힌다
LOGGER = "app.body_image_validation.remote"


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    return caplog


def phases(**overrides):
    return httpx.Timeout(**{"connect": LONG, "read": LONG, "write": LONG, "pool": LONG, **overrides})


def applied(request):
    return request.extensions["timeout"]


async def read_request(reader):
    head = await reader.readuntil(b"\r\n\r\n")
    length = next(
        int(line.split(b":")[1]) for line in head.split(b"\r\n") if line.lower().startswith(b"content-length")
    )
    await reader.readexactly(length)


async def serve(server, reader, writer):
    """keep-alive 로 요청을 받아 성공 응답을 보낸다."""
    while not reader.at_eof():
        await read_request(reader)
        server.requests += 1
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s" % (len(SUCCESS), SUCCESS))
        await writer.drain()


async def stay_silent(server, reader, writer):
    await asyncio.Event().wait()


class RawServer:
    """loopback 서버. 테스트와 같은 event loop 에서 돌고, 연결마다 first 또는 rest 처리를 고른다."""

    def __init__(self, first, rest=serve):
        self.first, self.rest = first, rest
        self.connections = self.requests = 0
        self.writers = []

    async def __aenter__(self):
        listener = socket.create_server(("127.0.0.1", 0))
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)  # 읽지 않으면 바로 쓰기가 막히게 한다
        self.server = await asyncio.start_server(self._accept, sock=listener)
        self.port = listener.getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        for writer in self.writers:
            writer.transport.abort()

    async def _accept(self, reader, writer):
        self.connections += 1
        self.writers.append(writer)
        handle = self.first if self.connections == 1 else self.rest
        try:
            await handle(self, reader, writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass  # 호출하는 쪽이 끊었다


async def open_service(server, timeout, scheme="http", transport=None, **kwargs):
    svc = service(SimpleNamespace(transport=transport), timeout=timeout, **kwargs)
    svc.endpoint = f"{scheme}://127.0.0.1:{server.port}{ENDPOINT_PATH}"
    return await opened(svc)


def idle(svc):
    return (svc.limiter.running, svc.limiter.waiting) == (0, 0)


def logged(logs, *keys):
    return tuple(summaries(logs)[-1][key] for key in keys)


# --- 단계별 timeout 이 각각 끝나는 경우. 실제 socket 으로 재현한다 ---


def test_connect_timeout_when_tls_never_finishes_sends_nothing(logs):
    async def scenario():
        async with RawServer(stay_silent) as server:
            svc = await open_service(server, phases(connect=SHORT), scheme="https")

            assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
            assert (server.connections, server.requests) == (1, 0)
            assert idle(svc)

    run(scenario)

    assert logged(logs, "outcome", "error", "timeout_stage") == ("network_timeout", "connect_timeout", "http")


def test_read_timeout_is_not_retried_and_the_next_request_uses_a_new_connection(logs):
    async def hang_after_the_request(server, reader, writer):
        await read_request(reader)
        server.requests += 1
        await asyncio.Event().wait()

    async def scenario():
        async with RawServer(hang_after_the_request) as server:
            svc = await open_service(server, phases(read=SHORT))

            assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
            assert (server.connections, server.requests) == (1, 1)  # POST 를 다시 보내지 않았다
            assert idle(svc)
            assert logged(logs, "outcome", "error") == ("network_timeout", "read_timeout")

            svc.timeout = phases()
            assert (await svc.avalidate(jpeg())).s3_key == KEY
            assert server.connections == 2

    run(scenario)


def test_write_timeout_when_the_server_stops_reading_is_not_retried(logs):
    async def scenario():
        async with RawServer(stay_silent) as server:
            svc = await open_service(server, phases(write=SHORT))
            image = jpeg() + b"\0" * LARGE_BODY

            assert await rejected(svc.avalidate(image)) == TIMED_OUT
            assert (server.connections, server.requests) == (1, 0)
            assert idle(svc)
            assert logged(logs, "outcome", "error") == ("network_timeout", "write_timeout")

            svc.timeout = phases()
            assert (await svc.avalidate(jpeg())).s3_key == KEY
            assert server.connections == 2

    run(scenario)


@pytest.mark.parametrize(("write", "succeeds"), [(LONG, True), (0.05, False)], ids=["longer", "shorter"])
def test_upload_held_back_by_a_cold_start_needs_a_longer_write_timeout(write, succeeds):
    # Modal 은 처리가 밀리면 TCP 로 backpressure 를 전한다. 그동안 write 는 진행되지 않는다.
    async def cold_start(server, reader, writer):
        await asyncio.sleep(0.2)
        await serve(server, reader, writer)

    async def scenario():
        async with RawServer(cold_start) as server:
            svc = await open_service(server, phases(write=write))
            image = jpeg() + b"\0" * LARGE_BODY

            if succeeds:
                assert (await svc.avalidate(image)).s3_key == KEY
            else:
                assert await rejected(svc.avalidate(image)) == TIMED_OUT

    run(scenario)


def pool_of_one():
    return httpx.AsyncHTTPTransport(limits=httpx.Limits(max_connections=1))


async def holding_the_only_connection(server, release):
    async def held(server, reader, writer):
        await read_request(reader)
        server.requests += 1
        await release.wait()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s" % (len(SUCCESS), SUCCESS))
        await writer.drain()
        await serve(server, reader, writer)

    server.first = held


def test_pool_timeout_never_reaches_the_worker_and_the_connection_stays_usable(logs):
    async def scenario():
        release = asyncio.Event()
        async with RawServer(serve) as server:
            await holding_the_only_connection(server, release)
            limiter = AsyncBodyValidationLimiter(2, 0)
            svc = await open_service(server, phases(pool=SHORT), transport=pool_of_one(), limiter=limiter)
            first = asyncio.create_task(svc.avalidate(jpeg()))
            while server.requests == 0:
                await asyncio.sleep(0.001)

            assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
            assert server.requests == 1  # 연결을 얻지 못해 전송되지 않았다
            assert logged(logs, "outcome", "error") == ("network_timeout", "pool_timeout")

            release.set()
            assert (await first).s3_key == KEY
            assert (await svc.avalidate(jpeg())).s3_key == KEY
            assert (server.connections, server.requests) == (1, 2)  # keep-alive 연결을 그대로 쓴다
            assert idle(svc)

    run(scenario)


def test_cancel_while_waiting_for_a_pool_connection_leaves_nothing_behind():
    async def scenario():
        release = asyncio.Event()
        async with RawServer(serve) as server:
            await holding_the_only_connection(server, release)
            limiter = AsyncBodyValidationLimiter(2, 0)
            svc = await open_service(server, phases(), transport=pool_of_one(), limiter=limiter)
            first = asyncio.create_task(svc.avalidate(jpeg()))
            while server.requests == 0:
                await asyncio.sleep(0.001)
            waiting = asyncio.create_task(svc.avalidate(jpeg()))
            while limiter.running < 2:
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.01)

            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting

            assert limiter.running == 1
            release.set()
            await first
            assert (await svc.avalidate(jpeg())).s3_key == KEY
            assert server.connections == 1
            assert idle(svc)

    run(scenario)


def test_cancel_during_a_blocked_upload_returns_the_slot():
    async def scenario():
        async with RawServer(stay_silent) as server:
            svc = await open_service(server, phases())
            call = asyncio.create_task(svc.avalidate(jpeg() + b"\0" * LARGE_BODY))
            while server.connections == 0:
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.01)

            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call

            assert idle(svc)

    run(scenario)


# --- 남은 전체 deadline 이 우선한다 ---


def clocked_service(handler, timeout, now):
    fake = SimpleNamespace(transport=httpx.MockTransport(handler))
    return service(fake, timeout=timeout, deadline=120.0, clock=lambda: now[0], wall_clock=lambda: 5000.0)


@pytest.mark.parametrize(
    ("left", "connect", "read", "write", "pool"),
    [
        (120.0, 10.0, 120.0, 60.0, 5.0),
        (30.0, 10.0, 30.0, 30.0, 5.0),
        (7.0, 7.0, 7.0, 7.0, 5.0),
        (3.0, 3.0, 3.0, 3.0, 3.0),
    ],
    ids=["all-time-left", "read-write-cut", "connect-cut", "all-cut"],
)
def test_each_phase_is_cut_to_the_time_left(left, connect, read, write, pool):
    now = [1000.0]
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=SUCCESS)

    async def scenario():
        svc = await opened(clocked_service(handler, httpx.Timeout(connect=10, read=120, write=60, pool=5), now))
        limit = RequestDeadline(120.0, clock=lambda: now[0])
        now[0] += 120.0 - left

        await svc.avalidate(jpeg(), limit)

    run(scenario)

    assert applied(calls[0]) == {"connect": connect, "read": read, "write": write, "pool": pool}


def test_redirect_get_gets_only_the_time_left_and_the_same_worker_deadline():
    now = [1000.0]
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            now[0] += 113.0
            return httpx.Response(303, headers={"location": POLL})
        return httpx.Response(200, content=SUCCESS)

    async def scenario():
        svc = await opened(clocked_service(handler, httpx.Timeout(connect=10, read=120, write=60, pool=5), now))

        assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)

    post_call, get_call = calls
    assert applied(post_call) == {"connect": 10.0, "read": 120.0, "write": 60.0, "pool": 5.0}
    assert applied(get_call) == {"connect": 7.0, "read": 7.0, "write": 7.0, "pool": 5.0}
    assert post_call.headers[worker_deadline.HEADER] == get_call.headers[worker_deadline.HEADER]


def test_no_redirect_get_after_the_deadline(logs):
    now = [1000.0]
    calls = []

    def handler(request):
        calls.append(request)
        now[0] += 120.0
        return httpx.Response(303, headers={"location": POLL})

    async def scenario():
        svc = await opened(clocked_service(handler, phases(), now))

        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
        assert idle(svc)

    run(scenario)

    assert [request.method for request in calls] == ["POST"]
    assert logged(logs, "outcome", "timeout_stage", "redirects") == ("deadline_exceeded", "http", "1")


def test_no_post_when_the_deadline_passes_right_before_it():
    now = [1000.0]
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=SUCCESS)

    async def scenario():
        svc = await opened(clocked_service(handler, phases(), now))
        real_network_timeout = svc._network_timeout

        def late(deadline_at):
            now[0] = deadline_at
            return real_network_timeout(deadline_at)

        svc._network_timeout = late

        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
        assert idle(svc)

    run(scenario)

    assert calls == []


@pytest.mark.parametrize(
    ("error", "expired", "outcome", "reason"),
    [
        (httpx.ConnectTimeout, False, "network_timeout", "connect_timeout"),
        (httpx.ReadTimeout, False, "network_timeout", "read_timeout"),
        (httpx.WriteTimeout, False, "network_timeout", "write_timeout"),
        (httpx.PoolTimeout, False, "network_timeout", "pool_timeout"),
        (httpx.ConnectTimeout, True, "deadline_exceeded", "deadline_exceeded"),
        (httpx.ReadTimeout, True, "deadline_exceeded", "deadline_exceeded"),
        (httpx.WriteTimeout, True, "deadline_exceeded", "deadline_exceeded"),
        (httpx.PoolTimeout, True, "deadline_exceeded", "deadline_exceeded"),
    ],
)
def test_timeout_cut_to_the_deadline_is_logged_as_the_deadline(logs, error, expired, outcome, reason):
    now = [1000.0]

    def handler(request):
        if expired:
            now[0] += 120.0
        raise error("timed out", request=request)

    async def scenario():
        svc = await opened(clocked_service(handler, phases(), now))

        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT

    run(scenario)

    assert logged(logs, "outcome", "error") == (outcome, reason)


@pytest.mark.parametrize("scheme", ["https", "http"], ids=["while-connecting", "while-reading"])
def test_deadline_ends_the_call_when_it_is_shorter_than_the_phase(logs, scheme):
    async def hang(server, reader, writer):
        if scheme == "http":
            await read_request(reader)
            server.requests += 1
        await asyncio.Event().wait()

    async def scenario():
        async with RawServer(hang) as server:
            svc = await open_service(server, phases(), scheme=scheme)
            started = time.monotonic()

            assert await rejected(svc.avalidate(jpeg(), spent_limit(120.0, left=0.1))) == TIMED_OUT

            assert time.monotonic() - started < 1.0  # 단계의 5초가 아니라 남은 0.1초에 끝났다
            assert idle(svc)

    run(scenario)

    assert logged(logs, "outcome", "error", "timeout_stage") == (
        "deadline_exceeded", "deadline_exceeded", "http",
    )


# --- 정상 경로·공유 client·설정 ---


def test_requests_within_the_limits_reuse_one_connection_with_the_shared_client():
    async def scenario():
        async with RawServer(serve) as server:
            svc = await open_service(server, phases(connect=0.5, write=0.5, pool=0.5))
            client = svc._shared_client

            for _ in range(3):
                assert (await svc.avalidate(jpeg())).s3_key == KEY

            assert (server.connections, server.requests) == (1, 3)
            assert svc._shared_client is client

    run(scenario)


def test_slow_response_within_the_read_timeout_is_kept_past_the_other_phases():
    async def slow(server, reader, writer):
        await read_request(reader)
        await asyncio.sleep(0.2)  # GPU 추론 중에는 아무 데이터도 오지 않는다
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s" % (len(SUCCESS), SUCCESS))
        await writer.drain()

    async def scenario():
        async with RawServer(slow) as server:
            svc = await open_service(server, phases(connect=0.05, write=0.05, pool=0.05))

            assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)


def test_redirect_policy_still_follows_303_with_separate_phases():
    async def scenario():
        fake = FakeWorker(steps=[redirect(POLL), OK])
        svc = await opened(service(fake, timeout=phases(connect=1.0)))

        assert (await svc.avalidate(jpeg())).s3_key == KEY
        assert [call.method for call in fake.calls] == ["POST", "GET"]
        assert all(applied(call)["connect"] == 1.0 for call in fake.calls)

    run(scenario)


@pytest.mark.parametrize(
    "timeout",
    [0, -1.0, httpx.Timeout(5.0, connect=0), httpx.Timeout(5.0, pool=None), httpx.Timeout(None)],
    ids=["zero", "negative", "one-phase-zero", "one-phase-unlimited", "unlimited"],
)
def test_every_phase_must_be_a_positive_limit(timeout):
    with pytest.raises(RemoteWorkerConfigError, match="단계마다"):
        service(FakeWorker(), timeout=timeout)


def test_policy_values_and_warmup_stay_as_decided(monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_MODE", "remote")
    monkeypatch.setattr(config, "REMOTE_URL", "https://example.modal.run")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_KEY", "k")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_SECRET", "s")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_API_KEY", "a")

    built = runtime_module.RemoteBodyImageRuntime().service
    warmup = create_warmup_client()

    assert built.timeout == httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=5.0)
    assert built.total_deadline == 120.0
    assert warmup.timeout == 90.0


# --- 실제 API: 응답 계약은 그대로다 ---


def test_api_read_timeout_keeps_the_504_contract_and_leaves_modal_running(deadline_app):  # noqa: F811
    server = deadline_app(30.0)
    server.service.timeout = phases(read=0.2)

    assert server.send().read().response() == (504, TIMEOUT)
    assert server.slots == (0, 0)
    assert (server.modal.started, server.modal.active, server.modal.finished) == (1, 1, 0)
