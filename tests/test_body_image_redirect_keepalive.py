import asyncio
import itertools
import json
import logging
import random
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from app.body_image_validation.async_concurrency import AsyncBodyValidationLimiter
from app.body_image_validation.exceptions import BodyImageValidationError
from tests.test_body_image_http_timeouts import (
    RawServer,
    open_service,
    phases,
    read_request,
    serve,
    stay_silent,
)
from tests.test_body_image_remote import (
    POLL,
    SUCCESS,
    UNAVAILABLE,
    URL,
    WORKER,
    FakeWorker,
    jpeg,
    post,
    redirect,
    remote_service,
    use,
)
from tests.test_body_image_remote import client as remote_client  # noqa: F401 (pytest fixture)
from tests.test_body_image_remote_async import (
    BAD_GATEWAY,
    KEY,
    OK,
    TIMED_OUT,
    opened,
    rejected,
    run,
    service,
)
from tests.test_body_image_remote_async import UNAVAILABLE as CONNECTION_FAILED
from tests.test_body_image_request_deadline import summaries

LOGGER = "app.body_image_validation.remote"
HOST = httpx.URL(WORKER).host
OK_RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s" % (len(SUCCESS), SUCCESS)
LARGE_BODY = 9_900_000
WAIT = 5.0


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    return caplog


def idle(svc):
    return (svc.limiter.running, svc.limiter.waiting) == (0, 0)


def worker_answering(handler):
    return SimpleNamespace(transport=httpx.MockTransport(handler))


def redirect_once(location, status=303):
    """첫 요청에 redirect, 그다음은 성공. 받은 요청을 calls 에 남긴다."""
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) > 1:
            return httpx.Response(200, content=SUCCESS)
        return httpx.Response(status, headers={} if location is None else {"location": location})

    return handler, calls


# --- 303 정상 경로 ---


def test_303_becomes_a_get_without_the_image_and_keeps_the_auth_headers():
    handler, calls = redirect_once(POLL)

    async def scenario():
        svc = await opened(service(worker_answering(handler)))

        assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)

    post_call, get_call = calls
    assert (post_call.method, get_call.method) == ("POST", "GET")
    assert str(get_call.url) == WORKER + POLL
    assert get_call.content == b""
    assert "content-type" not in get_call.headers and "content-length" not in get_call.headers
    for name in ("modal-key", "modal-secret", "authorization"):
        assert get_call.headers[name] == post_call.headers[name]


def test_get_after_a_303_carries_no_body_on_the_wire():
    # 로컬 서버는 https 가 아니어서 대상 판단만 건너뛴다. 나머지는 실제 소켓으로 나가는 그대로다.
    seen = {}

    async def post_then_get(server, reader, writer):
        await read_request(reader)
        writer.write(b"HTTP/1.1 303 See Other\r\nLocation: /poll\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        await asyncio.Event().wait()

    async def get_only(server, reader, writer):
        seen["head"] = await reader.readuntil(b"\r\n\r\n")
        writer.write(OK_RESPONSE)
        await writer.drain()
        await reader.read(1)

    async def scenario():
        async with RawServer(post_then_get, rest=get_only) as server:
            svc = await open_service(server, phases())
            svc._redirect_target = lambda current, location, visited, count: current.join(location)

            assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)

    head = seen["head"].lower()
    assert head.startswith(b"get /poll http/1.1")
    assert b"content-length" not in head and b"content-type" not in head
    assert b"bearer worker-key" in head


@pytest.mark.parametrize(
    "location",
    [
        POLL,
        f"https://{HOST}{POLL}",
        f"https://{HOST}:443{POLL}",
        f"https://{HOST.upper()}{POLL}",
        "?__modal_function_call_id=fc-1",
    ],
    ids=["relative", "absolute", "default-port", "upper-case-host", "query-only"],
)
def test_same_origin_forms_are_followed_with_the_bearer_key(location):
    handler, calls = redirect_once(location)

    async def scenario():
        svc = await opened(service(worker_answering(handler)))

        assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)

    assert calls[1].headers["authorization"] == "Bearer worker-key"
    assert calls[1].url.host == HOST


# --- 잘못된 Redirect: 502, 호출 1건, 원시 예외 없음 ---

REJECTED = {
    "missing": None,
    "empty": "",
    "bad-port": "https://x:notaport/",
    "port-out-of-range": f"https://{HOST}:99999/",
    "control-char": "/\x7f",
    "nul": "/\x00",
    "newline": "/\n",
    "tab": "\t",
    "unbalanced-bracket": "http://[::1",
    "too-long-url": "https://" + "a" * 70000 + "/",
    "too-long-path": "/" + "a" * 70000,
    "javascript": "javascript:alert(1)",
    "data": "data:text/plain,x",
    "backslashes": "https:\\\\evil.example.com",
    "idna": "https://xn--/",
    "no-host": "https:///x",
    "scheme-only": "https:",
    "plain-http": f"http://{HOST}{POLL}",
    "ftp": f"ftp://{HOST}/x",
    "file": "file:///etc/passwd",
    "other-origin": "https://evil.example.com/x",
    "protocol-relative": "//evil.example.com/x",
    "other-port": f"https://{HOST}:444{POLL}",
    "sub-domain-of-evil": f"https://{HOST}.evil.com{POLL}",
    "host-in-userinfo-of-evil": f"https://{HOST}@evil.com/",
    "fragment-trick": f"https://evil.com#{HOST}",
    "userinfo": f"https://user:pw@{HOST}{POLL}",
    "evil-as-userinfo": f"https://evil.com@{HOST}{POLL}",
}


@pytest.mark.parametrize("location", REJECTED.values(), ids=REJECTED.keys())
def test_invalid_or_foreign_location_is_502_and_nothing_more_is_sent(location):
    handler, calls = redirect_once(location)

    async def scenario():
        svc = await opened(service(worker_answering(handler)))

        assert await rejected(svc.avalidate(jpeg())) == BAD_GATEWAY
        assert idle(svc)

    run(scenario)

    assert [request.method for request in calls] == ["POST"]


@pytest.mark.parametrize("status", [301, 302, 307, 308])
@pytest.mark.parametrize("location", [POLL, "https://x:notaport/", "javascript:alert(1)"], ids=["valid", "port", "js"])
def test_other_redirect_statuses_are_502_whatever_the_location(status, location):
    handler, calls = redirect_once(location, status=status)

    async def scenario():
        svc = await opened(service(worker_answering(handler)))

        assert await rejected(svc.avalidate(jpeg())) == BAD_GATEWAY

    run(scenario)

    assert len(calls) == 1


def test_location_header_on_a_success_is_ignored():
    def handler(request):
        return httpx.Response(200, headers={"location": "https://x:notaport/"}, content=SUCCESS)

    async def scenario():
        svc = await opened(service(worker_answering(handler)))

        assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)


def test_random_locations_never_escape_as_raw_errors_or_leave_the_origin():
    rng = random.Random(20261101)
    pieces = [
        "https://", "http://", "//", "/", "\\", ":", "@", "?", "#", "[", "]", "%", "xn--", HOST, "evil.com", "443",
        "x", " ", "\t", "\x7f", "\x00", "..", "javascript:", "data:", "user:pw@", "é", "☃",
    ]
    locations = ["".join(rng.choices(pieces, k=rng.randint(1, 6))) for _ in range(800)]
    outcomes = set()

    async def one(location):
        handler, calls = redirect_once(location.encode())
        svc = await opened(service(worker_answering(handler)))
        try:
            await svc.avalidate(jpeg())
            outcomes.add(200)
        except BodyImageValidationError as error:
            outcomes.add(error.status_code)
        assert idle(svc)
        for request in calls[1:]:
            assert (request.url.scheme, request.url.host, request.url.port) == ("https", HOST, None)
            assert request.headers["authorization"] == "Bearer worker-key"

    async def scenario():
        for location in locations:
            await one(location)

    run(scenario)

    assert outcomes == {200, 502}


def test_invalid_location_never_reaches_the_logs_or_the_response(remote_client, monkeypatch, logs):  # noqa: F811
    secret = "fc-secret-value"
    location = f"https://x:notaport/?__modal_function_call_id={secret}"
    fake = FakeWorker(steps=[redirect(location)])
    use(monkeypatch, remote_service(fake))

    response = post(remote_client, jpeg())

    assert (response.status_code, response.json()) == (502, UNAVAILABLE)
    assert secret not in logs.text
    assert "notaport" not in logs.text
    assert "invalid Location header: status=303" in logs.text


def test_valid_redirect_url_and_cookie_stay_out_of_the_logs(logs):
    secret = "fc-secret-value"

    def handler(request):
        if request.method == "POST":
            headers = {"location": f"{URL}?__modal_function_call_id={secret}", "set-cookie": "sid=cookie-value; Path=/"}
            return httpx.Response(303, headers=headers)
        return httpx.Response(200, content=SUCCESS)

    async def scenario():
        svc = await opened(service(worker_answering(handler)))

        await svc.avalidate(jpeg())

    run(scenario)

    assert secret not in logs.text and "cookie-value" not in logs.text and "worker-key" not in logs.text


def test_bad_location_on_a_real_connection_leaves_the_only_connection_free():
    async def bad_redirect(server, reader, writer):
        await read_request(reader)
        writer.write(b"HTTP/1.1 303 See Other\r\nLocation: https://x:notaport/\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        await serve(server, reader, writer)

    async def scenario():
        async with RawServer(bad_redirect) as server:
            transport = httpx.AsyncHTTPTransport(limits=httpx.Limits(max_connections=1))
            svc = await open_service(server, phases(pool=0.5), transport=transport)

            assert await rejected(svc.avalidate(jpeg())) == BAD_GATEWAY
            assert (await svc.avalidate(jpeg())).s3_key == KEY  # 풀이 1개여도 PoolTimeout 이 나지 않는다
            assert idle(svc)

    run(scenario)


def test_polling_the_same_result_url_stays_allowed_and_limited():
    async def scenario():
        fake = FakeWorker(steps=[redirect(POLL), redirect(POLL), OK])
        svc = await opened(service(fake))

        assert (await svc.avalidate(jpeg())).s3_key == KEY
        assert [call.method for call in fake.calls] == ["POST", "GET", "GET"]

    run(scenario)


# --- keep-alive: 실제 TCP 서버 ---


def reused_by(second):
    """연결 하나로 첫 요청은 정상 응답하고, 두 번째 요청이 올 때 second 가 그 연결을 처리한다."""

    async def connection(server, reader, writer):
        await read_request(reader)
        server.requests += 1
        writer.write(OK_RESPONSE)
        await writer.drain()
        await second(server, reader, writer)

    return connection


async def fin_after_the_headers(server, reader, writer):
    await reader.readuntil(b"\r\n\r\n")
    server.arrived = server.arrived + 1
    writer.close()


async def rst_after_the_headers(server, reader, writer):
    await reader.readuntil(b"\r\n\r\n")
    server.arrived = server.arrived + 1
    writer.transport.abort()


async def rst_during_the_upload(server, reader, writer):
    await reader.readuntil(b"\r\n\r\n")
    await reader.readexactly(100_000)
    server.arrived = server.arrived + 1
    writer.transport.abort()


async def truncated_response(server, reader, writer):
    await read_request(reader)
    server.arrived = server.arrived + 1
    writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 500\r\n\r\n{"code"')
    await writer.drain()
    writer.close()


async def not_http(server, reader, writer):
    await read_request(reader)
    server.arrived = server.arrived + 1
    writer.write(b"NOT HTTP\r\n\r\n")
    await writer.drain()
    writer.close()


# 연결이 닫히는 순간 client 가 어디까지 보냈는지에 따라 쓰기 또는 읽기가 먼저 실패한다. httpcore 는 쓰기 실패를
# 삼키고 응답을 읽으므로 error 는 write_error 가 아니라 read_error 다.
ARRIVING = ("send_request_body", "receive_response_headers")
STALE_CASES = {
    "fin-as-the-request-arrives": (fin_after_the_headers, "read_error", ARRIVING, 0),
    "rst-as-the-request-arrives": (rst_after_the_headers, "read_error", ARRIVING, 0),
    "rst-during-the-upload": (rst_during_the_upload, "read_error", ("send_request_body",), LARGE_BODY),
    "truncated-response": (truncated_response, "remote_protocol_error", ("receive_response_body",), 0),
    "not-http": (not_http, "remote_protocol_error", ("receive_response_headers",), 0),
}


@pytest.mark.parametrize(("second", "error", "failed_at", "padding"), STALE_CASES.values(), ids=STALE_CASES.keys())
def test_reused_connection_failing_is_503_once_and_the_next_request_works(logs, second, error, failed_at, padding):
    async def scenario():
        async with RawServer(reused_by(second)) as server:
            server.arrived = 0
            # 연결 1개짜리 풀: 끊긴 연결이 풀에 남으면 다음 요청이 PoolTimeout 으로 실패한다.
            transport = httpx.AsyncHTTPTransport(limits=httpx.Limits(max_connections=1))
            svc = await open_service(server, phases(pool=0.5), transport=transport)
            client = svc._shared_client
            assert (await svc.avalidate(jpeg())).s3_key == KEY

            assert await rejected(svc.avalidate(jpeg() + b"\0" * padding)) == CONNECTION_FAILED

            assert server.arrived == 1  # 같은 POST 를 다시 보내지 않았다
            assert server.connections == 1
            assert idle(svc)
            logged = summaries(logs)[-1]
            assert (logged["outcome"], logged["error"]) == ("connection_failed", error)
            assert logged["connection"] == "reused"
            assert logged["failed_at"] in failed_at
            assert logged["remote_stopped"] == "unknown"  # 요청이 이미 Modal 에 닿았을 수 있다

            assert (await svc.avalidate(jpeg())).s3_key == KEY
            assert server.connections == 2
            assert summaries(logs)[-1]["connection"] == "new"
            assert svc._shared_client is client and not client.is_closed

    run(scenario)


def test_healthy_connection_is_reused_and_logged_as_reused(logs):
    async def scenario():
        async with RawServer(serve) as server:
            svc = await open_service(server, phases())

            for _ in range(3):
                assert (await svc.avalidate(jpeg())).s3_key == KEY

            assert (server.connections, server.requests) == (1, 3)

    run(scenario)

    assert [(s["connection"], s["failed_at"]) for s in summaries(logs)] == [
        ("new", "-"), ("reused", "-"), ("reused", "-"),
    ]


def test_connection_closed_while_idle_is_replaced_without_an_error(logs):
    async def close_when_idle(server, reader, writer):
        writer.close()

    async def scenario():
        async with RawServer(reused_by(close_when_idle)) as server:
            svc = await open_service(server, phases())
            assert (await svc.avalidate(jpeg())).s3_key == KEY
            await asyncio.sleep(0.02)  # FIN 이 client 에 닿은 뒤다

            assert (await svc.avalidate(jpeg())).s3_key == KEY

            assert server.connections == 2

    run(scenario)

    assert [s["connection"] for s in summaries(logs)] == ["new", "new"]


# 같은 event loop 의 서버로는 경합이 나지 않는다(종료가 항상 먼저 끝나 httpcore 가 닫힌 연결을 걸러 낸다).
# 그래서 서버를 별도 프로세스로 띄운다. 유휴 시간이 지나면 연결을 닫고, 받은 요청의 deadline header 를 기록한다.
RACE_SERVER = r"""
import asyncio, json, sys

IDLE = float(sys.argv[1])
seen, connections = [], [0]
ANSWER = sys.stdin.buffer.readline().rstrip(b"\n").decode("unicode_escape").encode("latin-1")


async def handle(reader, writer):
    connections[0] += 1
    served = False
    while True:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), IDLE if served else None)
        except (TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            writer.close()
            return
        lines = [line.lower().decode() for line in head.split(b"\r\n")]
        seen.append(next(line.split(":")[1].strip() for line in lines if line.startswith("x-body-image-deadline")))
        length = next(int(line.split(":")[1]) for line in lines if line.startswith("content-length"))
        try:
            await reader.readexactly(length)
            writer.write(ANSWER)
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            return
        served = True


async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    print(server.sockets[0].getsockname()[1], flush=True)
    await asyncio.get_running_loop().run_in_executor(None, sys.stdin.read)
    print(json.dumps({"seen": seen, "connections": connections[0]}), flush=True)


asyncio.run(main())
"""


class RaceServer:
    """별도 프로세스의 loopback 서버."""

    def __init__(self, idle):
        self.process = subprocess.Popen(
            [sys.executable, "-c", RACE_SERVER, str(idle)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        self.process.stdin.write(OK_RESPONSE.decode("latin-1").encode("unicode_escape").decode() + "\n")
        self.process.stdin.flush()
        self.port = int(self.process.stdout.readline())

    def finish(self):
        stats, _ = self.process.communicate(timeout=WAIT)
        return json.loads(stats)

    def kill(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.communicate()


def test_connection_closing_as_the_next_request_goes_out_is_503_and_never_resent(logs):
    # 서버가 닫는 시점 직전에 요청이 나가게 간격을 훑는다. 재사용 연결의 실패가 실제로 나올 때까지 반복한다.
    idle_close = 0.03
    max_attempts, enough_failures = 1200, 3
    server = RaceServer(idle_close)
    image = jpeg()

    async def scenario():
        svc = service(SimpleNamespace(transport=None), timeout=phases())
        svc.endpoint = f"http://127.0.0.1:{server.port}{URL}"
        await svc.aopen()
        client = svc._shared_client
        gaps = itertools.cycle(idle_close - ms / 1000 for ms in (6, 5, 4, 3, 2, 1))
        attempts = failed = 0

        while attempts < max_attempts and failed < enough_failures:
            attempts += 1
            try:
                await svc.avalidate(image)
            except BodyImageValidationError as error:
                assert (error.status_code, error.code) == CONNECTION_FAILED
                failed += 1
            await asyncio.sleep(next(gaps))

        assert idle(svc)
        assert svc._shared_client is client and not client.is_closed
        assert (await svc.avalidate(image)).s3_key == KEY  # 경합 뒤에도 다음 요청은 정상이다
        await svc.aclose()
        return attempts

    try:
        attempts = asyncio.run(scenario())
        stats = server.finish()
    finally:
        server.kill()

    records = summaries(logs)
    stale = [record for record in records if record["outcome"] == "connection_failed"]
    assert len(stale) == enough_failures, f"{attempts}번 시도에서 경합이 재현되지 않았다"
    assert {record["connection"] for record in stale} == {"reused"}
    assert {record["error"] for record in stale} <= {"read_error", "remote_protocol_error"}
    # 요청마다 deadline 값이 다르다. 같은 값이 두 번 왔다면 같은 POST 를 다시 보낸 것이다.
    assert len(stats["seen"]) == len(set(stats["seen"]))
    assert len(stats["seen"]) <= len(records)


def test_failure_on_one_connection_leaves_a_running_request_alone():
    async def scenario():
        release = asyncio.Event()
        plan = ["both", "both", "hold", "kill", "ok"]
        arrived = asyncio.Condition()

        async def handle(server, reader, writer):
            while True:
                await read_request(reader)
                step = plan.pop(0)
                async with arrived:
                    server.requests += 1
                    arrived.notify_all()
                if step == "kill":
                    writer.transport.abort()
                    return
                if step == "both":
                    async with arrived:
                        await arrived.wait_for(lambda: server.requests >= 2)
                elif step == "hold":
                    await release.wait()
                writer.write(OK_RESPONSE)
                await writer.drain()

        async with RawServer(handle, rest=handle) as server:
            limiter = AsyncBodyValidationLimiter(2, 0)
            svc = await open_service(server, phases(), limiter=limiter)
            client = svc._shared_client
            await asyncio.gather(svc.avalidate(jpeg()), svc.avalidate(jpeg()))  # 유휴 연결 2개를 만든다
            assert server.connections == 2

            running = asyncio.create_task(svc.avalidate(jpeg()))
            async with arrived:
                await arrived.wait_for(lambda: server.requests == 3)

            assert await rejected(svc.avalidate(jpeg())) == CONNECTION_FAILED

            assert not running.done()
            assert not client.is_closed
            release.set()
            assert (await running).s3_key == KEY
            assert (await svc.avalidate(jpeg())).s3_key == KEY
            assert idle(svc)

    run(scenario)


def test_client_closes_cleanly_after_connection_failures():
    async def scenario():
        async with RawServer(reused_by(rst_after_the_headers)) as server:
            server.arrived = 0
            svc = await open_service(server, phases())
            client = svc._shared_client
            await svc.avalidate(jpeg())
            await rejected(svc.avalidate(jpeg()))

            await svc.aclose()

            assert client.is_closed

    run(scenario)


# --- 요약 로그로 끊긴 단계를 구분한다 ---


def test_failed_step_tells_a_blocked_upload_from_a_silent_worker(logs):
    async def read_then_hang(server, reader, writer):
        await read_request(reader)
        await asyncio.Event().wait()

    async def scenario():
        async with RawServer(stay_silent, rest=read_then_hang) as server:
            svc = await open_service(server, phases(write=0.05, read=0.05))

            assert await rejected(svc.avalidate(jpeg() + b"\0" * LARGE_BODY)) == TIMED_OUT
            assert summaries(logs)[-1]["failed_at"] == "send_request_body"
            assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
            assert summaries(logs)[-1]["failed_at"] == "receive_response_headers"

    run(scenario)


def test_failed_step_names_the_tls_handshake(logs):
    async def scenario():
        async with RawServer(stay_silent) as server:
            svc = await open_service(server, phases(connect=0.05), scheme="https")

            assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT

    run(scenario)

    logged = summaries(logs)[-1]
    assert (logged["connection"], logged["failed_at"], logged["error"]) == ("new", "start_tls", "connect_timeout")


def test_redirect_get_opens_a_new_connection_and_is_logged_as_new(logs):
    # 303 의 본문을 읽지 않아 httpcore 가 그 연결을 닫는다. 정상 운영에서는 303 이 오지 않으므로 그대로 둔다.
    async def then_redirect(server, reader, writer):
        await read_request(reader)
        writer.write(b"HTTP/1.1 303 See Other\r\nLocation: /poll\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()

    async def get_only(server, reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(OK_RESPONSE)
        await writer.drain()

    async def scenario():
        async with RawServer(reused_by(then_redirect), rest=get_only) as server:
            svc = await open_service(server, phases())
            svc._redirect_target = lambda current, location, visited, count: current.join(location)
            await svc.avalidate(jpeg())
            assert summaries(logs)[-1]["connection"] == "new"

            await svc.avalidate(jpeg())

            assert server.connections == 2

    run(scenario)

    assert summaries(logs)[-1]["connection"] == "new"


def test_mock_transport_has_no_connection_detail(logs):
    async def scenario():
        svc = await opened(service(FakeWorker(steps=[OK])))

        await svc.avalidate(jpeg())

    run(scenario)

    assert (summaries(logs)[-1]["connection"], summaries(logs)[-1]["failed_at"]) == ("-", "-")
