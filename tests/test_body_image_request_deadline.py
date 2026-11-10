import asyncio
import logging
import re
import threading
import time

import pytest
from anyio import to_thread
from starlette.datastructures import UploadFile

from app import main
from app.body_image_validation import deadline as worker_deadline
from app.body_image_validation import remote as remote_module
from app.body_image_validation import router as validation_router
from app.body_image_validation import worker
from app.body_image_validation.async_concurrency import AsyncBodyValidationLimiter
from app.body_image_validation.concurrency import SlotWaitTimeout
from app.body_image_validation.deadline import SCOPE_KEY, RequestDeadline
from app.body_image_validation.deadline_middleware import BodyImageDeadlineMiddleware
from app.body_image_validation.exceptions import ServerBusyError
from app.body_image_validation.remote import ENDPOINT_PATH
from app.config import body_image_validation as config
from app.errors import error_response
from app.security import verify_internal_key
from tests.test_body_image_async_concurrency import wait_for_queue
from tests.test_body_image_c3_http import (  # noqa: F401 (pytest fixture)
    WAIT,
    Backend,
    live,
    multipart,
    wait_until,
)
from tests.test_body_image_remote import POLL, TIMEOUT, FakeWorker, jpeg, redirect
from tests.test_body_image_remote_async import KEY as S3_KEY
from tests.test_body_image_remote_async import OK, TIMED_OUT, opened, rejected, run, service
from tests.test_body_image_warmup_api import remote, runtime  # noqa: F401 (pytest fixture)

LOGGER = "app.body_image_validation.remote"
OVER_THE_SPOOL = 3_000_000  # starlette 는 1MB 를 넘는 업로드를 디스크로 옮기고 write·read·close 를 threadpool 에서 한다


def summaries(caplog):
    return [
        dict(pair.split("=", 1) for pair in r.getMessage().removeprefix("body image request: ").split())
        for r in caplog.records
        if r.getMessage().startswith("body image request:")
    ]


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    return caplog


@pytest.fixture
def remote_mode(monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_MODE", "remote")
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 120.0)


def scope(path=ENDPOINT_PATH, method="POST", kind="http"):
    return {"type": kind, "method": method, "path": path}


def bounded(call):
    """끝나지 않는 결함이 테스트를 멈추지 않게 시간 상한을 둔다."""

    async def limited():
        async with asyncio.timeout(WAIT):
            return await call

    return asyncio.run(limited())


def asgi_call(app, request=None, sent=None, receive=None):
    """middleware 를 ASGI 로 부른다. send 는 sent 에 쌓는다."""
    sent = [] if sent is None else sent

    async def send(message):
        sent.append(message)

    return app(request if request is not None else scope(), receive, send)


def statuses(sent):
    return [message["status"] for message in sent if message["type"] == "http.response.start"]


def spent_limit(total, left):
    """이미 시간을 쓰고 left 초가 남은 요청."""
    limit = RequestDeadline(total)
    limit.deadline_at = limit.started + left
    return limit


# --- 요청 진입 시각과 적용 범위 ---


def test_remote_post_starts_its_deadline_at_the_entry(remote_mode):
    seen = {}

    async def app(request, receive, send):
        seen.update(limit=request[SCOPE_KEY])

    middleware = BodyImageDeadlineMiddleware(app, clock=lambda: 1000.0)
    bounded(asgi_call(middleware))

    assert seen["limit"].started == 1000.0
    assert seen["limit"].deadline_at == 1120.0
    assert seen["limit"].source == "asgi"


@pytest.mark.parametrize(
    "request_scope",
    [
        scope(method="GET"),
        scope(path=ENDPOINT_PATH + "/warmup"),
        scope(path="/health", method="GET"),
        {"type": "websocket", "path": ENDPOINT_PATH},
        {"type": "lifespan"},
    ],
    ids=["other-method", "warmup", "other-path", "websocket", "lifespan"],
)
def test_other_requests_pass_through_untouched(remote_mode, request_scope):
    seen = {}
    receive, send = object(), object()

    async def app(request, app_receive, app_send):
        seen.update(request=request, receive=app_receive, send=app_send)

    bounded(BodyImageDeadlineMiddleware(app)(request_scope, receive, send))

    assert SCOPE_KEY not in seen["request"]
    assert seen["receive"] is receive and seen["send"] is send


def test_local_mode_has_no_deadline_and_can_run_past_it(monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_MODE", "local")
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 0.01)
    seen = {}

    async def app(request, receive, send):
        await asyncio.sleep(0.05)
        seen.update(request=request)

    bounded(asgi_call(BodyImageDeadlineMiddleware(app), sent=(sent := [])))

    assert SCOPE_KEY not in seen["request"]
    assert sent == []  # 기존처럼 제한시간 없이 끝까지 돈다


def test_only_the_ai_server_has_the_middleware_and_it_sits_inside_metrics():
    assert all(m.cls is not BodyImageDeadlineMiddleware for m in worker.create_app().user_middleware)
    # 마지막이 가장 안쪽이다. metrics 밖에 두면 여기서 낸 504 가 500 으로 집계된다.
    assert main.app.user_middleware[-1].cls is BodyImageDeadlineMiddleware


def test_middleware_gives_receive_to_the_app_without_reading_the_body(remote_mode):
    messages = [
        {"type": "http.request", "body": b"a", "more_body": True},
        {"type": "http.request", "body": b"b", "more_body": False},
    ]
    read = []

    async def receive():
        read.append("middleware")
        return messages[len(read) - 1]

    async def app(request, app_receive, send):
        assert read == []  # 앱이 부를 때까지 본문을 읽지 않는다
        await app_receive()
        await app_receive()

    bounded(asgi_call(BodyImageDeadlineMiddleware(app), receive=receive))

    assert len(read) == 2


def test_concurrent_requests_keep_their_own_deadlines(remote_mode, monkeypatch, logs):
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 0.05)
    limits = {}

    async def app(request, receive, send):
        limit = request[SCOPE_KEY]
        limits[request["name"]] = limit
        if request["name"] == "second":
            limit.hand_over()  # 이 요청은 service 가 이어받아 앞 요청의 제한시간과 무관하다
            await asyncio.sleep(0.1)
            await error_response(200, "OK", "ok")(request, receive, send)
            return
        await asyncio.sleep(1)

    async def scenario():
        middleware = BodyImageDeadlineMiddleware(app)
        first, second = [], []
        first_call = asyncio.create_task(asgi_call(middleware, {**scope(), "name": "first"}, first))
        await asyncio.sleep(0.02)
        await asyncio.gather(first_call, asgi_call(middleware, {**scope(), "name": "second"}, second))
        return first, second

    first, second = bounded(scenario())

    assert (statuses(first), statuses(second)) == ([504], [200])
    assert limits["first"] is not limits["second"]
    assert limits["first"].started < limits["second"].started


# --- service 가 이어받기 전의 제한시간 초과 ---


def test_stuck_request_is_504_in_the_common_format_and_the_app_is_cancelled(
    remote_mode, monkeypatch, logs
):
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 0.02)
    cancelled = []

    async def app(request, receive, send):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    bounded(asgi_call(BodyImageDeadlineMiddleware(app), sent=(sent := [])))

    assert statuses(sent) == [504]
    assert b'"code":"BODY_IMAGE_WORKER_TIMEOUT"' in sent[1]["body"]
    assert cancelled == [True]
    assert [(s["outcome"], s["code"], s["timeout_stage"]) for s in summaries(logs)] == [
        ("deadline_exceeded", "BODY_IMAGE_WORKER_TIMEOUT", "request_parse")
    ]
    assert "body image validation failed: BODY_IMAGE_WORKER_TIMEOUT" in logs.text  # 기존 오류 로그와 같은 문구


def test_504_does_not_wait_for_the_cancelled_app_to_clean_up(remote_mode, monkeypatch):
    # 업로드 파일 close 가 threadpool 을 기다리는 경우다. 응답은 먼저 나가고 middleware 는 정리까지 기다린다.
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 0.02)

    async def scenario():
        cleaning, finish_cleanup = asyncio.Event(), asyncio.Event()

        async def app(request, receive, send):
            try:
                await asyncio.sleep(5)
            finally:
                cleaning.set()
                await finish_cleanup.wait()

        sent = []
        call = asyncio.create_task(asgi_call(BodyImageDeadlineMiddleware(app), sent=sent))
        await cleaning.wait()
        while len(sent) < 2:
            await asyncio.sleep(0.001)
        assert statuses(sent) == [504]
        assert not call.done()
        finish_cleanup.set()
        await call

    bounded(scenario())


def test_started_response_gets_no_second_status(remote_mode, monkeypatch, logs):
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 0.02)

    async def app(request, receive, send):
        await send({"type": "http.response.start", "status": 400, "headers": []})
        await asyncio.sleep(5)

    bounded(asgi_call(BodyImageDeadlineMiddleware(app), sent=(sent := [])))

    assert statuses(sent) == [400]
    assert len(sent) == 1
    assert len(summaries(logs)) == 1


def test_app_sending_after_the_504_is_dropped(remote_mode, monkeypatch):
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 0.02)

    async def app(request, receive, send):
        try:
            await asyncio.sleep(5)
        finally:
            await send({"type": "http.response.start", "status": 500, "headers": []})

    bounded(asgi_call(BodyImageDeadlineMiddleware(app), sent=(sent := [])))

    assert statuses(sent) == [504]


def test_error_raised_while_the_cancelled_app_cleans_up_is_not_lost_or_raised(remote_mode, monkeypatch):
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 0.02)

    async def app(request, receive, send):
        try:
            await asyncio.sleep(5)
        finally:
            raise RuntimeError("cleanup")

    bounded(asgi_call(BodyImageDeadlineMiddleware(app), sent=(sent := [])))

    assert statuses(sent) == [504]


def test_hand_over_stops_the_guard_and_the_service_keeps_the_summary(remote_mode, monkeypatch, logs):
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 0.02)

    async def app(request, receive, send):
        request[SCOPE_KEY].hand_over()
        await asyncio.sleep(0.06)
        await error_response(200, "OK", "ok")(request, receive, send)

    bounded(asgi_call(BodyImageDeadlineMiddleware(app), sent=(sent := [])))

    assert statuses(sent) == [200]
    assert summaries(logs) == []


@pytest.mark.parametrize("error", [TimeoutError("app"), RuntimeError("app")], ids=["timeout", "other"])
def test_errors_from_the_app_are_not_turned_into_a_504(remote_mode, error):
    async def app(request, receive, send):
        raise error

    with pytest.raises(type(error)):
        bounded(asgi_call(BodyImageDeadlineMiddleware(app), sent=(sent := [])))

    assert sent == []


def test_cancel_before_the_service_cancels_the_app_and_logs_once(remote_mode, logs):
    async def scenario():
        started, cancelled = asyncio.Event(), []

        async def app(request, receive, send):
            started.set()
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        call = asyncio.create_task(asgi_call(BodyImageDeadlineMiddleware(app), sent=(sent := [])))
        await started.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert cancelled == [True]
        assert sent == []

    bounded(scenario())

    assert [(s["outcome"], s["timeout_stage"]) for s in summaries(logs)] == [("cancelled", "-")]


def test_cancel_after_the_hand_over_reaches_the_service_without_a_second_summary(remote_mode, logs):
    async def scenario():
        started, cancelled = asyncio.Event(), []

        async def app(request, receive, send):
            request[SCOPE_KEY].hand_over()
            started.set()
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        call = asyncio.create_task(asgi_call(BodyImageDeadlineMiddleware(app)))
        await started.wait()
        await asyncio.sleep(0)  # 대기하던 middleware 가 깨어나 앱을 기다리는 쪽으로 넘어간다
        await asyncio.sleep(0)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert cancelled == [True]

    bounded(scenario())

    assert summaries(logs) == []


# --- service 는 넘겨받은 제한시간을 쓴다 ---


def test_time_spent_before_the_service_comes_off_the_http_budget(logs):
    now = [1000.0]

    async def scenario():
        fake = FakeWorker(steps=[OK])
        svc = await opened(service(fake, timeout=120.0, deadline=120.0, clock=lambda: now[0],
                                   wall_clock=lambda: 5000.0))
        limit = RequestDeadline(120.0, clock=lambda: now[0])
        now[0] += 30.0

        result = await svc.avalidate(jpeg(), limit)

        assert result.s3_key == S3_KEY
        request = fake.calls[0]
        assert request.extensions["timeout"]["read"] == 90.0
        assert request.headers[worker_deadline.HEADER] == "5090.000"  # wall clock + 남은 시간. 시계 변환은 그대로다

    run(scenario)

    assert summaries(logs)[0]["budget"] == "90.000s"
    assert summaries(logs)[0]["total"] == "30.000s"
    assert summaries(logs)[0]["input_check"] == "0.000s"  # 각 단계는 자기 구간만 잰다
    assert summaries(logs)[0]["total_from"] == "asgi"


def test_direct_call_still_starts_the_deadline_itself(logs):
    now = [1000.0]

    async def scenario():
        fake = FakeWorker(steps=[OK])
        svc = await opened(service(fake, deadline=60.0, clock=lambda: now[0], wall_clock=lambda: 5000.0))

        await svc.avalidate(jpeg())

        assert fake.calls[0].headers[worker_deadline.HEADER] == "5060.000"

    run(scenario)

    logged = summaries(logs)[0]
    assert (logged["total_from"], logged["deadline"]) == ("service", "60.0s")
    assert (logged["request_parse"], logged["service_lookup"], logged["image_read"]) == ("-", "-", "-")


def test_expired_request_stops_before_the_input_check_and_modal(monkeypatch, logs):
    now = [1000.0]
    opened_images = []
    monkeypatch.setattr(remote_module, "open_image", lambda body: opened_images.append(body))

    async def scenario():
        fake = FakeWorker(steps=[OK])
        svc = await opened(service(fake, clock=lambda: now[0]))
        limit = RequestDeadline(120.0, clock=lambda: now[0])
        limit.enter("service_lookup")
        limit.enter("image_read")
        now[0] += 120.0

        assert await rejected(svc.avalidate(jpeg(), limit)) == TIMED_OUT
        assert fake.calls == []
        assert (svc.limiter.running, svc.limiter.waiting) == (0, 0)

    run(scenario)

    assert opened_images == []
    logged = summaries(logs)[0]
    assert (logged["outcome"], logged["timeout_stage"]) == ("deadline_exceeded", "image_read")
    assert logged["input_check"] == "-"


def test_queue_limit_is_kept_when_more_time_is_left_than_the_queue_allows():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1, max_wait_seconds=0.02)
        fake = FakeWorker(steps=[OK])
        svc = await opened(service(fake, limiter=limiter))

        async with limiter.slot():
            with pytest.raises(SlotWaitTimeout) as caught:
                await svc.avalidate(jpeg(), spent_limit(120.0, left=60.0))

        assert isinstance(caught.value, ServerBusyError)
        assert fake.calls == []

    run(scenario)


def test_queue_ends_at_the_deadline_when_less_time_is_left_than_the_queue_allows(logs):
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1, max_wait_seconds=10.0)
        fake = FakeWorker(steps=[OK])
        svc = await opened(service(fake, limiter=limiter))

        async with limiter.slot():
            assert await rejected(svc.avalidate(jpeg(), spent_limit(120.0, left=0.05))) == TIMED_OUT

        assert limiter.waiting == 0
        assert fake.calls == []

    run(scenario)

    assert summaries(logs)[0]["timeout_stage"] == "queue"


def test_queue_time_comes_off_the_same_budget():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1, max_wait_seconds=10.0)
        fake = FakeWorker(steps=[OK])
        svc = await opened(service(fake, limiter=limiter, timeout=120.0))
        limit = spent_limit(120.0, left=20.0)
        before = time.time()

        async with limiter.slot():
            call = asyncio.create_task(svc.avalidate(jpeg(), limit))
            await wait_for_queue(limiter, 1)
            await asyncio.sleep(0.05)
        await call

        request = fake.calls[0]
        assert request.extensions["timeout"]["read"] <= 20.0 - 0.05
        assert float(request.headers[worker_deadline.HEADER]) <= before + 20.0 + 0.01

    run(scenario)


def test_redirect_keeps_the_time_left_at_the_entry():
    now = [1000.0]

    async def scenario():
        fake = FakeWorker(steps=[redirect(POLL), OK])
        svc = await opened(service(fake, timeout=120.0, deadline=120.0, clock=lambda: now[0],
                                   wall_clock=lambda: 5000.0))
        limit = RequestDeadline(120.0, clock=lambda: now[0])
        now[0] += 100.0

        await svc.avalidate(jpeg(), limit)

        post_call, get_call = fake.calls
        assert (post_call.method, get_call.method) == ("POST", "GET")
        assert post_call.headers[worker_deadline.HEADER] == get_call.headers[worker_deadline.HEADER]
        assert get_call.extensions["timeout"]["read"] == 20.0

    run(scenario)


def test_deadline_during_the_http_call_returns_the_slot_and_names_the_stage(logs):
    async def scenario():
        fake = FakeWorker(steps=[(5, 200, {}, b"")])
        svc = await opened(service(fake))

        assert await rejected(svc.avalidate(jpeg(), spent_limit(120.0, left=0.05))) == TIMED_OUT
        assert (svc.limiter.running, svc.limiter.waiting) == (0, 0)

    run(scenario)

    logged = summaries(logs)[0]
    assert (logged["timeout_stage"], logged["http_cancelled"]) == ("http", "True")
    assert logged["remote_stopped"] == "unknown"  # 연결을 끊었다고 Modal 작업이 멈췄다고 보지 않는다


def test_cancelled_call_with_a_limit_returns_the_slot(logs):
    async def scenario():
        fake = FakeWorker(steps=[(5, 200, {}, b"")])
        svc = await opened(service(fake))
        call = asyncio.create_task(svc.avalidate(jpeg(), spent_limit(120.0, left=60.0)))
        while not fake.calls:
            await asyncio.sleep(0.001)

        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        assert (svc.limiter.running, svc.limiter.waiting) == (0, 0)

    run(scenario)

    assert [s["outcome"] for s in summaries(logs)] == ["cancelled"]


# --- 실제 uvicorn + main.app ---


def split_upload(image):
    """본문 앞부분과 나머지. 나머지를 보내기 전까지 요청은 Multipart 수신 단계에 머문다."""
    raw = multipart(image)
    return raw[: len(raw) - 64], raw[len(raw) - 64 :]


class Saturated:
    """서버 event loop 의 threadpool 토큰을 모두 잡아 둔다. 해제 전 응답이 오면 포화 중에 응답한 것이다."""

    def __init__(self):
        self.release = threading.Event()
        self.limiter = None

    async def hold(self):
        self.limiter = to_thread.current_default_thread_limiter()
        self.limiter.total_tokens = 1
        asyncio.get_running_loop().create_task(to_thread.run_sync(self.release.wait))
        await asyncio.sleep(0)

    def restore(self, server):
        self.release.set()

        async def restore():
            await to_thread.run_sync(lambda: None)
            self.limiter.total_tokens = 40

        asyncio.run_coroutine_threadsafe(restore(), server.service._shared_loop).result(WAIT)
        assert self.limiter.borrowed_tokens == 0  # 취소돼도 토큰이 남지 않는다


@pytest.fixture
def deadline_app(live, monkeypatch):  # noqa: F811
    def start(total):
        monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", total)
        monkeypatch.setattr(config, "REMOTE_READ_TIMEOUT_SECONDS", total)
        return live()

    return start


def test_normal_request_keeps_the_response_and_logs_every_stage(deadline_app, logs):
    server = deadline_app(30.0)
    server.modal.release.set()

    status, body = server.send().read().response()

    assert status == 200
    assert body["code"] == "BODY_IMAGE_UPLOAD_SUCCESS"
    assert body["data"] == {"s3_key": S3_KEY}
    assert len(summaries(logs)) == 1
    logged = summaries(logs)[0]
    assert (logged["outcome"], logged["total_from"], logged["timeout_stage"]) == ("response", "asgi", "-")
    for stage in ("request_parse", "service_lookup", "image_read", "input_check", "queue_wait", "http"):
        assert logged[stage].endswith("s") and logged[stage] != "-"
    assert float(logged["total"][:-1]) + 0.005 >= sum(  # 값마다 소수 3자리로 반올림돼 있다
        float(logged[stage][:-1]) for stage in ("request_parse", "service_lookup", "image_read", "http")
    )


def test_slow_upload_is_counted_in_the_deadline_and_the_worker_header(deadline_app, logs):
    server = deadline_app(30.0)
    server.modal.release.set()
    head, rest = split_upload(jpeg())
    sent_at = time.time()

    request = Backend(server.port, head)
    time.sleep(0.2)
    request.sock.sendall(rest)
    assert request.read().response()[0] == 200

    assert float(summaries(logs)[0]["request_parse"][:-1]) >= 0.2
    assert float(summaries(logs)[0]["budget"][:-1]) <= 30.0 - 0.2
    header = {name.lower(): value for name, value in server.modal.headers[0].items()}
    assert float(header["x-body-image-deadline"]) <= sent_at + 30.0 + 0.05  # 업로드 뒤 30초가 아니다


def test_upload_that_never_finishes_is_504_and_modal_is_not_called(deadline_app, logs):
    server = deadline_app(0.3)
    server.modal.release.set()
    head, _ = split_upload(jpeg())

    status, body = Backend(server.port, head).read().response()

    assert (status, body) == (504, TIMEOUT)
    assert server.modal.started == 0
    assert server.slots == (0, 0)
    assert [s["timeout_stage"] for s in summaries(logs)] == ["request_parse"]
    assert server.send().read().response()[0] == 200  # 다음 요청은 정상이다


def test_upload_that_arrives_after_the_504_does_not_reach_modal(deadline_app):
    server = deadline_app(0.2)
    server.modal.release.set()
    # keep-alive 연결이면 uvicorn 이 남은 본문을 버린다. 보내는 쪽은 끊기지 않고 504 를 읽는다.
    head, rest = split_upload(jpeg())
    head = head.replace(b"Connection: close\r\n", b"")
    request = Backend(server.port, head)
    request.read()
    wait_until(lambda: b"\r\n\r\n" in request.raw)

    request.sock.sendall(rest)
    time.sleep(0.1)

    assert request.raw.startswith(b"HTTP/1.1 504")
    assert server.modal.started == 0


@pytest.mark.parametrize("padding", [0, OVER_THE_SPOOL], ids=["in-memory", "on-disk"])
def test_saturated_threadpool_before_the_request_is_504_while_it_is_still_full(
    deadline_app, logs, padding
):
    server = deadline_app(0.3)
    saturated = Saturated()
    asyncio.run_coroutine_threadsafe(saturated.hold(), server.service._shared_loop).result(WAIT)
    try:
        request = Backend(server.port, b"")
        sender = threading.Thread(target=request.sock.sendall, args=(multipart(jpeg() + b"\0" * padding),))
        sender.start()

        status, body = request.read().response()
        sender.join(WAIT)
        assert (status, body) == (504, TIMEOUT)
        assert not saturated.release.is_set()
    finally:
        saturated.restore(server)
    server.modal.release.set()
    time.sleep(0.05)
    assert server.modal.started == 0
    assert [s["timeout_stage"] for s in summaries(logs)] == ["request_parse"]


def test_threadpool_saturated_after_parsing_with_a_file_on_disk_is_still_504_in_time(
    deadline_app, logs
):
    # 취소 정리의 UploadFile.close 가 토큰을 기다리는 경우다. 응답은 그 정리를 기다리지 않는다.
    server = deadline_app(0.3)
    saturated = Saturated()
    main.app.dependency_overrides[verify_internal_key] = saturated.hold
    try:
        status, body = server.send(jpeg() + b"\0" * OVER_THE_SPOOL).read().response()
        assert (status, body) == (504, TIMEOUT)
        assert not saturated.release.is_set()
    finally:
        main.app.dependency_overrides.clear()
        saturated.restore(server)
    assert server.modal.started == 0
    assert [s["timeout_stage"] for s in summaries(logs)] == ["service_lookup"]


def test_service_lookup_finishing_after_the_deadline_does_not_call_modal(deadline_app, monkeypatch, logs):
    server = deadline_app(0.2)
    server.modal.release.set()
    entered, release = threading.Event(), threading.Event()
    real_get_service = validation_router.get_service

    def stuck_get_service():
        entered.set()
        release.wait(WAIT)
        return real_get_service()

    monkeypatch.setattr(validation_router, "get_service", stuck_get_service)
    try:
        status, body = server.send().read().response()
        assert (status, body) == (504, TIMEOUT)
        assert entered.is_set() and not release.is_set()  # 도는 thread 는 끊을 수 없어도 응답은 먼저 나간다
    finally:
        release.set()
    time.sleep(0.05)
    assert server.modal.started == 0
    logged = summaries(logs)
    assert [(s["timeout_stage"], s["image_read"]) for s in logged] == [("service_lookup", "-")]


def test_stuck_file_read_is_504_without_calling_modal(deadline_app, monkeypatch, logs):
    server = deadline_app(0.2)
    server.modal.release.set()

    async def stuck_read(self, size=-1):
        await asyncio.sleep(WAIT)

    monkeypatch.setattr(UploadFile, "read", stuck_read)

    assert server.send().read().response() == (504, TIMEOUT)
    assert server.modal.started == 0
    assert [s["timeout_stage"] for s in summaries(logs)] == ["image_read"]


def test_time_lost_before_the_service_ends_at_the_same_deadline_in_the_input_check(
    deadline_app, monkeypatch, logs
):
    server = deadline_app(0.3)
    release = threading.Event()
    real_open = remote_module.open_image

    def stuck_open(body):
        release.wait(WAIT)
        return real_open(body)

    monkeypatch.setattr(remote_module, "open_image", stuck_open)
    head, rest = split_upload(jpeg())
    request = Backend(server.port, head)
    time.sleep(0.15)
    request.sock.sendall(rest)
    started = time.monotonic()
    try:
        status, body = request.read().response()
    finally:
        release.set()

    assert (status, body) == (504, TIMEOUT)
    assert time.monotonic() - started < 0.25  # 입력 검사에 새 0.3초가 주어지지 않았다
    assert server.modal.started == 0
    assert [s["timeout_stage"] for s in summaries(logs)] == ["input_check"]


def test_middleware_504_is_counted_as_504_in_metrics(deadline_app):
    server = deadline_app(0.2)
    server.modal.release.set()
    head, _ = split_upload(jpeg())

    def counted_504():
        # 지표는 프로세스 전역이다. 앞 테스트의 504 와 섞이지 않게 증가분을 본다.
        request = Backend(server.port, b"GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        request.read().reader.join(WAIT)
        label = f'http_requests_total{{handler="{ENDPOINT_PATH}",method="POST",status="504"}} '
        found = re.search(re.escape(label) + r"([0-9.]+)", request.raw.decode())
        return float(found.group(1)) if found else 0.0

    before = counted_504()
    assert Backend(server.port, head).read().response()[0] == 504

    wait_until(lambda: counted_504() == before + 1)


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        (multipart(b"x").replace(b"Authorization: Bearer", b"Authorization: Bad"), 401),
        (multipart(b"x").replace(b'name="image"', b'name="other"'), 400),
    ],
    ids=["bad-key", "missing-file"],
)
def test_rejections_before_the_service_keep_their_response_and_log_no_summary(
    deadline_app, logs, payload, status
):
    server = deadline_app(30.0)

    assert Backend(server.port, payload).read().response()[0] == status
    assert summaries(logs) == []
    assert server.modal.started == 0
