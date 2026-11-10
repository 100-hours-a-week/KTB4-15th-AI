import asyncio
import contextlib
import logging
import threading
import time
from contextlib import asynccontextmanager
from email import message_from_bytes

import httpx
import pytest
from anyio import to_thread

from app.body_image_validation import deadline as worker_deadline
from app.body_image_validation import remote
from app.body_image_validation.async_concurrency import AsyncBodyValidationLimiter
from app.body_image_validation.concurrency import BodyValidationLimiter, SlotWaitTimeout
from app.body_image_validation.exceptions import BodyImageValidationError, ServerBusyError
from app.body_image_validation.models import BodyImageValidationResult
from app.body_image_validation.remote import MAX_REDIRECTS, RemoteBodyImageValidationService
from app.config import body_image_validation as config
from tests.test_body_image_async_concurrency import run as bounded_run
from tests.test_body_image_async_concurrency import state, wait_for_queue, wait_until
from tests.test_body_image_concurrent_api import HeldWorker
from tests.test_body_image_deadline import CancelTrackingWorker
from tests.test_body_image_image_io import png_header_only
from tests.test_body_image_remote import (
    LIMIT,
    POLL,
    SUCCESS,
    SUCCESS_CODE,
    URL,
    WORKER,
    BodyStream,
    FakeWorker,
    common,
    gif,
    jpeg,
    local_worker,  # noqa: F401 (pytest fixture)
    redirect,
    sized,
)
from tests.test_body_image_remote import tcp_service as sync_tcp_service

KEY = "body-images/abc.png"
BUSY = (429, "SERVER_BUSY")
BAD_GATEWAY = (502, "BODY_IMAGE_WORKER_UNAVAILABLE")
UNAVAILABLE = (503, "BODY_IMAGE_WORKER_UNAVAILABLE")
TIMED_OUT = (504, "BODY_IMAGE_WORKER_TIMEOUT")
OK = (0, 200, {}, SUCCESS)


def service(fake, limiter=None, timeout=30.0, deadline=60.0, **kwargs):
    return RemoteBodyImageValidationService(
        WORKER,
        proxy_key="wk-test",
        proxy_secret="ws-test",
        api_key="worker-key",
        timeout=timeout,
        total_deadline=deadline,
        limiter=limiter or AsyncBodyValidationLimiter(1, 1, max_wait_seconds=10.0),
        transport=fake.transport,
        **kwargs,
    )


_opened = []


async def opened(service_):
    """공유 client 를 연다. run 이 시나리오가 끝날 때 닫는다."""
    await service_.aopen()
    _opened.append(service_)
    return service_


def run(scenario):
    async def closing_what_it_opened():
        try:
            await scenario()
        finally:
            while _opened:
                # 테스트가 실패해 호출이 남아 있어도 기다리지 않는다(시간이 끝나면 바로 닫는다).
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(1.0):
                        await _opened.pop().aclose()

    bounded_run(closing_what_it_opened)


def failure(error):
    return error.status_code, error.code


async def rejected(call):
    """시작한 호출이 낸 BodyImageValidationError 의 (status, code)."""
    with pytest.raises(BodyImageValidationError) as exc_info:
        await call
    return failure(exc_info.value)


# --- 정상·redirect·크기 제한·오류 매핑·재시도 없음 ---

MESSAGE = "머리부터 발끝까지 모두 나오도록 전신을 촬영해주세요."
POLICY_CASES = {
    "success": (lambda: [OK], KEY, 1),
    "303-then-200": (lambda: [redirect(POLL), OK], KEY, 2),
    "303-polling": (lambda: [redirect(POLL), redirect(POLL), OK], KEY, 3),
    "303-then-user-error": (
        lambda: [redirect(POLL), (0, 422, {}, common("FULL_BODY_NOT_VISIBLE", MESSAGE))],
        (422, "FULL_BODY_NOT_VISIBLE", MESSAGE),
        2,
    ),
    "other-origin": (lambda: [redirect("https://evil.example.com/x"), OK], BAD_GATEWAY, 1),
    "downgrade-to-http": (lambda: [redirect(WORKER.replace("https", "http") + POLL), OK], BAD_GATEWAY, 1),
    "307-is-not-followed": (lambda: [redirect(POLL, status=307), OK], BAD_GATEWAY, 1),
    "303-without-location": (lambda: [(0, 303, {}, b""), OK], BAD_GATEWAY, 1),
    "back-to-endpoint": (lambda: [redirect(URL), OK], BAD_GATEWAY, 1),
    "too-many-redirects": (lambda: [redirect(POLL)], BAD_GATEWAY, MAX_REDIRECTS + 1),
    "64kb-accepted": (
        lambda: [(0, 200, {}, sized(LIMIT, SUCCESS_CODE, {"s3_key": KEY}))], KEY, 1
    ),
    "over-64kb": (
        lambda: [(0, 200, {}, sized(LIMIT + 1, SUCCESS_CODE, {"s3_key": KEY}))], BAD_GATEWAY, 1
    ),
    "over-64kb-error": (
        lambda: [(0, 422, {}, sized(LIMIT + 1, "FULL_BODY_NOT_VISIBLE"))], BAD_GATEWAY, 1
    ),
    "user-error-is-relayed": (
        lambda: [(0, 422, {}, common("IMAGE_TOO_DARK", "dark"))], (422, "IMAGE_TOO_DARK", "dark"), 1
    ),
    "server-busy-is-relayed": (
        lambda: [(0, 429, {}, common("SERVER_BUSY", "busy"))], (429, "SERVER_BUSY", "busy"), 1
    ),
    "worker-500-is-relayed": (
        lambda: [(0, 500, {}, common("IMAGE_UPLOAD_FAILED", "m"))], (500, "IMAGE_UPLOAD_FAILED", "m"), 1
    ),
    "refused": (lambda: [httpx.ConnectError("refused"), OK], UNAVAILABLE, 1),
    "reset": (lambda: [httpx.RemoteProtocolError("closed"), OK], UNAVAILABLE, 1),
    "reset-in-body": (
        lambda: [(0, 200, {}, BodyStream(b'{"code":', httpx.ReadError("reset"))), OK], UNAVAILABLE, 1
    ),
    "upstream-503": (lambda: [(0, 503, {}, b"unavailable"), OK], UNAVAILABLE, 1),
    "read-timeout": (lambda: [httpx.ReadTimeout("t"), OK], TIMED_OUT, 1),
    "connect-timeout": (lambda: [httpx.ConnectTimeout("t"), OK], TIMED_OUT, 1),
    "pool-timeout": (lambda: [httpx.PoolTimeout("t"), OK], TIMED_OUT, 1),
    "timeout-in-body": (
        lambda: [(0, 200, {}, BodyStream(b'{"code":', httpx.ReadTimeout("t"))), OK], TIMED_OUT, 1
    ),
    "upstream-504": (lambda: [(0, 504, {}, b"timeout"), OK], TIMED_OUT, 1),
    "upstream-502": (lambda: [(0, 502, {}, b"bad gateway"), OK], BAD_GATEWAY, 1),
    "proxy-401": (lambda: [(0, 401, {}, b"modal-http: missing credentials"), OK], BAD_GATEWAY, 1),
    "worker-403": (lambda: [(0, 403, {}, common("FORBIDDEN")), OK], BAD_GATEWAY, 1),
    "404": (lambda: [(0, 404, {}, common("NOT_FOUND")), OK], BAD_GATEWAY, 1),
    "invalid-request": (lambda: [(0, 400, {}, common("INVALID_REQUEST")), OK], BAD_GATEWAY, 1),
    "200-not-json": (lambda: [(0, 200, {}, b"not json"), OK], BAD_GATEWAY, 1),
    "200-without-key": (
        lambda: [(0, 200, {}, common(SUCCESS_CODE, data={})), OK], BAD_GATEWAY, 1
    ),
    "unexpected-error": (lambda: [RuntimeError("bug"), OK], "RuntimeError", 1),
}


@pytest.mark.parametrize(("steps", "expected", "requests"), POLICY_CASES.values(), ids=POLICY_CASES)
def test_avalidate_applies_the_remote_policy(steps, expected, requests):
    worker = FakeWorker(steps=steps())
    outcome = {}

    async def scenario():
        svc = await opened(service(worker))
        try:
            outcome["value"] = (await svc.avalidate(jpeg())).s3_key
        except BodyImageValidationError as error:
            outcome["value"] = (error.status_code, error.code, error.message)
        except Exception as error:  # noqa: BLE001 - 예상 밖 오류도 그대로 올라와야 한다
            outcome["value"] = type(error).__name__
        assert state(svc.limiter) == (0, 0)

    run(scenario)

    # (status, code) 만 적은 경우는 공통 시스템 오류 문구(message 없음)다.
    assert outcome["value"] == ((*expected, None) if len(expected) == 2 else expected)
    assert len(worker.calls) == requests  # 재시도가 없다


def test_success_returns_the_result_object():
    async def scenario():
        result = await (await opened(service(FakeWorker()))).avalidate(jpeg())

        assert result == BodyImageValidationResult(s3_key=KEY)

    run(scenario)


def test_request_carries_the_auth_headers_timeout_and_the_original_image():
    fake = FakeWorker()
    body = jpeg()

    async def scenario():
        await (await opened(service(fake, timeout=42.0))).avalidate(body)

    run(scenario)

    [request] = fake.calls
    assert str(request.url) == WORKER + URL
    assert request.method == "POST"
    assert request.headers["modal-key"] == "wk-test"
    assert request.headers["modal-secret"] == "ws-test"
    assert request.headers["authorization"] == "Bearer worker-key"
    assert 41.5 < request.extensions["timeout"]["read"] <= 42.0  # 남은 deadline 60초보다 작음
    message = message_from_bytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    [part] = message.get_payload()
    assert part.get_param("name", header="content-disposition") == "image"
    assert part.get_content_type() == "image/jpeg"
    assert part.get_payload(decode=True) == body


def test_deadline_header_is_absolute_and_the_same_after_a_redirect():
    fake = FakeWorker(steps=[redirect(POLL), OK])

    async def scenario():
        await (await opened(service(fake, deadline=50.0, wall_clock=lambda: 1_000_000.0))).avalidate(jpeg())

    run(scenario)

    post_request, get_request = fake.calls
    assert 1_000_049.9 < float(post_request.headers[worker_deadline.HEADER]) <= 1_000_050.0
    assert get_request.headers[worker_deadline.HEADER] == post_request.headers[worker_deadline.HEADER]
    assert get_request.headers["modal-key"] == "wk-test"
    assert get_request.headers["authorization"] == "Bearer worker-key"
    assert get_request.content == b""  # 이미지를 다시 보내지 않는다


def test_http_client_ignores_the_environment_and_does_not_follow_redirects(monkeypatch):
    created = []
    real_client = httpx.AsyncClient

    class Spy(real_client):
        def __init__(self, **kwargs):
            created.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Spy)

    async def scenario():
        await (await opened(service(FakeWorker()))).avalidate(jpeg())

    run(scenario)

    assert created
    for kwargs in created:
        assert kwargs["follow_redirects"] is False
        assert kwargs["trust_env"] is False


def test_http_runs_on_the_callers_event_loop():
    loops = []

    class LoopRecordingWorker(FakeWorker):
        async def _handle(self, request):
            loops.append(asyncio.get_running_loop())
            return await super()._handle(request)

    async def scenario():
        await (await opened(service(LoopRecordingWorker()))).avalidate(jpeg())

        assert loops == [asyncio.get_running_loop()]  # asyncio.run 으로 새 loop 를 만들지 않는다

    run(scenario)


# --- 입력 검사 ---


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (b"", 400, "IMAGE_EMPTY"),
        (b"not-an-image", 400, "INVALID_IMAGE"),
        (gif(), 400, "IMAGE_FORMAT_UNSUPPORTED"),
        (b"\x00" * (config.MAX_IMAGE_BYTES + 1), 413, "IMAGE_TOO_LARGE"),
        (jpeg((300, 400)), 422, "IMAGE_RESOLUTION_TOO_SMALL"),
        (png_header_only(15055, 10000), 422, "IMAGE_RESOLUTION_TOO_LARGE"),
    ],
    ids=["empty", "not-an-image", "unsupported-format", "too-large", "too-small", "too-many-pixels"],
)
def test_invalid_uploads_are_rejected_without_a_slot_or_the_worker(body, status, code):
    fake = FakeWorker()

    async def scenario():
        svc = await opened(service(fake))
        assert await rejected(svc.avalidate(body)) == (status, code)
        assert state(svc.limiter) == (0, 0)

    run(scenario)

    assert fake.calls == []


def test_invalid_upload_gets_its_own_error_while_the_queue_is_full():
    async def scenario():
        worker = HeldWorker()
        svc = await opened(service(worker))
        first = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(worker.entered.is_set)
        second = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_for_queue(svc.limiter, 1)

        assert await rejected(svc.avalidate(b"not-an-image")) == (400, "INVALID_IMAGE")

        assert state(svc.limiter) == (1, 1)
        worker.released.set()
        await asyncio.gather(first, second)

    run(scenario)


class SlowOpen:
    """입력 검사가 event 를 받을 때까지 thread 를 잡고 있게 한다. 실행한 thread 와 이미지 닫힘을 기록한다."""

    def __init__(self, monkeypatch, wait=2.0):
        self.entered, self.released = threading.Event(), threading.Event()
        self.threads, self.closed = [], []
        self.wait = wait
        self.real_open = remote.open_image
        monkeypatch.setattr(remote, "open_image", self)

    def __call__(self, body):
        self.threads.append(threading.get_ident())
        self.entered.set()
        self.released.wait(self.wait)
        source = self.real_open(body)
        real_close = source.close

        def close():
            self.closed.append(True)
            real_close()

        source.close = close
        return source


def test_input_check_runs_off_the_event_loop_and_closes_the_image(monkeypatch):
    slow = SlowOpen(monkeypatch, wait=0.5)

    async def scenario():
        task = asyncio.create_task((await opened(service(FakeWorker()))).avalidate(jpeg()))
        await wait_until(slow.entered.is_set)
        for _ in range(20):
            await asyncio.sleep(0.001)
        assert not task.done()  # 입력 검사가 thread 를 잡고 있는 동안 loop 는 계속 돈다
        slow.released.set()

        assert (await task).s3_key == KEY
        assert slow.threads and slow.threads[0] != threading.get_ident()

    run(scenario)

    assert slow.closed == [True]


def test_failed_input_check_runs_off_the_event_loop_too(monkeypatch):
    threads = []
    real_open = remote.open_image

    def recording_open(body):
        threads.append(threading.get_ident())
        return real_open(body)

    monkeypatch.setattr(remote, "open_image", recording_open)

    async def scenario():
        assert await rejected((await opened(service(FakeWorker()))).avalidate(b"not-an-image")) == (400, "INVALID_IMAGE")
        assert threads[0] != threading.get_ident()

    run(scenario)


def test_time_spent_on_the_input_check_comes_out_of_the_deadline(monkeypatch):
    now = [1000.0]
    real_open = remote.open_image

    def slow_open(body):
        now[0] += 8.0
        return real_open(body)

    monkeypatch.setattr(remote, "open_image", slow_open)
    fake = FakeWorker()

    async def scenario():
        svc = await opened(service(fake, deadline=50.0, clock=lambda: now[0], wall_clock=lambda: 1_000_000.0))
        await svc.avalidate(jpeg())

    run(scenario)

    [request] = fake.calls
    assert request.extensions["timeout"]["read"] == pytest.approx(30.0)  # 네트워크 timeout 30초 < 남은 42초
    assert float(request.headers[worker_deadline.HEADER]) == pytest.approx(1_000_042.0)


def test_deadline_spent_on_the_input_check_is_504_without_calling_the_worker(monkeypatch):
    now = [1000.0]
    real_open = remote.open_image

    def slow_open(body):
        now[0] += 50.0
        return real_open(body)

    monkeypatch.setattr(remote, "open_image", slow_open)
    fake = FakeWorker()

    async def scenario():
        svc = await opened(service(fake, deadline=50.0, clock=lambda: now[0]))
        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
        assert state(svc.limiter) == (0, 0)

    run(scenario)

    assert fake.calls == []


def test_deadline_spent_on_the_input_check_is_504_even_when_the_queue_is_full(monkeypatch):
    now = [1000.0]
    real_open = remote.open_image

    def slow_open(body):
        now[0] += 50.0
        return real_open(body)

    monkeypatch.setattr(remote, "open_image", slow_open)

    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 0)
        svc = await opened(service(FakeWorker(), limiter=limiter, deadline=50.0, clock=lambda: now[0]))
        async with limiter.slot():  # 대기열이 찼어도 429 가 아니라 504 다
            assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT

    run(scenario)


def test_waiting_for_the_threadpool_counts_against_the_deadline():
    fake = FakeWorker()

    async def scenario():
        threads = to_thread.current_default_thread_limiter()
        threads.total_tokens = 1  # 이 event loop 의 threadpool 만 줄인다
        other_request = object()
        threads.acquire_on_behalf_of_nowait(other_request)  # 다른 요청이 threadpool 을 다 쓰고 있다
        try:
            svc = await opened(service(fake, deadline=0.05))
            started = time.monotonic()
            assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
            assert time.monotonic() - started < 1.0
            assert state(svc.limiter) == (0, 0)
        finally:
            threads.release_on_behalf_of(other_request)

    run(scenario)

    assert fake.calls == []


def test_stuck_input_check_ends_at_the_deadline_and_still_closes_the_image(monkeypatch):
    slow = SlowOpen(monkeypatch)
    fake = FakeWorker()

    async def scenario():
        svc = await opened(service(fake, deadline=0.05))
        started = time.monotonic()
        try:
            assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
            assert time.monotonic() - started < 1.0  # thread 가 끝나기를 기다리지 않는다
        finally:
            slow.released.set()
        await wait_until(lambda: slow.closed)
        assert state(svc.limiter) == (0, 0)

    run(scenario)

    assert fake.calls == []


# --- 비동기 limiter: 실행 1 + 대기 1, 대기 제한 ---


def test_one_runs_one_waits_and_the_third_is_rejected_at_once():
    async def scenario():
        worker = HeldWorker()
        svc = await opened(service(worker))
        first = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(worker.entered.is_set)
        second = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_for_queue(svc.limiter, 1)

        with pytest.raises(ServerBusyError) as exc_info:
            await svc.avalidate(jpeg())

        assert type(exc_info.value) is ServerBusyError
        assert failure(exc_info.value) == BUSY
        assert state(svc.limiter) == (1, 1)
        assert worker.arrivals == 1  # 대기·초과 요청은 Modal 을 부르지 않았다
        worker.released.set()
        results = await asyncio.gather(first, second)
        assert [result.s3_key for result in results] == [KEY, KEY]
        assert worker.arrivals == 2
        assert state(svc.limiter) == (0, 0)

    run(scenario)


def test_waiting_past_the_queue_limit_is_429_without_calling_the_worker():
    async def scenario():
        worker = HeldWorker()
        svc = await opened(service(worker, limiter=AsyncBodyValidationLimiter(1, 1, max_wait_seconds=0.02)))
        first = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(worker.entered.is_set)

        with pytest.raises(SlotWaitTimeout) as exc_info:
            await svc.avalidate(jpeg())

        assert failure(exc_info.value) == BUSY
        assert state(svc.limiter) == (1, 0)
        assert worker.arrivals == 1
        worker.released.set()
        await first

    run(scenario)


def test_deadline_ending_in_the_queue_is_504_rather_than_429():
    async def scenario():
        worker = HeldWorker()
        limiter = AsyncBodyValidationLimiter(1, 1, max_wait_seconds=10.0)
        running = await opened(service(worker, limiter=limiter))
        short = await opened(service(worker, limiter=limiter, deadline=0.02))  # deadline 이 대기 제한보다 짧다
        first = asyncio.create_task(running.avalidate(jpeg()))
        await wait_until(worker.entered.is_set)

        assert await rejected(short.avalidate(jpeg())) == TIMED_OUT

        assert state(limiter) == (1, 0)
        assert worker.arrivals == 1
        worker.released.set()
        await first

    run(scenario)


def test_no_time_left_after_the_queue_is_504_and_the_slot_is_returned():
    now = [1000.0]

    class SlowSlot(AsyncBodyValidationLimiter):
        @asynccontextmanager
        async def slot(self, max_wait=None):
            async with super().slot(max_wait):
                now[0] += 50.0
                yield

    fake = FakeWorker()

    async def scenario():
        svc = await opened(service(fake, limiter=SlowSlot(1, 1), deadline=50.0, clock=lambda: now[0]))
        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
        assert state(svc.limiter) == (0, 0)

    run(scenario)

    assert fake.calls == []


def test_time_in_the_queue_comes_out_of_the_deadline():
    scale = 0.01
    fake = FakeWorker(steps=[(8 * scale, 200, {}, SUCCESS), OK])

    async def scenario():
        svc = await opened(service(fake, timeout=100 * scale, deadline=50 * scale))
        first = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(lambda: svc.limiter.running == 1)
        second = asyncio.create_task(svc.avalidate(jpeg()))
        await asyncio.gather(first, second)

    run(scenario)

    budget = fake.calls[1].extensions["timeout"]["read"]
    assert 38 * scale < budget <= 42 * scale + 0.02  # 50 - 8 ≈ 42초만 쓸 수 있다


def test_waiting_requests_call_the_worker_in_arrival_order():
    async def scenario():
        worker = HeldWorker()
        svc = await opened(service(worker, limiter=AsyncBodyValidationLimiter(1, 2)))
        bodies = [jpeg((600, 800 + index)) for index in range(3)]
        tasks = []
        for index, body in enumerate(bodies):
            tasks.append(asyncio.create_task(svc.avalidate(body)))
            if index == 0:
                await wait_until(worker.entered.is_set)
            else:
                await wait_for_queue(svc.limiter, index)
        worker.released.set()
        await asyncio.gather(*tasks)

        assert [bodies.index(_image_of(call)) for call in worker.calls] == [0, 1, 2]

    run(scenario)


def _image_of(request):
    message = message_from_bytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    return message.get_payload()[0].get_payload(decode=True)


# --- 시간 초과와 취소 ---


def test_worker_slower_than_the_deadline_is_504_and_the_http_call_is_cancelled():
    fake = CancelTrackingWorker(steps=[(5.0, 200, {}, SUCCESS), OK])

    async def scenario():
        svc = await opened(service(fake, timeout=0.05, deadline=0.05))

        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT

        assert fake.cancelled is True and fake.completed is False
        assert len(fake.calls) == 1
        assert state(svc.limiter) == (0, 0)

    run(scenario)


def test_cancelled_http_call_stays_a_cancellation_and_returns_the_slot(caplog):
    fake = CancelTrackingWorker(steps=[(5.0, 200, {}, SUCCESS), OK])

    async def scenario():
        svc = await opened(service(fake))
        task = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(lambda: fake.calls)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):  # 503·504 로 바뀌지 않는다
            await task

        assert fake.cancelled is True
        assert state(svc.limiter) == (0, 0)
        assert (await svc.avalidate(jpeg())).s3_key == KEY  # 슬롯이 돌아왔다
        assert len(fake.calls) == 2

    with caplog.at_level(logging.INFO, logger="app.body_image_validation.remote"):
        run(scenario)

    summary = next(r.getMessage() for r in caplog.records if "body image request:" in r.getMessage())
    assert "outcome=cancelled" in summary
    assert "http_cancelled=True" in summary  # 이 쪽 연결은 끊었다
    assert "remote_cancel=not_requested" in summary
    assert "remote_stopped=unknown" in summary  # Modal 작업이 멈췄는지는 알 수 없다
    for secret in ("wk-test", "ws-test", "worker-key"):
        assert secret not in caplog.text


def test_cancelling_the_running_request_starts_the_waiting_one_with_one_slot():
    async def scenario():
        worker = HeldWorker()
        svc = await opened(service(worker))
        first = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(worker.entered.is_set)
        second = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_for_queue(svc.limiter, 1)

        first.cancel()

        await wait_until(lambda: worker.arrivals == 2)
        assert state(svc.limiter) == (1, 0)  # 두 번 반환했다면 running 은 0 이다
        with pytest.raises(asyncio.CancelledError):
            await first
        worker.released.set()
        assert (await second).s3_key == KEY
        assert state(svc.limiter) == (0, 0)

    run(scenario)


def test_cancelled_while_queued_leaves_without_calling_the_worker():
    async def scenario():
        worker = HeldWorker()
        svc = await opened(service(worker))
        first = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(worker.entered.is_set)
        second = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_for_queue(svc.limiter, 1)

        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second

        assert state(svc.limiter) == (1, 0)
        worker.released.set()
        await first
        assert worker.arrivals == 1
        assert state(svc.limiter) == (0, 0)

    run(scenario)


def test_cancelled_during_the_input_check_takes_no_slot_and_closes_the_image(monkeypatch):
    slow = SlowOpen(monkeypatch)
    fake = FakeWorker()

    async def scenario():
        svc = await opened(service(fake))
        task = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(slow.entered.is_set)

        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            slow.released.set()

        await wait_until(lambda: slow.closed)
        assert state(svc.limiter) == (0, 0)

    run(scenario)

    assert fake.calls == []


def test_cancel_delivered_with_the_deadline_stays_a_cancellation(monkeypatch):
    real_on_timeout = asyncio.timeouts.Timeout._on_timeout

    def timeout_then_cancel(timeout):
        real_on_timeout(timeout)  # deadline 이 끝난 그 step 에 바깥 취소도 온다
        timeout._task.cancel()

    monkeypatch.setattr(asyncio.timeouts.Timeout, "_on_timeout", timeout_then_cancel)
    fake = FakeWorker(steps=[(5.0, 200, {}, SUCCESS)])

    async def scenario():
        svc = await opened(service(fake, timeout=30.0, deadline=0.05))

        with pytest.raises(asyncio.CancelledError):  # 504 로 바뀌지 않는다
            await svc.avalidate(jpeg())

        assert state(svc.limiter) == (0, 0)

    run(scenario)

    assert len(fake.calls) == 1


# --- limiter ---


def test_avalidate_refuses_the_sync_limiter_before_calling_the_worker():
    fake = FakeWorker()

    async def scenario():
        svc = await opened(service(fake, limiter=BodyValidationLimiter(1, 1)))
        with pytest.raises(TypeError):
            await svc.avalidate(jpeg())
        assert state(svc.limiter) == (0, 0)

    run(scenario)

    assert fake.calls == []


def test_a_service_stays_on_the_event_loop_it_first_ran_on():
    fake = FakeWorker()
    svc = service(fake)

    with asyncio.Runner() as first, asyncio.Runner() as second:
        first.run(svc.aopen())
        first.run(svc.avalidate(jpeg()))
        with pytest.raises(RuntimeError, match="event loop"):  # 503 등으로 숨기지 않는다
            second.run(svc.avalidate(jpeg()))
        first.run(svc.aclose())

    assert len(fake.calls) == 1


# --- 실제 TCP 연결 ---


def tcp_service(server, **kwargs):
    kwargs.setdefault("limiter", AsyncBodyValidationLimiter(1, 0))
    return sync_tcp_service(server, **kwargs)


@pytest.mark.parametrize("keep_alive", [True, False], ids=["keep-alive", "server-closes"])
def test_consecutive_requests_work_over_real_connections(local_worker, keep_alive):  # noqa: F811
    local_worker.keep_alive = keep_alive

    async def scenario():
        svc = await opened(tcp_service(local_worker))
        results = [await svc.avalidate(jpeg()) for _ in range(6)]

        assert [result.s3_key for result in results] == [KEY] * 6
        assert state(svc.limiter) == (0, 0)

    run(scenario)

    assert local_worker.requests == 6  # 재시도 없음


def test_refused_connection_is_503_over_a_real_socket(local_worker):  # noqa: F811
    async def scenario():
        svc = await opened(tcp_service(local_worker))
        svc.endpoint = "http://127.0.0.1:1" + URL

        assert await rejected(svc.avalidate(jpeg())) == UNAVAILABLE
        assert state(svc.limiter) == (0, 0)

    run(scenario)

    assert local_worker.requests == 0


def test_request_after_a_deadline_works_over_real_connections(local_worker):  # noqa: F811
    local_worker.delays = [1.0]

    async def scenario():
        # 첫 요청이 서버에 닿은 뒤 끊기도록 제한을 넉넉히 둔다(먼저 끊기면 지연을 다음 요청이 받는다).
        svc = await opened(tcp_service(local_worker, timeout=0.4, deadline=0.4))

        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
        svc.total_deadline, svc.timeout = 5.0, httpx.Timeout(5.0)  # 짧은 제한은 첫 요청만을 위한 것이다
        assert (await svc.avalidate(jpeg())).s3_key == KEY

    run(scenario)

    assert local_worker.requests == 2


def test_event_loop_keeps_running_during_a_real_request(local_worker):  # noqa: F811
    local_worker.delays = [0.1]

    async def scenario():
        svc = await opened(tcp_service(local_worker))
        task = asyncio.create_task(svc.avalidate(jpeg()))
        ticks = 0
        while not task.done():
            ticks += 1
            await asyncio.sleep(0.001)

        assert ticks > 20
        assert task.result().s3_key == KEY

    run(scenario)
