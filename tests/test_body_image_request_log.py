import asyncio
import io
import logging
import re
from contextlib import asynccontextmanager

import httpx
import pytest

from app.body_image_validation import remote as remote_module
from app.body_image_validation.async_concurrency import AsyncBodyValidationLimiter
from app.body_image_validation.concurrency import SlotWaitTimeout
from app.body_image_validation.exceptions import ServerBusyError
from app.config.logging_config import LOG_FORMAT, KstDefaultFormatter
from tests.test_body_image_async_concurrency import wait_for_queue, wait_until
from tests.test_body_image_concurrent_api import HeldWorker
from tests.test_body_image_remote import (
    AUTH,
    POLL,
    SUCCESS,
    URL,
    WORKER,
    FakeWorker,
    common,
    jpeg,
    post,
    redirect,
    remote_service,
    use,
)
from tests.test_body_image_remote import client as remote_client  # noqa: F401 (pytest fixture)
from tests.test_body_image_remote_async import (
    OK,
    TIMED_OUT,
    UNAVAILABLE,
    local_worker,  # noqa: F401 (pytest fixture)
    opened,
    rejected,
    run,
    service,
    tcp_service,
)

LOGGER = "app.body_image_validation.remote"
SECRET_TEXT = f"{WORKER}{POLL} key wk-test ws-test worker-key Bearer worker-key"


def summaries(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("body image request:")]


def fields(line):
    return dict(pair.split("=", 1) for pair in line.removeprefix("body image request: ").split())


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    return caplog


# --- HTTP 오류 분류: 내부 로그는 원인별로, 외부 응답은 기존 그대로 ---

ERROR_CASES = [
    (httpx.ConnectError, "connection_failed", "connect_error", UNAVAILABLE),
    (httpx.ReadError, "connection_failed", "read_error", UNAVAILABLE),
    (httpx.WriteError, "connection_failed", "write_error", UNAVAILABLE),
    (httpx.CloseError, "connection_failed", "close_error", UNAVAILABLE),
    (httpx.RemoteProtocolError, "connection_failed", "remote_protocol_error", UNAVAILABLE),
    (httpx.LocalProtocolError, "connection_failed", "local_protocol_error", UNAVAILABLE),
    (httpx.ProxyError, "connection_failed", "proxy_error", UNAVAILABLE),
    (httpx.UnsupportedProtocol, "connection_failed", "unsupported_protocol", UNAVAILABLE),
    (httpx.TransportError, "connection_failed", "transport_error", UNAVAILABLE),
    (httpx.ConnectTimeout, "network_timeout", "connect_timeout", TIMED_OUT),
    (httpx.ReadTimeout, "network_timeout", "read_timeout", TIMED_OUT),
    (httpx.WriteTimeout, "network_timeout", "write_timeout", TIMED_OUT),
    (httpx.PoolTimeout, "network_timeout", "pool_timeout", TIMED_OUT),
    (httpx.TimeoutException, "network_timeout", "timeout", TIMED_OUT),
]


@pytest.mark.parametrize(
    ("error", "outcome", "error_type", "public"), ERROR_CASES, ids=[case[2] for case in ERROR_CASES]
)
def test_http_errors_are_classified_and_the_public_response_is_unchanged(
    logs, error, outcome, error_type, public
):
    fake = FakeWorker(error=error(SECRET_TEXT))

    async def scenario():
        svc = await opened(service(fake))
        assert await rejected(svc.avalidate(jpeg())) == public

    run(scenario)

    [line] = summaries(logs)
    logged = fields(line)
    assert (logged["outcome"], logged["error"]) == (outcome, error_type)
    assert logged["code"] == public[1]
    assert logged["http_cancelled"] == str(outcome == "network_timeout")
    assert len(fake.calls) == 1  # 재시도 없음


def test_every_httpx_transport_error_has_its_own_name():
    def concrete(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from concrete(sub)

    named = {case[0] for case in ERROR_CASES}
    missing = [cls.__name__ for cls in concrete(httpx.TransportError) if cls.__mro__[0] not in named]
    abstract = {"NetworkError", "ProtocolError"}

    assert set(missing) == abstract  # 묶음 클래스 외에는 httpx 를 올려도 이름 없이 넘어가지 않는다


def test_subclass_of_a_known_error_keeps_its_name():
    class CustomRead(httpx.ReadError):
        pass

    assert remote_module._error_type(CustomRead("x")) == "read_error"


# --- 요청별 요약 ---


def test_one_summary_per_request_with_a_new_id(logs):
    async def scenario():
        svc = await opened(service(FakeWorker()))
        for _ in range(3):
            await svc.avalidate(jpeg())

    run(scenario)

    lines = [fields(line) for line in summaries(logs)]
    assert len(lines) == 3
    assert len({line["request_id"] for line in lines}) == 3
    assert all(re.fullmatch(r"[0-9a-f]{16}", line["request_id"]) for line in lines)


def test_each_stage_is_timed_on_the_injected_clock(logs, monkeypatch):
    now = [1000.0]
    real_open = remote_module.open_image

    def slow_open(body):
        now[0] += 1.0
        return real_open(body)

    class SlowSlot(AsyncBodyValidationLimiter):
        @asynccontextmanager
        async def slot(self, max_wait=None):
            now[0] += 2.0
            async with super().slot(max_wait):
                yield

    class SlowWorker(FakeWorker):
        async def _handle(self, request):
            now[0] += 3.0
            return await super()._handle(request)

    monkeypatch.setattr(remote_module, "open_image", slow_open)

    async def scenario():
        svc = await opened(
            service(SlowWorker(), limiter=SlowSlot(1, 1), deadline=60.0, clock=lambda: now[0])
        )
        await svc.avalidate(jpeg())

    run(scenario)

    logged = fields(summaries(logs)[0])
    assert logged["input_check"] == "1.000s"
    assert logged["queue_wait"] == "2.000s"
    assert logged["http"] == "3.000s"
    assert logged["total"] == "6.000s"
    assert logged["budget"] == "57.000s"
    assert (logged["outcome"], logged["error"]) == ("response", "-")
    assert (logged["status"], logged["code"]) == ("200", "BODY_IMAGE_UPLOAD_SUCCESS")


def test_stages_that_did_not_run_are_not_given_a_time(logs):
    async def scenario():
        svc = await opened(service(FakeWorker()))
        await rejected(svc.avalidate(b"not-an-image"))

    run(scenario)

    logged = fields(summaries(logs)[0])
    assert logged["outcome"] == "invalid_input"
    assert logged["code"] == "INVALID_IMAGE"
    assert logged["input_check"] != "-"
    assert (logged["queue_wait"], logged["http"], logged["budget"], logged["status"]) == ("-",) * 4


def test_queue_full_and_queue_timeout_are_told_apart(logs):
    async def scenario():
        worker = HeldWorker()
        svc = await opened(service(worker, limiter=AsyncBodyValidationLimiter(1, 1, max_wait_seconds=0.02)))
        running = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(worker.entered.is_set)
        with pytest.raises(SlotWaitTimeout):
            await svc.avalidate(jpeg())

        waiting = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_for_queue(svc.limiter, 1)
        with pytest.raises(ServerBusyError):
            await svc.avalidate(jpeg())
        worker.released.set()
        await running
        await asyncio.gather(waiting, return_exceptions=True)

    run(scenario)

    outcomes = [fields(line)["outcome"] for line in summaries(logs)]
    assert outcomes[0] == "queue_timeout"
    assert "queue_full" in outcomes
    for line in map(fields, summaries(logs)):
        if line["outcome"].startswith("queue_"):
            assert line["code"] == "SERVER_BUSY"
            assert line["http"] == "-"


def test_worker_response_is_logged_with_its_status_and_the_final_code(logs):
    steps = [(0, 422, {}, common("IMAGE_TOO_DARK", "dark")), (0, 200, {}, b"not json")]

    async def scenario():
        svc = await opened(service(FakeWorker(steps=steps)))
        await rejected(svc.avalidate(jpeg()))
        await rejected(svc.avalidate(jpeg()))

    run(scenario)

    relayed, invalid = map(fields, summaries(logs))
    assert (relayed["outcome"], relayed["status"], relayed["code"]) == ("response", "422", "IMAGE_TOO_DARK")
    assert (invalid["outcome"], invalid["status"], invalid["code"]) == (
        "response", "200", "BODY_IMAGE_WORKER_UNAVAILABLE",
    )


def test_redirect_count_and_blocked_redirect_are_logged(logs):
    steps = [redirect(POLL), OK, redirect("https://evil.example.com/x")]

    async def scenario():
        svc = await opened(service(FakeWorker(steps=steps)))
        await svc.avalidate(jpeg())
        await rejected(svc.avalidate(jpeg()))

    run(scenario)

    followed, blocked = map(fields, summaries(logs))
    assert (followed["redirects"], followed["status"]) == ("1", "200")
    assert (blocked["outcome"], blocked["status"], blocked["redirects"]) == ("error", "303", "1")
    assert blocked["code"] == "BODY_IMAGE_WORKER_UNAVAILABLE"


def test_unexpected_error_logs_only_its_kind(logs):
    fake = FakeWorker(error=RuntimeError(SECRET_TEXT))

    async def scenario():
        svc = await opened(service(fake))
        with pytest.raises(RuntimeError):
            await svc.avalidate(jpeg())

    run(scenario)

    logged = fields(summaries(logs)[0])
    assert (logged["outcome"], logged["error"]) == ("error", "RuntimeError")
    assert "wk-test" not in summaries(logs)[0]


# --- deadline 과 취소: 어느 단계에서 끝났는지 ---


def test_deadline_in_each_stage_is_logged_as_deadline_exceeded(logs, monkeypatch):
    now = [1000.0]
    real_open = remote_module.open_image

    def slow_open(body):
        now[0] += 50.0
        return real_open(body)

    class SlowSlot(AsyncBodyValidationLimiter):
        @asynccontextmanager
        async def slot(self, max_wait=None):
            async with super().slot(max_wait):
                now[0] += 50.0
                yield

    async def scenario():
        # HTTP 호출 중
        fake = FakeWorker(steps=[(5.0, 200, {}, SUCCESS)])
        svc = await opened(service(fake, timeout=30.0, deadline=0.02))
        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
        # 대기열을 나온 직후
        svc = await opened(service(FakeWorker(), limiter=SlowSlot(1, 1), deadline=50.0, clock=lambda: now[0]))
        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT
        # 입력 검사 뒤
        monkeypatch.setattr(remote_module, "open_image", slow_open)
        svc = await opened(service(FakeWorker(), deadline=50.0, clock=lambda: now[0]))
        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT

    run(scenario)

    in_http, after_queue, after_input = map(fields, summaries(logs))
    for line in (in_http, after_queue, after_input):
        assert (line["outcome"], line["error"]) == ("deadline_exceeded", "deadline_exceeded")
    assert (in_http["http_cancelled"], in_http["remote_stopped"]) == ("True", "unknown")
    assert (after_queue["http"], after_queue["http_cancelled"]) == ("-", "False")
    assert (after_input["queue_wait"], after_input["remote_stopped"]) == ("-", "n/a")


def test_deadline_shorter_than_the_network_timeout_is_not_a_read_timeout(logs):
    async def scenario():
        fake = FakeWorker(steps=[(5.0, 200, {}, SUCCESS)])
        svc = await opened(service(fake, timeout=30.0, deadline=0.02))
        await rejected(svc.avalidate(jpeg()))

    run(scenario)

    assert fields(summaries(logs)[0])["error"] == "deadline_exceeded"


@pytest.mark.parametrize("stage", ["queue", "http"])
def test_cancellation_is_logged_with_the_stage_it_stopped(logs, stage):
    async def scenario():
        worker = HeldWorker()
        svc = await opened(service(worker))
        running = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(worker.entered.is_set)
        target = running
        if stage == "queue":
            target = asyncio.create_task(svc.avalidate(jpeg()))
            await wait_for_queue(svc.limiter, 1)

        target.cancel()
        with pytest.raises(asyncio.CancelledError):
            await target

        worker.released.set()
        await asyncio.gather(running, return_exceptions=True)

    run(scenario)

    cancelled = next(line for line in map(fields, summaries(logs)) if line["outcome"] == "cancelled")
    assert cancelled["error"] == "cancelled"
    assert cancelled["http_cancelled"] == str(stage == "http")
    assert (cancelled["http"] == "-") == (stage == "queue")
    assert cancelled["queue_wait"] != "-"
    assert cancelled["code"] == "-"  # 응답을 만들지 않았다


# --- 로그에 남기지 않는 것 ---


@pytest.fixture
def real_log():
    """운영과 같은 포매터의 최종 출력을 모은다(예외 traceback 포함)."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(KstDefaultFormatter(LOG_FORMAT, use_colors=False))
    logger = logging.getLogger("app")
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield stream
    logger.removeHandler(handler)
    logger.setLevel(level)


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout],
    ids=["connect", "read", "protocol", "read-timeout", "connect-timeout"],
)
def test_exception_text_does_not_reach_the_final_log_output(remote_client, monkeypatch, real_log, error):  # noqa: F811
    fake = FakeWorker(error=error(SECRET_TEXT))
    use(monkeypatch, remote_service(fake))

    response = post(remote_client, jpeg())

    assert response.status_code in (503, 504)
    text = real_log.getvalue()
    assert "body image request:" in text
    assert "body image validation failed" in text
    for secret in ("wk-test", "ws-test", "worker-key", "fc-test123", "__modal_function_call_id", "modal.run"):
        assert secret not in text


def test_public_error_does_not_carry_the_httpx_exception():
    async def scenario():
        svc = await opened(service(FakeWorker(error=httpx.ConnectError(SECRET_TEXT))))
        with pytest.raises(remote_module.RelayedWorkerError) as exc_info:
            await svc.avalidate(jpeg())

        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True

    run(scenario)


def test_request_id_stays_out_of_the_api_response(remote_client, monkeypatch, logs):  # noqa: F811
    use(monkeypatch, remote_service(FakeWorker()))

    response = remote_client.post(URL, files={"image": ("body.jpg", jpeg(), "image/jpeg")}, headers=AUTH)

    request_id = fields(summaries(logs)[0])["request_id"]
    assert set(response.json()) == {"code", "data", "message"}
    assert request_id not in response.text
    assert request_id not in str(response.headers)


# --- 실제 소켓의 분류 ---


def test_real_refused_connection_is_connect_error(logs, local_worker):  # noqa: F811
    async def scenario():
        svc = await opened(tcp_service(local_worker))
        svc.endpoint = "http://127.0.0.1:1" + URL
        await rejected(svc.avalidate(jpeg()))

    run(scenario)

    assert fields(summaries(logs)[0])["error"] == "connect_error"


def test_real_stalled_response_is_read_timeout_before_the_deadline(logs, local_worker):  # noqa: F811
    local_worker.delays = [0.4]

    async def scenario():
        svc = await opened(tcp_service(local_worker, timeout=0.05, deadline=5.0))
        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT

    run(scenario)

    logged = fields(summaries(logs)[0])
    assert (logged["outcome"], logged["error"]) == ("network_timeout", "read_timeout")


def test_real_stalled_response_is_deadline_exceeded_when_the_deadline_comes_first(logs, local_worker):  # noqa: F811
    local_worker.delays = [0.4]

    async def scenario():
        svc = await opened(tcp_service(local_worker, timeout=5.0, deadline=0.05))
        assert await rejected(svc.avalidate(jpeg())) == TIMED_OUT

    run(scenario)

    assert fields(summaries(logs)[0])["error"] == "deadline_exceeded"


def test_real_connection_dropped_without_a_response_is_remote_protocol_error(logs, local_worker):  # noqa: F811
    local_worker.drop_next = True

    async def scenario():
        svc = await opened(tcp_service(local_worker))
        assert await rejected(svc.avalidate(jpeg())) == UNAVAILABLE

    run(scenario)

    assert fields(summaries(logs)[0])["error"] == "remote_protocol_error"
    assert local_worker.requests == 1


def test_client_close_is_logged_with_what_was_in_flight(logs):
    async def scenario():
        worker = HeldWorker()
        svc = await opened(service(worker))
        running = asyncio.create_task(svc.avalidate(jpeg()))
        await wait_until(worker.entered.is_set)
        closing = asyncio.create_task(svc.aclose())
        await asyncio.sleep(0.01)
        worker.released.set()
        await asyncio.gather(running, closing)

    run(scenario)

    [line] = [r.getMessage() for r in logs.records if "client closing" in r.getMessage()]
    assert "in_flight=1 left=0" in line
