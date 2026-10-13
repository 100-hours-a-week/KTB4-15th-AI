"""대기열 최대 대기시간과, 대기열 대기까지 포함한 요청당 하나의 전체 deadline(remote 모드).

시간은 1초 → 10ms 로 줄여 실제 Condition.wait·asyncio.sleep·asyncio.timeout 을 그대로 쓴다(SCALE).
설정: 실행 1 + 대기 1, 대기열 최대 10초, 전체 deadline 50초.
"""

import threading
import time

import httpx
import pytest

from app.body_image_validation import deadline as worker_deadline
from app.body_image_validation import router
from app.body_image_validation.concurrency import BodyValidationLimiter, SlotWaitTimeout
from app.body_image_validation.exceptions import BodyImageValidationError, ServerBusyError
from app.body_image_validation.remote import RemoteBodyImageValidationService
from tests.test_body_image_remote import POLL, SUCCESS, TIMEOUT, FakeWorker, jpeg, redirect, use
from tests.test_body_image_remote import client as remote_client  # noqa: F401 (pytest fixture)
from tests.test_body_image_remote import post as remote_post

S = 0.01  # 1초 → 10ms
QUEUE_WAIT = 10 * S
DEADLINE = 50 * S
BUSY = {"code": "SERVER_BUSY", "data": None,
        "message": "현재 이미지 처리 요청이 많습니다. 잠시 후 다시 시도해주세요."}


class TrackingWorker(FakeWorker):
    """동시에 worker 에 들어와 있는 요청 수의 최댓값을 잰다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    async def _handle(self, request):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            return await super()._handle(request)
        finally:
            with self._lock:
                self.in_flight -= 1


def service(fake, deadline=DEADLINE, queue_wait=QUEUE_WAIT, limiter=None, timeout=100 * S):
    return RemoteBodyImageValidationService(
        "https://example--lookddak-body-image-staging.modal.run",
        proxy_key="wk-test",
        proxy_secret="ws-test",
        api_key="worker-key",
        timeout=timeout,
        total_deadline=deadline,
        limiter=limiter or BodyValidationLimiter(1, 1, max_wait_seconds=queue_wait),
        transport=fake.transport,
    )


def ok(delay_seconds):
    return (delay_seconds * S, 200, {}, SUCCESS)


class Call:
    """service.validate 를 다른 thread 에서 실행하고 결과·예외·시간을 남긴다."""

    def __init__(self, svc, start_delay=0.0):
        self.svc, self.result, self.error, self.elapsed = svc, None, None, None
        self.thread = threading.Thread(target=self._run, args=(start_delay,))

    def _run(self, start_delay):
        time.sleep(start_delay)
        started = time.monotonic()
        try:
            self.result = self.svc.validate(jpeg())
        except BaseException as error:  # noqa: BLE001 - 결과로 확인한다
            self.error = error
        self.elapsed = time.monotonic() - started

    def start(self):
        self.thread.start()
        return self

    def join(self):
        self.thread.join(5)
        assert not self.thread.is_alive()
        return self


def wait_until(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "상태가 바뀌지 않았다"
        time.sleep(0.001)


def idle(limiter):
    return (limiter.running, limiter.waiting) == (0, 0)


# --- 1~4. 대기열 ---


def test_single_request_runs_immediately():
    fake = FakeWorker(steps=[ok(1)])
    svc = service(fake)

    call = Call(svc).start().join()

    assert call.result.s3_key == "body-images/abc.png"
    assert call.elapsed < 5 * S + 0.05
    assert idle(svc.limiter)


def test_waiting_request_runs_after_the_running_one():
    fake = FakeWorker(steps=[ok(5)])
    svc = service(fake)

    a = Call(svc).start()
    wait_until(lambda: svc.limiter.running == 1)
    b = Call(svc).start()
    wait_until(lambda: svc.limiter.waiting == 1)
    a.join(), b.join()

    assert a.result and b.result
    assert len(fake.calls) == 2
    assert b.elapsed >= 3 * S  # A 가 끝날 때까지 기다렸다
    assert idle(svc.limiter)


def test_waiting_more_than_10s_is_429_and_the_queue_is_restored():
    fake = FakeWorker(steps=[ok(20)])  # A 가 20초 동안 슬롯을 잡는다
    svc = service(fake)

    a = Call(svc).start()
    wait_until(lambda: svc.limiter.running == 1)
    b = Call(svc).start().join()

    assert isinstance(b.error, SlotWaitTimeout) and isinstance(b.error, ServerBusyError)
    assert (b.error.status_code, b.error.code) == (429, "SERVER_BUSY")
    assert 9 * S <= b.elapsed < 15 * S  # 10초 근처에서 포기했다(A 의 20초를 기다리지 않음)
    assert svc.limiter.waiting == 0  # 대기열 자리를 돌려줬다
    a.join()
    assert a.result
    assert len(fake.calls) == 1  # B 는 Modal 을 부르지 않았다
    assert idle(svc.limiter)


def test_third_request_is_rejected_immediately_when_queue_is_full():
    fake = FakeWorker(steps=[ok(20)])
    svc = service(fake)

    a = Call(svc).start()
    wait_until(lambda: svc.limiter.running == 1)
    b = Call(svc).start()
    wait_until(lambda: svc.limiter.waiting == 1)
    c = Call(svc).start().join()

    assert type(c.error) is ServerBusyError  # 대기시간 초과가 아니라 대기열이 찬 것
    assert c.elapsed < 2 * S
    a.join(), b.join()
    assert idle(svc.limiter)


# --- 5~7. 대기열 대기도 하나의 전체 deadline 안에 ---


def test_8s_in_queue_leaves_only_42s_for_modal():
    fake = FakeWorker(steps=[ok(8), ok(1)])
    svc = service(fake)

    t0 = time.time()
    a = Call(svc).start()
    wait_until(lambda: svc.limiter.running == 1)
    b = Call(svc).start()
    a.join(), b.join()

    assert a.result and b.result
    b_request = fake.calls[1]
    budget = b_request.extensions["timeout"]["read"]  # min(네트워크 timeout 100초, 남은 deadline)
    assert 38 * S < budget <= 42 * S + 0.02  # 50 - 8 ≈ 42초만 쓸 수 있다
    sent = float(b_request.headers[worker_deadline.HEADER])
    assert sent <= t0 + DEADLINE + 0.05  # worker deadline 도 B 의 도착 기준 50초. 58초로 늘지 않는다


def test_queue_wait_plus_execution_over_50s_is_504():
    fake = FakeWorker(steps=[ok(8), ok(45)])  # B: 8초 대기 + 45초 실행 = 53초
    svc = service(fake)

    a = Call(svc).start()
    wait_until(lambda: svc.limiter.running == 1)
    b = Call(svc).start()
    a.join(), b.join()

    assert isinstance(b.error, BodyImageValidationError)
    assert (b.error.status_code, b.error.code) == (504, "BODY_IMAGE_WORKER_TIMEOUT")
    assert b.elapsed < 50 * S + 0.1  # 53초가 아니라 50초 근처에서 끝났다
    assert idle(svc.limiter)


def test_deadline_ending_in_queue_is_504_without_calling_modal():
    # deadline(5초)이 대기열 제한(10초)보다 짧으면 deadline 이 먼저 끝난다 → 429 가 아니라 504
    fake = FakeWorker(steps=[ok(20)])
    limiter = BodyValidationLimiter(1, 1, max_wait_seconds=QUEUE_WAIT)
    svc_a = service(fake, limiter=limiter)  # A: deadline 50초, 20초 실행
    svc_b = service(fake, deadline=5 * S, limiter=limiter)  # B: deadline 5초(대기열 제한 10초보다 짧음)

    a = Call(svc_a).start()
    wait_until(lambda: limiter.running == 1)
    b = Call(svc_b).start().join()

    assert (b.error.status_code, b.error.code) == (504, "BODY_IMAGE_WORKER_TIMEOUT")
    assert b.elapsed < 9 * S
    assert limiter.waiting == 0
    a.join()
    assert a.result
    assert len(fake.calls) == 1  # B 는 Modal 을 부르지 않았다


# --- 8~10, 18. 슬롯·대기열 복구 ---


@pytest.mark.parametrize(
    ("step", "status"),
    [
        (ok(1), None),
        ((0, 422, {}, b'{"code":"FULL_BODY_NOT_VISIBLE","data":null,"message":"m"}'), 422),
        ((0, 502, {}, b"bad gateway"), 502),
        (httpx.ConnectError("refused"), 503),
        ((0, 504, {}, b"gateway timeout"), 504),
        ((80 * S, 200, {}, SUCCESS), 504),  # 전체 deadline 초과
        (RuntimeError("unexpected"), None),  # 예상하지 못한 예외
    ],
    ids=["200", "422", "502", "503", "504", "deadline", "exception"],
)
def test_slot_is_returned_after_every_outcome(step, status):
    fake = FakeWorker(steps=[step])
    svc = service(fake)

    call = Call(svc).start().join()

    if status is not None:
        assert call.error.status_code == status
    assert idle(svc.limiter)


def test_waiting_request_takes_the_slot_after_a_deadline_failure():
    fake = FakeWorker(steps=[(80 * S, 200, {}, SUCCESS), ok(1)])
    svc = service(fake)

    a = Call(svc).start()
    wait_until(lambda: svc.limiter.running == 1)
    b = Call(svc).start()
    a.join(), b.join()

    # A 는 50초에 504 로 끝났고 B 는 그때 슬롯을 얻는다. 단, B 의 대기는 10초를 넘었으므로 429 다.
    assert a.error.status_code == 504
    assert isinstance(b.error, SlotWaitTimeout)
    assert idle(svc.limiter)


# --- 11~13. 기존 동작 ---


def test_redirect_still_works_inside_the_remaining_deadline():
    fake = FakeWorker(steps=[redirect(POLL, delay=5 * S), ok(1)])
    svc = service(fake)

    call = Call(svc).start().join()

    assert call.result.s3_key == "body-images/abc.png"
    assert [c.method for c in fake.calls] == ["POST", "GET"]
    assert fake.calls[0].headers[worker_deadline.HEADER] == fake.calls[1].headers[worker_deadline.HEADER]


def test_api_responses_keep_the_contract(remote_client, monkeypatch):  # noqa: F811
    limiter = BodyValidationLimiter(1, 1, max_wait_seconds=QUEUE_WAIT)
    fake = FakeWorker(steps=[ok(1)])
    use(monkeypatch, service(fake, limiter=limiter))

    with limiter.slot():  # 다른 요청이 실행 중 → 이 요청은 10초 대기 후 429
        busy = remote_post(remote_client, jpeg())
    assert busy.status_code == 429
    assert busy.json() == BUSY

    ok_response = remote_post(remote_client, jpeg())
    assert ok_response.status_code == 200
    assert ok_response.json()["code"] == "BODY_IMAGE_UPLOAD_SUCCESS"

    use(monkeypatch, service(FakeWorker(steps=[(80 * S, 200, {}, SUCCESS)]), limiter=limiter))
    late = remote_post(remote_client, jpeg())
    assert late.status_code == 504
    assert late.json() == TIMEOUT


# --- 17. 경쟁 상태 ---


def test_concurrent_requests_never_exceed_one_running_and_one_waiting():
    fake = TrackingWorker(steps=[ok(3)])
    svc = service(fake, queue_wait=4 * S)
    waiting_seen = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            waiting_seen.append((svc.limiter.running, svc.limiter.waiting))
            time.sleep(0.0005)

    watcher = threading.Thread(target=watch)
    watcher.start()
    calls = [Call(svc, start_delay=i * 0.001).start() for i in range(12)]
    for call in calls:
        call.join()
    stop.set()
    watcher.join()

    succeeded = [c for c in calls if c.result is not None]
    busy = [c for c in calls if isinstance(c.error, ServerBusyError)]
    assert len(succeeded) + len(busy) == 12
    assert succeeded and busy
    assert fake.max_in_flight == 1  # Modal 에는 한 번에 한 요청만
    assert len(fake.calls) == len(succeeded)  # 거절된 요청은 Modal 을 부르지 않았다
    assert all(0 <= r <= 1 and 0 <= w <= 1 for r, w in waiting_seen)
    assert idle(svc.limiter)


# --- local 모드는 기존 정책(대기 시간 제한 없음), remote 모드만 10초 ---


def test_local_default_limiter_keeps_unlimited_waiting(monkeypatch):
    from app.body_image_validation.service import BodyImageValidationService
    from app.config import body_image_validation as config

    monkeypatch.setattr(config, "MAX_QUEUE_WAIT_SECONDS", 0.01)
    local = BodyImageValidationService(object(), object(), object(), object())

    assert local.limiter.max_wait_seconds is None  # local 모드에는 대기 시간 제한을 걸지 않는다


def test_local_waiting_request_waits_past_the_remote_limit():
    from app.body_image_validation.service import BodyImageValidationService
    from tests.test_body_image_validation_service import (
        FakeBackgroundRemover,
        FakePersonDetector,
        FakePoseDetector,
        FakeStorage,
        image_bytes,
    )

    local = BodyImageValidationService(
        FakePersonDetector(), FakePoseDetector(), FakeBackgroundRemover(), FakeStorage()
    )
    release = threading.Event()
    holder = threading.Thread(target=lambda: _hold(local.limiter, release))
    holder.start()
    wait_until(lambda: local.limiter.running == 1)
    result = {}
    waiter = threading.Thread(target=lambda: result.update(key=local.validate(image_bytes()).s3_key))
    waiter.start()
    wait_until(lambda: local.limiter.waiting == 1)
    time.sleep(QUEUE_WAIT * 2)  # remote 대기 제한(10초 상당)을 넘겨도 local 요청은 계속 기다린다
    assert local.limiter.waiting == 1
    release.set()
    holder.join(5), waiter.join(5)

    assert result["key"].startswith("body-images/")
    assert idle(local.limiter)


def _hold(limiter, release):
    with limiter.slot():
        release.wait(5)


def test_remote_runtime_limiter_uses_the_configured_wait(monkeypatch):
    from app.body_image_validation import runtime
    from app.config import body_image_validation as config
    from app.config import settings

    monkeypatch.setattr(config, "VALIDATION_MODE", "remote")
    monkeypatch.setattr(config, "REMOTE_URL", "https://example.modal.run")
    monkeypatch.setattr(config, "MAX_QUEUE_WAIT_SECONDS", 10.0)
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 50.0)
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_KEY", "k")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_SECRET", "s")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_API_KEY", "a")

    remote = runtime.RemoteBodyImageRuntime().service

    assert remote.limiter.max_wait_seconds == 10.0
    assert remote.total_deadline == 50.0
    assert (remote.limiter._max_running, remote.limiter._max_waiting) == (1, 1)
    assert router  # router 는 그대로 쓴다(계약 유지)
