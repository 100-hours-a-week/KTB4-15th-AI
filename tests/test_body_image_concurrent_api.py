import asyncio
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.body_image_validation import router as validation_router
from app.body_image_validation.models import BodyImageValidationResult
from app.body_image_validation.runtime import BodyImageRuntimeManager
from app.config import body_image_validation as config
from tests.test_body_image_queue_deadline import BUSY, wait_until
from tests.test_body_image_remote import FakeWorker, jpeg
from tests.test_body_image_validation_service import FakePersonDetector, image_bytes
from tests.test_body_image_validation_service import service as local_service
from tests.test_body_image_warmup_api import (  # noqa: F401 (pytest fixture)
    AUTH,
    VALIDATE_URL,
    remote,
    runtime,
)

QUEUE_WAIT = 0.5
DEADLINE = 3.0
ANSWER_WITHIN = 2.0  # DEADLINE 보다 짧다. 이 안에 응답이 오지 않으면 막힌 것으로 본다


class WorkerSlot(httpx.AsyncBaseTransport):
    """공유 client 를 만들 때 넘기는 transport. 테스트가 고른 가짜 worker 로 요청을 보낸다."""

    def __init__(self):
        self.worker = None

    async def handle_async_request(self, request):
        return await self.worker.transport.handle_async_request(request)


class HeldWorker(FakeWorker):
    """released 전까지 응답하지 않는 Modal worker 자리. 서버 loop·별도 loop 어디서 돌아도 되게 Event 를 polling 한다."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arrivals = 0
        self.entered = threading.Event()
        self.released = threading.Event()

    async def _handle(self, request):
        self.arrivals += 1
        self.entered.set()
        while not self.released.is_set():
            await asyncio.sleep(0.001)
        return await super()._handle(request)


class HeldDetector(FakePersonDetector):
    """local 검증의 동기 작업(모델 추론) 자리. released 전까지 호출한 thread 를 막는다."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.released = threading.Event()

    def detect(self, image):
        self.entered.set()
        self.released.wait(DEADLINE)
        return super().detect(image)


class Call(threading.Thread):
    """요청 하나를 별도 thread 에서 보낸다."""

    def __init__(self, send):
        super().__init__(daemon=True)
        self.send = send
        self.response = None
        self.start()

    def run(self):
        self.response = self.send()

    def result(self):
        self.join(ANSWER_WITHIN)
        assert not self.is_alive(), "응답이 오지 않았다"
        return self.response


class Api:
    def __init__(self, client, manager, slot):
        self.client, self.manager, self.slot = client, manager, slot
        self.held = []

    @property
    def limiter(self):
        return self.manager.runtime.service.limiter

    def install(self, worker):
        self.slot.worker = worker
        self.held.append(worker)
        return worker

    def post(self, body=None):
        files = {"image": ("body.jpg", jpeg() if body is None else body, "image/jpeg")}
        return self.client.post(VALIDATE_URL, files=files, headers=AUTH)

    def start(self, body=None):
        return Call(lambda: self.post(body))

    def health(self):
        return Call(lambda: self.client.get("/health")).result()

    def fill_the_queue(self, worker):
        """실행 1 + 대기 1 을 채운다."""
        first = self.start()
        assert worker.entered.wait(ANSWER_WITHIN)
        second = self.start()
        wait_until(lambda: self.limiter.waiting == 1)
        return first, second


@pytest.fixture
def api(remote, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "MAX_CONCURRENT_BODY_VALIDATIONS", 1)
    monkeypatch.setattr(config, "MAX_WAITING_BODY_VALIDATIONS", 1)
    monkeypatch.setattr(config, "MAX_QUEUE_WAIT_SECONDS", QUEUE_WAIT)
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", DEADLINE)
    monkeypatch.setattr(config, "REMOTE_READ_TIMEOUT_SECONDS", DEADLINE)
    slot = WorkerSlot()
    manager = BodyImageRuntimeManager(remote_transport=slot)
    monkeypatch.setattr(main, "runtime_manager", manager)
    monkeypatch.setattr(validation_router, "runtime_manager", manager)
    with TestClient(main.app, raise_server_exceptions=False) as client:
        test_api = Api(client, manager, slot)
        try:
            yield test_api
        finally:
            for worker in test_api.held:
                worker.released.set()


# --- 실행 1 + 대기 1 ---


def test_one_runs_one_waits_and_the_third_is_rejected_at_once(api):
    worker = api.install(HeldWorker())
    first, second = api.fill_the_queue(worker)

    third = api.post()

    assert third.status_code == 429
    assert third.json() == BUSY
    assert worker.arrivals == 1  # 대기·초과 요청은 Modal 을 부르지 않았다
    worker.released.set()
    assert [first.result().status_code, second.result().status_code] == [200, 200]
    assert worker.arrivals == 2  # 대기하던 요청은 앞 요청이 끝난 뒤 실행됐다
    wait_until(lambda: (api.limiter.running, api.limiter.waiting) == (0, 0))


def test_waiting_past_the_queue_limit_is_429_and_the_queue_recovers(api):
    worker = api.install(HeldWorker())
    first = api.start()
    assert worker.entered.wait(ANSWER_WITHIN)

    started = time.monotonic()
    waited = api.post()
    elapsed = time.monotonic() - started

    assert waited.status_code == 429
    assert waited.json() == BUSY
    assert QUEUE_WAIT * 0.9 <= elapsed < DEADLINE  # 앞 요청을 끝까지 기다리지 않고 대기 제한에서 포기했다
    assert worker.arrivals == 1
    worker.released.set()
    assert first.result().status_code == 200
    assert api.post().status_code == 200  # 슬롯과 대기열 자리가 돌아왔다


def test_invalid_upload_gets_its_own_error_while_the_queue_is_full(api):
    worker = api.install(HeldWorker())
    first, second = api.fill_the_queue(worker)

    response = api.post(b"not-an-image")

    assert (response.status_code, response.json()["code"]) == (400, "INVALID_IMAGE")
    assert api.limiter.waiting == 1  # 입력 오류는 대기열을 건드리지 않는다
    worker.released.set()
    first.result(), second.result()


# --- 검증이 도는 동안에도 서버는 다른 요청에 응답한다 ---


def test_server_stays_responsive_while_the_worker_call_runs(api):
    worker = api.install(HeldWorker())
    running = api.start()
    assert worker.entered.wait(ANSWER_WITHIN)

    assert api.health().status_code == 200

    worker.released.set()
    assert running.result().status_code == 200


def test_server_stays_responsive_while_a_request_waits_in_the_queue(api):
    worker = api.install(HeldWorker())
    first, second = api.fill_the_queue(worker)

    assert api.health().status_code == 200
    assert api.limiter.waiting == 1  # 대기가 끝난 뒤가 아니라 대기하는 동안 응답했다

    worker.released.set()
    first.result(), second.result()


def test_local_validation_returns_the_key_and_stays_off_the_event_loop(api, monkeypatch):
    detector = HeldDetector()
    local = local_service(person=detector)
    monkeypatch.setattr(validation_router, "get_service", lambda: local)
    running = api.start(image_bytes())
    try:
        assert detector.entered.wait(ANSWER_WITHIN)

        assert api.health().status_code == 200
    finally:
        detector.released.set()

    response = running.result()
    assert response.status_code == 200
    assert response.json()["data"]["s3_key"].startswith("body-images/")


def test_local_service_runs_in_the_threadpool(api, monkeypatch):
    threads = []

    class LocalStub:
        def validate(self, body):
            threads.append(threading.get_ident())
            return BodyImageValidationResult(s3_key="body-images/local.png")

    monkeypatch.setattr(validation_router, "get_service", LocalStub)
    loop_thread = api.client.portal.call(threading.get_ident)

    response = api.post(image_bytes())

    assert response.json()["data"] == {"s3_key": "body-images/local.png"}
    assert threads and threads[0] != loop_thread


def test_slow_local_runtime_start_stays_off_the_event_loop(api, monkeypatch):
    entered, released = threading.Event(), threading.Event()
    local = local_service()

    def slow_start():
        entered.set()
        released.wait(DEADLINE)
        return local

    monkeypatch.setattr(validation_router, "get_service", slow_start)
    running = api.start(image_bytes())
    try:
        assert entered.wait(ANSWER_WITHIN)

        assert api.health().status_code == 200  # 모델을 올리는 동안에도 서버는 응답한다
    finally:
        released.set()

    assert running.result().status_code == 200


def test_rejected_upload_is_answered_from_the_shared_router_in_both_modes(api, monkeypatch):
    local = local_service()

    remote_response = api.post(b"not-an-image")
    monkeypatch.setattr(validation_router, "get_service", lambda: local)
    local_response = api.post(b"not-an-image")

    assert remote_response.status_code == local_response.status_code == 400
    assert remote_response.json() == local_response.json()
