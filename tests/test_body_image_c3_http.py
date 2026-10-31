import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import uvicorn

from app import main
from app.body_image_validation import remote as remote_module
from app.body_image_validation import router as validation_router
from app.body_image_validation.remote import ENDPOINT_PATH, RemoteBodyImageValidationService
from app.body_image_validation.runtime import BodyImageRuntimeManager
from app.config import body_image_validation as config
from tests.test_body_image_remote import SUCCESS, jpeg
from tests.test_body_image_warmup_api import (  # noqa: F401 (pytest fixture)
    KEY,
    PROXY_SECRET,
    remote,
    runtime,
)

WAIT = 5.0


class ModalHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.server.connections += 1

    def do_POST(self):
        length = int(self.headers["content-length"])
        self.rfile.read(length)
        modal = self.server
        with modal.lock:
            modal.started += 1
            modal.active += 1
            modal.max_active = max(modal.max_active, modal.active)
            modal.headers.append(dict(self.headers.items()))
            modal.lengths.append(length)
        modal.release.wait(WAIT * 4)  # AI Server 가 연결을 끊어도 GPU 작업은 멈추지 않는다
        with modal.lock:
            modal.active -= 1
            modal.finished += 1
        self.send_response(200)
        self.send_header("Content-Length", str(len(SUCCESS)))
        self.send_header("Set-Cookie", f"sid={modal.finished}; Path=/")
        self.end_headers()
        try:
            self.wfile.write(SUCCESS)
        except OSError:
            pass  # deadline 으로 AI Server 가 먼저 끊었다

    def log_message(self, *args):
        pass


class ModalStub(ThreadingHTTPServer):
    """Modal 자리의 loopback 서버. 실제 Modal 을 부르지 않는다."""

    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), ModalHandler)
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.started = self.finished = self.active = self.max_active = self.connections = 0
        self.headers = []
        self.lengths = []


def multipart(image):
    boundary = "bnd"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="b.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image + f"\r\n--{boundary}--\r\n".encode()
    head = (
        f"POST {ENDPOINT_PATH} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {KEY}\r\n"
        f"Content-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode()
    return head + body


class Backend:
    """Backend 의 요청 하나. 실제 TCP 로 보내고, 응답을 받거나 도중에 연결을 끊는다."""

    def __init__(self, port, payload):
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.sock.sendall(payload)
        self.raw = b""
        self.reader = None

    def disconnect(self):
        self.sock.close()

    def read(self):
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        return self

    def _read(self):
        self.sock.settimeout(WAIT * 2)
        try:
            while chunk := self.sock.recv(65536):
                self.raw += chunk
        except OSError:
            pass
        self.sock.close()

    def response(self):
        self.reader.join(WAIT * 2)
        assert not self.reader.is_alive(), "응답이 오지 않았다"
        head, _, body = self.raw.partition(b"\r\n\r\n")
        return int(head.split(b" ", 2)[1]), json.loads(body)


def wait_until(predicate):
    end = time.monotonic() + WAIT
    while not predicate():
        assert time.monotonic() < end, "상태가 바뀌지 않았다"
        time.sleep(0.002)


class Live:
    """실제 uvicorn 위의 main.app(remote 모드). 공유 client 는 lifespan 이 연다."""

    def __init__(self, monkeypatch, graceful=None):
        self.manager = BodyImageRuntimeManager()
        monkeypatch.setattr(main, "runtime_manager", self.manager)
        monkeypatch.setattr(validation_router, "runtime_manager", self.manager)
        self.modal = ModalStub()
        threading.Thread(target=self.modal.serve_forever, daemon=True).start()
        self.server = uvicorn.Server(
            uvicorn.Config(
                main.app, host="127.0.0.1", port=0, log_level="critical",
                timeout_graceful_shutdown=graceful,
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        wait_until(lambda: self.server.started)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        self.service = self.manager.get_service()
        self.client = self.service._shared_client
        self.service.endpoint = f"http://127.0.0.1:{self.modal.server_port}{ENDPOINT_PATH}"

    @property
    def limiter(self):
        return self.service.limiter

    @property
    def slots(self):
        return self.limiter.running, self.limiter.waiting

    def wait_for_queue(self, size):
        wait_until(lambda: self.limiter.waiting == size)

    def send(self, image=None):
        return Backend(self.port, multipart(jpeg() if image is None else image))

    def stop(self):
        self.modal.release.set()
        self.server.should_exit = True
        self.thread.join(WAIT * 2)
        self.modal.shutdown()
        self.modal.server_close()


@pytest.fixture
def live(remote, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "MAX_CONCURRENT_BODY_VALIDATIONS", 1)
    monkeypatch.setattr(config, "MAX_WAITING_BODY_VALIDATIONS", 1)
    monkeypatch.setattr(config, "MAX_QUEUE_WAIT_SECONDS", 10.0)
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 30.0)
    monkeypatch.setattr(config, "REMOTE_TIMEOUT_SECONDS", 30.0)
    started = []

    def start(**kwargs):
        started.append(Live(monkeypatch, **kwargs))
        return started[-1]

    yield start
    for server in started:
        server.stop()


# --- 실제 HTTP API ---


def test_repeated_api_calls_reuse_one_tcp_connection_to_modal(live):
    server = live()
    server.modal.release.set()

    responses = [server.send().read().response() for _ in range(5)]

    assert [status for status, _ in responses] == [200] * 5
    assert responses[-1][1]["data"] == {"s3_key": "body-images/abc.png"}
    assert (server.modal.started, server.modal.connections) == (5, 1)


def test_remote_api_runs_avalidate_without_a_new_event_loop_or_client(live, monkeypatch):
    server = live()
    server.modal.release.set()
    calls = []
    real_avalidate = RemoteBodyImageValidationService.avalidate

    async def recording_avalidate(self, body):
        calls.append(asyncio.get_running_loop())
        return await real_avalidate(self, body)

    def must_not_run(*args, **kwargs):
        raise AssertionError("remote 경로에서 asyncio.run 을 쓰면 안 된다")

    monkeypatch.setattr(RemoteBodyImageValidationService, "avalidate", recording_avalidate)
    monkeypatch.setattr(asyncio, "run", must_not_run)
    monkeypatch.setattr(RemoteBodyImageValidationService, "_build_client", must_not_run)

    statuses = [server.send().read().response()[0] for _ in range(2)]

    assert statuses == [200, 200]
    assert len(calls) == 2 and calls[0] is calls[1]
    assert server.service._shared_client is server.client


def test_auth_headers_stay_and_cookies_do_not_cross_requests(live):
    server = live()
    server.modal.release.set()

    for _ in range(3):
        server.send().read().response()

    assert server.modal.connections == 1
    for headers in server.modal.headers:
        names = {name.lower(): value for name, value in headers.items()}
        assert names["modal-key"] == "wk-test"
        assert names["modal-secret"] == PROXY_SECRET
        assert names["authorization"] == "Bearer worker-key"
        assert "x-body-image-deadline" in names
        assert "cookie" not in names  # 앞 요청이 받은 Set-Cookie 가 실리지 않는다


def test_one_runs_one_waits_and_the_third_is_429(live):
    server = live()
    first = server.send().read()
    wait_until(lambda: server.modal.started == 1)
    second = server.send().read()
    wait_until(lambda: server.limiter.waiting == 1)

    status, body = server.send().read().response()

    assert status == 429
    assert body["code"] == "SERVER_BUSY"
    assert server.modal.started == 1
    server.modal.release.set()
    assert [first.response()[0], second.response()[0]] == [200, 200]
    assert server.modal.max_active == 1


def test_waiting_requests_reach_modal_in_arrival_order(live, monkeypatch):
    monkeypatch.setattr(config, "MAX_WAITING_BODY_VALIDATIONS", 2)
    server = live()
    images = [jpeg((600, 800)), jpeg((640, 900)), jpeg((700, 1000))]
    requests = [server.send(images[0]).read()]
    wait_until(lambda: server.modal.started == 1)
    for queued, image in enumerate(images[1:], start=1):
        requests.append(server.send(image).read())
        server.wait_for_queue(queued)

    server.modal.release.set()

    assert [request.response()[0] for request in requests] == [200, 200, 200]
    assert server.modal.lengths == sorted(server.modal.lengths)  # 이미지가 뒤로 갈수록 크다
    assert server.modal.max_active == 1


def test_event_loop_answers_while_a_request_waits_for_modal_and_the_queue(live):
    server = live()
    first = server.send().read()
    wait_until(lambda: server.modal.started == 1)
    second = server.send().read()
    wait_until(lambda: server.limiter.waiting == 1)

    health = socket.create_connection(("127.0.0.1", server.port), timeout=WAIT)
    health.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
    answer = health.recv(4096)
    health.close()

    assert answer.startswith(b"HTTP/1.1 200")
    assert server.slots == (1, 1)  # 대기가 끝난 뒤가 아니라 기다리는 동안 응답했다
    server.modal.release.set()
    first.response(), second.response()


def test_stuck_input_check_ends_at_the_deadline_without_calling_modal(live, monkeypatch):
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 0.2)
    monkeypatch.setattr(config, "REMOTE_TIMEOUT_SECONDS", 0.2)
    server = live()
    release = threading.Event()
    real_open = remote_module.open_image

    def stuck_open(body):
        release.wait(WAIT)
        return real_open(body)

    monkeypatch.setattr(remote_module, "open_image", stuck_open)
    try:
        status, body = server.send().read().response()
    finally:
        release.set()

    assert status == 504
    assert body["code"] == "BODY_IMAGE_WORKER_TIMEOUT"
    assert server.modal.started == 0


# --- Backend 연결 종료: 호출은 끝까지 가고 슬롯은 Modal 작업이 끝날 때 돌아온다(A 방식과 같다) ---


def test_disconnect_after_modal_started_keeps_the_slot_until_modal_finishes(live):
    server = live()
    first = server.send()
    wait_until(lambda: server.modal.started == 1)

    first.disconnect()
    time.sleep(0.2)

    assert server.slots == (1, 0)  # 끊겨도 취소되지 않고 슬롯을 잡고 있다
    assert server.modal.active == 1
    second = server.send().read()
    wait_until(lambda: server.limiter.waiting == 1)
    assert server.modal.started == 1  # 다음 요청은 앞 작업이 끝날 때까지 기다린다
    server.modal.release.set()
    assert second.response()[0] == 200
    assert server.modal.max_active == 1  # Modal 에서 겹치지 않았다
    assert server.modal.finished == 2


def test_disconnect_while_queued_still_calls_modal_after_the_running_request(live):
    server = live()
    first = server.send().read()
    wait_until(lambda: server.modal.started == 1)
    queued = server.send()
    wait_until(lambda: server.limiter.waiting == 1)

    queued.disconnect()
    time.sleep(0.2)

    assert server.slots == (1, 1)  # 대기열을 떠나지 않는다
    server.modal.release.set()
    assert first.response()[0] == 200
    wait_until(lambda: server.modal.finished == 2)
    wait_until(lambda: server.slots == (0, 0))
    assert server.modal.max_active == 1


def test_disconnect_before_the_post_still_calls_modal(live, monkeypatch):
    server = live()
    entered, proceed = threading.Event(), threading.Event()
    real_open = remote_module.open_image

    def slow_open(body):
        entered.set()
        proceed.wait(WAIT)
        return real_open(body)

    monkeypatch.setattr(remote_module, "open_image", slow_open)
    request = server.send()
    assert entered.wait(WAIT)

    request.disconnect()
    time.sleep(0.2)
    assert server.modal.started == 0
    proceed.set()
    server.modal.release.set()

    wait_until(lambda: server.modal.finished == 1)
    wait_until(lambda: server.slots == (0, 0))


# --- deadline: 슬롯은 돌아오지만 이미 시작한 Modal 작업은 멈추지 않는다(A 방식에도 있던 위험) ---


def test_after_a_deadline_the_next_request_can_overlap_the_running_modal_job(live, monkeypatch):
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 0.3)
    monkeypatch.setattr(config, "REMOTE_TIMEOUT_SECONDS", 0.3)
    server = live()

    status, body = server.send().read().response()

    assert status == 504
    assert body["code"] == "BODY_IMAGE_WORKER_TIMEOUT"
    assert server.slots == (0, 0)  # AI Server 의 슬롯은 돌아왔다
    assert (server.modal.active, server.modal.finished) == (1, 0)  # Modal 작업은 계속 돈다
    server.send()
    wait_until(lambda: server.modal.started == 2)
    assert server.modal.max_active == 2  # 슬롯은 1건이어도 Modal 에서는 2건이 겹친다


# --- 서버 종료 ---


def test_graceful_shutdown_finishes_the_running_request_then_closes_the_client(live):
    server = live()
    first = server.send().read()
    wait_until(lambda: server.modal.started == 1)

    server.server.should_exit = True
    time.sleep(0.3)

    assert server.thread.is_alive()  # 진행 중인 요청을 기다린다
    assert not server.client.is_closed
    server.modal.release.set()
    assert first.response()[0] == 200
    server.thread.join(WAIT)
    assert not server.thread.is_alive()
    assert server.client.is_closed
    assert server.manager.runtime is None


def test_forced_shutdown_cancels_the_call_returns_the_slot_and_leaves_modal_running(live):
    server = live(graceful=1)
    server.send().read()
    wait_until(lambda: server.modal.started == 1)
    limiter = server.limiter

    server.server.should_exit = True
    server.thread.join(WAIT)

    assert not server.thread.is_alive()
    assert (limiter.running, limiter.waiting) == (0, 0)
    assert server.client.is_closed
    assert (server.modal.active, server.modal.finished) == (1, 0)  # 취소해도 Modal 작업은 멈추지 않는다
