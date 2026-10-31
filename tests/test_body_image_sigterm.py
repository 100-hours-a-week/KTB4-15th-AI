import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.test_body_image_c3_http import KEY, Backend, ModalStub, multipart, wait_until
from tests.test_body_image_remote import jpeg

ROOT = Path(__file__).resolve().parents[1]
STOP_WITHIN = 15.0
# 실제 uvicorn 프로세스가 띄우는 앱. 운영 앱에 Modal 주소만 loopback 으로 바꾸고 외부 연결을 막는다.
PROBE_APP = """
import ipaddress
import os
import socket

_connect = socket.socket.connect


def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        host = address[0]
        if host != "localhost" and not ipaddress.ip_address(host.split("%")[0]).is_loopback:
            raise ConnectionRefusedError("외부 연결 차단")
    return _connect(self, address)


socket.socket.connect = connect

from app.body_image_validation import remote

_init = remote.RemoteBodyImageValidationService.__init__


def init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    self.endpoint = os.environ["PROBE_MODAL_URL"]


remote.RemoteBodyImageValidationService.__init__ = init

from app.main import app  # noqa: E402,F401
"""


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Process:
    """Dockerfile CMD 와 같은 `uvicorn <app> --host --port` 프로세스. signal 은 uvicorn 의 기본 처리를 그대로 쓴다."""

    def __init__(self, tmp_path, *options):
        (tmp_path / "probe_app.py").write_text(PROBE_APP)
        self.modal = ModalStub()
        threading.Thread(target=self.modal.serve_forever, daemon=True).start()
        self.port = free_port()
        self.log_path = tmp_path / "server.log"
        env = {
            **os.environ,
            "PYTHONPATH": f"{ROOT}{os.pathsep}{tmp_path}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "BODY_IMAGE_VALIDATION_MODE": "remote",
            "BODY_IMAGE_REMOTE_URL": "https://example--probe.modal.run",
            "BODY_IMAGE_REMOTE_PROXY_KEY": "wk-test",
            "BODY_IMAGE_REMOTE_PROXY_SECRET": "ws-test",
            "BODY_IMAGE_REMOTE_API_KEY": "worker-key",
            "INTERNAL_API_KEY": KEY,
            "AUTH_DISABLED": "false",
            "CHECKPOINT_DSN": "",
            "S3_BUCKET": "",
            "RUNWARE_VTON_API_KEY": "test",
            "RUNWARE_LLM_API_KEY": "test",
            "PROBE_MODAL_URL": f"http://127.0.0.1:{self.modal.server_port}/api/v1/validation/body-image",
        }
        self.log = self.log_path.open("w")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "probe_app:app", "--host", "127.0.0.1",
             "--port", str(self.port), *options],
            cwd=tmp_path, env=env, stdout=self.log, stderr=subprocess.STDOUT,
        )
        self._wait_for_port()

    def _wait_for_port(self):
        end = time.monotonic() + 60.0
        while time.monotonic() < end:
            assert self.process.poll() is None, self.output()
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.05)
        raise AssertionError("서버가 뜨지 않았다")

    def output(self):
        self.log.flush()
        return self.log_path.read_text()

    def send(self):
        return Backend(self.port, multipart(jpeg()))

    def terminate(self):
        self.process.send_signal(signal.SIGTERM)

    def wait(self):
        self.process.wait(STOP_WITHIN)

    def accepts_connections(self):
        try:
            socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
            return True
        except OSError:
            return False

    def cleanup(self):
        self.modal.release.set()
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(5)
        self.log.close()
        self.modal.shutdown()
        self.modal.server_close()


@pytest.fixture
def uvicorn_process(tmp_path):
    started = []

    def start(*options):
        started.append(Process(tmp_path, *options))
        return started[-1]

    yield start
    for process in started:
        process.cleanup()


def test_sigterm_finishes_the_running_request_and_refuses_new_ones(uvicorn_process):
    server = uvicorn_process()
    running = server.send().read()
    wait_until(lambda: server.modal.started == 1)

    server.terminate()
    wait_until(lambda: not server.accepts_connections())  # 새 연결을 받지 않는다

    assert server.process.poll() is None  # 진행 중인 요청을 기다린다
    assert server.modal.finished == 0
    server.modal.release.set()
    status, body = running.response()
    server.wait()

    assert status == 200
    assert body["data"] == {"s3_key": "body-images/abc.png"}
    output = server.output()
    assert "outcome=response" in output
    assert "body image worker client closing: in_flight=0 left=0" in output  # 요청이 끝난 뒤 닫았다
    for secret in ("wk-test", "ws-test", "worker-key"):
        assert secret not in output


def test_sigterm_on_an_idle_server_closes_the_client_and_exits(uvicorn_process):
    server = uvicorn_process()
    server.modal.release.set()
    assert server.send().read().response()[0] == 200

    server.terminate()
    server.wait()

    assert "body image worker client closing: in_flight=0 left=0" in server.output()
    assert "body image runtime close failed" not in server.output()


def test_graceful_timeout_cancels_the_call_while_modal_keeps_running(uvicorn_process):
    server = uvicorn_process("--timeout-graceful-shutdown", "1")
    server.send().read()
    wait_until(lambda: server.modal.started == 1)

    started = time.monotonic()
    server.terminate()
    server.wait()
    elapsed = time.monotonic() - started

    assert elapsed < 5.0
    assert (server.modal.active, server.modal.finished) == (1, 0)  # 취소해도 Modal 작업은 멈추지 않는다
    output = server.output()
    assert "outcome=cancelled" in output
    assert "http_cancelled=True" in output
    assert "remote_stopped=unknown" in output
    # 취소 직후 lifespan 종료가 시작돼 호출 1건이 잡혔다가, client 를 닫기 전에 빠진다.
    assert "body image worker client closing: in_flight=1 left=0" in output
