"""AI Server(remote 모드) → Modal worker HTTP 연동. 실제 Modal 은 부르지 않고 httpx MockTransport 를 쓴다."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from email import message_from_bytes
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.body_image_validation import router, runtime, worker
from app.body_image_validation.concurrency import BodyValidationLimiter
from app.body_image_validation.remote import (
    RemoteBodyImageValidationService,
    RemoteWorkerConfigError,
)
from app.body_image_validation.runtime import BodyImageRuntimeError, BodyImageRuntimeManager
from app.config import body_image_validation as config
from app.config import settings

URL = "/api/v1/validation/body-image"
KEY = "ai-server-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
WORKER = "https://example--lookddak-body-image-staging.modal.run"
ROOT = Path(__file__).resolve().parents[1]


def jpeg(size=(600, 800)):
    output = BytesIO()
    Image.new("RGB", size, (180, 170, 160)).save(output, format="JPEG")
    return output.getvalue()


def common(code, message="msg", data=None):
    return json.dumps({"code": code, "data": data, "message": message}).encode()


SUCCESS = common(
    "BODY_IMAGE_UPLOAD_SUCCESS", "전신 사진 검증에 성공했습니다.", {"s3_key": "body-images/abc.png"}
)


class FakeWorker:
    """Modal worker 자리의 httpx MockTransport. 받은 요청을 기록하고 정해 둔 응답이나 예외를 차례로 낸다.

    steps 의 각 항목은 (지연 초, status, headers, body) 또는 예외다. 요청 수가 steps 보다 많으면 마지막 항목을
    반복한다. 지연은 asyncio.sleep 이라 deadline(asyncio.timeout)이 실제로 취소할 수 있다.
    """

    def __init__(self, status=200, body=None, error=None, steps=None):
        if steps is None:
            steps = [error if error is not None else (0, status, {}, SUCCESS if body is None else body)]
        self.steps = steps
        self.calls = []
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request):
        await request.aread()
        self.calls.append(request)
        step = self.steps[min(len(self.calls), len(self.steps)) - 1]
        if isinstance(step, BaseException):
            raise step
        delay, status, headers, body = step
        if delay:
            await asyncio.sleep(delay)
        return httpx.Response(status, headers=headers, content=body)


def remote_service(fake, limiter=None, timeout=30.0, deadline=60.0):
    return RemoteBodyImageValidationService(
        WORKER,
        proxy_key="wk-test",
        proxy_secret="ws-test",
        api_key="worker-key",
        timeout=timeout,
        total_deadline=deadline,
        limiter=limiter or BodyValidationLimiter(1, 1),
        transport=fake.transport,
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", KEY)
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)
    with TestClient(worker.create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def post(client, body):
    return client.post(URL, files={"image": ("body.jpg", body, "image/jpeg")}, headers=AUTH)


def use(monkeypatch, service):
    monkeypatch.setattr(router, "get_service", lambda: service)


# --- 요청 구성과 인증 헤더 ---


def test_request_carries_both_auth_layers_and_the_original_image(client, monkeypatch):
    fake = FakeWorker()
    use(monkeypatch, remote_service(fake, timeout=42.0))
    body = jpeg()

    post(client, body)

    [request] = fake.calls
    assert str(request.url) == WORKER + URL
    assert request.method == "POST"
    assert request.headers["modal-key"] == "wk-test"
    assert request.headers["modal-secret"] == "ws-test"
    assert request.headers["authorization"] == "Bearer worker-key"  # Backend 의 키가 아니라 worker 용 키
    timeouts = request.extensions["timeout"]
    assert 41.5 < timeouts["read"] <= 42.0  # 네트워크 timeout(남은 deadline 60초보다 작음)
    message = message_from_bytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    [part] = message.get_payload()
    assert part.get_param("name", header="content-disposition") == "image"
    assert part.get_content_type() == "image/jpeg"
    assert part.get_payload(decode=True) == body


def test_trailing_slash_in_the_base_url_is_ignored():
    service = RemoteBodyImageValidationService(
        WORKER + "/", proxy_key="k", proxy_secret="s", api_key="a", timeout=1, total_deadline=1,
        limiter=BodyValidationLimiter(1, 1),
    )
    assert service.endpoint == WORKER + URL


# --- 정상·검증 실패 응답 전달 ---


def test_success_keeps_the_existing_contract(client, monkeypatch):
    use(monkeypatch, remote_service(FakeWorker()))

    response = post(client, jpeg())

    assert response.status_code == 200
    assert response.json() == {
        "code": "BODY_IMAGE_UPLOAD_SUCCESS",
        "data": {"s3_key": "body-images/abc.png"},
        "message": "전신 사진 검증에 성공했습니다.",
    }


@pytest.mark.parametrize(
    ("status", "code", "message"),
    [
        (422, "FULL_BODY_NOT_VISIBLE", "머리부터 발끝까지 모두 나오도록 전신을 촬영해주세요."),
        (422, "IMAGE_TOO_DARK", "사진이 너무 어둡습니다. 밝은 곳에서 다시 촬영해주세요."),
        (400, "IMAGE_DECODE_FAILED", "이미지가 손상되어 처리할 수 없습니다. 다른 사진을 업로드해주세요."),
        (413, "IMAGE_TOO_LARGE", "이미지 크기는 2MB 이하여야 합니다."),
        (429, "SERVER_BUSY", "현재 이미지 처리 요청이 많습니다. 잠시 후 다시 시도해주세요."),
        (500, "IMAGE_UPLOAD_FAILED", "전신 사진 처리 중 서버 오류가 발생했습니다."),
        (500, "S3_CONFIG_ERROR", "전신 사진 처리 중 서버 오류가 발생했습니다."),
    ],
)
def test_worker_errors_in_the_common_format_are_relayed(client, monkeypatch, status, code, message):
    use(monkeypatch, remote_service(FakeWorker(status=status, body=common(code, message))))

    response = post(client, jpeg())

    assert response.status_code == status
    assert response.json() == {"code": code, "data": None, "message": message}


# --- 연결 실패·시간 초과·5xx·설정 문제 ---

UNAVAILABLE = {"code": "BODY_IMAGE_WORKER_UNAVAILABLE", "data": None,
               "message": "전신 사진 처리 중 서버 오류가 발생했습니다."}
TIMEOUT = {"code": "BODY_IMAGE_WORKER_TIMEOUT", "data": None,
           "message": "전신 사진 처리 중 서버 오류가 발생했습니다."}


@pytest.mark.parametrize(
    ("fake", "status", "expected"),
    [
        # 시간 초과 → 504
        (FakeWorker(error=httpx.ReadTimeout("timed out")), 504, TIMEOUT),
        (FakeWorker(error=httpx.ConnectTimeout("timed out")), 504, TIMEOUT),
        (FakeWorker(status=504, body=b"gateway timeout"), 504, TIMEOUT),
        # worker 에 닿지 못함 → 503
        (FakeWorker(error=httpx.ConnectError("refused")), 503, UNAVAILABLE),
        (FakeWorker(error=httpx.ConnectError("name resolution failed")), 503, UNAVAILABLE),
        (FakeWorker(error=httpx.RemoteProtocolError("server disconnected")), 503, UNAVAILABLE),
        (FakeWorker(status=503, body=common("SERVICE_UNAVAILABLE")), 503, UNAVAILABLE),
        (FakeWorker(status=503, body=b"<html>unavailable</html>"), 503, UNAVAILABLE),
        # 정상 worker 응답이 아님 → 502
        (FakeWorker(status=502, body=b"<html>bad gateway</html>"), 502, UNAVAILABLE),
        (FakeWorker(status=401, body=b"modal-http: missing credentials for proxy authorization\n"), 502, UNAVAILABLE),
        (FakeWorker(status=401, body=common("UNAUTHORIZED", "인증에 실패했습니다.")), 502, UNAVAILABLE),
        (FakeWorker(status=403, body=common("FORBIDDEN")), 502, UNAVAILABLE),
        (FakeWorker(status=404, body=common("NOT_FOUND")), 502, UNAVAILABLE),
        (FakeWorker(status=405, body=common("METHOD_NOT_ALLOWED")), 502, UNAVAILABLE),
        (FakeWorker(status=400, body=common("INVALID_REQUEST")), 502, UNAVAILABLE),
        (FakeWorker(status=422, body=b"not json"), 502, UNAVAILABLE),
        (FakeWorker(status=200, body=b"not json"), 502, UNAVAILABLE),
        (FakeWorker(status=200, body=json.dumps({"data": {"s3_key": "k"}}).encode()), 502, UNAVAILABLE),
        (FakeWorker(status=200, body=common("BODY_IMAGE_UPLOAD_SUCCESS", data={})), 502, UNAVAILABLE),
        (FakeWorker(status=200, body=common("OTHER_CODE", data={"s3_key": "k"})), 502, UNAVAILABLE),
    ],
    ids=["timeout", "connect-timeout", "upstream-504", "refused", "dns", "reset", "upstream-503-json",
         "upstream-503-html", "upstream-502", "proxy-401", "worker-401", "403", "404", "405",
         "invalid-request", "422-not-json", "200-not-json", "200-not-common-format", "200-without-key",
         "200-other-code"],
)
def test_worker_failures_map_to_gateway_statuses(client, monkeypatch, fake, status, expected):
    use(monkeypatch, remote_service(fake))

    response = post(client, jpeg())

    assert response.status_code == status
    assert response.json() == expected
    assert len(fake.calls) == 1  # 재시도하지 않는다(같은 사진을 다시 처리·업로드할 수 있어서)


@pytest.mark.parametrize(
    ("fake", "layer"),
    [
        (FakeWorker(status=401, body=b"modal-http: missing credentials for proxy authorization\n"),
         "modal proxy"),
        (FakeWorker(status=401, body=common("UNAUTHORIZED", "인증에 실패했습니다.")), "worker bearer"),
    ],
)
def test_auth_failures_name_the_layer_without_logging_secrets(client, monkeypatch, caplog, fake, layer):
    use(monkeypatch, remote_service(fake))

    with caplog.at_level(logging.ERROR, logger="app.body_image_validation.remote"):
        post(client, jpeg())

    assert f"layer={layer}" in caplog.text
    for secret in ("wk-test", "ws-test", "worker-key", KEY):
        assert secret not in caplog.text


# --- 요청 전 검사와 동시 실행 제한 ---


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (b"", 400, "IMAGE_EMPTY"),
        (b"not-an-image", 400, "INVALID_IMAGE"),
        (jpeg((300, 400)), 422, "IMAGE_RESOLUTION_TOO_SMALL"),
    ],
)
def test_invalid_uploads_are_rejected_without_waking_the_worker(client, monkeypatch, body, status, code):
    fake = FakeWorker()
    use(monkeypatch, remote_service(fake))

    response = post(client, body)

    assert (response.status_code, response.json()["code"]) == (status, code)
    assert fake.calls == []


def test_full_limiter_answers_429_without_calling_the_worker(client, monkeypatch):
    limiter = BodyValidationLimiter(1, 0)
    fake = FakeWorker()
    use(monkeypatch, remote_service(fake, limiter=limiter))

    with limiter.slot():  # 다른 요청이 worker 응답을 기다리는 중
        response = post(client, jpeg())

    assert response.status_code == 429
    assert response.json()["code"] == "SERVER_BUSY"
    assert fake.calls == []
    assert (limiter.running, limiter.waiting) == (0, 0)


def test_limiter_is_released_after_a_worker_failure(client, monkeypatch):
    limiter = BodyValidationLimiter(1, 0)
    use(monkeypatch, remote_service(FakeWorker(error=httpx.ReadTimeout("t")), limiter=limiter))

    response = post(client, jpeg())

    assert response.status_code == 504
    assert (limiter.running, limiter.waiting) == (0, 0)
    assert post(client, jpeg()).status_code == 504  # 슬롯이 돌아와 다음 요청도 worker 까지 간다(429 아님)


# --- 설정 ---


@pytest.mark.parametrize(
    "overrides",
    [{"base_url": ""}, {"proxy_key": ""}, {"proxy_secret": ""}, {"api_key": ""},
     {"base_url": "http://example.modal.run"}, {"timeout": 0}, {"total_deadline": 0}],
)
def test_incomplete_or_unsafe_settings_are_rejected(overrides):
    options = {"base_url": WORKER, "proxy_key": "k", "proxy_secret": "s", "api_key": "a", "timeout": 5,
               "total_deadline": 5}
    options.update(overrides)
    base_url = options.pop("base_url")

    with pytest.raises(RemoteWorkerConfigError):
        RemoteBodyImageValidationService(base_url, limiter=BodyValidationLimiter(1, 1), **options)


@pytest.fixture
def remote_mode(monkeypatch):
    """모델·S3·GPU 를 만드는 생성자를 호출되면 실패하게 바꾸고 remote 모드로 둔다."""

    def must_not_build(*args, **kwargs):
        raise AssertionError("remote 모드에서는 로컬 모델·S3 를 만들지 않는다")

    for name in ("S3ImageStorage", "MediaPipePersonDetector", "MediaPipePoseDetector",
                 "RembgBackgroundRemover", "GpuJpegDecoder", "get_s3_bucket"):
        monkeypatch.setattr(runtime, name, must_not_build)
    monkeypatch.setattr(config, "VALIDATION_MODE", "remote")
    monkeypatch.setattr(config, "REMOTE_URL", WORKER)
    monkeypatch.setattr(config, "REMOTE_TIMEOUT_SECONDS", 30.0)
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 60.0)
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_KEY", "wk-test")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_PROXY_SECRET", "ws-test")
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_API_KEY", "worker-key")
    monkeypatch.setattr(settings, "S3_BUCKET", "")  # remote 에서는 S3 설정이 없어도 된다


def test_remote_mode_builds_only_the_http_client(remote_mode):
    manager = BodyImageRuntimeManager()

    manager.start()

    assert isinstance(manager.get_service(), RemoteBodyImageValidationService)
    assert manager.get_service().endpoint == WORKER + URL


def test_remote_mode_with_missing_settings_is_runtime_unavailable(remote_mode, monkeypatch, client):
    monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_API_KEY", "")
    manager = BodyImageRuntimeManager()
    monkeypatch.setattr(router, "runtime_manager", manager)

    response = post(client, jpeg())

    assert response.status_code == 500
    assert response.json()["code"] == "BODY_IMAGE_RUNTIME_UNAVAILABLE"


def test_unknown_mode_is_runtime_unavailable(monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_MODE", "remot")
    manager = BodyImageRuntimeManager()

    with pytest.raises(BodyImageRuntimeError):
        manager.get_service()


def test_remote_runtime_rejects_prebuilt_models(remote_mode):
    manager = BodyImageRuntimeManager()

    manager.start(person_detector=object(), pose_detector=object())

    assert manager.runtime is None


def test_remote_mode_imports_no_model_libraries():
    # 새 프로세스에서 AI Server 앱과 remote runtime 을 띄운 뒤, 모델 라이브러리가 import 되지 않았는지 본다.
    code = (
        "import sys, json\n"
        "from app.body_image_validation.runtime import runtime_manager\n"
        "import app.main\n"
        "runtime_manager.start()\n"
        "svc = runtime_manager.get_service()\n"
        "print(json.dumps({'service': type(svc).__name__,"
        " 'loaded': sorted(m for m in ('mediapipe', 'rembg', 'onnxruntime', 'torch', 'cv2')"
        " if m in sys.modules)}))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT), "BODY_IMAGE_VALIDATION_MODE": "remote",
           "BODY_IMAGE_REMOTE_URL": WORKER, "BODY_IMAGE_REMOTE_PROXY_KEY": "k",
           "BODY_IMAGE_REMOTE_PROXY_SECRET": "s", "BODY_IMAGE_REMOTE_API_KEY": "a"}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                            check=False, timeout=120)

    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "service": "RemoteBodyImageValidationService", "loaded": [],
    }


# --- Modal worker 는 local 만 ---


def test_worker_refuses_to_start_in_remote_mode(remote_mode):
    with pytest.raises(worker.WorkerConfigError):
        worker.start_gpu(object(), object())


def test_default_mode_is_local(monkeypatch):
    import importlib

    monkeypatch.delenv("BODY_IMAGE_VALIDATION_MODE", raising=False)
    try:
        assert importlib.reload(config).VALIDATION_MODE == "local"
    finally:
        monkeypatch.undo()
        importlib.reload(config)



# --- Modal 303 redirect ---

POLL = "/api/v1/validation/body-image?__modal_function_call_id=fc-test123"


def redirect(location, delay=0.0, status=303):
    return (delay, status, {"location": location}, b"")


def test_303_is_followed_with_get_and_returns_the_final_result(client, monkeypatch):
    fake = FakeWorker(steps=[redirect(POLL), (0, 200, {}, SUCCESS)])
    use(monkeypatch, remote_service(fake))

    response = post(client, jpeg())

    assert response.status_code == 200
    assert response.json()["data"] == {"s3_key": "body-images/abc.png"}
    first, second = fake.calls
    assert first.method == "POST" and second.method == "GET"
    assert str(second.url) == WORKER + POLL  # Modal 이 준 결과 조회 query 를 그대로 쓴다
    assert second.content == b""  # 이미지를 다시 보내지 않는다
    assert "content-type" not in second.headers and second.headers.get("content-length") in (None, "0")
    # 같은 origin 이므로 Modal proxy·worker 인증은 유지한다
    assert second.headers["modal-key"] == "wk-test"
    assert second.headers["authorization"] == "Bearer worker-key"


def test_absolute_same_origin_location_is_followed(client, monkeypatch):
    fake = FakeWorker(steps=[redirect(WORKER + POLL), (0, 200, {}, SUCCESS)])
    use(monkeypatch, remote_service(fake))

    assert post(client, jpeg()).status_code == 200
    assert len(fake.calls) == 2


def test_repeated_303_to_the_same_poll_url_is_modal_polling(client, monkeypatch):
    fake = FakeWorker(steps=[redirect(POLL), redirect(POLL), (0, 200, {}, SUCCESS)])
    use(monkeypatch, remote_service(fake))

    assert post(client, jpeg()).status_code == 200
    assert [c.method for c in fake.calls] == ["POST", "GET", "GET"]


@pytest.mark.parametrize(
    "location",
    [
        "https://evil.example.com/api/v1/validation/body-image?__modal_function_call_id=fc-test123",
        "http://example--lookddak-body-image-staging.modal.run" + POLL,  # https → http
        "https://example--lookddak-body-image-staging.modal.run:8443" + POLL,  # 다른 port
        "//evil.example.com/steal",
    ],
    ids=["other-host", "downgrade-to-http", "other-port", "scheme-relative"],
)
def test_redirect_to_another_origin_is_blocked_before_sending(client, monkeypatch, caplog, location):
    fake = FakeWorker(steps=[redirect(location), (0, 200, {}, SUCCESS)])
    use(monkeypatch, remote_service(fake))

    with caplog.at_level(logging.DEBUG):
        response = post(client, jpeg())

    assert response.status_code == 502
    assert response.json()["code"] == "BODY_IMAGE_WORKER_UNAVAILABLE"
    assert len(fake.calls) == 1  # 다른 origin 으로는 요청 자체를 보내지 않는다(인증 헤더 유출 없음)
    assert "different origin" in caplog.text
    for secret in ("wk-test", "ws-test", "worker-key", KEY, "evil.example.com", "fc-test123"):
        assert secret not in caplog.text


@pytest.mark.parametrize("status", [301, 302, 307, 308, 300, 304])
def test_other_redirect_statuses_are_not_followed(client, monkeypatch, status):
    fake = FakeWorker(steps=[redirect(POLL, status=status), (0, 200, {}, SUCCESS)])
    use(monkeypatch, remote_service(fake))

    response = post(client, jpeg())

    assert response.status_code == 502
    assert len(fake.calls) == 1  # 307/308 로 이미지를 다시 보내지 않는다


def test_303_without_location_is_bad_gateway(client, monkeypatch):
    fake = FakeWorker(steps=[(0, 303, {}, b"")])
    use(monkeypatch, remote_service(fake))

    assert post(client, jpeg()).status_code == 502
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "steps",
    [
        [redirect(POLL), redirect("/api/v1/validation/body-image?other=1"), redirect(POLL)],
        [redirect("/api/v1/validation/body-image")],  # query 없는 원래 endpoint 로 돌아감
    ],
    ids=["cycle", "back-to-endpoint"],
)
def test_redirect_loops_are_blocked(client, monkeypatch, steps):
    fake = FakeWorker(steps=steps + [(0, 200, {}, SUCCESS)])
    use(monkeypatch, remote_service(fake))

    assert post(client, jpeg()).status_code == 502


def test_too_many_redirects_are_bad_gateway(client, monkeypatch):
    from app.body_image_validation.remote import MAX_REDIRECTS

    fake = FakeWorker(steps=[redirect(POLL)])  # 계속 303
    use(monkeypatch, remote_service(fake))

    response = post(client, jpeg())

    assert response.status_code == 502
    assert len(fake.calls) == MAX_REDIRECTS + 1  # POST 1 + GET MAX_REDIRECTS


def test_relayed_user_error_after_redirect_keeps_its_status(client, monkeypatch):
    fake = FakeWorker(steps=[redirect(POLL), (0, 422, {}, common("FULL_BODY_NOT_VISIBLE", "m"))])
    use(monkeypatch, remote_service(fake))

    response = post(client, jpeg())

    assert response.status_code == 422
    assert response.json()["code"] == "FULL_BODY_NOT_VISIBLE"


# --- 전체 deadline (Modal 의 150초 303 을 줄인 시간으로 재현) ---

SCALE = 0.005  # 1초 → 5ms. asyncio.sleep 과 asyncio.timeout 을 그대로 써서 실제 취소 경로를 검사한다.


def modal_slow_steps():
    """0초 POST → 150초 303 → 결과 조회 GET 이 20초 뒤(누적 170초) 200."""
    return [redirect(POLL, delay=150 * SCALE), (20 * SCALE, 200, {}, SUCCESS)]


def test_scenario_a_deadline_200_waits_through_the_303_and_returns_the_key(client, monkeypatch):
    fake = FakeWorker(steps=modal_slow_steps())
    use(monkeypatch, remote_service(fake, timeout=180 * SCALE, deadline=200 * SCALE))

    started = time.monotonic()
    response = post(client, jpeg())
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert response.json()["data"]["s3_key"] == "body-images/abc.png"
    assert [c.method for c in fake.calls] == ["POST", "GET"]
    assert elapsed < 200 * SCALE + 0.5


def test_scenario_b_deadline_90_ends_at_90_with_504(client, monkeypatch):
    fake = FakeWorker(steps=modal_slow_steps())
    limiter = BodyValidationLimiter(1, 0)
    use(monkeypatch, remote_service(fake, limiter=limiter, timeout=90 * SCALE, deadline=90 * SCALE))

    started = time.monotonic()
    response = post(client, jpeg())
    elapsed = time.monotonic() - started

    assert response.status_code == 504
    assert response.json() == TIMEOUT
    assert len(fake.calls) == 1  # 303 을 받기 전에 끝났다
    assert elapsed < 150 * SCALE  # 150초(303)까지 기다리지 않고 deadline 에서 끊었다
    assert (limiter.running, limiter.waiting) == (0, 0)  # deadline 뒤에도 슬롯을 돌려준다


def test_deadline_does_not_restart_after_a_redirect(client, monkeypatch):
    # 각 구간(120초)은 deadline(200초)보다 짧지만 합(240초)은 넘는다 → 504
    fake = FakeWorker(steps=[redirect(POLL, delay=120 * SCALE), (120 * SCALE, 200, {}, SUCCESS)])
    use(monkeypatch, remote_service(fake, timeout=180 * SCALE, deadline=200 * SCALE))

    response = post(client, jpeg())

    assert response.status_code == 504
    assert [c.method for c in fake.calls] == ["POST", "GET"]


def test_network_timeout_after_redirect_is_504(client, monkeypatch):
    fake = FakeWorker(steps=[redirect(POLL), httpx.ReadTimeout("timed out")])
    use(monkeypatch, remote_service(fake))

    assert post(client, jpeg()).json() == TIMEOUT


def test_per_request_network_timeout_never_exceeds_the_remaining_deadline(client, monkeypatch):
    fake = FakeWorker(steps=[redirect(POLL, delay=150 * SCALE), (0, 200, {}, SUCCESS)])
    use(monkeypatch, remote_service(fake, timeout=180 * SCALE, deadline=200 * SCALE))

    post(client, jpeg())

    post_timeout = fake.calls[0].extensions["timeout"]["read"]
    get_timeout = fake.calls[1].extensions["timeout"]["read"]
    assert post_timeout == pytest.approx(180 * SCALE, abs=0.05)  # 네트워크 timeout
    assert get_timeout <= (200 - 150) * SCALE + 0.05  # 남은 deadline 으로 줄었다


def test_call_summary_is_logged_without_secrets_or_urls(client, monkeypatch, caplog):
    fake = FakeWorker(steps=[redirect(POLL), (0, 200, {}, SUCCESS)])
    use(monkeypatch, remote_service(fake))

    with caplog.at_level(logging.INFO, logger="app.body_image_validation.remote"):
        post(client, jpeg())

    [summary] = [r.getMessage() for r in caplog.records if "body image worker call" in r.getMessage()]
    assert "outcome=response" in summary and "status=200" in summary and "redirects=1" in summary
    assert "elapsed=" in summary
    for secret in ("wk-test", "ws-test", "worker-key", KEY, "fc-test123", "__modal_function_call_id", WORKER):
        assert secret not in caplog.text


@pytest.mark.parametrize(
    ("fake", "outcome"),
    [
        (FakeWorker(steps=[(1.0, 200, {}, SUCCESS)]), "deadline_exceeded"),
        (FakeWorker(error=httpx.ReadTimeout("t")), "network_timeout:ReadTimeout"),
        (FakeWorker(error=httpx.ConnectError("refused")), "connection_failed:ConnectError"),
    ],
)
def test_timeout_kinds_are_logged(client, monkeypatch, caplog, fake, outcome):
    use(monkeypatch, remote_service(fake, timeout=0.1, deadline=0.1))

    with caplog.at_level(logging.INFO, logger="app.body_image_validation.remote"):
        post(client, jpeg())

    assert f"outcome={outcome}" in caplog.text


def test_misconfigured_network_timeout_below_modal_redirect_is_warned(caplog):
    with caplog.at_level(logging.WARNING, logger="app.body_image_validation.remote"):
        remote_service(FakeWorker(), timeout=90, deadline=200)

    assert "before Modal's 303 redirect" in caplog.text
