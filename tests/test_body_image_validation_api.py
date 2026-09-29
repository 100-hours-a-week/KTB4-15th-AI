import threading
import time
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.body_image_validation import router
from app.body_image_validation.concurrency import BodyValidationLimiter
from app.body_image_validation.exceptions import BodyImageSystemError, user_error
from app.body_image_validation.models import BodyImageValidationResult
from app.body_image_validation.runtime import BodyImageRuntimeError
from app.body_image_validation.service import BodyImageValidationService
from app.clients.s3 import S3ConfigError
from app.config import settings
from app.main import app
from tests.test_body_image_image_io import png_header_only

URL = "/api/v1/validation/body-image"
KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class StubService:
    def __init__(self, result=None, error=None):
        self.result = result or BodyImageValidationResult(s3_key="body-images/example.png")
        self.error = error
        self.calls = []

    def validate(self, body):
        self.calls.append(body)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "CHECKPOINT_DSN", "")
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", KEY)
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "test-vton-key")
    monkeypatch.setenv("RUNWARE_LLM_API_KEY", "test-llm-key")
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def use_service(monkeypatch, service):
    monkeypatch.setattr(router, "get_service", lambda: service)


def test_success_contract_and_multipart_fields(client, monkeypatch):
    service = StubService()
    use_service(monkeypatch, service)

    response = client.post(
        URL,
        files={"image": ("body.png", b"image-body", "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 200
    assert response.json() == {
        "code": "BODY_IMAGE_UPLOAD_SUCCESS",
        "data": {"s3_key": "body-images/example.png"},
        "message": "전신 사진 검증에 성공했습니다.",
    }
    assert service.calls == [b"image-body"]


def test_user_failure_returns_first_reason(client, monkeypatch):
    use_service(monkeypatch, StubService(error=user_error("FULL_BODY_NOT_VISIBLE")))

    response = client.post(
        URL,
        files={"image": ("body.png", b"image-body", "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "FULL_BODY_NOT_VISIBLE",
        "data": None,
        "message": "머리부터 발끝까지 모두 나오도록 전신을 촬영해주세요.",
    }


def test_too_dark_image_is_user_failure(client, monkeypatch):
    use_service(monkeypatch, StubService(error=user_error("IMAGE_TOO_DARK")))

    response = client.post(
        URL,
        files={"image": ("body.png", b"image-body", "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "IMAGE_TOO_DARK",
        "data": None,
        "message": "사진이 너무 어둡습니다. 밝은 곳에서 다시 촬영해주세요.",
    }


def test_system_failure_is_distinct_from_user_validation(client, monkeypatch):
    use_service(
        monkeypatch,
        StubService(error=BodyImageSystemError("PERSON_DETECTION_FAILED")),
    )

    response = client.post(
        URL,
        files={"image": ("body.png", b"image-body", "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": "PERSON_DETECTION_FAILED",
        "data": None,
        "message": "전신 사진 처리 중 서버 오류가 발생했습니다.",
    }


def test_request_does_not_need_user_id_and_ignores_a_legacy_one(client, monkeypatch):
    service = StubService()
    use_service(monkeypatch, service)

    response = client.post(
        URL,
        data={"user_id": "7"},
        files={"image": ("body.png", b"image-body", "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 200
    assert response.json()["data"]["s3_key"] == "body-images/example.png"
    assert service.calls == [b"image-body"]


def test_missing_s3_bucket_is_s3_config_error_before_any_processing(client, monkeypatch):
    def misconfigured():
        raise S3ConfigError("S3_BUCKET 환경변수가 설정되지 않았습니다.")

    monkeypatch.setattr(router, "get_service", misconfigured)

    response = client.post(
        URL,
        files={"image": ("body.png", b"image-body", "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": "S3_CONFIG_ERROR",
        "data": None,
        "message": "전신 사진 처리 중 서버 오류가 발생했습니다.",
    }


def test_unavailable_runtime_is_a_system_failure_not_a_success(client, monkeypatch):
    def unavailable():
        raise BodyImageRuntimeError("전신 이미지 모델 파일이 없습니다.")

    monkeypatch.setattr(router, "get_service", unavailable)

    response = client.post(
        URL,
        files={"image": ("body.png", b"image-body", "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": "BODY_IMAGE_RUNTIME_UNAVAILABLE",
        "data": None,
        "message": "전신 사진 처리 중 서버 오류가 발생했습니다.",
    }


class RecordingStage:
    """detector/pose/rembg/S3 자리. 호출되면 기록만 한다."""

    def __init__(self, calls, name):
        self._calls = calls
        self._name = name

    def _record(self, *args, **kwargs):
        self._calls.append(self._name)
        raise AssertionError(f"{self._name} 는 호출되면 안 된다")

    detect = remove = upload_bytes = _record


def test_huge_resolution_upload_is_422_before_any_model_or_s3(client, monkeypatch):
    calls = []
    use_service(
        monkeypatch,
        BodyImageValidationService(
            RecordingStage(calls, "person_detector"),
            RecordingStage(calls, "pose_detector"),
            RecordingStage(calls, "rembg"),
            RecordingStage(calls, "s3"),
        ),
    )

    response = client.post(
        URL,
        # 약 60바이트 파일에 15055 x 10000 = 150,550,000 픽셀을 선언한다(장애 재현 크기).
        files={"image": ("huge.png", png_header_only(15055, 10000), "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "IMAGE_RESOLUTION_TOO_LARGE",
        "data": None,
        "message": "이미지 해상도가 너무 높습니다. 더 작은 해상도의 사진을 업로드해주세요.",
    }
    assert calls == []


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (b"not-an-image", 400, "INVALID_IMAGE"),
        (png_header_only(480, 640), 400, "IMAGE_DECODE_FAILED"),
    ],
    ids=["not-an-image", "damaged-body"],
)
def test_user_file_problems_are_user_failures_before_any_model(
    client, monkeypatch, body, status, code
):
    calls = []
    use_service(
        monkeypatch,
        BodyImageValidationService(
            RecordingStage(calls, "person_detector"),
            RecordingStage(calls, "pose_detector"),
            RecordingStage(calls, "rembg"),
            RecordingStage(calls, "s3"),
        ),
    )

    response = client.post(
        URL, files={"image": ("body.png", body, "image/png")}, headers=AUTH
    )

    assert response.status_code == status
    assert response.json()["code"] == code
    assert response.json()["data"] is None
    assert isinstance(response.json()["message"], str)
    assert calls == []


def test_internal_decode_error_is_a_500_internal_server_error_not_a_user_error(
    client, monkeypatch
):
    def broken_exif_transpose(image):
        raise RuntimeError("internal bug")

    monkeypatch.setattr("app.body_image_validation.image_io.ImageOps.exif_transpose",
                        broken_exif_transpose)
    calls = []
    use_service(
        monkeypatch,
        BodyImageValidationService(
            RecordingStage(calls, "person_detector"),
            RecordingStage(calls, "pose_detector"),
            RecordingStage(calls, "rembg"),
            RecordingStage(calls, "s3"),
        ),
    )
    good_image = BytesIO()
    Image.new("RGB", (480, 640), "white").save(good_image, format="PNG")

    response = client.post(
        URL, files={"image": ("body.png", good_image.getvalue(), "image/png")}, headers=AUTH
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": "INTERNAL_SERVER_ERROR",
        "data": None,
        "message": "서버 내부 오류가 발생했습니다.",
    }
    assert calls == []


def test_unexpected_image_open_error_is_a_500_internal_server_error(client, monkeypatch):
    def broken_open(*args, **kwargs):
        raise RuntimeError("internal bug in Image.open")

    monkeypatch.setattr("app.body_image_validation.image_io.Image.open", broken_open)
    calls = []
    use_service(
        monkeypatch,
        BodyImageValidationService(
            RecordingStage(calls, "person_detector"),
            RecordingStage(calls, "pose_detector"),
            RecordingStage(calls, "rembg"),
            RecordingStage(calls, "s3"),
        ),
    )

    response = client.post(
        URL, files={"image": ("body.png", b"any-bytes", "image/png")}, headers=AUTH
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": "INTERNAL_SERVER_ERROR",
        "data": None,
        "message": "서버 내부 오류가 발생했습니다.",
    }
    assert calls == []


def test_authentication_is_required(client, monkeypatch):
    service = StubService()
    use_service(monkeypatch, service)

    response = client.post(
        URL,
        files={"image": ("body.png", b"image-body", "image/png")},
    )

    assert response.status_code == 401
    assert service.calls == []


def test_missing_multipart_field_is_invalid_request(client):
    response = client.post(URL, headers=AUTH)

    assert response.status_code == 400
    assert response.json() == {
        "code": "INVALID_REQUEST",
        "data": None,
        "message": "입력값이 올바르지 않습니다.",
    }


# --- 긴급 안정화: 2MB 제한과 동시 실행 제한(실행 1 + 대기 1, 초과 시 429) ---


def test_upload_over_2mb_is_413_image_too_large(client, monkeypatch):
    calls = []
    use_service(
        monkeypatch,
        BodyImageValidationService(
            RecordingStage(calls, "person_detector"),
            RecordingStage(calls, "pose_detector"),
            RecordingStage(calls, "rembg"),
            RecordingStage(calls, "s3"),
        ),
    )

    response = client.post(
        URL,
        files={"image": ("big.png", b"\x00" * (2 * 1024 * 1024 + 1), "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 413
    assert response.json() == {
        "code": "IMAGE_TOO_LARGE",
        "data": None,
        "message": "이미지 크기는 2MB 이하여야 합니다.",
    }
    assert calls == []


def _occupy(limiter):
    """실행 1 + 대기 1 을 채운다. 돌려준 release() 를 부르면 둘 다 끝난다."""
    release = threading.Event()
    running = threading.Event()

    def hold():
        with limiter.slot():
            running.set()
            release.wait(5)

    threads = [threading.Thread(target=hold, daemon=True) for _ in range(2)]
    threads[0].start()
    assert running.wait(5)
    threads[1].start()
    deadline = time.monotonic() + 5
    while limiter.waiting != 1:
        assert time.monotonic() < deadline
        time.sleep(0.001)

    def finish():
        release.set()
        for thread in threads:
            thread.join(5)

    return finish


def _valid_png():
    output = BytesIO()
    Image.new("RGB", (480, 640), "white").save(output, format="PNG")
    return output.getvalue()


def test_third_request_is_429_server_busy_while_one_runs_and_one_waits(client, monkeypatch):
    calls = []
    limiter = BodyValidationLimiter(1, 1)
    use_service(
        monkeypatch,
        BodyImageValidationService(
            RecordingStage(calls, "person_detector"),
            RecordingStage(calls, "pose_detector"),
            RecordingStage(calls, "rembg"),
            RecordingStage(calls, "s3"),
            limiter=limiter,
        ),
    )
    finish = _occupy(limiter)
    try:
        response = client.post(
            URL, files={"image": ("body.png", _valid_png(), "image/png")}, headers=AUTH
        )
    finally:
        finish()

    assert response.status_code == 429
    assert response.json() == {
        "code": "SERVER_BUSY",
        "data": None,
        "message": "현재 이미지 처리 요청이 많습니다. 잠시 후 다시 시도해주세요.",
    }
    assert calls == []
    assert (limiter.running, limiter.waiting) == (0, 0)


def test_invalid_image_gets_its_own_error_even_when_the_queue_is_full(client, monkeypatch):
    limiter = BodyValidationLimiter(1, 1)
    use_service(
        monkeypatch,
        BodyImageValidationService(
            RecordingStage([], "person_detector"),
            RecordingStage([], "pose_detector"),
            RecordingStage([], "rembg"),
            RecordingStage([], "s3"),
            limiter=limiter,
        ),
    )
    finish = _occupy(limiter)
    try:
        response = client.post(
            URL, files={"image": ("body.png", b"", "image/png")}, headers=AUTH
        )
    finally:
        finish()

    assert response.status_code == 400
    assert response.json()["code"] == "IMAGE_EMPTY"
