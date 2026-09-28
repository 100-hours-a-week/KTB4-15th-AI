from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.body_image_validation import router
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
        "code": 200,
        "message": "body_image_validation_success",
        "data": {"s3_key": "body-images/example.png"},
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
        "code": 422,
        "message": "body_image_validation_failed",
        "data": {
            "reason_code": "FULL_BODY_NOT_VISIBLE",
            "reason": "머리부터 발끝까지 모두 나오도록 전신을 촬영해주세요.",
        },
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
        "code": 422,
        "message": "body_image_validation_failed",
        "data": {
            "reason_code": "IMAGE_TOO_DARK",
            "reason": "사진이 너무 어둡습니다. 밝은 곳에서 다시 촬영해주세요.",
        },
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
        "code": 500,
        "message": "body_image_validation_system_failed",
        "data": {"reason_code": "PERSON_DETECTION_FAILED"},
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
        "code": 500,
        "message": "body_image_validation_system_failed",
        "data": {"reason_code": "S3_CONFIG_ERROR"},
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
        "code": 500,
        "message": "body_image_validation_system_failed",
        "data": {"reason_code": "BODY_IMAGE_RUNTIME_UNAVAILABLE"},
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
        "code": 422,
        "message": "body_image_validation_failed",
        "data": {
            "reason_code": "IMAGE_RESOLUTION_TOO_LARGE",
            "reason": "이미지 해상도가 너무 높습니다. 더 작은 해상도의 사진을 업로드해주세요.",
        },
    }
    assert calls == []


@pytest.mark.parametrize(
    ("body", "status", "message", "reason_code"),
    [
        (b"not-an-image", 400, "body_image_validation_failed", "INVALID_IMAGE"),
        (png_header_only(480, 640), 400, "body_image_validation_failed", "IMAGE_DECODE_FAILED"),
    ],
    ids=["not-an-image", "damaged-body"],
)
def test_user_file_problems_are_user_failures_before_any_model(
    client, monkeypatch, body, status, message, reason_code
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
    assert response.json()["message"] == message
    assert response.json()["data"]["reason_code"] == reason_code
    assert calls == []


def test_internal_decode_error_is_a_500_processing_failure_not_a_user_error(
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
        "code": 500,
        "message": "body_image_validation_system_failed",
        "data": {"reason_code": "IMAGE_PROCESSING_FAILED"},
    }
    assert calls == []


def test_unexpected_image_open_error_is_a_500_processing_failure(client, monkeypatch):
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
        "code": 500,
        "message": "body_image_validation_system_failed",
        "data": {"reason_code": "IMAGE_PROCESSING_FAILED"},
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
    assert response.json() == {"code": 400, "message": "invalid_request", "data": None}
