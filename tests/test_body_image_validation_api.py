import pytest
from fastapi.testclient import TestClient

from app.body_image_validation import router
from app.body_image_validation.exceptions import BodyImageSystemError, user_error
from app.body_image_validation.models import BodyImageValidationResult
from app.body_image_validation.runtime import BodyImageRuntimeError
from app.clients.s3 import S3ConfigError
from app.config import settings
from app.main import app

URL = "/api/v1/validation/body-image"
KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class StubService:
    def __init__(self, result=None, error=None):
        self.result = result or BodyImageValidationResult(
            s3_key="body-images/example.png", warnings=[]
        )
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
    service = StubService(
        BodyImageValidationResult(
            s3_key="body-images/example.png", warnings=["IMAGE_TOO_DARK"]
        )
    )
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
        "data": {
            "s3_key": "body-images/example.png",
            "warnings": ["IMAGE_TOO_DARK"],
        },
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
