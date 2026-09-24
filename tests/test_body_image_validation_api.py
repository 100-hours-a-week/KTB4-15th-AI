import pytest
from fastapi.testclient import TestClient

from app.body_image_validation import router
from app.body_image_validation.exceptions import BodyImageSystemError, user_error
from app.body_image_validation.models import BodyImageValidationResult
from app.config import settings
from app.main import app

URL = "/api/v1/validation/body-image"
KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class StubService:
    def __init__(self, result=None, error=None):
        self.result = result or BodyImageValidationResult(
            s3_key="users/7/body-images/example.png", warnings=[]
        )
        self.error = error
        self.calls = []

    def validate(self, user_id, body):
        self.calls.append((user_id, body))
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
            s3_key="users/7/body-images/example.png", warnings=["IMAGE_TOO_DARK"]
        )
    )
    use_service(monkeypatch, service)

    response = client.post(
        URL,
        data={"user_id": "7"},
        files={"image": ("body.png", b"image-body", "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "message": "body_image_validation_success",
        "data": {
            "s3_key": "users/7/body-images/example.png",
            "warnings": ["IMAGE_TOO_DARK"],
        },
    }
    assert service.calls == [(7, b"image-body")]


def test_user_failure_returns_first_reason(client, monkeypatch):
    use_service(monkeypatch, StubService(error=user_error("FULL_BODY_NOT_VISIBLE")))

    response = client.post(
        URL,
        data={"user_id": "7"},
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
        data={"user_id": "7"},
        files={"image": ("body.png", b"image-body", "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": 500,
        "message": "body_image_validation_system_failed",
        "data": {"reason_code": "PERSON_DETECTION_FAILED"},
    }


def test_authentication_is_required(client, monkeypatch):
    service = StubService()
    use_service(monkeypatch, service)

    response = client.post(
        URL,
        data={"user_id": "7"},
        files={"image": ("body.png", b"image-body", "image/png")},
    )

    assert response.status_code == 401
    assert service.calls == []


def test_missing_multipart_field_is_invalid_request(client):
    response = client.post(URL, data={"user_id": "7"}, headers=AUTH)

    assert response.status_code == 400
    assert response.json() == {"code": 400, "message": "invalid_request", "data": None}
