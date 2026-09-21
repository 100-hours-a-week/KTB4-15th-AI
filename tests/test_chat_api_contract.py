"""Backend 와의 HTTP 계약 (단계1 §6, §9).

오류 형식과 인증은 Backend 가 코드로 분기하는 값이라 형태가 바뀌면 바로 알아야 한다.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

KEY = "test-internal-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", KEY)
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)
    monkeypatch.setattr(settings, "CHECKPOINT_DSN", "")
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_missing_credentials_is_401(client):
    response = client.post(
        "/api/v1/chat/stream",
        json={"chat_id": 1, "user_id": 1, "message": "안녕"},
    )
    assert response.status_code == 401
    assert response.json() == {"code": 401, "message": "unauthorized", "data": None}


def test_wrong_key_is_401(client):
    response = client.post(
        "/api/v1/chat/stream",
        json={"chat_id": 1, "user_id": 1, "message": "안녕"},
        headers={"Authorization": "Bearer nope"},
    )
    assert response.status_code == 401


def test_schema_violation_is_400_invalid_request(client):
    response = client.post(
        "/api/v1/chat/stream",
        json={"chat_id": "abc", "user_id": 1, "message": "안녕"},
        headers=AUTH,
    )
    assert response.status_code == 400
    assert response.json() == {"code": 400, "message": "invalid_request", "data": None}


def test_missing_required_field_is_400(client):
    response = client.post("/api/v1/chat/stream", json={"chat_id": 1}, headers=AUTH)
    assert response.status_code == 400
    assert response.json()["message"] == "invalid_request"


def test_wishlist_needs_exactly_ten_products(client):
    response = client.post(
        "/api/v1/chat/stream",
        json={
            "chat_id": 1,
            "user_id": 1,
            "message": "찜 추천",
            "source_type": "WISHLIST",
            "product_ids": ["1", "2"],
        },
        headers=AUTH,
    )
    assert response.status_code == 400
    assert response.json()["message"] == "invalid_wishlist_products"


def test_message_longer_than_500_is_rejected(client):
    # Backend 의 ChatMessageCreateRequest(@Size(max = 500)) 와 같은 상한
    response = client.post(
        "/api/v1/chat/stream",
        json={"chat_id": 1, "user_id": 1, "message": "가" * 501},
        headers=AUTH,
    )
    assert response.status_code == 400
    assert response.json()["message"] == "invalid_request"


def test_unknown_source_type_is_rejected(client):
    # Backend 의 ChatSourceType 은 GENERAL / WISHLIST 두 개뿐이다.
    response = client.post(
        "/api/v1/chat/stream",
        json={"chat_id": 1, "user_id": 1, "message": "안녕", "source_type": "chat"},
        headers=AUTH,
    )
    assert response.status_code == 400
    assert response.json()["message"] == "invalid_request"


def test_delete_returns_the_common_shape(client):
    response = client.delete("/api/v1/chat/1", headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {"code": 200, "message": "chat_deleted", "data": None}


def test_delete_also_requires_credentials(client):
    assert client.delete("/api/v1/chat/1").status_code == 401


def test_unexpected_error_is_500_internal_server_error(client, monkeypatch):
    from app.chat import controller

    async def boom(*args, **kwargs):
        raise RuntimeError("어딘가 터짐")

    monkeypatch.setattr(controller, "find_pre_stream_error", boom)

    response = client.post(
        "/api/v1/chat/stream",
        json={"chat_id": 1, "user_id": 1, "message": "안녕"},
        headers=AUTH,
    )
    assert response.status_code == 500
    assert response.json() == {"code": 500, "message": "internal_server_error", "data": None}
