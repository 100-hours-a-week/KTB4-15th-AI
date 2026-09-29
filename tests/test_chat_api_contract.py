"""Backend 와의 HTTP 계약 (팀 공통 응답 형식, 단계1 §9).

오류 형식과 인증은 Backend 가 code 로 분기하는 값이라 형태가 바뀌면 바로 알아야 한다.
모든 응답 본문은 {"code": str, "data": ..., "message": str} 이다.
"""

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.clients.llm import LLMError
from app.config import settings
from app.config.checkpointer import get_graph
from app.main import app
from app.recommendation import RecommendationError, WishlistCommentError

KEY = "test-internal-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
INVALID_REQUEST_BODY = {
    "code": "INVALID_REQUEST",
    "data": None,
    "message": "입력값이 올바르지 않습니다.",
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", KEY)
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)
    monkeypatch.setattr(settings, "CHECKPOINT_DSN", "")
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "test-vton-key")
    monkeypatch.setenv("RUNWARE_LLM_API_KEY", "test-llm-key")
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_missing_credentials_is_401(client):
    response = client.post(
        "/api/v1/chat/stream",
        json={"chat_id": 1, "user_id": 1, "message": "안녕"},
    )
    assert response.status_code == 401
    assert response.json() == {
        "code": "UNAUTHORIZED",
        "data": None,
        "message": "인증에 실패했습니다.",
    }


def test_wrong_key_is_401(client):
    response = client.post(
        "/api/v1/chat/stream",
        json={"chat_id": 1, "user_id": 1, "message": "안녕"},
        headers={"Authorization": "Bearer nope"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


def test_schema_violation_is_400_invalid_request(client):
    response = client.post(
        "/api/v1/chat/stream",
        json={"chat_id": "abc", "user_id": 1, "message": "안녕"},
        headers=AUTH,
    )
    assert response.status_code == 400
    assert response.json() == INVALID_REQUEST_BODY
    assert "detail" not in response.json()


def test_missing_required_field_is_400(client):
    response = client.post("/api/v1/chat/stream", json={"chat_id": 1}, headers=AUTH)
    assert response.status_code == 400
    assert response.json() == INVALID_REQUEST_BODY


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
    assert response.json() == {
        "code": "INVALID_WISHLIST_PRODUCTS",
        "data": {"chat_id": 1},
        "message": "찜 상품 목록이 올바르지 않습니다.",
    }


def test_message_longer_than_500_is_rejected(client):
    # Backend 의 ChatMessageCreateRequest(@Size(max = 500)) 와 같은 상한
    response = client.post(
        "/api/v1/chat/stream",
        json={"chat_id": 1, "user_id": 1, "message": "가" * 501},
        headers=AUTH,
    )
    assert response.status_code == 400
    assert response.json() == INVALID_REQUEST_BODY


def test_unknown_source_type_is_rejected(client):
    # Backend 의 ChatSourceType 은 GENERAL / WISHLIST 두 개뿐이다.
    response = client.post(
        "/api/v1/chat/stream",
        json={"chat_id": 1, "user_id": 1, "message": "안녕", "source_type": "chat"},
        headers=AUTH,
    )
    assert response.status_code == 400
    assert response.json() == INVALID_REQUEST_BODY


def test_delete_success_without_data_has_null_data(client):
    response = client.delete("/api/v1/chat/1", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "code": "CHAT_DELETE_SUCCESS",
        "data": None,
        "message": "대화가 삭제되었습니다.",
    }
    assert list(body) == ["code", "data", "message"]


def test_delete_also_requires_credentials(client):
    assert client.delete("/api/v1/chat/1").status_code == 401


def test_unexpected_error_is_500_without_internal_details(client, monkeypatch, caplog):
    from app.chat import controller

    async def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(controller, "find_pre_stream_error", boom)

    with caplog.at_level(logging.ERROR):
        response = client.post(
            "/api/v1/chat/stream",
            json={"chat_id": 1, "user_id": 1, "message": "안녕"},
            headers=AUTH,
        )
    assert response.status_code == 500
    assert response.json() == {
        "code": "INTERNAL_SERVER_ERROR",
        "data": None,
        "message": "서버 내부 오류가 발생했습니다.",
    }
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text
    # 원인은 서버 로그에 남는다.
    assert "secret internal detail" in caplog.text


def test_unknown_route_is_404_in_the_common_shape(client):
    response = client.get("/api/v1/does-not-exist", headers=AUTH)
    assert response.status_code == 404
    assert response.json() == {
        "code": "NOT_FOUND",
        "data": None,
        "message": "요청한 리소스를 찾을 수 없습니다.",
    }


def test_wrong_method_is_405_in_the_common_shape(client):
    response = client.get("/api/v1/sync-fitting", headers=AUTH)
    assert response.status_code == 405
    assert response.json() == {
        "code": "METHOD_NOT_ALLOWED",
        "data": None,
        "message": "허용되지 않은 요청 메서드입니다.",
    }


class FailingGraph:
    """token 하나를 보낸 뒤 지정한 오류를 내는 그래프 대역. 외부 LLM 을 부르지 않는다."""

    def __init__(self, error):
        self._error = error

    async def aget_state(self, config):
        return SimpleNamespace(values={})

    async def astream(self, state, config, stream_mode):
        yield {"event": "token", "content": "안녕"}
        raise self._error


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (LLMError("boom"), "LLM_GENERATION_FAILED"),
        (WishlistCommentError("boom"), "WISHLIST_COMMENT_GENERATION_FAILED"),
        (RecommendationError("boom"), "RECOMMENDATION_SEARCH_FAILED"),
    ],
)
def test_stream_error_event_uses_an_uppercase_code(client, error, code):
    app.dependency_overrides[get_graph] = lambda: FailingGraph(error)
    try:
        response = client.post(
            "/api/v1/chat/stream",
            json={"chat_id": 1, "user_id": 1, "message": "안녕"},
            headers=AUTH,
        )
    finally:
        app.dependency_overrides.pop(get_graph, None)

    events = {
        block.split("\n")[0].removeprefix("event: "): json.loads(block.split("data: ", 1)[1])
        for block in response.text.strip().split("\n\n")
    }
    assert events["error"]["code"] == code
    assert "done" in events
