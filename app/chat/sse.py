"""SSE 이벤트 포맷. 단계1 §7 "대화 및 상품 추천 스트리밍"의 계약을 여기서만 만든다."""

import json

TOKEN = "token"
PRODUCTS = "products"
DONE = "done"
ERROR = "error"
# 단계1 §7의 4종에 없는 이벤트다. 찜 추천처럼 오래 걸리는 경로의 진행 상황을 보낸다.
# Backend / Frontend와 합의가 필요하고, 모르는 이벤트는 무시하는 것이 전제다.
STATUS = "status"


def format_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def token(chat_id: int, content: str) -> str:
    return format_event(TOKEN, {"chat_id": chat_id, "content": content})


def products(chat_id: int, items: list[dict]) -> str:
    """Top 3가 확정된 뒤 한 번만 보낸다. JSON을 토큰 단위로 쪼개지 않는다."""
    return format_event(PRODUCTS, {"chat_id": chat_id, "products": items})


def status(chat_id: int, label: str) -> str:
    """진행 상황 라벨. 내부 노드 이름을 그대로 내보내지 않는다.

    노드 이름을 보내면 그래프 구조가 Frontend와의 계약이 되어, 노드를 쪼개거나
    합칠 때마다 화면 문구가 따라 깨진다.
    """
    return format_event(STATUS, {"chat_id": chat_id, "label": label})


def done(chat_id: int, content: str = "") -> str:
    """현재 SSE 응답 종료.

    content 에는 이 턴에 흘려보낸 말풍선 전체 문장이 담긴다 (2026-09-17 결정).
    Backend 는 token 을 이어붙여 저장하되 이 값으로 대조할 수 있다.
    추천만 하고 끝난 턴처럼 말풍선이 없었으면 빈 문자열이다.
    """
    return format_event(DONE, {"chat_id": chat_id, "content": content})


def error(chat_id: int, code: str, message: str) -> str:
    return format_event(ERROR, {"chat_id": chat_id, "code": code, "message": message})
