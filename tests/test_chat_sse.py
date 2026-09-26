"""SSE 이벤트 포맷. Backend 와의 계약이라 형태가 바뀌면 바로 알아야 한다."""

import json

from app.chat import sse


def test_event_ends_with_blank_line():
    raw = sse.token(123, "안녕")
    assert raw.startswith("event: token\n")
    assert raw.endswith("\n\n")


def test_korean_is_not_escaped():
    raw = sse.token(123, "안녕")
    assert "안녕" in raw


def test_done_carries_full_sentence():
    raw = sse.done(123, "이 조건으로 추천해 드릴까요?")
    data = json.loads(raw.split("data: ", 1)[1])
    assert data == {"chat_id": 123, "content": "이 조건으로 추천해 드릴까요?", "products": []}


def test_done_without_speech_is_empty_string():
    data = json.loads(sse.done(123).split("data: ", 1)[1])
    assert data["content"] == ""


def test_done_always_has_products_key():
    # 일반 대화 턴에도 products 키가 있어야 한다 (2026-09-27 Backend 합의)
    data = json.loads(sse.done(123, "안녕하세요").split("data: ", 1)[1])
    assert data["products"] == []


def test_done_carries_recommended_products():
    items = [{"product_id": "0000001", "product_name": "셔츠", "price": 39000, "item_type": "TOP"}]
    data = json.loads(sse.done(123, sse.SEARCH_DONE_CONTENT, items).split("data: ", 1)[1])
    assert data == {"chat_id": 123, "content": "조건에 맞는 옷을 검색해봤습니다.", "products": items}


def test_products_event_is_one_payload():
    items = [{"product_id": "0000001", "product_name": "셔츠"}]
    raw = sse.products(123, items)
    assert raw.count("data: ") == 1
    assert json.loads(raw.split("data: ", 1)[1])["products"] == items
