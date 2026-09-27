"""done 이벤트에 무엇이 실리는가 (2026-09-27 Backend 합의).

- 모든 턴의 done 에 products 키가 있다. 추천이 없으면 빈 배열이다.
- 검색 기반 추천 턴은 products 이벤트와 같은 목록을 done 에도 싣고, content 는 고정 문구다.
- 그 밖의 턴의 content 는 지금처럼 흘려보낸 말풍선 전체다.
"""

import asyncio
import json

from app.chat import sse
from app.chat.controller import stream_chat
from app.chat.schemas import ChatRequest

PRODUCTS = [
    {
        "product_code": "4097486",
        "product_name": "리넨 블랜디드 커브드 팬츠",
        "image_url": "https://example.com/1.jpg",
        "detail_url": "https://example.com/p/1",
        "color": "블랙",
        "price": 25420,
        "item_type": "BOTTOM",
        "llm_comment": "일상적인 캐주얼 스타일에 적합하다.",
    }
]


class FakeGraph:
    """그래프 대신 정해 둔 custom 청크를 흘린다."""

    def __init__(self, chunks):
        self.chunks = chunks

    async def astream(self, state, config, stream_mode):
        for chunk in self.chunks:
            yield chunk


def _events(chunks) -> list[tuple[str, dict]]:
    async def run():
        request = ChatRequest(chat_id=7, user_id=1, message="응")
        return [raw async for raw in stream_chat(FakeGraph(chunks), request)]

    events = []
    for raw in asyncio.run(run()):
        head, data = raw.strip().split("\n", 1)
        events.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_general_chat_done_has_empty_products():
    events = _events([
        {"event": sse.TOKEN, "content": "안녕"},
        {"event": sse.TOKEN, "content": "하세요"},
    ])

    name, done = events[-1]
    assert name == sse.DONE
    assert done == {"chat_id": 7, "content": "안녕하세요", "products": []}


def test_search_turn_done_repeats_products_with_fixed_content():
    events = _events([
        {"event": sse.PRODUCTS, "products": PRODUCTS, "done_content": sse.SEARCH_DONE_CONTENT},
    ])

    assert [name for name, _ in events] == [sse.PRODUCTS, sse.DONE]
    assert events[0][1]["products"] == PRODUCTS  # products 이벤트는 그대로 나간다
    assert events[1][1] == {
        "chat_id": 7,
        "content": "조건에 맞는 옷을 검색해봤습니다.",
        "products": PRODUCTS,
    }


def test_search_with_no_result_still_uses_fixed_content():
    events = _events([
        {"event": sse.PRODUCTS, "products": [], "done_content": sse.SEARCH_DONE_CONTENT},
    ])

    assert events[-1][1] == {"chat_id": 7, "content": sse.SEARCH_DONE_CONTENT, "products": []}


def test_products_without_done_content_keep_spoken_content():
    # 찜 추천처럼 고정 문구를 정하지 않은 경로는 말풍선 전체(없으면 빈 문자열)를 그대로 쓴다
    events = _events([
        {"event": sse.STATUS, "label": "찜한 상품을 살펴보고 있어요"},
        {"event": sse.PRODUCTS, "products": PRODUCTS},
    ])

    assert events[-1][1] == {"chat_id": 7, "content": "", "products": PRODUCTS}
