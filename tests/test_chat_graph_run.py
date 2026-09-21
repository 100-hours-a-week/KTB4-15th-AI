"""그래프를 실제로 한 턴 돌린다.

컴파일만으로는 분기가 검증되지 않는다. 분기 함수의 반환값과 path map 이 어긋나도
컴파일은 통과하고, 실행할 때가 되어서야 KeyError 로 터진다.
"""

import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from app import recommendation
from app.chat import sse
from app.chat.graph import build_graph
from app.chat.graph.state import initial_state

PRODUCTS = [
    {
        "product_id": "0000011",
        "product_name": "오버핏 후드 집업",
        "image_url": "https://example.com/11.jpg",
        "detail_url": "https://example.com/p/11",
        "color": "IVORY",
        "llm_comment": "최근 찜한 상품과 비슷한 실루엣입니다.",
    }
]


def _collect(graph, state, thread_id):
    async def run():
        config = {"configurable": {"thread_id": thread_id}}
        return [chunk async for chunk in graph.astream(state, config, stream_mode="custom")]

    return asyncio.run(run())


def test_chip_click_reaches_the_wishlist_node(monkeypatch):
    async def fake_wishlist(*, product_ids, on_progress=None):
        if on_progress is not None:
            on_progress("찜한 상품을 살펴보고 있어요")
        return PRODUCTS

    monkeypatch.setattr(recommendation, "recommend_from_wishlist", fake_wishlist)

    graph = build_graph(InMemorySaver())
    state = initial_state(
        "내 찜 목록으로 추천받기",
        source_type="WISHLIST",
        product_ids=[str(i) for i in range(10)],
    )

    events = _collect(graph, state, "chip-turn")

    assert [event["event"] for event in events] == [sse.STATUS, sse.PRODUCTS]
    assert events[-1]["products"] == PRODUCTS


def test_recommendation_is_left_in_the_conversation(monkeypatch):
    async def fake_wishlist(*, product_ids, on_progress=None):
        return PRODUCTS

    monkeypatch.setattr(recommendation, "recommend_from_wishlist", fake_wishlist)

    graph = build_graph(InMemorySaver())
    state = initial_state("내 찜 목록으로 추천받기", source_type="WISHLIST", product_ids=["1"])
    config = {"configurable": {"thread_id": "chip-turn-2"}}

    async def run():
        await graph.ainvoke(state, config)
        return await graph.aget_state(config)

    snapshot = asyncio.run(run())
    messages = snapshot.values["messages"]

    assert messages[0]["content"] == "내 찜 목록으로 추천받기"
    assert "오버핏 후드 집업 (0000011)" in messages[-1]["content"]
    assert snapshot.values["awaiting_confirm"] is False
