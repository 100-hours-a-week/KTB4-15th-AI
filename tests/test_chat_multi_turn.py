"""여러 턴에 걸친 대화.

checkpointer 가 thread_id(= chat_id) 별로 State 를 복원하는지, 조건이 턴을 넘어
쌓이는지, 확인 플래그가 다음 턴에서 해석되는지를 실제 그래프 실행으로 확인한다.

LLM 은 가짜로 바꾼다. 여기서 보려는 것은 모델의 판단이 아니라 턴 사이에 무엇이
남고 무엇이 프롬프트로 들어가는지다.
"""

import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from app import recommendation
from app.chat.graph import build_graph
from app.chat.graph.state import initial_state
from app.clients import llm as llm_module

THREAD = {"configurable": {"thread_id": "multi-turn"}}

PRODUCTS = [
    {
        "product_code": "0000001",
        "product_name": "오버핏 코튼 셔츠",
        "image_url": "https://example.com/1.jpg",
        "detail_url": "https://example.com/p/1",
        "color": "NAVY",
        "llm_comment": "차분한 네이비 셔츠입니다.",
    }
]


class FakeLLM:
    """정해진 순서대로 답하고, 받은 프롬프트를 모아둔다."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def complete_json(self, messages, **kwargs):
        self.prompts.append(messages)
        return self.replies.pop(0)

    async def stream_text(self, messages, **kwargs):
        for piece in ("이 조건으로 ", "추천해 드릴까요?"):
            yield piece


def _run(graph, message):
    async def turn():
        return [
            chunk
            async for chunk in graph.astream(
                initial_state(message), THREAD, stream_mode="custom"
            )
        ]

    return asyncio.run(turn())


def _state(graph):
    return asyncio.run(graph.aget_state(THREAD)).values


def test_three_turns_share_one_conversation(monkeypatch):
    fake = FakeLLM(
        [
            # 1턴: 추천 의도 + 카테고리만 말함
            {
                "intent": "recommend",
                "metadata": {"category": "셔츠"},
                "semantic_query": "소개팅에 입기 좋은 단정한 분위기",
            },
            # 2턴: 거절하면서 조건을 더 말함
            {
                "intent": "reject_with_conditions",
                "metadata": {"color": "블랙", "max_price": 50000},
                "dislikes": [{"field": "color", "value": "레드"}],
            },
            # 3턴: 동의
            {"intent": "confirm", "metadata": {}},
        ]
    )
    monkeypatch.setattr(llm_module.llm, "complete_json", fake.complete_json)
    monkeypatch.setattr(llm_module.llm, "stream_text", fake.stream_text)

    async def fake_search(**kwargs):
        fake_search.called_with = kwargs
        return PRODUCTS

    monkeypatch.setattr(recommendation, "search_products", fake_search)

    graph = build_graph(InMemorySaver())

    # --- 1턴
    _run(graph, "소개팅에 입을 셔츠 추천해줘")
    state = _state(graph)
    assert state["awaiting_confirm"] is True
    assert state["conditions"]["category"] == "셔츠"

    # --- 2턴: 직전 턴이 요약 확인이었다는 사실이 프롬프트에 실려야 한다
    _run(graph, "아니, 검정색으로 5만원 이하")
    second_prompt = str(fake.prompts[1])
    assert "'직전 턴에 봇이 조건을 요약하고 확인을 물었는가': True" in second_prompt
    assert "셔츠" in second_prompt  # 1턴 조건이 복원되어 함께 들어갔다

    state = _state(graph)
    assert state["conditions"] == {
        "category": "셔츠",
        "color": "블랙",
        "max_price": 50000,
        "dislikes": [{"field": "color", "value": "레드"}],
    }
    assert state["awaiting_confirm"] is True  # 재요약이라 다시 켜진다

    # --- 3턴: "응" 한 마디로 검색까지 간다
    events = _run(graph, "응")
    assert events[-1]["products"] == PRODUCTS

    # 세 턴에 걸쳐 모인 조건이 그대로 검색에 넘어간다
    assert fake_search.called_with["category"] == "셔츠"
    assert fake_search.called_with["color"] == "블랙"
    assert fake_search.called_with["max_price"] == 50000

    state = _state(graph)
    assert state["awaiting_confirm"] is False
    assert "오버핏 코튼 셔츠 (0000001)" in state["messages"][-1]["content"]
    # 사용자 3번 + 봇 3번(요약·요약·추천기록)
    assert len(state["messages"]) == 6


def test_other_chat_room_does_not_see_this_conversation(monkeypatch):
    fake = FakeLLM([{"intent": "chat", "metadata": {}}])
    monkeypatch.setattr(llm_module.llm, "complete_json", fake.complete_json)
    monkeypatch.setattr(llm_module.llm, "stream_text", fake.stream_text)

    graph = build_graph(InMemorySaver())

    async def turn(thread_id):
        config = {"configurable": {"thread_id": thread_id}}
        await graph.ainvoke(initial_state("안녕"), config)
        return (await graph.aget_state(config)).values

    values = asyncio.run(turn("room-a"))
    assert len(values["messages"]) == 2

    other = asyncio.run(graph.aget_state({"configurable": {"thread_id": "room-b"}}))
    assert other.values == {}
