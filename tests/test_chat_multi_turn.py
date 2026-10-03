"""여러 턴에 걸친 대화.

checkpointer 가 thread_id(= chat_id) 별로 State 를 복원하는지, 조건이 턴을 넘어
쌓이는지, 봇이 물어 둔 질문이 다음 턴에서 해석되는지를 실제 그래프 실행으로 확인한다.

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
            # 1턴: 카테고리와 분위기를 말함
            {
                "answer": "none",
                "metadata": {"category": "셔츠"},
                "mood": "소개팅에 입기 좋은 단정한 분위기",
            },
            # 2턴: 거절하면서 조건을 더 말함
            {
                "answer": "no",
                "metadata": {"color": "블랙", "max_price": 50000},
                "dislikes": [{"field": "color", "value": "레드"}],
            },
            # 3턴: 동의
            {"answer": "yes", "metadata": {}},
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
    assert state["pending_question"] == "confirm_summary"
    assert state["conditions"]["category"] == "셔츠"

    # --- 2턴: 직전 턴이 요약 확인이었다는 사실이 프롬프트에 실려야 한다
    _run(graph, "아니, 검정색으로 5만원 이하")
    second_prompt = str(fake.prompts[1])
    assert "봇이 지금까지의 조건을 요약하고, 이 조건으로 추천해도 될지 물었다." in second_prompt
    assert "셔츠" in second_prompt  # 1턴 조건이 복원되어 함께 들어갔다

    state = _state(graph)
    assert state["conditions"] == {
        "category": "셔츠",
        "color": "블랙",
        "min_price": 0,  # 가격은 LLM 값이 아니라 발화 "5만원 이하"에서 규칙으로 뽑는다
        "max_price": 50000,
        "dislikes": [{"field": "color", "value": "레드"}],
    }
    assert state["pending_question"] == "confirm_summary"  # 거절했지만 조건을 말해서 재요약

    # --- 3턴: "응" 한 마디로 검색까지 간다
    events = _run(graph, "응")
    assert events[-1]["products"] == PRODUCTS

    # 세 턴에 걸쳐 모인 조건이 그대로 검색에 넘어간다
    assert fake_search.called_with["category"] == "셔츠"
    assert fake_search.called_with["color"] == "블랙"
    assert fake_search.called_with["min_price"] == 0
    assert fake_search.called_with["max_price"] == 50000

    state = _state(graph)
    assert state["pending_question"] is None
    assert "오버핏 코튼 셔츠 (0000001)" in state["messages"][-1]["content"]
    # 사용자 3번 + 봇 3번(요약·요약·추천기록)
    assert len(state["messages"]) == 6


def test_other_chat_room_does_not_see_this_conversation(monkeypatch):
    fake = FakeLLM([{"answer": "none", "metadata": {}}])
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


def test_condition_change_after_ask_change_goes_back_to_summary(monkeypatch):
    """요약 → "ㄴㄴ" → 되묻기 → "가을 말고 여름" 이 일반 대화로 빠지지 않는다 (2026-09-28 QA)."""
    fake = FakeLLM(
        [
            {"answer": "none", "metadata": {}, "mood": "가을 데이트 룩"},
            {"answer": "no", "metadata": {}},
            {
                "answer": "no",
                "metadata": {},
                "mood": "여름 데이트 룩",
            },
        ]
    )
    monkeypatch.setattr(llm_module.llm, "complete_json", fake.complete_json)
    monkeypatch.setattr(llm_module.llm, "stream_text", fake.stream_text)

    graph = build_graph(InMemorySaver())

    _run(graph, "가을에 데이트할 때 입을 옷")
    _run(graph, "ㄴㄴ")
    assert _state(graph)["pending_question"] == "ask_change"

    _run(graph, "가을 말고 여름")
    third_prompt = str(fake.prompts[2])
    assert "봇이 추천 조건 중 무엇을 바꾸고 싶은지 물었다." in third_prompt

    state = _state(graph)
    assert state["said_conditions"] is True
    assert state["mood"] == "여름 데이트 룩"
    assert state["pending_question"] == "confirm_summary"  # 재요약했다
