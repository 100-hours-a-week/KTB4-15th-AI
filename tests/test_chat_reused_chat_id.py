"""재사용된 chat_id 로 찜 추천이 올 때 (2026-10-18 결정).

첫 메시지인지는 Backend 가 판단한다. 찜 추천 요청인데 checkpoint 에 대화가 남아 있으면
거절하지 않고 지운 뒤 시작한다. 예전 채팅방의 대화가 새 채팅방에 섞이면 안 된다.
"""

import asyncio
import logging

from langgraph.checkpoint.memory import InMemorySaver

from app.chat import controller
from app.chat.graph import build_graph
from app.chat.schemas import ChatRequest

WISHLIST_IDS = [str(code) for code in range(1, 11)]
OLD_STATE = {
    "messages": [
        {"role": "user", "content": "예전 채팅방의 질문"},
        {"role": "assistant", "content": "예전 채팅방의 답"},
    ],
    "shown_product_codes": ["0000001"],
    "pending_question": "confirm_result",
}


def _request(source_type, product_ids=()):
    return ChatRequest(
        chat_id=7,
        user_id=1,
        message="찜 기반 추천",
        source_type=source_type,
        product_ids=list(product_ids),
    )


def _graph_with_old_chat(request):
    graph = build_graph(InMemorySaver())
    config = controller.thread_config(request)
    asyncio.run(graph.aupdate_state(config, OLD_STATE, as_node="chat"))
    return graph, config


def test_wishlist_request_is_not_rejected_on_a_used_chat_id():
    request = _request("WISHLIST", WISHLIST_IDS)
    _graph_with_old_chat(request)

    assert asyncio.run(controller.find_pre_stream_error(request)) is None


def test_wishlist_request_clears_the_old_chat(caplog):
    request = _request("WISHLIST", WISHLIST_IDS)
    graph, config = _graph_with_old_chat(request)

    with caplog.at_level(logging.WARNING):
        asyncio.run(controller.clear_reused_chat_state(graph, request))

    values = asyncio.run(graph.aget_state(config)).values
    assert not values.get("messages")
    assert not values.get("shown_product_codes")
    assert values.get("pending_question") is None
    # 재사용 여부는 로그로 남기되, 예전 대화 내용은 남기지 않는다.
    assert "chat_id=7" in caplog.text
    assert "예전 채팅방" not in caplog.text


def test_general_request_keeps_the_conversation():
    request = _request("GENERAL")
    graph, config = _graph_with_old_chat(request)

    asyncio.run(controller.clear_reused_chat_state(graph, request))

    values = asyncio.run(graph.aget_state(config)).values
    assert values["messages"] == OLD_STATE["messages"]
    assert values["shown_product_codes"] == ["0000001"]


def test_wishlist_request_on_a_new_chat_id_does_nothing(caplog):
    graph = build_graph(InMemorySaver())
    request = _request("WISHLIST", WISHLIST_IDS)

    with caplog.at_level(logging.WARNING):
        asyncio.run(controller.clear_reused_chat_state(graph, request))

    assert "clearing" not in caplog.text
