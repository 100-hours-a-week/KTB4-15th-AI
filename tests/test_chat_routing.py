"""입구 분기와 intent 분기.

분기 함수는 노드 이름이 아니라 path map 의 **키**를 돌려줘야 한다. LangGraph 가 그
반환값을 map 에서 다시 찾기 때문이다. 이 규칙이 깨지면 그래프를 실제로 돌릴 때
KeyError 로 터진다.
"""

import pytest

from app.chat.graph.routing import ENTRY_ROUTES, ROUTES, entry_route, route

INTENTS = ("chat", "recommend", "reject_with_conditions", "reject_only", "confirm")


@pytest.mark.parametrize("source_type", ["GENERAL", "WISHLIST"])
def test_entry_route_returns_a_key_of_the_path_map(source_type):
    assert entry_route({"source_type": source_type}) in ENTRY_ROUTES


@pytest.mark.parametrize("intent", INTENTS)
def test_route_returns_a_key_of_the_path_map(intent):
    assert route({"intent": intent}) in ROUTES


def test_chip_click_goes_to_wishlist():
    assert ENTRY_ROUTES[entry_route({"source_type": "WISHLIST"})] == "wishlist"


def test_normal_message_goes_to_analyze():
    assert ENTRY_ROUTES[entry_route({"source_type": "GENERAL"})] == "analyze"


def test_missing_source_type_is_treated_as_general():
    assert ENTRY_ROUTES[entry_route({})] == "analyze"


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("chat", "chat"),
        ("recommend", "summarize"),
        ("reject_with_conditions", "summarize"),
        ("reject_only", "ask_change"),
        ("confirm", "search"),
    ],
)
def test_intent_maps_to_node(intent, expected):
    assert ROUTES[route({"intent": intent})] == expected


def test_every_route_target_is_a_real_node():
    # builder 가 등록하는 노드 이름과 어긋나면 그래프 컴파일이 깨진다.
    assert set(ROUTES.values()) <= {"chat", "summarize", "ask_change", "search"}
    assert set(ENTRY_ROUTES.values()) <= {"analyze", "wishlist"}
