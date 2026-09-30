"""입구 분기와 analyze 이후 분기.

분기 함수는 노드 이름이 아니라 path map 의 **키**를 돌려줘야 한다. LangGraph 가 그
반환값을 map 에서 다시 찾기 때문이다. 이 규칙이 깨지면 그래프를 실제로 돌릴 때
KeyError 로 터진다.

analyze 이후 갈 곳은 2026-09-30 에 합의한 표를 그대로 따른다. 이 파일이 그 표의 사본이다.
"""

import pytest

from app.chat.graph.routing import ENTRY_ROUTES, ROUTES, entry_route, route

PENDING = ("confirm_summary", "ask_change", None)
ANSWERS = ("yes", "no", "none")


@pytest.mark.parametrize("source_type", ["GENERAL", "WISHLIST"])
def test_entry_route_returns_a_key_of_the_path_map(source_type):
    assert entry_route({"source_type": source_type}) in ENTRY_ROUTES


@pytest.mark.parametrize("pending", PENDING)
@pytest.mark.parametrize("answer", ANSWERS)
@pytest.mark.parametrize("said_conditions", [True, False])
def test_route_returns_a_key_of_the_path_map(pending, answer, said_conditions):
    state = {"pending_question": pending, "answer": answer, "said_conditions": said_conditions}
    assert route(state) in ROUTES


def test_chip_click_goes_to_wishlist():
    assert ENTRY_ROUTES[entry_route({"source_type": "WISHLIST"})] == "wishlist"


def test_normal_message_goes_to_analyze():
    assert ENTRY_ROUTES[entry_route({"source_type": "GENERAL"})] == "analyze"


def test_missing_source_type_is_treated_as_general():
    assert ENTRY_ROUTES[entry_route({})] == "analyze"


@pytest.mark.parametrize("pending", PENDING)
@pytest.mark.parametrize("answer", ANSWERS)
def test_saying_conditions_always_resummarizes(pending, answer):
    """조건을 말했으면 예/아니오와 상관없이 기존 조건의 수정으로 보고 다시 요약한다."""
    state = {"pending_question": pending, "answer": answer, "said_conditions": True}
    assert ROUTES[route(state)] == "summarize"


@pytest.mark.parametrize(
    ("pending", "answer", "expected"),
    [
        ("confirm_summary", "yes", "search"),
        ("confirm_summary", "no", "ask_change"),
        ("confirm_summary", "none", "chat"),
        ("ask_change", "yes", "search"),
        ("ask_change", "no", "chat"),  # 같은 질문을 되풀이하지 않는다
        ("ask_change", "none", "chat"),
        (None, "yes", "chat"),  # 묻지 않았으면 답은 보지 않는다
        (None, "no", "chat"),
        (None, "none", "chat"),  # 조건 없는 추천 요청은 chat 이 조건을 되묻는다
    ],
)
def test_answer_without_conditions(pending, answer, expected):
    state = {"pending_question": pending, "answer": answer, "said_conditions": False}
    assert ROUTES[route(state)] == expected


@pytest.mark.parametrize("answer", ["maybe", None, "YES", ""])
@pytest.mark.parametrize("pending", PENDING)
def test_unknown_answer_goes_to_chat(pending, answer):
    """LLM 은 목록 밖 값·null 을 낼 수 있다. KeyError 없이 chat 으로 간다."""
    state = {"pending_question": pending, "answer": answer, "said_conditions": False}
    assert ROUTES[route(state)] == "chat"


def test_missing_fields_go_to_chat():
    assert ROUTES[route({})] == "chat"


def test_every_route_target_is_a_real_node():
    # builder 가 등록하는 노드 이름과 어긋나면 그래프 컴파일이 깨진다.
    assert set(ROUTES.values()) <= {"chat", "summarize", "ask_change", "search"}
    assert set(ENTRY_ROUTES.values()) <= {"analyze", "wishlist"}
