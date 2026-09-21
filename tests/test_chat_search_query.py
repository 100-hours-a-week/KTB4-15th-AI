"""검색에 넘길 의미 질의.

사용자가 분위기를 말하지 않은 턴에서도 임베딩에 빈 문자열이 가지 않아야 한다.
"""

from app.chat.graph.nodes import _search_query


def test_uses_what_the_user_said():
    state = {
        "semantic_query": "소개팅에 입기 좋은 단정한 분위기",
        "conditions": {"category": "셔츠"},
    }
    assert _search_query(state) == "소개팅에 입기 좋은 단정한 분위기"


def test_falls_back_to_the_category_phrase():
    # "청바지 하나 보여줘" — 분위기를 말하지 않은 턴
    state = {"semantic_query": "", "conditions": {"category": "데님"}}
    assert _search_query(state) == "데님 팬츠"


def test_color_is_included_when_there_is_nothing_else():
    state = {"semantic_query": "", "conditions": {"color": "베이지", "category": "니트"}}
    assert _search_query(state) == "베이지 니트"


def test_whitespace_only_counts_as_empty():
    state = {"semantic_query": "   ", "conditions": {"category": "셔츠"}}
    assert _search_query(state) == "셔츠"


def test_nothing_to_build_from_stays_empty():
    # 조건도 분위기도 없는 턴. 여기서 지어내면 사용자가 말하지 않은 검색이 된다.
    assert _search_query({"semantic_query": "", "conditions": {}}) == ""
