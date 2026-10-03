"""검색에 넘길 의미 질의.

색 + 세부 카테고리 + 분위기를 잇는다 (2026-10-03, QA 10번). 카테고리 문구는 둘 다 없을 때만 쓴다.
사용자가 분위기를 말하지 않은 턴에서도 임베딩에 빈 문자열이 가지 않아야 한다.
"""

from app.chat.graph.nodes import _merge_query_words, _search_query


def test_joins_every_part_the_user_said():
    state = {
        "detail_category": "자켓",
        "mood": "데이트",
        "conditions": {"color": "블랙", "category": "아우터"},
    }
    assert _search_query(state) == "블랙 재킷 데이트"  # 자켓 → 재킷


def test_category_phrase_is_left_out_when_detail_exists():
    # "슬랙스 팬츠" 로 붙이면 슬랙스 신호가 묻힌다 (0/10)
    state = {"detail_category": "슬랙스", "mood": "출근", "conditions": {"category": "팬츠"}}
    assert _search_query(state) == "슬랙스 출근"


def test_mood_only_does_not_add_the_category_phrase():
    state = {"mood": "데이트", "conditions": {"color": "블랙", "category": "아우터"}}
    assert _search_query(state) == "블랙 데이트"


def test_user_spelling_is_converted_only_in_the_query():
    state = {"detail_category": "후드티", "conditions": {"category": "스웨트/후디"}}
    assert _search_query(state) == "후디"
    assert state["detail_category"] == "후드티"  # 상태는 사용자 표기 그대로


def test_falls_back_to_the_category_phrase():
    # "청바지 하나 보여줘" — 세부 카테고리도 분위기도 말하지 않은 턴
    state = {"detail_category": "", "mood": "", "conditions": {"category": "데님"}}
    assert _search_query(state) == "데님 팬츠"


def test_whitespace_only_counts_as_empty():
    state = {"detail_category": "  ", "mood": "   ", "conditions": {"category": "셔츠"}}
    assert _search_query(state) == "셔츠"


def test_nothing_to_build_from_stays_empty():
    # 조건도 분위기도 없는 턴. 여기서 지어내면 사용자가 말하지 않은 검색이 된다.
    assert _search_query({"conditions": {}}) == ""


def _merge(previous: dict, raw: dict, category: str | None):
    state = {"conditions": {"category": previous.get("category")}, **previous}
    return _merge_query_words(state, raw, {"category": category})


def test_detail_survives_a_turn_that_does_not_change_category():
    # "데이트할 때 입을 자켓" → "5만원 이하로"
    previous = {"category": "아우터", "detail_category": "자켓", "mood": "데이트"}
    assert _merge(previous, {}, "아우터") == ("자켓", "데이트")


def test_detail_is_cleared_when_category_changes_but_mood_stays():
    # "와이드 팬츠" 보다가 "셔츠도"
    previous = {"category": "팬츠", "detail_category": "와이드", "mood": "출근"}
    assert _merge(previous, {}, "셔츠") == ("", "출근")


def test_new_words_overwrite_in_the_same_turn_as_a_category_change():
    previous = {"category": "팬츠", "detail_category": "와이드", "mood": "출근"}
    raw = {"detail_category": "린넨", "mood": "여행"}
    assert _merge(previous, raw, "셔츠") == ("린넨", "여행")
