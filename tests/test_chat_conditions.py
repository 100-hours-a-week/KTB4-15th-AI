"""조건 병합. 대화가 여러 턴에 걸쳐 조건을 쌓는 부분이다."""

from app.chat.graph.nodes import _merge_conditions


def test_unmentioned_condition_survives():
    current = {"color": "블랙", "category": "셔츠"}
    merged = _merge_conditions(current, {"max_price": 50000}, None)
    assert merged == {"color": "블랙", "category": "셔츠", "max_price": 50000}


def test_null_does_not_erase_existing_value():
    merged = _merge_conditions({"color": "블랙"}, {"color": None}, None)
    assert merged["color"] == "블랙"


def test_mentioned_condition_is_overwritten():
    merged = _merge_conditions({"color": "블랙"}, {"color": "네이비"}, None)
    assert merged["color"] == "네이비"


def test_dislikes_accumulate_without_duplicates():
    first = _merge_conditions({}, None, [{"field": "color", "value": "레드"}])
    second = _merge_conditions(first, None, [{"field": "color", "value": "레드"}])
    third = _merge_conditions(second, None, [{"field": "style", "value": "오버핏"}])
    assert third["dislikes"] == [
        {"field": "color", "value": "레드"},
        {"field": "style", "value": "오버핏"},
    ]


def test_original_conditions_are_not_mutated():
    current = {"color": "블랙", "dislikes": [{"field": "color", "value": "레드"}]}
    _merge_conditions(current, {"color": "네이비"}, [{"field": "color", "value": "핑크"}])
    assert current == {"color": "블랙", "dislikes": [{"field": "color", "value": "레드"}]}


def test_new_price_range_overwrites_both_bounds():
    current = {"min_price": 50000, "max_price": 59999}
    merged = _merge_conditions(current, {"min_price": 0, "max_price": 30000}, None)
    assert merged == {"min_price": 0, "max_price": 30000}


def test_price_comes_from_rule_not_llm():
    from app.chat.graph.nodes import _said_conditions, postprocess_analysis

    raw = {"answer": "yes", "metadata": {"color": None, "max_price": 30000}}
    applied = postprocess_analysis(raw, "응, 근데 5만원 이하로")
    assert applied["metadata"] == {"color": None, "min_price": 0, "max_price": 50000}
    assert _said_conditions(applied)
    assert raw["metadata"] == {"color": None, "max_price": 30000}  # 원본은 그대로


def test_unknown_category_keeps_the_previous_one():
    from app.chat.graph.nodes import _said_conditions, postprocess_analysis

    raw = {"answer": "none", "metadata": {"category": "원피스"}}
    cleaned = postprocess_analysis(raw, "원피스 보여줘")
    assert cleaned["metadata"]["category"] is None
    assert not _said_conditions(cleaned)  # 목록 밖 값만 말했으면 조건을 말한 턴이 아니다
    merged = _merge_conditions({"category": "셔츠"}, cleaned["metadata"], None)
    assert merged["category"] == "셔츠"


def test_allowed_category_passes():
    from app.chat.graph.nodes import postprocess_analysis

    cleaned = postprocess_analysis({"metadata": {"category": "하의"}}, "하의 보여줘")
    assert cleaned["metadata"]["category"] == "하의"
