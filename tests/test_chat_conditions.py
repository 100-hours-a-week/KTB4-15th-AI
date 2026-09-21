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
