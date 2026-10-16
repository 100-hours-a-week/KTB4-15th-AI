"""찜 기반 추천 — 최빈값, 갈래 선별, 쿼리 조립, 후보 합치기, 추천 이유의 부분 실패.

DB 와 임베딩 API 없이 돌 수 있는 부분만 본다. LLM 은 가짜로 바꿔 끼운다.
"""

import asyncio
import random

import pytest

from app.chat import vocab
from app.clients.llm import LLMError
from app.recommendation import prompts
from app.recommendation import wishlist as wishlist_module
from app.recommendation.types import WishlistCommentError
from app.recommendation.wishlist import (
    build_branch_query,
    merge_candidates,
    most_common,
    pick_branches,
    write_reasons,
)


def _row(code, color, sub_category, description=None):
    return {
        "product_code": code,
        "color": color,
        "sub_category": sub_category,
        "image_description": description or f"설명{code}",
    }


# --- 최빈값 ---


def test_most_common_does_not_count_etc():
    values = ["기타", "기타", "기타", "블랙", "블랙"]
    assert most_common(values, random.Random(0)) == "블랙"


def test_most_common_is_none_when_only_etc_or_empty():
    assert most_common(["기타", "기타"], random.Random(0)) is None
    assert most_common([], random.Random(0)) is None
    assert most_common([None, ""], random.Random(0)) is None


def test_tie_is_broken_randomly_among_the_tied_values_only():
    values = ["블랙", "블랙", "네이비", "네이비", "화이트"]
    picked = {most_common(values, random.Random(seed)) for seed in range(30)}
    assert picked == {"블랙", "네이비"}


def test_same_seed_gives_same_pick_regardless_of_input_order():
    values = ["블랙", "네이비", "블랙", "네이비"]
    for seed in range(10):
        assert most_common(values, random.Random(seed)) == most_common(
            list(reversed(values)), random.Random(seed)
        )


# --- 갈래 선별 ---


def test_each_branch_takes_the_wishlist_items_with_its_value():
    wishlist = [
        _row(1, "블랙", "크루넥"),
        _row(2, "블랙", "크루넥"),
        _row(3, "네이비", "브이넥"),
        _row(4, "블랙", "트렌치/맥코트"),
        _row(5, "화이트", "긴소매 셔츠"),
    ]

    branches = pick_branches(wishlist, random.Random(0))

    assert branches == [
        {"kind": "color", "value": "블랙", "descriptions": ["설명1", "설명2", "설명4"]},
        # 크루넥·브이넥은 둘 다 "니트" 묶음이다
        {"kind": "category", "value": "니트", "descriptions": ["설명1", "설명2", "설명3"]},
    ]


def test_branch_without_a_countable_value_is_skipped():
    # 색은 전부 '기타', 카테고리는 오타라 '기타'(테님 재킷)가 아닌 것이 셔츠 하나
    wishlist = [
        _row(1, "기타", "테님 재킷"),
        _row(2, "기타", "테님 재킷"),
        _row(3, "기타", "긴소매 셔츠"),
    ]

    branches = pick_branches(wishlist, random.Random(0))

    assert [branch["kind"] for branch in branches] == ["category"]
    assert branches[0]["value"] == "셔츠"


def test_no_branch_when_nothing_can_be_counted():
    assert pick_branches([_row(1, "기타", "테님 재킷")], random.Random(0)) == []


# --- 쿼리 ---


def _where(sql: str) -> str:
    return sql.split("WHERE", 1)[1].split("ORDER BY", 1)[0]


def test_color_branch_filters_by_main_color_only():
    sql, params = build_branch_query(kind="color", value="블랙", top_k=3, exclude_codes=["1"])
    where = _where(sql)
    assert "color = %(color)s" in where
    # 서브 색(로고·배색)까지 보는 colors 배열로 걸지 않는다
    assert "colors" not in where
    assert params["color"] == "블랙"


def test_category_branch_expands_to_its_source_categories():
    sql, params = build_branch_query(kind="category", value="니트", top_k=3, exclude_codes=["1"])
    assert "sub_category = ANY(%(category_sources)s)" in _where(sql)
    assert params["category_sources"] == vocab.category_sources("니트")


@pytest.mark.parametrize("kind, value", [("color", "블랙"), ("category", "니트")])
def test_wishlist_and_sold_out_are_excluded_before_limit(kind, value):
    codes = [str(code) for code in range(10)]
    sql, params = build_branch_query(kind=kind, value=value, top_k=3, exclude_codes=codes)
    where = _where(sql)
    # WHERE 에서 빼야 LIMIT 3 이 남은 후보 안에서 잘린다
    assert "NOT (product_code = ANY(%(exclude_codes)s::bigint[]))" in where
    assert "NOT is_sold_out" in where
    assert "embedding IS NOT NULL" in where
    assert params["exclude_codes"] == codes
    assert params["top_k"] == 3


@pytest.mark.parametrize("kind", ["color", "category"])
def test_no_other_metadata_filter(kind):
    sql, _ = build_branch_query(kind=kind, value="블랙", top_k=3, exclude_codes=[])
    assert "price" not in _where(sql)


def test_values_never_enter_the_sql_text():
    sql, _ = build_branch_query(
        kind="color", value="블랙'; DROP TABLE products; --", top_k=3, exclude_codes=["1'--"]
    )
    assert "DROP TABLE" not in sql
    assert "1'--" not in sql


def test_query_returns_what_the_reason_prompt_needs():
    sql, _ = build_branch_query(kind="color", value="블랙", top_k=3, exclude_codes=[])
    select = sql.split("FROM", 1)[0]
    assert "image_description" in select
    assert "description_summary" in select  # _to_product 가 읽는다


# --- 후보 합치기 ---


def _branch(kind, value):
    return {"kind": kind, "value": value, "features": f"{value} 특징"}


def test_duplicate_keeps_one_and_gets_both_triggers():
    branches = [_branch("color", "블랙"), _branch("category", "니트")]
    results = [
        [{"product_code": 11}, {"product_code": 12}],
        [{"product_code": 12}, {"product_code": 13}],
    ]

    candidates = merge_candidates(branches, results)

    assert [c["row"]["product_code"] for c in candidates] == [11, 12, 13]
    assert [t["kind"] for t in candidates[1]["triggers"]] == ["color", "category"]
    assert [t["kind"] for t in candidates[0]["triggers"]] == ["color"]
    assert [t["kind"] for t in candidates[2]["triggers"]] == ["category"]


def test_trigger_carries_the_branch_features():
    candidates = merge_candidates([_branch("color", "블랙")], [[{"product_code": 1}]])
    assert candidates[0]["triggers"] == [
        {"kind": "color", "value": "블랙", "features": "블랙 특징"}
    ]


# --- 추천 이유 ---


class _FakeLLM:
    """설명에 failing 문자열이 들어 있는 상품만 실패한다."""

    def __init__(self, failing: set[str]):
        self.failing = failing

    async def complete_json(self, messages, *, max_output_tokens, reasoning_effort="none"):
        content = messages[-1]["content"]
        if any(word in content for word in self.failing):
            raise LLMError("boom")
        description = content.rsplit("\n", 1)[-1]
        return {prompts.REASON_KEY: f"{description} 이유"}


def _candidate(code, description):
    return {
        "row": {"product_code": code, "image_description": description},
        "triggers": [_branch("color", "블랙")],
    }


def test_failed_reason_is_left_empty_and_others_are_kept(monkeypatch):
    monkeypatch.setattr(wishlist_module, "llm", _FakeLLM({"설명B"}))
    candidates = [_candidate(1, "설명A"), _candidate(2, "설명B"), _candidate(3, "설명C")]

    reasons = asyncio.run(write_reasons(candidates))

    assert reasons == ["설명A 이유", "", "설명C 이유"]


def test_all_reasons_failing_raises_wishlist_comment_error(monkeypatch):
    monkeypatch.setattr(wishlist_module, "llm", _FakeLLM({"설명"}))
    candidates = [_candidate(1, "설명A"), _candidate(2, "설명B")]

    with pytest.raises(WishlistCommentError):
        asyncio.run(write_reasons(candidates))


def test_no_candidates_is_not_a_comment_failure(monkeypatch):
    monkeypatch.setattr(wishlist_module, "llm", _FakeLLM(set()))
    assert asyncio.run(write_reasons([])) == []


# --- 프롬프트 ---


def test_reason_prompt_carries_every_trigger_and_the_description():
    messages = prompts.reason_prompt(
        [_branch("color", "블랙"), _branch("category", "니트")], "블랙 컬러의 크루넥 니트다."
    )
    user = messages[-1]["content"]
    assert "색상 '블랙'" in user and "블랙 특징" in user
    assert "종류 '니트'" in user and "니트 특징" in user
    assert user.endswith("블랙 컬러의 크루넥 니트다.")


def test_features_prompt_lists_every_description():
    user = prompts.features_prompt(["설명1", "설명2", "설명3"])[-1]["content"]
    assert user.count("\n- ") == 3


def test_prompts_ask_for_the_keys_the_code_reads():
    assert f'"{prompts.FEATURES_KEY}"' in prompts.features_prompt(["x"])[0]["content"]
    assert f'"{prompts.REASON_KEY}"' in prompts.reason_prompt([_branch("color", "블랙")], "x")[0]["content"]
