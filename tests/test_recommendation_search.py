"""검색 기반 추천 — 색상 쿼터와 쿼리 조립.

DB 와 임베딩 API 없이 돌 수 있는 두 순수 함수만 본다.
쿼터 사례는 2026-09-23 적재한 10건 표본의 실제 분포다.
"""

from app.recommendation.search import _to_product, apply_color_quota, build_query


def _rows(main: int, sub: int) -> list[dict]:
    # build_query 의 ORDER BY 와 같이 메인 → 서브, 각 무리 안에서 거리순
    rows = [{"id": f"m{i}", "is_main": True} for i in range(main)]
    rows += [{"id": f"s{i}", "is_main": False} for i in range(sub)]
    return rows


def _pick(main: int, sub: int, top_k: int = 3) -> list[str]:
    return [r["id"] for r in apply_color_quota(_rows(main, sub), top_k=top_k, color_given=True)]


def test_two_main_and_one_sub_when_both_are_plenty():
    assert _pick(main=5, sub=5) == ["m0", "m1", "s0"]


def test_main_fills_the_sub_slot_when_there_is_no_sub():
    # 블랙: 메인 6 / 서브 0
    assert _pick(main=6, sub=0) == ["m0", "m1", "m2"]


def test_sub_does_not_grow_when_main_is_short():
    # 화이트: 메인 0 / 서브 5 — 서브로 채우지 않고 1개만 내보낸다
    assert _pick(main=0, sub=5) == ["s0"]


def test_returns_fewer_than_top_k_when_both_are_short():
    # 그린·베이지: 메인 1 / 서브 1
    assert _pick(main=1, sub=1) == ["m0", "s0"]
    # 네이비: 메인 2 / 서브 0
    assert _pick(main=2, sub=0) == ["m0", "m1"]


def test_no_color_condition_takes_top_k_by_distance():
    rows = [{"id": f"r{i}", "is_main": True} for i in range(5)]
    picked = apply_color_quota(rows, top_k=3, color_given=False)
    assert [r["id"] for r in picked] == ["r0", "r1", "r2"]


def _query(**overrides):
    kwargs = {"color": None, "category": None, "max_price": None, "dislikes": [], "top_k": 3}
    kwargs.update(overrides)
    return build_query(**kwargs)


def test_color_is_included_through_the_colors_array():
    sql, params = _query(color="화이트")
    assert "colors && ARRAY[%(color)s]" in sql
    assert params["color"] == "화이트"


def test_color_dislike_is_null_safe_and_main_only():
    sql, params = _query(dislikes=[{"field": "color", "value": "레드"},
                                   {"field": "color", "value": "핑크"}])
    assert "color IS DISTINCT FROM %(dislike_color_0)s" in sql
    assert "color IS DISTINCT FROM %(dislike_color_1)s" in sql
    assert (params["dislike_color_0"], params["dislike_color_1"]) == ("레드", "핑크")
    # NULL 과 만나면 행을 조용히 버리는 비교를 쓰지 않는다
    assert "NOT IN" not in sql
    assert "<>" not in sql
    # 제외는 배색까지 보지 않는다 — 레드 로고가 있는 블랙 셔츠는 남는다
    assert "NOT (colors" not in sql


def test_values_never_enter_the_sql_text():
    sql, _ = _query(color="블랙'; DROP TABLE products; --",
                    dislikes=[{"field": "color", "value": "레드'--"}])
    assert "DROP TABLE" not in sql
    assert "레드'" not in sql


def test_category_expands_to_its_source_categories():
    sql, params = _query(category="쇼츠")
    assert "sub_category = ANY(%(category_sources)s)" in sql
    assert params["category_sources"] == ["쇼트"]


def test_rows_without_embedding_are_never_candidates():
    sql, _ = _query()
    assert "embedding IS NOT NULL" in sql


def _row(main_category: str) -> dict:
    return {
        "product_code": 4097486, "product_name": "팬츠", "image_url": "i", "detail_url": "d",
        "color": "블랙", "price": 25420, "main_category": main_category,
        "description_summary": "요약",
    }


def test_item_type_is_sent_as_top_or_bottom():
    # Backend 합의 (2026-09-27): DB 의 한글 대분류가 아니라 TOP / BOTTOM 으로 보낸다
    assert _to_product(_row("상의"))["item_type"] == "TOP"
    assert _to_product(_row("하의"))["item_type"] == "BOTTOM"
