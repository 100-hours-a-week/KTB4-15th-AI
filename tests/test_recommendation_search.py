"""검색 기반 추천 — 색상 쿼터와 쿼리 조립.

DB 와 임베딩 API 없이 돌 수 있는 두 순수 함수만 본다.
쿼터 사례는 2026-09-23 적재한 10건 표본의 실제 분포다. 규칙은 2026-09-30 "메인 우선"으로 바뀌었다.
"""

from app.recommendation.search import _to_product, apply_color_quota, build_query


def _rows(main: int, sub: int) -> list[dict]:
    # build_query 의 ORDER BY 와 같이 메인 → 서브, 각 무리 안에서 거리순
    rows = [{"id": f"m{i}", "is_main": True} for i in range(main)]
    rows += [{"id": f"s{i}", "is_main": False} for i in range(sub)]
    return rows


def _pick(main: int, sub: int, top_k: int = 3) -> list[str]:
    return [r["id"] for r in apply_color_quota(_rows(main, sub), top_k=top_k)]


def test_only_main_when_main_is_plenty():
    # QA "검정 상의" — 메인 일치가 충분하면 서브 일치는 나가지 않는다
    assert _pick(main=5, sub=5) == ["m0", "m1", "m2"]


def test_sub_fills_only_what_main_cannot():
    # 조건이 좁아 메인 일치가 1개뿐
    assert _pick(main=1, sub=5) == ["m0", "s0", "s1"]
    # 화이트: 메인 0 / 서브 5
    assert _pick(main=0, sub=5) == ["s0", "s1", "s2"]


def test_returns_fewer_than_top_k_when_both_are_short():
    # 그린·베이지: 메인 1 / 서브 1
    assert _pick(main=1, sub=1) == ["m0", "s0"]
    assert _pick(main=0, sub=0) == []


def test_no_color_condition_takes_top_k_by_distance():
    rows = [{"id": f"r{i}", "is_main": True} for i in range(5)]
    picked = apply_color_quota(rows, top_k=3)
    assert [r["id"] for r in picked] == ["r0", "r1", "r2"]


def _query(**overrides):
    kwargs = {"color": None, "category": None, "min_price": None, "max_price": None, "dislikes": [], "top_k": 3}
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


def test_price_range_uses_both_bounds():
    sql, params = _query(min_price=50000, max_price=59999)
    assert "price >= %(min_price)s" in sql
    assert "price <= %(max_price)s" in sql
    assert params["min_price"] == 50000
    assert params["max_price"] == 59999


def test_zero_min_price_is_still_a_bound():
    sql, params = _query(min_price=0, max_price=50000)
    assert "price >= %(min_price)s" in sql
    assert params["min_price"] == 0


def test_broad_category_filters_by_grouped_sub_categories():
    _, params = _query(category="상의")
    assert "반소매 티셔츠" in params["category_sources"]
    assert "긴소매 셔츠" in params["category_sources"]
    assert "카디건" in params["category_sources"]
    # 아우터와 하의는 빠진다
    assert "롱코트" not in params["category_sources"]
    assert "와이드 팬츠" not in params["category_sources"]


def test_broad_category_dislike_uses_the_same_groups():
    _, params = _query(dislikes=[{"field": "category", "value": "하의"}])
    assert "데님 팬츠" in params["dislike_category_0"]
    assert "쇼트" in params["dislike_category_0"]
    assert "반소매 티셔츠" not in params["dislike_category_0"]


def test_unknown_category_is_left_out_of_the_filter():
    # 운영 대화에 남은 옛 이름 등 — 0건 대신 카테고리 조건 없이 검색한다
    sql, params = _query(category="없는 이름")
    assert "sub_category = ANY" not in sql
    assert "category_sources" not in params
