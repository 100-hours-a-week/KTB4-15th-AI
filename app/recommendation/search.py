"""검색 기반 추천.

대화에서 누적한 조건으로 metadata filter + pgvector 유사도 검색을 수행한다.
그래프의 search 노드가 이 함수를 호출한다. 흐름에 갈래가 없으므로 노드로 쪼개지 않는다.

필터와 유사도 정렬을 한 쿼리에서 끝낸다 (2026-09-23 결정). 검색한 뒤에 거르면 몇 개가
남을지 몰라 top_k 를 얼마나 부풀려 뽑을지가 문제가 되는데, 한 쿼리면 그 문제가 없다.

색상 규칙 (2026-09-23 결정)
    포함  colors && ARRAY[색]          로고·배색으로만 겹쳐도 걸린다
    제외  color IS DISTINCT FROM 색     메인 색만 본다. "레드 싫어"는 옷 전반의 색을 말한다
    쿼터  메인 일치 top_k-1 + 서브 일치 1
          서브가 없으면 메인으로 채우고, 메인이 모자라도 서브를 늘리지 않는다 (top_k 미만 반환)

제외 조건에 NOT IN / <> 를 쓰지 않는다. NULL 과 비교하면 unknown 이 되어 행이 조용히
빠지고, 목록에 NULL 이 하나라도 끼면 NOT IN 은 0건을 돌려준다.
"""

import psycopg
from psycopg.rows import dict_row

from app.chat import vocab
from app.clients.llm import LLMError, llm
from app.config.database import get_async_pool

from .types import ITEM_TYPES, Product, RecommendationError

# 색 조건이 있을 때 서브 일치(로고·배색으로만 겹침)에 내주는 자리 수.
SUB_COLOR_SLOTS = 1

_COLUMNS = (
    "product_code, product_name, image_url, detail_url, color, price, main_category, "
    "description_summary"
)


def build_query(
    *,
    color: str | None,
    category: str | None,
    max_price: int | None,
    dislikes: list[dict],
    top_k: int,
) -> tuple[str, dict]:
    """조건을 SQL 한 벌로 만든다. 값은 전부 파라미터로 넘기고 문자열에 끼워 넣지 않는다.

    메인 일치와 서브 일치를 각각 거리순으로 top_k 개까지 가져온다. 몇 개씩 쓸지는
    apply_color_quota 가 정한다.
    """
    where = ["embedding IS NOT NULL"]
    params: dict = {"top_k": top_k}

    if color:
        where.append("colors && ARRAY[%(color)s]::text[]")
        params["color"] = color
    if category:
        where.append("sub_category = ANY(%(category_sources)s)")
        params["category_sources"] = vocab.CATEGORY_GROUPS.get(category, [])
    if max_price is not None:
        where.append("price <= %(max_price)s")
        params["max_price"] = max_price

    color_dislikes = [d["value"] for d in dislikes if d.get("field") == "color"]
    for index, value in enumerate(color_dislikes):
        key = f"dislike_color_{index}"
        where.append(f"color IS DISTINCT FROM %({key})s")
        params[key] = value

    category_dislikes = [d["value"] for d in dislikes if d.get("field") == "category"]
    for index, value in enumerate(category_dislikes):
        key = f"dislike_category_{index}"
        # sub_category 는 NOT NULL 이라 여기서는 unknown 이 생기지 않는다
        where.append(f"NOT (sub_category = ANY(%({key})s))")
        params[key] = vocab.CATEGORY_GROUPS.get(value, [])

    # 색 조건이 없으면 전부 한 무리(메인)로 본다
    is_main = "color IS NOT DISTINCT FROM %(color)s" if color else "TRUE"

    sql = f"""
WITH candidates AS (
    SELECT {_COLUMNS},
           {is_main} AS is_main,
           embedding <=> %(query_vector)s::vector AS distance
      FROM products
     WHERE {" AND ".join(where)}
), ranked AS (
    SELECT *, row_number() OVER (PARTITION BY is_main ORDER BY distance) AS rank
      FROM candidates
)
SELECT {_COLUMNS}, is_main, distance
  FROM ranked
 WHERE rank <= %(top_k)s
 ORDER BY is_main DESC, distance
"""
    return sql, params


# sabu: 쿼터의 경계 — top_k 가 SUB_COLOR_SLOTS 이하(예: 1)이면 메인 자리는 몇 개가 되고,
#       메인 일치가 충분히 있는데도 서브 일치 하나만 나가는 게 "메인 우선" 결정과 맞나?
def apply_color_quota(rows: list[dict], *, top_k: int, color_given: bool) -> list[dict]:
    """거리순으로 정렬된 후보에서 최종 top_k 를 고른다.

    rows 는 메인 → 서브, 각 무리 안에서는 거리순이어야 한다 (build_query 의 ORDER BY).
    """
    if not color_given:
        return rows[:top_k]
    main = [row for row in rows if row["is_main"]]
    sub = [row for row in rows if not row["is_main"]][:SUB_COLOR_SLOTS]
    return main[: top_k - len(sub)] + sub


# sabu: 목록 밖 대분류 — main_category 에 "상의"/"하의" 말고 다른 값(예: 나중에 분리할 "아우터")이
#       들어오면 이 함수는 어떻게 되고, 그 예외는 search_products 의 except 안에서 나나 밖에서 나나?
def _to_product(row: dict) -> Product:
    return Product(
        product_code=str(row["product_code"]),
        product_name=row["product_name"],
        image_url=row["image_url"],
        detail_url=row["detail_url"],
        color=row["color"] or "",
        price=row["price"],
        item_type=ITEM_TYPES[row["main_category"]],
        llm_comment=row["description_summary"] or "",
    )


# sabu: 빈 질의 — nodes._search_query 는 조건도 분위기도 없는 턴에 "" 를 돌려준다(테스트가 그걸 지킨다).
#       그 "" 가 여기 오면 임베딩 API 는 무엇을 하고, 사용자는 무엇을 보게 되나?
# sabu: 가격 제외 — FILTERABLE_FIELDS 에 max_price 가 있어서 {"field": "max_price", ...} 인 dislike 가
#       여기까지 넘어온다. 그런데 "가격이 싫다"는 무슨 조건인가? 지금 코드는 그걸 어떻게 처리하고 있나?
# sabu: 품절 — 단계3 은 품절 상태를 metadata filter 로 적어 두었는데, 이 쿼리는 is_sold_out 을 보나?
async def search_products(
    *,
    color: str | None,
    category: str | None,
    max_price: int | None,
    dislikes: list[dict],
    semantic_query: str,
    top_k: int,
) -> list[Product]:
    """metadata filter + pgvector 유사도 검색으로 Top K를 고른다.

    dislikes 는 [{"field": "color", "value": "레드"}] 형태이며,
    field 가 vocab.FILTERABLE_FIELDS 안에 있는 항목만 필터로 쓸 수 있다.

    llm_comment 는 사전 생성한 description_summary 를 그대로 사용하므로
    이 경로에서는 LLM 을 호출하지 않는다 (단계1 §7, 단계5 §3.2).
    임베딩 API 는 질의 벡터를 만들 때 한 번 부른다.
    """
    sql, params = build_query(
        color=color,
        category=category,
        max_price=max_price,
        dislikes=dislikes,
        top_k=top_k,
    )
    try:
        # 임베딩을 먼저 만든다. API 를 기다리는 동안 pool 의 connection 을 붙잡지 않기 위해서다.
        vector = await llm.embed(semantic_query)
        params["query_vector"] = str(vector)
        async with (
            get_async_pool().connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
    except (LLMError, psycopg.Error) as error:
        raise RecommendationError(f"상품 검색에 실패했습니다: {error}") from error

    picked = apply_color_quota(rows, top_k=top_k, color_given=bool(color))
    return [_to_product(row) for row in picked]
