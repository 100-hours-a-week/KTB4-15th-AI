"""찜 목록 기반 추천.

최근 찜 상품 10개로 최대 6개를 추천한다 (단계1 §7, 단계5 §3.3).
그래프의 wishlist 노드가 이 함수를 호출한다.

흐름 (2026-10-15 결정)
    1. 찜 10개의 색·카테고리·설명을 읽는다
    2. 최빈 색상과 최빈 카테고리를 센다. 색은 DB color(이미 대분류), 카테고리는
       vocab.normalize_category(sub_category) 10종이다. 둘 다 '기타'는 세지 않고,
       동률이면 그중 하나를 랜덤으로 고른다.
    3. 갈래마다 그 값을 가진 찜 상품의 설명으로 공통 특징 질의를 만든다 (LLM, 두 갈래 병렬)
    4. 질의를 임베딩해 갈래당 3개를 찾는다
         색 갈래       color = 최빈 색 (메인 색만)
         카테고리 갈래 sub_category 가 최빈 카테고리에 속함
       단계5 §3.3 은 찜 추천에 metadata filter 를 쓰지 않는다고 적었지만 2026-10-15 에 바꿨다.
       "자주 찜한 블랙"을 계기로 추천하면 사용자는 블랙 옷을 기대한다.
       찜 10개와 품절은 WHERE 에서 뺀다. 질의가 찜 상품 설명에서 나와 그대로 두면 찜 상품이
       상위를 차지하고, 검색한 뒤에 빼면 3개보다 적게 남는다.
    5. 두 갈래에 다 나온 상품은 하나만 남기고 두 갈래의 계기를 다 준다
    6. 고른 상품에 추천 이력을 남긴다 (가격 변동 알림이 이 표시로 고른다)
    7. 상품마다 추천 이유를 LLM 으로 쓴다 (병렬)
"""

import asyncio
import logging
import random
from collections import Counter
from collections.abc import Callable

import psycopg
from psycopg.rows import dict_row

from app.chat import vocab
from app.clients.llm import LLMError, llm
from app.config import settings
from app.config.database import get_async_pool

from . import prompts
from .search import _COLUMNS, MARK_RECOMMENDED_SQL, _to_product
from .types import Product, RecommendationError, WishlistCommentError

logger = logging.getLogger(__name__)

# 최빈값을 셀 때 빼는 값. 색은 대분류 밖의 색(로즈골드 등), 카테고리는 묶음에 없는 sub_category 다.
UNCOUNTED = "기타"

WISHLIST_SQL = """
SELECT product_code, color, sub_category, image_description
  FROM products
 WHERE product_code = ANY(%(codes)s::bigint[])
"""


def most_common(values: list[str | None], rng: random.Random) -> str | None:
    """가장 많이 나온 값. '기타'와 빈 값은 세지 않는다. 셀 값이 없으면 None.

    동률이면 그중 하나를 랜덤으로 고른다 (2026-10-15 결정). 후보를 정렬한 뒤 고르므로
    같은 시드면 입력 순서와 상관없이 같은 값이 나온다.
    """
    counts = Counter(value for value in values if value and value != UNCOUNTED)
    if not counts:
        return None
    top = max(counts.values())
    return rng.choice(sorted(value for value, count in counts.items() if count == top))


def _category_of(row: dict) -> str:
    return vocab.normalize_category(row["sub_category"])


# sabu: 1개짜리 갈래 — 찜 10개의 색이 전부 다르면 색 갈래의 descriptions 는 몇 개인가?
#       설명 하나에서 "여러 설명에 함께 나타나는 특징"은 무엇이 되나?
def pick_branches(wishlist: list[dict], rng: random.Random) -> list[dict]:
    """최빈 색상 갈래와 최빈 카테고리 갈래. 셀 값이 없는 갈래는 만들지 않는다.

    갈래 하나는 {"kind", "value", "descriptions"} 다. descriptions 는 그 값을 가진
    찜 상품의 image_description 이다.
    """
    branches = []
    for kind, key in (("color", lambda row: row["color"]), ("category", _category_of)):
        value = most_common([key(row) for row in wishlist], rng)
        if value is None:
            continue
        descriptions = [row["image_description"] for row in wishlist if key(row) == value]
        branches.append({"kind": kind, "value": value, "descriptions": descriptions})
    return branches


def build_branch_query(
    *, kind: str, value: str, top_k: int, exclude_codes: list[str]
) -> tuple[str, dict]:
    """한 갈래의 검색 SQL. 값은 전부 파라미터로 넘기고 문자열에 끼워 넣지 않는다.

    색 갈래는 메인 색(color)만 본다. colors(서브 포함)로 걸면 로고만 블랙인 옷이
    "자주 찜한 블랙"으로 나간다 (검색 추천의 2026-10-01 "메인 우선"과 같은 이유).
    """
    where = [
        "embedding IS NOT NULL",
        "NOT is_sold_out",
        # product_code 는 bigint PK 라 NULL 이 없다. 찜 목록은 문자열로 오므로 형을 맞춘다
        "NOT (product_code = ANY(%(exclude_codes)s::bigint[]))",
    ]
    params: dict = {"top_k": top_k, "exclude_codes": list(exclude_codes)}
    if kind == "color":
        where.append("color = %(color)s")
        params["color"] = value
    else:
        where.append("sub_category = ANY(%(category_sources)s)")
        params["category_sources"] = vocab.category_sources(value)

    sql = f"""
SELECT {_COLUMNS}, image_description
  FROM products
 WHERE {" AND ".join(where)}
 ORDER BY embedding <=> %(query_vector)s::vector
 LIMIT %(top_k)s
"""
    return sql, params


def merge_candidates(branches: list[dict], results: list[list[dict]]) -> list[dict]:
    """갈래별 검색 결과를 합친다.

    두 갈래에 다 나온 상품은 처음 나온 자리에 하나만 남기고, 계기(triggers)에 두 갈래를
    다 담는다 (2026-10-15 결정). 순서는 갈래 순서(색 → 카테고리), 갈래 안에서는 거리순이다.
    branches 는 features 까지 채워진 갈래다.
    """
    merged: dict[int, dict] = {}
    for branch, rows in zip(branches, results, strict=True):
        trigger = {"kind": branch["kind"], "value": branch["value"], "features": branch["features"]}
        for row in rows:
            candidate = merged.setdefault(row["product_code"], {"row": row, "triggers": []})
            candidate["triggers"].append(trigger)
    return list(merged.values())


# sabu: 지워진 상품 — 10-06 재적재로 products 는 50,000 → 44,075행이 됐다. 그 사이 빠진 상품을
#       찜해 둔 사용자가 오면 여기서 몇 행이 나오고, 누가 그걸 알아채나? (단계1 의 404 wishlist_product_not_found)
# sabu: 형식 — product_ids 에 "abc" 가 오면 Backend 는 어떤 code 를 받나? 단계1 은 무엇을 약속했나?
async def _fetch_wishlist(product_ids: list[str]) -> list[dict]:
    try:
        async with (
            get_async_pool().connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(WISHLIST_SQL, {"codes": list(product_ids)})
            return await cursor.fetchall()
    except psycopg.Error as error:
        raise RecommendationError(f"찜 상품을 읽지 못했습니다: {error}") from error


# sabu: 한 갈래만 실패 — 두 갈래 중 하나의 호출만 실패하면 recommend_from_wishlist 의 gather 는
#       어떻게 되고, 사용자는 무엇을 보게 되나? 남은 한 갈래로라도 추천하는 게 나은가?
async def _extract_features(branch: dict) -> str:
    result = await llm.complete_json(
        prompts.features_prompt(branch["descriptions"]),
        max_output_tokens=settings.WISHLIST_FEATURES_MAX_OUTPUT_TOKENS,
    )
    return result[prompts.FEATURES_KEY]


# sabu: 키 없는 답 — LLM 이 {"reason": ...} 이 아닌 다른 키로 답하면 이 함수는 무엇을 던지고,
#       write_reasons 는 그걸 "실패한 상품 하나"로 보나 "전체 실패"로 보나? 같은 모양의 자리가 이 파일에 또 있나?
async def _write_reason(candidate: dict) -> str:
    result = await llm.complete_json(
        prompts.reason_prompt(candidate["triggers"], candidate["row"]["image_description"]),
        max_output_tokens=settings.WISHLIST_REASON_MAX_OUTPUT_TOKENS,
    )
    return result[prompts.REASON_KEY]


async def write_reasons(candidates: list[dict]) -> list[str]:
    """상품마다 추천 이유를 병렬로 쓴다 (2026-09-17 결정).

    실패한 상품은 빈 문자열로 두고 나머지는 그대로 추천한다. 전부 실패했을 때만
    WishlistCommentError 를 던진다.
    """
    results = await asyncio.gather(
        *(_write_reason(candidate) for candidate in candidates), return_exceptions=True
    )
    reasons = []
    for candidate, result in zip(candidates, results, strict=True):
        if isinstance(result, LLMError):
            # 일부 실패는 위로 올라가지 않아 controller 의 로그에 남지 않는다. 여기서 남긴다.
            logger.warning(
                "wishlist reason failed: product_code=%s error=%s",
                candidate["row"]["product_code"],
                result,
            )
            reasons.append("")
        elif isinstance(result, BaseException):
            raise result
        else:
            reasons.append(result)

    if candidates and not any(reasons):
        raise WishlistCommentError("추천 이유를 하나도 만들지 못했습니다")
    return reasons


# sabu: 표시 뒤 실패 — 추천 이력은 이유를 쓰기 전에 남긴다. 이유가 전부 실패해 사용자가 아무것도
#       못 보면 그 상품들의 표시는 맞는 기록인가? (search.py 의 "추천 표시 실패" 마커와 같은 질문의 반대편)
async def recommend_from_wishlist(
    *,
    product_ids: list[str],
    on_progress: Callable[[str], None] | None = None,
    rng: random.Random | None = None,
) -> list[Product]:
    """최빈 색상 갈래와 최빈 카테고리 갈래로 나누어 추천한다.

    한 건당 LLM 호출이 최대 8회다 (공통 특징 추출 2 + 추천 이유 6). 둘 다 병렬이라
    사용자가 기다리는 것은 LLM 두 번 분량이다.

    셀 수 있는 색·카테고리가 하나도 없거나 찾은 상품이 없으면 빈 목록을 돌려준다
    (단계1 의 422 는 폐기하고 검색 추천과 같이 빈 배열로 통일했다).

    on_progress 는 진행 라벨을 받는 선택 콜백이다. 라벨 문구와 부를 시점은 Backend 와
    합의하지 않아 아직 부르지 않는다.
    rng 는 동률을 고를 때 쓴다. 테스트에서 시드를 고정하려고 받는다.
    """
    rng = rng or random.Random()
    wishlist = await _fetch_wishlist(product_ids)
    branches = pick_branches(wishlist, rng)
    if not branches:
        return []

    # 두 갈래는 서로 독립이라 병렬로 부른다 (2026-10-15 결정)
    features = await asyncio.gather(*(_extract_features(branch) for branch in branches))
    for branch, text in zip(branches, features, strict=True):
        branch["features"] = text

    try:
        # 임베딩을 먼저 만든다. API 를 기다리는 동안 pool 의 connection 을 붙잡지 않기 위해서다.
        vectors = await asyncio.gather(*(llm.embed(text) for text in features))
        async with (
            get_async_pool().connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            results = []
            for branch, vector in zip(branches, vectors, strict=True):
                sql, params = build_branch_query(
                    kind=branch["kind"],
                    value=branch["value"],
                    top_k=settings.WISHLIST_TOP_K_PER_BRANCH,
                    exclude_codes=product_ids,
                )
                params["query_vector"] = str(vector)
                await cursor.execute(sql, params)
                results.append(await cursor.fetchall())

            candidates = merge_candidates(branches, results)
            if candidates:
                codes = [candidate["row"]["product_code"] for candidate in candidates]
                await cursor.execute(MARK_RECOMMENDED_SQL, {"codes": codes})
    except (LLMError, psycopg.Error) as error:
        raise RecommendationError(f"찜 기반 상품 검색에 실패했습니다: {error}") from error

    reasons = await write_reasons(candidates)

    products = []
    for candidate, reason in zip(candidates, reasons, strict=True):
        product = _to_product(candidate["row"])
        product["recommended_reason"] = reason
        products.append(product)
    return products
