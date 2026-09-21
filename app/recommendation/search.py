"""검색 기반 추천.

대화에서 누적한 조건으로 metadata filter + pgvector 유사도 검색을 수행한다.
그래프의 search 노드가 이 함수를 호출한다. 흐름에 갈래가 없으므로 노드로 쪼개지 않는다.
"""

from .types import Product


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
    """
    # sabu: 제외 조건의 쿼리 모양 — dislikes 를 pgvector 검색에 어떻게 싣지?
    #       metadata filter 와 같은 한 번의 쿼리에 넣을 수 있나, 아니면 검색 뒤에 걸러내나?
    #       후자면 top_k 를 몇 개로 뽑아야 거르고도 3개가 남지?
    #       (PostgreSQL + pgvector 를 실제로 연결한 뒤에 정한다)
    raise NotImplementedError("검색 구현 예정")
