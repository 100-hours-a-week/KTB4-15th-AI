"""찜 목록 기반 추천.

최근 찜 상품 10개로 최대 6개를 추천한다 (단계1 §7, 단계5 §3.3).
그래프의 wishlist 노드가 이 함수를 호출한다.
"""

from collections.abc import Callable

from .types import Product


async def recommend_from_wishlist(
    *,
    product_ids: list[str],
    on_progress: Callable[[str], None] | None = None,
) -> list[Product]:
    """최빈 색상 갈래와 최빈 카테고리 갈래로 나누어 추천한다.

    한 건당 LLM 호출이 최대 8회다 (공통 특징 추출 2 + 개인화 comment 6).
    comment 6회는 병렬로 생성한다 (2026-09-17 결정). 갈래별 공통 특징 추출 2회의
    병렬 여부는 미정이다.

    개별 상품의 comment 생성이 실패하면 그 상품의 llm_comment 를 빈 문자열로 두고
    나머지는 그대로 추천한다. 전부 실패한 경우에만 WishlistCommentError 를 던진다.

    on_progress 는 진행 라벨을 받는 선택 콜백이다. 화면에 그대로 나가는 문구이므로
    내부 단계 이름이 아니라 사용자용 라벨을 넘긴다.
    """
    raise NotImplementedError("찜 추천 구현 예정")
