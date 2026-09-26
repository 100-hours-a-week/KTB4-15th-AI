"""추천 결과 형식과 예외."""

from typing import TypedDict


class RecommendationError(RuntimeError):
    """검색 실패. 스트림 안에서는 recommendation_search_failed로 나간다."""


class WishlistCommentError(RecommendationError):
    """찜 추천의 개인화 comment를 하나도 만들지 못한 경우.

    기존 REST 명세의 502 wishlist_comment_generation_failed에 대응한다.
    상품 일부만 실패한 경우에는 이 예외를 던지지 않는다. 실패한 상품은
    llm_comment를 빈 문자열로 채워 추천 결과에 포함한다 (2026-09-17 결정).
    """


class Product(TypedDict):
    """단계1 §7의 products 계약."""

    product_code: str
    product_name: str
    image_url: str
    detail_url: str
    color: str
    price: int
    # 상품의 대분류. DB 의 main_category("상의"/"하의")를 ITEM_TYPES 로 바꿔 보낸다 (2026-09-27 Backend 합의).
    item_type: str
    # 검색 기반은 사전 생성한 description_summary를 그대로 쓴다.
    # 찜 기반은 요청 시점에 생성하며, 그 상품의 생성이 실패하면 빈 문자열이 온다.
    llm_comment: str


# DB main_category → API item_type
ITEM_TYPES = {"상의": "TOP", "하의": "BOTTOM"}
