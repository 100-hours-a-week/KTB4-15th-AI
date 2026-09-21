"""Recommendation Module.

검색 기반·찜 기반 추천 로직. 외부 서비스를 감싸는 clients/ 와 달리 우리가 직접
구현하는 코드다.

두 추천 모두 LangGraph 노드가 아니라 일반 함수다 (단계4 §4). 그래프의 search /
wishlist 노드는 이 함수를 호출하고 결과를 SSE 로 흘리는 일만 한다. 검색 중간값
(query vector, 후보 목록)이 ChatState 와 checkpointer 에 쌓이지 않도록 하기 위해서다.
"""

from .search import search_products
from .types import Product, RecommendationError, WishlistCommentError
from .wishlist import recommend_from_wishlist

__all__ = [
    "Product",
    "RecommendationError",
    "WishlistCommentError",
    "recommend_from_wishlist",
    "search_products",
]
