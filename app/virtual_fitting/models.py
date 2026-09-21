"""가상피팅에서 사용하는 상품 도메인 모델."""

from dataclasses import dataclass

TOP_CATEGORY = "상의"
BOTTOM_CATEGORY = "하의"


@dataclass(frozen=True)
class FittingProduct:
    """AI PostgreSQL products 테이블에서 가상피팅에 필요한 컬럼만 담는다."""

    product_code: str
    image_url: str | None
    main_category: str
    sub_category: str
