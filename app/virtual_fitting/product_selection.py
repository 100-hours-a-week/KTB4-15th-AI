"""요청된 product_code 를 조회하고 조합을 검증한 뒤 상의 → 하의 순으로 정렬한다."""

from typing import Sequence

from app.virtual_fitting.exceptions import (
    InvalidFittingCombinationError,
    ProductImageMissingError,
    ProductNotFoundError,
)
from app.virtual_fitting.models import BOTTOM_CATEGORY, TOP_CATEGORY, FittingProduct
from app.virtual_fitting.repositories.product_repository import ProductRepository

_CATEGORY_ORDER = {TOP_CATEGORY: 0, BOTTOM_CATEGORY: 1}
_MAX_PRODUCTS = 2


def validate_and_sort_products(
    products: Sequence[FittingProduct],
) -> list[FittingProduct]:
    """상의 1 / 하의 1 / 상의 1 + 하의 1 만 허용하고 상의 → 하의 순으로 반환한다."""
    if not 1 <= len(products) <= _MAX_PRODUCTS:
        raise InvalidFittingCombinationError(
            f"상품은 1개 이상 {_MAX_PRODUCTS}개 이하여야 합니다: {len(products)}개"
        )

    for product in products:
        if product.main_category not in _CATEGORY_ORDER:
            raise InvalidFittingCombinationError(
                f"가상피팅은 상의/하의만 지원합니다: "
                f"{product.product_code} ({product.main_category!r})"
            )

    categories = [product.main_category for product in products]
    if len(set(categories)) != len(categories):
        raise InvalidFittingCombinationError(
            f"같은 카테고리 상품을 함께 선택할 수 없습니다: {categories}"
        )

    for product in products:
        if not product.image_url or not product.image_url.strip():
            raise ProductImageMissingError(product.product_code)

    return sorted(products, key=lambda product: _CATEGORY_ORDER[product.main_category])


def select_fitting_products(
    product_codes: Sequence[str],
    repository: ProductRepository,
) -> list[FittingProduct]:
    """조회 → 존재 확인 → 조합 검증 → 상의 → 하의 정렬.

    반환 리스트의 순서가 Pruna 의 garment_images[i] ↔ prompt 의 Garment (i+1) 순서다.
    """
    found = {
        int(product.product_code): product
        for product in repository.find_by_codes(product_codes)
    }
    missing = [code for code in product_codes if int(code) not in found]
    if missing:
        raise ProductNotFoundError(missing)

    return validate_and_sort_products([found[int(code)] for code in product_codes])
