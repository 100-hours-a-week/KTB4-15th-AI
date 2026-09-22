from collections.abc import Sequence
from typing import Any

import psycopg

from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.models import FittingProduct

_SELECT_COLUMNS = (
    "product_code",
    "image_url",
    "main_category",
    "sub_category",
)


def build_select_sql() -> str:
    columns = ", ".join(_SELECT_COLUMNS)
    return f"SELECT {columns} FROM products WHERE product_code = ANY(%s)"


def _row_to_product(row: Sequence[Any]) -> FittingProduct:
    product_code, image_url, main_category, sub_category = row
    return FittingProduct(
        product_code=str(product_code),
        image_url=image_url,
        main_category=main_category,
        sub_category=sub_category,
    )


class ProductRepository:
    """connection 의 생명주기(commit/close)는 호출하는 쪽이 관리한다."""

    def __init__(self, connection: Any):
        self._connection = connection

    def find_by_codes(self, product_codes: Sequence[str]) -> list[FittingProduct]:
        """존재하는 상품만 반환한다. 순서는 보장하지 않으며 누락 판단은 호출자가 한다."""
        codes = sorted({int(code) for code in product_codes})
        if not codes:
            return []
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(build_select_sql(), (codes,))
                rows = cursor.fetchall()
        except psycopg.Error as error:
            raise FittingDatabaseError("AI PostgreSQL 상품 조회에 실패했습니다.") from error
        return [_row_to_product(row) for row in rows]
