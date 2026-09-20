"""실제 PostgreSQL 없이 ProductRepository 의 SQL/파라미터/row 변환을 검증한다."""

from app.virtual_fitting.models import FittingProduct
from app.virtual_fitting.repositories.product_repository import (
    ProductRepository,
    build_select_sql,
)


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    def execute(self, sql, params):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, rows):
        self.cursor_instance = FakeCursor(rows)

    def cursor(self):
        return self.cursor_instance


TOP_ROW = (
    4120937,
    "https://img.29cm.co.kr/item/top.jpg",
    "상의",
    "스웨트셔츠",
)


def test_find_by_codes_returns_products():
    repository = ProductRepository(FakeConnection([TOP_ROW]))

    products = repository.find_by_codes(["4120937"])

    assert products == [
        FittingProduct(
            product_code="4120937",
            image_url="https://img.29cm.co.kr/item/top.jpg",
            main_category="상의",
            sub_category="스웨트셔츠",
        )
    ]


def test_find_by_codes_uses_single_parameterized_query():
    connection = FakeConnection([TOP_ROW])

    ProductRepository(connection).find_by_codes(["4120937", "3", "4120937"])

    [(sql, params)] = connection.cursor_instance.executed
    assert sql == build_select_sql()
    assert "product_code = ANY(%s)" in sql
    assert params == ([3, 4120937],)


def test_find_by_codes_returns_empty_when_not_found():
    assert ProductRepository(FakeConnection([])).find_by_codes(["9"]) == []


def test_find_by_codes_skips_query_for_empty_input():
    connection = FakeConnection([])

    assert ProductRepository(connection).find_by_codes([]) == []
    assert connection.cursor_instance.executed == []


def test_select_only_needed_columns():
    sql = build_select_sql()
    for column in ("product_code", "image_url", "main_category", "sub_category"):
        assert column in sql
    assert "description_summary" not in sql
    assert "*" not in sql
