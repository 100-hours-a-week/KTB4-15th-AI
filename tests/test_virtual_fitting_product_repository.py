"""실제 PostgreSQL 없이 ProductRepository 의 SQL/파라미터/row 변환을 검증한다."""

from contextlib import contextmanager

import psycopg
import pytest

from app.virtual_fitting.exceptions import FittingDatabaseError, ProductNotFoundError
from app.virtual_fitting.models import FittingProduct
from app.virtual_fitting.product_selection import select_fitting_products
from app.virtual_fitting.repositories.product_repository import (
    ProductRepository,
    build_select_sql,
)

SECRET_URL = "postgresql://user:secret-password@db-host:5432/db"


class FakeCursor:
    def __init__(self, rows, execute_error=None, fetch_error=None):
        self._rows = rows
        self._execute_error = execute_error
        self._fetch_error = fetch_error
        self.executed = []

    def execute(self, sql, params):
        self.executed.append((sql, params))
        if self._execute_error is not None:
            raise self._execute_error

    def fetchall(self):
        if self._fetch_error is not None:
            raise self._fetch_error
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, rows, execute_error=None, fetch_error=None, cursor_error=None):
        self.cursor_instance = FakeCursor(rows, execute_error, fetch_error)
        self._cursor_error = cursor_error

    def cursor(self):
        if self._cursor_error is not None:
            raise self._cursor_error
        return self.cursor_instance


class FakePool:
    def __init__(self, connection):
        self.connection_instance = connection

    @contextmanager
    def connection(self):
        yield self.connection_instance


def repository(connection):
    return ProductRepository(FakePool(connection))


TOP_ROW = (
    4120937,
    "https://img.29cm.co.kr/item/top.jpg",
    "상의",
    "스웨트셔츠",
)


def test_find_by_codes_returns_products():
    product_repository = repository(FakeConnection([TOP_ROW]))

    products = product_repository.find_by_codes(["4120937"])

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

    repository(connection).find_by_codes(["4120937", "3", "4120937"])

    [(sql, params)] = connection.cursor_instance.executed
    assert sql == build_select_sql()
    assert "product_code = ANY(%s)" in sql
    assert params == ([3, 4120937],)


def test_find_by_codes_returns_empty_when_not_found():
    assert repository(FakeConnection([])).find_by_codes(["9"]) == []


def test_find_by_codes_skips_query_for_empty_input():
    connection = FakeConnection([])

    assert repository(connection).find_by_codes([]) == []
    assert connection.cursor_instance.executed == []


def test_select_only_needed_columns():
    sql = build_select_sql()
    for column in ("product_code", "image_url", "main_category", "sub_category"):
        assert column in sql
    assert "description_summary" not in sql
    assert "*" not in sql


# --- DB 오류와 "조회 결과 없음"은 다르다 ---


@pytest.mark.parametrize(
    "failure",
    [
        {"execute_error": psycopg.OperationalError(f"connection lost {SECRET_URL}")},
        {"fetch_error": psycopg.InterfaceError(f"cursor closed {SECRET_URL}")},
        {"cursor_error": psycopg.OperationalError(f"server closed {SECRET_URL}")},
        {"execute_error": psycopg.errors.UndefinedTable(f"relation missing {SECRET_URL}")},
    ],
    ids=["execute", "fetch", "cursor", "undefined-table"],
)
def test_psycopg_error_becomes_a_database_error_without_details(failure):
    product_repository = repository(FakeConnection([TOP_ROW], **failure))

    with pytest.raises(FittingDatabaseError) as exc_info:
        product_repository.find_by_codes(["4120937"])

    error = exc_info.value
    assert (error.status_code, error.message) == (500, "database_error")
    assert "secret-password" not in str(error) + repr(error)
    assert "db-host" not in str(error) + repr(error)
    assert isinstance(error.__cause__, psycopg.Error)


def test_non_database_errors_are_not_hidden_as_database_errors():
    product_repository = repository(FakeConnection([], execute_error=RuntimeError("bug")))

    with pytest.raises(RuntimeError, match="bug"):
        product_repository.find_by_codes(["1"])


def test_invalid_product_code_is_a_programming_error_not_a_database_error():
    with pytest.raises(ValueError):
        repository(FakeConnection([])).find_by_codes(["abc"])


def test_successful_query_with_no_rows_is_not_a_database_error():
    product_repository = repository(FakeConnection([]))

    assert product_repository.find_by_codes(["999"]) == []
    with pytest.raises(ProductNotFoundError) as exc_info:
        select_fitting_products(["999"], product_repository)
    assert not isinstance(exc_info.value, FittingDatabaseError)
    assert (exc_info.value.status_code, exc_info.value.message) == (404, "product_not_found")
