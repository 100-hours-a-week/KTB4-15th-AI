import pytest

from app.virtual_fitting.exceptions import (
    InvalidFittingCombinationError,
    ProductImageMissingError,
    ProductNotFoundError,
)
from app.virtual_fitting.product_selection import (
    select_fitting_products,
    validate_and_sort_products,
)
from tests.virtual_fitting_fixtures import make_bottom, make_product, make_top


class FakeRepository:
    def __init__(self, *products):
        self._products = {int(p.product_code): p for p in products}

    def find_by_codes(self, product_codes):
        return [self._products[int(c)] for c in product_codes if int(c) in self._products]


# --- validate_and_sort_products ---


def test_single_top():
    top = make_top()
    assert validate_and_sort_products([top]) == [top]


def test_single_bottom():
    bottom = make_bottom()
    assert validate_and_sort_products([bottom]) == [bottom]


def test_top_and_bottom():
    top, bottom = make_top(), make_bottom()
    assert validate_and_sort_products([top, bottom]) == [top, bottom]


def test_sorts_bottom_first_input_to_top_then_bottom():
    top, bottom = make_top(), make_bottom()
    assert validate_and_sort_products([bottom, top]) == [top, bottom]


def test_rejects_top_and_top():
    with pytest.raises(InvalidFittingCombinationError):
        validate_and_sort_products([make_top("1"), make_top("2")])


def test_rejects_bottom_and_bottom():
    with pytest.raises(InvalidFittingCombinationError):
        validate_and_sort_products([make_bottom("1"), make_bottom("2")])


def test_rejects_three_or_more():
    with pytest.raises(InvalidFittingCombinationError):
        validate_and_sort_products([make_top("1"), make_bottom("2"), make_top("3")])


def test_rejects_empty():
    with pytest.raises(InvalidFittingCombinationError):
        validate_and_sort_products([])


def test_rejects_unsupported_category():
    with pytest.raises(InvalidFittingCombinationError):
        validate_and_sort_products([make_product("1", main_category="신발")])


@pytest.mark.parametrize("image_url", [None, "", "   "])
def test_rejects_missing_image_url(image_url):
    with pytest.raises(ProductImageMissingError) as exc_info:
        validate_and_sort_products([make_top("7", image_url=image_url)])
    assert exc_info.value.product_code == "7"


def test_error_codes_follow_api_spec():
    assert InvalidFittingCombinationError.status_code == 422
    assert InvalidFittingCombinationError.message == "invalid_fitting_combination"
    assert ProductNotFoundError.status_code == 404
    assert ProductNotFoundError.message == "product_not_found"


# --- select_fitting_products ---


def test_select_returns_products_sorted_top_then_bottom():
    top, bottom = make_top("1"), make_bottom("2")
    assert select_fitting_products(["2", "1"], FakeRepository(top, bottom)) == [top, bottom]


def test_select_single_product():
    top = make_top("1")
    assert select_fitting_products(["1"], FakeRepository(top)) == [top]


def test_select_raises_when_product_not_found():
    with pytest.raises(ProductNotFoundError) as exc_info:
        select_fitting_products(["1", "999"], FakeRepository(make_top("1")))
    assert exc_info.value.product_codes == ["999"]


def test_select_same_product_twice_is_rejected_as_same_category():
    with pytest.raises(InvalidFittingCombinationError):
        select_fitting_products(["1", "1"], FakeRepository(make_top("1")))


def test_select_leading_zero_code_matches_db_code():
    # DB 는 BIGINT 라 "0001" 도 1 번 상품이다.
    top = make_top("1")
    assert select_fitting_products(["0001"], FakeRepository(top)) == [top]
