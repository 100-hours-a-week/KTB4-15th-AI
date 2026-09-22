"""실제 DB, Pruna, OpenAI 없이 Service 흐름을 fake 의존성으로 검증한다."""

import pytest

from app.virtual_fitting.exceptions import (
    FittingModelError,
    FittingTimeoutError,
    InvalidFittingCombinationError,
    ProductNotFoundError,
)
from app.virtual_fitting.models import VirtualFittingResult
from app.virtual_fitting.prompt import build_fitting_prompt
from app.virtual_fitting.providers.base import FittingProvider, FittingResult
from app.virtual_fitting.providers.comment import (
    MOCK_COMMENT,
    MOCK_TITLE,
    CommentProvider,
    MockCommentProvider,
)
from app.virtual_fitting.schemas import SyncFittingRequest
from app.virtual_fitting.service import VirtualFittingService
from tests.virtual_fitting_fixtures import make_bottom, make_top

USER_IMAGE_URL = "https://example.com/users/15/body.png"
RESULT_URL = "https://example.com/result.jpg"


class FakeRepository:
    """ProductRepository 대역. 조회 요청 코드를 기록한다."""

    def __init__(self, *products):
        self._products = {int(p.product_code): p for p in products}
        self.requested_codes = []

    def find_by_codes(self, product_codes):
        self.requested_codes.append(list(product_codes))
        return [self._products[int(c)] for c in product_codes if int(c) in self._products]


class FakeFittingProvider:
    def __init__(self, result_image_url=RESULT_URL, error=None):
        self._result_image_url = result_image_url
        self._error = error
        self.inputs = []

    def try_on(self, fitting_input):
        self.inputs.append(fitting_input)
        if self._error is not None:
            raise self._error
        return FittingResult(result_image_url=self._result_image_url)


class FakeCommentProvider:
    def __init__(self):
        self.comment_calls = []
        self.title_calls = []

    def generate_comment(self, products):
        self.comment_calls.append(list(products))
        return "fake comment"

    def generate_title(self, comment):
        self.title_calls.append(comment)
        return f"title of ({comment})"


def _request(*codes):
    return SyncFittingRequest(
        user_image_url=USER_IMAGE_URL,
        products=[{"product_code": code} for code in codes],
    )


def _service(repository, fitting_provider=None, comment_provider=None):
    return VirtualFittingService(
        repository,
        fitting_provider or FakeFittingProvider(),
        comment_provider or MockCommentProvider(),
    )


TOP = make_top("1", sub_category="후디", image_url="https://img/top.jpg")
BOTTOM = make_bottom("2", sub_category="데님 팬츠", image_url="https://img/bottom.jpg")


def test_request_product_codes_are_passed_to_repository():
    repository = FakeRepository(TOP, BOTTOM)

    _service(repository).fit(_request("2", "1"))

    assert repository.requested_codes == [["2", "1"]]


def test_garments_are_ordered_top_then_bottom_regardless_of_request_order():
    provider = FakeFittingProvider()

    _service(FakeRepository(TOP, BOTTOM), provider).fit(_request("2", "1"))

    [fitting_input] = provider.inputs
    assert list(fitting_input.garment_image_urls) == ["https://img/top.jpg", "https://img/bottom.jpg"]


def test_prompt_and_user_image_are_passed_to_fitting_provider():
    provider = FakeFittingProvider()

    _service(FakeRepository(TOP, BOTTOM), provider).fit(_request("2", "1"))

    [fitting_input] = provider.inputs
    assert fitting_input.person_image_url == USER_IMAGE_URL
    assert fitting_input.prompt == build_fitting_prompt([TOP, BOTTOM])
    assert fitting_input.prompt.index("hoodie") < fitting_input.prompt.index("jeans")


def test_single_product_is_passed_alone():
    provider = FakeFittingProvider()

    _service(FakeRepository(BOTTOM), provider).fit(_request("2"))

    [fitting_input] = provider.inputs
    assert list(fitting_input.garment_image_urls) == ["https://img/bottom.jpg"]
    assert fitting_input.prompt == build_fitting_prompt([BOTTOM])


def test_result_image_url_comes_from_fitting_provider():
    provider = FakeFittingProvider(result_image_url="https://example.com/other.jpg")

    result = _service(FakeRepository(TOP), provider).fit(_request("1"))

    assert result.result_image_url == "https://example.com/other.jpg"


def test_mock_comment_and_title_are_included_in_result():
    result = _service(FakeRepository(TOP, BOTTOM)).fit(_request("1", "2"))

    assert result == VirtualFittingResult(
        result_image_url=RESULT_URL, llm_comment=MOCK_COMMENT, llm_title=MOCK_TITLE
    )


def test_comment_provider_is_replaceable_and_title_is_built_from_comment():
    comment_provider = FakeCommentProvider()

    result = _service(FakeRepository(TOP, BOTTOM), comment_provider=comment_provider).fit(
        _request("2", "1")
    )

    assert comment_provider.comment_calls == [[TOP, BOTTOM]]
    assert comment_provider.title_calls == ["fake comment"]
    assert result.llm_comment == "fake comment"
    assert result.llm_title == "title of (fake comment)"


def test_fakes_satisfy_provider_interfaces():
    assert isinstance(FakeFittingProvider(), FittingProvider)
    assert isinstance(FakeCommentProvider(), CommentProvider)


def test_selection_errors_propagate_unchanged_and_skip_downstream_calls():
    provider = FakeFittingProvider()
    comment_provider = FakeCommentProvider()

    with pytest.raises(ProductNotFoundError) as not_found:
        _service(FakeRepository(TOP), provider, comment_provider).fit(_request("1", "999"))
    assert not_found.value.product_codes == ["999"]

    with pytest.raises(InvalidFittingCombinationError):
        _service(FakeRepository(TOP, make_top("3")), provider, comment_provider).fit(
            _request("1", "3")
        )

    assert provider.inputs == []
    assert comment_provider.comment_calls == []


@pytest.mark.parametrize(
    "error", [FittingModelError("boom"), FittingTimeoutError("slow")]
)
def test_fitting_provider_errors_propagate_unchanged(error):
    comment_provider = FakeCommentProvider()

    with pytest.raises(type(error)) as exc_info:
        _service(
            FakeRepository(TOP), FakeFittingProvider(error=error), comment_provider
        ).fit(_request("1"))

    assert exc_info.value is error
    assert comment_provider.comment_calls == []
