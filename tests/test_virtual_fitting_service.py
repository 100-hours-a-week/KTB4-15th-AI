"""실제 DB, Pruna, OpenAI 없이 Service 흐름을 fake 의존성으로 검증한다."""

import logging

import pytest

from app.clients.s3 import ImageStorageError
from app.virtual_fitting.exceptions import (
    FittingImageStorageError,
    FittingModelError,
    FittingPostprocessError,
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
from app.virtual_fitting.service import (
    FALLBACK_COMMENT,
    FALLBACK_TITLE,
    VirtualFittingService,
)
from tests.virtual_fitting_fixtures import make_bottom, make_top

USER_IMAGE_URL = "https://example.com/users/15/body.png"
RESULT_URL = "https://example.com/result.jpg"
RESULT_KEY = "virtual-fitting/results/result.jpg"


class FakeRepository:
    """ProductRepository 대역. 조회 요청 코드를 기록한다."""

    def __init__(self, *products):
        self._products = {int(p.product_code): p for p in products}
        self.requested_codes = []

    def find_by_codes(self, product_codes):
        self.requested_codes.append(list(product_codes))
        return [self._products[int(c)] for c in product_codes if int(c) in self._products]


class FakeFittingProvider:
    def __init__(self, result_image_url=RESULT_URL, error=None, events=None):
        self._result_image_url = result_image_url
        self._error = error
        self._events = events if events is not None else []
        self.inputs = []

    def try_on(self, fitting_input):
        self._events.append("vton")
        self.inputs.append(fitting_input)
        if self._error is not None:
            raise self._error
        return FittingResult(result_image_url=self._result_image_url)


class FakeCommentProvider:
    def __init__(self, comment_error=None, title_error=None, events=None):
        self._comment_error = comment_error
        self._title_error = title_error
        self._events = events if events is not None else []
        self.comment_calls = []
        self.title_calls = []

    def generate_comment(self, result_image_url, description_summaries):
        self._events.append("comment")
        self.comment_calls.append((result_image_url, list(description_summaries)))
        if self._comment_error is not None:
            raise self._comment_error
        return "fake comment"

    def generate_title(self, comment):
        self._events.append("title")
        self.title_calls.append(comment)
        if self._title_error is not None:
            raise self._title_error
        return f"title of ({comment})"


class FakeImageStorage:
    def __init__(self, error=None, events=None):
        self.error = error
        self._events = events if events is not None else []
        self.urls = []

    def store_remote_image(self, url, key_prefix):
        self._events.append("s3")
        if self.error:
            raise self.error
        self.urls.append((url, key_prefix))
        return RESULT_KEY


def _request(*codes):
    return SyncFittingRequest(
        user_image_url=USER_IMAGE_URL,
        products=[{"product_code": code} for code in codes],
    )


def _service(repository, fitting_provider=None, comment_provider=None, image_storage=None):
    return VirtualFittingService(
        repository,
        fitting_provider or FakeFittingProvider(),
        comment_provider or MockCommentProvider(),
        image_storage or FakeImageStorage(),
    )


TOP = make_top(
    "1",
    sub_category="후디",
    image_url="https://img/top.jpg",
    description_summary="상의 설명 요약",
)
BOTTOM = make_bottom(
    "2",
    sub_category="데님 팬츠",
    image_url="https://img/bottom.jpg",
    description_summary="하의 설명 요약",
)


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


def test_result_image_is_stored_and_s3_key_is_returned():
    provider = FakeFittingProvider(result_image_url="https://example.com/other.jpg")

    result = _service(FakeRepository(TOP), provider).fit(_request("1"))

    assert result.result_image_key == RESULT_KEY


def test_only_the_provider_result_image_is_stored_under_the_fitting_prefix():
    storage = FakeImageStorage()
    provider = FakeFittingProvider(result_image_url="https://example.com/other.jpg")

    _service(FakeRepository(TOP, BOTTOM), provider, image_storage=storage).fit(_request("1", "2"))

    # 사용자 원본/상품 이미지가 아니라 provider 결과 이미지 하나만, 가상피팅 prefix 로 저장한다.
    assert storage.urls == [("https://example.com/other.jpg", "virtual-fitting/results")]


def test_mock_comment_and_title_are_included_in_result():
    result = _service(FakeRepository(TOP, BOTTOM)).fit(_request("1", "2"))

    assert result == VirtualFittingResult(
        result_image_key=RESULT_KEY, llm_comment=MOCK_COMMENT, llm_title=MOCK_TITLE
    )


def test_comment_uses_vton_result_url_and_description_summaries_in_garment_order():
    comment_provider = FakeCommentProvider()
    provider = FakeFittingProvider(result_image_url="https://im.runware.ai/image/vton.jpg")

    result = _service(
        FakeRepository(TOP, BOTTOM), provider, comment_provider=comment_provider
    ).fit(_request("2", "1"))

    # S3 key(RESULT_KEY)가 아니라 VTON 원본 URL 을, 상의 → 하의 순서의 설명과 함께 넘긴다.
    assert comment_provider.comment_calls == [
        ("https://im.runware.ai/image/vton.jpg", ["상의 설명 요약", "하의 설명 요약"])
    ]
    # title 에는 comment 만 넘긴다(이미지, 상품 설명 없음).
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


def test_s3_storage_failure_has_a_specific_domain_error():
    storage = FakeImageStorage(ImageStorageError("s3 unavailable"))

    with pytest.raises(FittingImageStorageError):
        _service(FakeRepository(TOP), image_storage=storage).fit(_request("1"))


# --- 실행 순서와 LLM fallback ---


def _run_with_events(comment_provider_kwargs=None, storage_error=None):
    events = []
    comment_provider = FakeCommentProvider(**(comment_provider_kwargs or {}), events=events)
    storage = FakeImageStorage(storage_error, events=events)
    service = _service(
        FakeRepository(TOP, BOTTOM),
        FakeFittingProvider(result_image_url=RESULT_URL, events=events),
        comment_provider,
        storage,
    )
    return service, comment_provider, storage, events


def test_llm_runs_before_s3_and_real_comment_and_title_are_returned():
    service, _, storage, events = _run_with_events()

    result = service.fit(_request("1", "2"))

    assert events == ["vton", "comment", "title", "s3"]
    assert storage.urls == [(RESULT_URL, "virtual-fitting/results")]
    assert result == VirtualFittingResult(
        result_image_key=RESULT_KEY,
        llm_comment="fake comment",
        llm_title="title of (fake comment)",
    )


def test_comment_failure_falls_back_skips_title_and_still_stores_the_image(caplog):
    service, comment_provider, storage, events = _run_with_events(
        {"comment_error": FittingPostprocessError("Runware LLM HTTP 오류: 500")}
    )

    with caplog.at_level(logging.ERROR, logger="app.virtual_fitting.service"):
        result = service.fit(_request("1", "2"))

    assert events == ["vton", "comment", "s3"]
    assert comment_provider.title_calls == []
    assert storage.urls == [(RESULT_URL, "virtual-fitting/results")]
    assert result == VirtualFittingResult(
        result_image_key=RESULT_KEY, llm_comment=FALLBACK_COMMENT, llm_title=FALLBACK_TITLE
    )
    assert "LLM postprocess failed" in caplog.text
    assert "FittingPostprocessError" in caplog.text  # traceback 이 남는다


def test_title_failure_discards_the_generated_comment_and_falls_back_as_a_pair():
    service, _, storage, events = _run_with_events(
        {"title_error": FittingPostprocessError("Runware LLM 응답이 완성되지 않았습니다")}
    )

    result = service.fit(_request("1", "2"))

    assert events == ["vton", "comment", "title", "s3"]
    assert storage.urls == [(RESULT_URL, "virtual-fitting/results")]
    assert result.llm_comment == FALLBACK_COMMENT != "fake comment"
    assert result.llm_title == FALLBACK_TITLE
    assert result.result_image_key == RESULT_KEY


@pytest.mark.parametrize(
    "comment_provider_kwargs",
    [{}, {"comment_error": FittingPostprocessError("llm down")}],
    ids=["llm-ok", "llm-fallback"],
)
def test_s3_failure_after_llm_is_not_hidden(comment_provider_kwargs):
    service, _, _, events = _run_with_events(
        comment_provider_kwargs, storage_error=ImageStorageError("s3 unavailable")
    )

    with pytest.raises(FittingImageStorageError):
        service.fit(_request("1", "2"))

    assert events[-1] == "s3"


@pytest.mark.parametrize(
    "error",
    [ValueError("description_summaries 가 비어 있다"), RuntimeError("bug")],
    ids=["invalid-db-data", "programming-error"],
)
def test_non_postprocess_errors_are_not_turned_into_fallback(error):
    service, _, storage, _ = _run_with_events({"comment_error": error})

    with pytest.raises(type(error)):
        service.fit(_request("1", "2"))

    assert storage.urls == []
