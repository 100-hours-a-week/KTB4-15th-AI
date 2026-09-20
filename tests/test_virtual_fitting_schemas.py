import pytest
from pydantic import ValidationError

from app.virtual_fitting.schemas import SyncFittingRequest, SyncFittingResponse

USER_IMAGE_URL = "https://example.com/users/15/body-no-bg.png"


def _request(*codes, **overrides):
    payload = {
        "user_image_url": USER_IMAGE_URL,
        "products": [{"product_code": code} for code in codes],
    }
    payload.update(overrides)
    return SyncFittingRequest(**payload)


def test_accepts_one_product():
    assert len(_request("1234333").products) == 1


def test_accepts_two_products():
    request = _request("1234333", "5678111")
    assert [p.product_code for p in request.products] == ["1234333", "5678111"]


def test_rejects_empty_products():
    with pytest.raises(ValidationError):
        _request()


def test_rejects_three_products():
    with pytest.raises(ValidationError):
        _request("1", "2", "3")


def test_rejects_missing_products():
    with pytest.raises(ValidationError):
        SyncFittingRequest(user_image_url=USER_IMAGE_URL)


def test_rejects_missing_user_image_url():
    with pytest.raises(ValidationError):
        SyncFittingRequest(products=[{"product_code": "1"}])


@pytest.mark.parametrize("url", ["", "ftp://example.com/a.png", "not-a-url"])
def test_rejects_invalid_user_image_url(url):
    with pytest.raises(ValidationError):
        _request("1", user_image_url=url)


def test_rejects_missing_product_code():
    with pytest.raises(ValidationError):
        SyncFittingRequest(user_image_url=USER_IMAGE_URL, products=[{}])


@pytest.mark.parametrize("code", ["", "abc", "12 3", "-1", "1" * 19])
def test_rejects_non_numeric_product_code(code):
    with pytest.raises(ValidationError):
        _request(code)


def test_response_default_code_and_message():
    response = SyncFittingResponse(
        data={
            "result_image_url": "https://example.com/result.jpg",
            "llm_title": "제목",
            "llm_comment": "코멘트",
        }
    )
    assert response.code == 200
    assert response.message == "fitting_succeeded"
    assert response.data.llm_title == "제목"
