"""POST /api/v1/sync-fitting 의 HTTP 계약 (단계1 §V1 동기 가상 피팅 요청).

Service 와 외부 서비스를 fake 로 바꿔 API 계층만 본다. 실제 DB, Runware, Pruna 는 부르지 않는다.
"""

from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.virtual_fitting import router
from app.virtual_fitting.exceptions import (
    FittingModelError,
    FittingTimeoutError,
    InvalidFittingCombinationError,
    ProductImageMissingError,
    ProductNotFoundError,
    UnsupportedSubCategoryError,
)
from app.virtual_fitting.models import VirtualFittingResult
from app.virtual_fitting.prompt import build_fitting_prompt
from app.virtual_fitting.providers.base import FittingResult
from app.virtual_fitting.providers.comment import MOCK_COMMENT, MOCK_TITLE, MockCommentProvider
from app.virtual_fitting.providers.runware import RunwareConfigError, RunwarePrunaProvider
from app.virtual_fitting.repositories.product_repository import ProductRepository
from app.virtual_fitting.service import VirtualFittingService
from tests.virtual_fitting_fixtures import make_bottom, make_top

KEY = "test-internal-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
URL = "/api/v1/sync-fitting"
BODY = {
    "user_image_url": "https://example.com/person.jpg",
    "products": [{"product_code": "123"}, {"product_code": "456"}],
}
RESULT = VirtualFittingResult(
    result_image_url="https://im.runware.ai/result.jpg",
    llm_comment="테스트 코멘트",
    llm_title="테스트 제목",
)


class StubService:
    """VirtualFittingService 대역. 받은 요청을 기록하고, 정해 둔 결과나 예외를 낸다."""

    def __init__(self, result=RESULT, error=None):
        self._result = result
        self._error = error
        self.requests = []

    def fit(self, request):
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._result


class FakeRepository:
    def __init__(self, *products):
        self._products = {int(p.product_code): p for p in products}

    def find_by_codes(self, product_codes):
        return [self._products[int(c)] for c in product_codes if int(c) in self._products]


class FakeFittingProvider:
    def __init__(self):
        self.inputs = []

    def try_on(self, fitting_input):
        self.inputs.append(fitting_input)
        return FittingResult(result_image_url="https://im.runware.ai/fake.jpg")


class FakeConnection:
    def __init__(self):
        self.autocommit = False
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    """Provider 가 실제 HTTP 를 보내려 하면 테스트가 실패한다."""

    def forbidden(*args, **kwargs):
        raise AssertionError("외부 HTTP 요청이 시도되었다")

    monkeypatch.setattr("app.virtual_fitting.providers.runware.urlopen", forbidden)
    monkeypatch.setattr("app.virtual_fitting.providers.pruna.urlopen", forbidden)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", KEY)
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)
    monkeypatch.setattr(settings, "CHECKPOINT_DSN", "")
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def use_service(monkeypatch):
    """라우터가 조립하는 Service 를 바꾼다. 조립 횟수는 opened 에 쌓인다."""
    opened = []

    def use(service):
        def open_stub():
            opened.append(1)
            return nullcontext(service)

        monkeypatch.setattr(router, "open_virtual_fitting_service", open_stub)
        return opened

    return use


def test_success_returns_the_common_shape(client, use_service):
    service = StubService()
    use_service(service)

    response = client.post(URL, json=BODY, headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "message": "fitting_succeeded",
        "data": {
            "result_image_url": "https://im.runware.ai/result.jpg",
            "llm_title": "테스트 제목",
            "llm_comment": "테스트 코멘트",
        },
    }


def test_request_is_passed_to_service_as_is(client, use_service):
    service = StubService()
    use_service(service)

    client.post(URL, json=BODY, headers=AUTH)

    [request] = service.requests
    assert request.user_image_url == "https://example.com/person.jpg"
    assert [p.product_code for p in request.products] == ["123", "456"]


def test_full_flow_with_real_service_and_fake_externals(client, use_service):
    top = make_top("1", sub_category="스웨트셔츠", image_url="https://img/top.jpg")
    bottom = make_bottom("2", sub_category="슬림 팬츠", image_url="https://img/bottom.jpg")
    provider = FakeFittingProvider()
    use_service(
        VirtualFittingService(FakeRepository(top, bottom), provider, MockCommentProvider())
    )

    response = client.post(
        URL,
        json={
            "user_image_url": "https://example.com/person.jpg",
            "products": [{"product_code": "2"}, {"product_code": "1"}],
        },
        headers=AUTH,
    )

    assert response.json() == {
        "code": 200,
        "message": "fitting_succeeded",
        "data": {
            "result_image_url": "https://im.runware.ai/fake.jpg",
            "llm_title": MOCK_TITLE,
            "llm_comment": MOCK_COMMENT,
        },
    }
    [fitting_input] = provider.inputs
    assert list(fitting_input.garment_image_urls) == ["https://img/top.jpg", "https://img/bottom.jpg"]
    assert fitting_input.prompt == build_fitting_prompt([top, bottom])


def test_missing_credentials_is_401_and_service_is_not_opened(client, use_service):
    opened = use_service(StubService())

    response = client.post(URL, json=BODY)

    assert response.status_code == 401
    assert response.json() == {"code": 401, "message": "unauthorized", "data": None}
    assert opened == []


@pytest.mark.parametrize(
    "body",
    [
        {"user_image_url": "https://example.com/p.jpg"},
        {"user_image_url": "https://example.com/p.jpg", "products": []},
        {
            "user_image_url": "https://example.com/p.jpg",
            "products": [{"product_code": "1"}, {"product_code": "2"}, {"product_code": "3"}],
        },
        {"user_image_url": "https://example.com/p.jpg", "products": [{"product_code": "abc"}]},
        {"user_image_url": "https://example.com/p.jpg", "products": [{}]},
        {"user_image_url": "not-a-url", "products": [{"product_code": "1"}]},
        {"products": [{"product_code": "1"}]},
    ],
    ids=[
        "no-products",
        "empty-products",
        "three-products",
        "bad-product-code",
        "no-product-code",
        "bad-user-image-url",
        "no-user-image-url",
    ],
)
def test_invalid_request_is_400_and_service_is_not_opened(client, use_service, body):
    service = StubService()
    opened = use_service(service)

    response = client.post(URL, json=body, headers=AUTH)

    assert response.status_code == 400
    assert response.json() == {"code": 400, "message": "invalid_request", "data": None}
    assert opened == []
    assert service.requests == []


@pytest.mark.parametrize(
    ("error", "status", "message"),
    [
        (ProductNotFoundError(["999"]), 404, "product_not_found"),
        (InvalidFittingCombinationError("같은 카테고리"), 422, "invalid_fitting_combination"),
        (ProductImageMissingError("1"), 422, "product_image_missing"),
        (UnsupportedSubCategoryError("없는 카테고리"), 500, "unsupported_sub_category"),
        (FittingModelError("boom"), 502, "fitting_model_failed"),
        (FittingTimeoutError("slow"), 504, "fitting_timeout"),
    ],
)
def test_domain_error_becomes_the_common_error_shape(client, use_service, error, status, message):
    use_service(StubService(error=error))

    response = client.post(URL, json=BODY, headers=AUTH)

    assert response.status_code == status
    assert response.json() == {"code": status, "message": message, "data": None}


def test_unexpected_error_is_500_internal_server_error(client, use_service):
    use_service(StubService(error=RuntimeError("어딘가 터짐")))

    response = client.post(URL, json=BODY, headers=AUTH)

    assert response.status_code == 500
    assert response.json() == {"code": 500, "message": "internal_server_error", "data": None}


def test_endpoint_is_in_the_openapi_surface():
    assert "post" in app.openapi()["paths"][URL]


# --- Service 조립과 DB connection 생명주기 ---


@pytest.fixture
def connection(monkeypatch):
    conn = FakeConnection()
    monkeypatch.setattr(router, "get_connection", lambda: conn)
    monkeypatch.setenv("RUNWARE_API_KEY", "test-runware-key")
    return conn


def test_service_is_assembled_from_the_v1_components(connection):
    with router.open_virtual_fitting_service() as service:
        assert isinstance(service, VirtualFittingService)
        assert isinstance(service.repository, ProductRepository)
        assert isinstance(service.fitting_provider, RunwarePrunaProvider)
        assert isinstance(service.comment_provider, MockCommentProvider)
        # 트랜잭션을 연 채로 Runware 응답을 기다리지 않는다.
        assert connection.autocommit is True
        assert connection.closed is False

    assert connection.closed is True


def test_connection_is_closed_when_the_request_fails(connection):
    with pytest.raises(FittingTimeoutError), router.open_virtual_fitting_service():
        raise FittingTimeoutError("slow")

    assert connection.closed is True


def test_connection_is_closed_when_assembly_fails(connection, monkeypatch):
    monkeypatch.delenv("RUNWARE_API_KEY")

    with pytest.raises(RunwareConfigError), router.open_virtual_fitting_service():
        pass

    assert connection.closed is True
