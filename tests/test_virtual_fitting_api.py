"""POST /api/v1/sync-fitting 의 HTTP 계약 (단계1 §V1 동기 가상 피팅 요청).

Service 와 외부 서비스를 fake 로 바꿔 API 계층만 본다. 실제 DB, Runware, Pruna 는 부르지 않는다.
"""

import logging
from contextlib import contextmanager, nullcontext

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.clients.s3 import S3ImageStorage
from app.config import settings
from app.main import app
from app.virtual_fitting import router
from app.virtual_fitting.exceptions import (
    FittingDatabaseError,
    FittingImageStorageError,
    FittingModelError,
    FittingPostprocessError,
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
from app.virtual_fitting.providers.runware_comment import RunwareCommentProvider
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
    result_image_key="virtual-fitting/results/result.jpg",
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


class FakePool:
    def __init__(self, connection):
        self._connection = connection

    @contextmanager
    def connection(self):
        try:
            yield self._connection
        finally:
            self._connection.close()


class FakeImageStorage:
    def store_remote_image(self, url, key_prefix):
        return "virtual-fitting/results/fake.jpg"


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    """Provider 가 실제 HTTP 를 보내려 하면 테스트가 실패한다."""

    def forbidden(*args, **kwargs):
        raise AssertionError("외부 HTTP 요청이 시도되었다")

    monkeypatch.setattr("app.virtual_fitting.providers.runware.urlopen", forbidden)
    monkeypatch.setattr("app.virtual_fitting.providers.pruna.urlopen", forbidden)
    monkeypatch.setattr("app.virtual_fitting.providers.runware_comment.urlopen", forbidden)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", KEY)
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)
    monkeypatch.setattr(settings, "CHECKPOINT_DSN", "")
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "test-vton-key")
    monkeypatch.setenv("RUNWARE_LLM_API_KEY", "test-llm-key")
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
            "result_image_key": "virtual-fitting/results/result.jpg",
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
        VirtualFittingService(
            FakeRepository(top, bottom), provider, MockCommentProvider(), FakeImageStorage()
        )
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
            "result_image_key": "virtual-fitting/results/fake.jpg",
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
        (FittingPostprocessError("llm failed"), 500, "fitting_postprocess_failed"),
        (FittingDatabaseError("db failed"), 500, "database_error"),
        (FittingImageStorageError("s3 failed"), 500, "fitting_image_storage_failed"),
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
    monkeypatch.setattr(router, "get_connection_pool", lambda: FakePool(conn))
    monkeypatch.setattr(router, "S3ImageStorage", FakeImageStorage)
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "test-vton-key")
    return conn


def test_service_is_assembled_from_the_v1_components(connection):
    with router.open_virtual_fitting_service() as service:
        assert isinstance(service, VirtualFittingService)
        assert isinstance(service.repository, ProductRepository)
        assert isinstance(service.fitting_provider, RunwarePrunaProvider)
        assert isinstance(service.comment_provider, MockCommentProvider)
        # description_summary 가 DB 에 생기기 전까지 production 은 Mock 을 쓴다 (A-2 에서 교체).
        assert not isinstance(service.comment_provider, RunwareCommentProvider)
        # Service 조립만으로는 pool에서 connection을 빌리지 않는다.
        assert connection.autocommit is False
        assert connection.closed is False

    assert connection.closed is False


def test_connection_is_not_acquired_when_failure_happens_before_query(connection):
    with pytest.raises(FittingTimeoutError), router.open_virtual_fitting_service():
        raise FittingTimeoutError("slow")

    assert connection.closed is False


def test_connection_is_not_acquired_when_assembly_fails(connection, monkeypatch):
    monkeypatch.delenv("RUNWARE_VTON_API_KEY")

    with pytest.raises(RunwareConfigError), router.open_virtual_fitting_service():
        pass

    assert connection.closed is False


# --- PostgreSQL 오류 (실제 DB 없이 fake connection 으로) ---

SECRET_URL = "postgresql://user:secret-password@db-host:5432/db"
LIBPQ_MESSAGE = 'connection to server at "db-host" (10.0.0.5), port 5432 failed: Connection refused'
DB_ERROR_BODY = {"code": 500, "message": "database_error", "data": None}
TOP_ROW = (1, "https://img/top.jpg", "상의", "스웨트셔츠")
BOTTOM_ROW = (2, "https://img/bottom.jpg", "하의", "슬림 팬츠")


class RowsCursor:
    def __init__(self, rows, error=None):
        self._rows = rows
        self._error = error

    def execute(self, sql, params):
        if self._error is not None:
            raise self._error

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class DbConnection(FakeConnection):
    """실제 ProductRepository 가 쓰는 cursor 를 가진 fake. query 오류나 결과 행을 정할 수 있다."""

    def __init__(self, rows=(), error=None):
        super().__init__()
        self._rows = list(rows)
        self._error = error

    def cursor(self):
        return RowsCursor(self._rows, self._error)


class RaisingFittingProvider:
    def __init__(self, error):
        self._error = error

    def try_on(self, fitting_input):
        raise self._error


@pytest.fixture
def real_assembly(monkeypatch):
    """실제 open_virtual_fitting_service 를 쓰되 DB connection 과 VTON Provider 만 바꾼다."""
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "test-vton-key")

    def use(connection, provider=None):
        monkeypatch.setattr(router, "get_connection_pool", lambda: FakePool(connection))
        monkeypatch.setattr(router, "S3ImageStorage", FakeImageStorage)
        if provider is not None:
            monkeypatch.setattr(router, "RunwarePrunaProvider", lambda: provider)
        return connection

    return use


def _fitting_request(client, *codes):
    body = {
        "user_image_url": "https://example.com/person.jpg",
        "products": [{"product_code": code} for code in codes],
    }
    return client.post(URL, json=body, headers=AUTH)


def test_connection_failure_is_500_database_error_without_details(client, monkeypatch, caplog):
    monkeypatch.setenv("DATABASE_URL", SECRET_URL)
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "test-vton-key")
    attempts = []

    def failing_pool():
        attempts.append(1)
        raise psycopg.OperationalError(LIBPQ_MESSAGE)

    monkeypatch.setattr(router, "get_connection_pool", failing_pool)

    with caplog.at_level(logging.ERROR):
        response = _fitting_request(client, "1")

    assert response.status_code == 500
    assert response.json() == DB_ERROR_BODY
    for leaked in (
        "secret-password",
        "db-host",
        "10.0.0.5",
        "5432",
        "Connection refused",
        SECRET_URL,
    ):
        assert leaked not in response.text
    assert len(attempts) == 1  # 재시도 없음
    # 서버 로그에는 원인이 남되(traceback 체인), DATABASE_URL 과 비밀번호는 없다.
    assert "database_error" in caplog.text
    assert "OperationalError" in caplog.text
    assert "secret-password" not in caplog.text
    assert SECRET_URL not in caplog.text


def test_connection_failure_never_tries_to_close_a_missing_connection(client, monkeypatch):
    closes = []
    monkeypatch.setattr(FakeConnection, "close", lambda self: closes.append(self))

    def failing():
        raise psycopg.OperationalError(LIBPQ_MESSAGE)

    monkeypatch.setattr(router, "get_connection_pool", failing)

    response = _fitting_request(client, "1")

    assert response.json() == DB_ERROR_BODY
    assert closes == []


def test_query_failure_is_500_database_error_and_the_connection_is_closed(
    client, real_assembly, caplog
):
    connection = real_assembly(DbConnection(error=psycopg.OperationalError(LIBPQ_MESSAGE)))

    with caplog.at_level(logging.ERROR):
        response = _fitting_request(client, "1")

    assert response.status_code == 500
    assert response.json() == DB_ERROR_BODY
    assert "db-host" not in response.text and "Connection refused" not in response.text
    assert connection.closed is True
    assert "OperationalError" in caplog.text


def test_query_that_succeeds_with_no_rows_is_404_not_a_database_error(client, real_assembly):
    connection = real_assembly(DbConnection(rows=[]))

    response = _fitting_request(client, "999")

    assert response.status_code == 404
    assert response.json() == {"code": 404, "message": "product_not_found", "data": None}
    assert connection.closed is True


def test_successful_lookup_runs_the_whole_flow_and_closes_the_connection(client, real_assembly):
    provider = FakeFittingProvider()
    connection = real_assembly(DbConnection(rows=[TOP_ROW, BOTTOM_ROW]), provider)

    response = _fitting_request(client, "2", "1")

    assert response.status_code == 200
    assert response.json()["data"] == {
        "result_image_key": "virtual-fitting/results/fake.jpg",
        "llm_title": MOCK_TITLE,
        "llm_comment": MOCK_COMMENT,
    }
    [fitting_input] = provider.inputs
    assert list(fitting_input.garment_image_urls) == ["https://img/top.jpg", "https://img/bottom.jpg"]
    assert connection.closed is True


def test_missing_s3_bucket_is_s3_config_error_before_any_external_call(
    client, real_assembly, monkeypatch, caplog
):
    provider = FakeFittingProvider()
    connection = real_assembly(DbConnection(rows=[TOP_ROW, BOTTOM_ROW]), provider)
    # real_assembly 는 S3ImageStorage 를 fake 로 바꾸므로, 이 테스트만 진짜 클래스로 되돌린다.
    monkeypatch.setattr(router, "S3ImageStorage", S3ImageStorage)
    monkeypatch.setattr(settings, "S3_BUCKET", "")

    def forbidden_client():
        raise AssertionError("S3 설정이 없으면 boto3 client 도 만들면 안 된다")

    monkeypatch.setattr("app.clients.s3.get_s3_client", forbidden_client)

    with caplog.at_level(logging.ERROR):
        response = _fitting_request(client, "1", "2")

    assert response.status_code == 500
    assert response.json() == {
        "code": 500,
        "message": "fitting_image_storage_failed",
        "data": {"reason_code": "S3_CONFIG_ERROR"},
    }
    assert provider.inputs == []  # 외부 가상피팅 API 는 부르지 않았다
    assert connection.closed is False  # 상품 조회(DB)까지도 가지 않았다
    assert "S3_BUCKET" in caplog.text


@pytest.mark.parametrize(
    ("error", "status", "message"),
    [
        (FittingModelError("boom"), 502, "fitting_model_failed"),
        (FittingTimeoutError("slow"), 504, "fitting_timeout"),
        (RuntimeError("어딘가 터짐"), 500, "internal_server_error"),
    ],
    ids=["vton-error", "vton-timeout", "unexpected"],
)
def test_the_connection_is_closed_when_fitting_fails(
    client, real_assembly, error, status, message
):
    connection = real_assembly(
        DbConnection(rows=[TOP_ROW, BOTTOM_ROW]), RaisingFittingProvider(error)
    )

    response = _fitting_request(client, "1", "2")

    assert response.status_code == status
    assert response.json() == {"code": status, "message": message, "data": None}
    assert connection.closed is True


@pytest.mark.parametrize(
    ("rows", "status", "message"),
    [
        (
            [TOP_ROW, (3, "https://img/top2.jpg", "상의", "후디")],
            422,
            "invalid_fitting_combination",
        ),
        ([(1, None, "상의", "스웨트셔츠")], 422, "product_image_missing"),
        ([(1, "https://img/x.jpg", "상의", "없는 카테고리")], 500, "unsupported_sub_category"),
    ],
    ids=["combination", "image-missing", "unsupported-sub-category"],
)
def test_existing_domain_errors_keep_their_meaning(client, real_assembly, rows, status, message):
    real_assembly(DbConnection(rows=rows), FakeFittingProvider())
    codes = [str(row[0]) for row in rows]

    response = _fitting_request(client, *codes)

    assert response.status_code == status
    assert response.json() == {"code": status, "message": message, "data": None}
