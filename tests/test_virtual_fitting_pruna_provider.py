"""실제 Pruna API 를 호출하지 않고 fake opener 로 Provider 를 검증한다."""

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from app.virtual_fitting.exceptions import FittingModelError, FittingTimeoutError
from app.virtual_fitting.providers.base import (
    FittingInput,
    FittingProvider,
    FittingResult,
)
from app.virtual_fitting.providers.pruna import (
    PRUNA_ENDPOINT,
    PrunaConfigError,
    PrunaDirectProvider,
    build_payload,
)

API_KEY = "test-secret-key"
RESULT_URL = "https://api.pruna.ai/v1/predictions/delivery/xezq/abc/output.jpg"


class FakeResponse:
    def __init__(self, payload):
        self._body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self._body


class FakeOpener:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        if self._error is not None:
            raise self._error
        return FakeResponse(self._result)


def _input():
    return FittingInput(
        person_image_url="https://example.com/user.png",
        garment_image_urls=["https://img/top.jpg", "https://img/bottom.jpg"],
        prompt="Garment 1: 후디.\nGarment 2: 데님 팬츠.\nDress the person with the provided garments.",
    )


def _provider(opener, **kwargs):
    return PrunaDirectProvider(API_KEY, opener=opener, **kwargs)


def _http_error(code, body=b'{"error": "boom"}'):
    return HTTPError(PRUNA_ENDPOINT, code, "err", {}, io.BytesIO(body))


# --- request 변환 ---


def test_payload_maps_fields_to_pruna_names():
    fitting_input = _input()
    assert build_payload(fitting_input) == {
        "input": {
            "person_image": "https://example.com/user.png",
            "garment_images": ["https://img/top.jpg", "https://img/bottom.jpg"],
            "prompt": fitting_input.prompt,
        }
    }


def test_payload_keeps_garment_image_order():
    urls = ["https://img/1.jpg", "https://img/2.jpg"]
    payload = build_payload(FittingInput("https://u", urls, "p"))
    assert payload["input"]["garment_images"] == urls
    reversed_payload = build_payload(FittingInput("https://u", urls[::-1], "p"))
    assert reversed_payload["input"]["garment_images"] == urls[::-1]


def test_payload_accepts_tuple_and_emits_list():
    payload = build_payload(FittingInput("https://u", ("https://img/1.jpg",), "p"))
    assert payload["input"]["garment_images"] == ["https://img/1.jpg"]


def test_try_on_sends_sync_post_with_headers_and_timeout():
    opener = FakeOpener({"status": "succeeded", "generation_url": RESULT_URL})

    _provider(opener, timeout=12.5).try_on(_input())

    [(request, timeout)] = opener.calls
    assert request.full_url == PRUNA_ENDPOINT
    assert request.get_method() == "POST"
    assert request.get_header("Apikey") == API_KEY
    assert request.get_header("Model") == "p-image-try-on"
    assert request.get_header("Try-sync") == "true"
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 12.5
    assert json.loads(request.data) == build_payload(_input())


# --- response parsing ---


def test_returns_internal_result_type():
    opener = FakeOpener({"status": "succeeded", "generation_url": RESULT_URL, "extra": 1})

    result = _provider(opener).try_on(_input())

    assert result == FittingResult(result_image_url=RESULT_URL)


def test_accepts_response_without_status_field():
    opener = FakeOpener({"generation_url": RESULT_URL})
    assert _provider(opener).try_on(_input()).result_image_url == RESULT_URL


# --- 오류 처리 ---


@pytest.mark.parametrize("code", [400, 401, 429, 500, 503])
def test_http_error_becomes_model_error(code):
    with pytest.raises(FittingModelError) as exc_info:
        _provider(FakeOpener(error=_http_error(code))).try_on(_input())
    assert str(code) in str(exc_info.value)
    assert exc_info.value.status_code == 502
    assert exc_info.value.message == "fitting_model_failed"


def test_http_504_becomes_timeout_error():
    with pytest.raises(FittingTimeoutError):
        _provider(FakeOpener(error=_http_error(504))).try_on(_input())


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), TimeoutError(), URLError(TimeoutError("timed out"))],
)
def test_timeout_becomes_timeout_error(error):
    with pytest.raises(FittingTimeoutError) as exc_info:
        _provider(FakeOpener(error=error)).try_on(_input())
    assert exc_info.value.status_code == 504
    assert exc_info.value.message == "fitting_timeout"


@pytest.mark.parametrize(
    "error", [URLError("name resolution failed"), ConnectionResetError("reset")]
)
def test_network_failure_becomes_model_error(error):
    with pytest.raises(FittingModelError):
        _provider(FakeOpener(error=error)).try_on(_input())


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"\xff\xfe",
        b"[]",
        b'"text"',
        b"null",
    ],
)
def test_malformed_response_becomes_model_error(body):
    with pytest.raises(FittingModelError):
        _provider(FakeOpener(body)).try_on(_input())


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "succeeded"},
        {"status": "succeeded", "generation_url": ""},
        {"status": "succeeded", "generation_url": None},
        {"status": "succeeded", "generation_url": 123},
        {"status": "succeeded", "generation_url": "not-a-url"},
        {},
    ],
)
def test_missing_result_image_becomes_model_error(payload):
    with pytest.raises(FittingModelError):
        _provider(FakeOpener(payload)).try_on(_input())


@pytest.mark.parametrize("status", ["failed", "processing", "starting"])
def test_non_succeeded_status_becomes_model_error(status):
    payload = {"status": status, "generation_url": RESULT_URL}
    with pytest.raises(FittingModelError):
        _provider(FakeOpener(payload)).try_on(_input())


def test_api_key_is_not_leaked_in_error_messages():
    with pytest.raises(FittingModelError) as exc_info:
        _provider(FakeOpener(error=_http_error(401))).try_on(_input())
    assert API_KEY not in str(exc_info.value)


# --- API Key ---


def test_reads_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("PRUNA_API_KEY", f"  {API_KEY}  ")
    opener = FakeOpener({"generation_url": RESULT_URL})

    PrunaDirectProvider(opener=opener).try_on(_input())

    [(request, _)] = opener.calls
    assert request.get_header("Apikey") == API_KEY


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_api_key_raises_config_error(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PRUNA_API_KEY", raising=False)
    else:
        monkeypatch.setenv("PRUNA_API_KEY", value)
    with pytest.raises(PrunaConfigError):
        PrunaDirectProvider()


# --- 공통 인터페이스 ---


def test_direct_provider_satisfies_fitting_provider_interface():
    provider = _provider(FakeOpener({"generation_url": RESULT_URL}))
    assert isinstance(provider, FittingProvider)


def test_service_can_use_provider_through_interface_only():
    def run(fitting_provider: FittingProvider) -> str:
        return fitting_provider.try_on(_input()).result_image_url

    assert run(_provider(FakeOpener({"generation_url": RESULT_URL}))) == RESULT_URL
