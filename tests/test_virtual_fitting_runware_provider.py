"""실제 Runware API 를 호출하지 않고 fake opener 로 Provider 를 검증한다."""

import io
import json
import uuid
from urllib.error import HTTPError, URLError

import pytest

from app.virtual_fitting.exceptions import FittingModelError, FittingTimeoutError
from app.virtual_fitting.providers.base import FittingInput, FittingProvider, FittingResult
from app.virtual_fitting.providers.runware import (
    RUNWARE_ENDPOINT,
    RunwareConfigError,
    RunwarePrunaProvider,
    build_payload,
)

API_KEY = "test-secret-key"
RESULT_URL = "https://im.runware.ai/image/os/a14d18/ws/2/ii/f1e2.jpg"
SUCCESS = {"data": [{"taskType": "imageInference", "imageURL": RESULT_URL}]}


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
        prompt=(
            "Dress the person in the sweatshirt from the first garment image "
            "and the slim pants from the second garment image."
        ),
    )


def _provider(opener, **kwargs):
    return RunwarePrunaProvider(API_KEY, opener=opener, **kwargs)


def _http_error(code, body=b'{"errors": [{"message": "boom"}]}'):
    return HTTPError(RUNWARE_ENDPOINT, code, "err", {}, io.BytesIO(body))


# --- request 변환 ---


def test_payload_is_task_array_with_person_and_garment_roles():
    fitting_input = _input()

    [task] = build_payload(fitting_input, "task-1")

    assert task["taskType"] == "imageInference"
    assert task["taskUUID"] == "task-1"
    assert task["model"] == "prunaai:p-image@try-on"
    assert task["positivePrompt"] == fitting_input.prompt
    assert task["deliveryMethod"] == "sync"
    assert task["outputType"] == "URL"
    assert task["inputs"]["referenceImages"] == [
        {"image": "https://example.com/user.png", "role": "person"},
        {"image": "https://img/top.jpg", "role": "garment"},
        {"image": "https://img/bottom.jpg", "role": "garment"},
    ]


def test_payload_keeps_garment_image_order():
    urls = ["https://img/1.jpg", "https://img/2.jpg"]
    for ordered in (urls, urls[::-1]):
        [task] = build_payload(FittingInput("https://u", ordered, "p"), "t")
        garments = [r["image"] for r in task["inputs"]["referenceImages"] if r["role"] == "garment"]
        assert garments == ordered


def test_try_on_sends_post_with_bearer_auth_and_timeout():
    opener = FakeOpener(SUCCESS)

    _provider(opener, timeout=12.5).try_on(_input())

    [(request, timeout)] = opener.calls
    assert request.full_url == RUNWARE_ENDPOINT
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == f"Bearer {API_KEY}"
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 12.5
    [task] = json.loads(request.data)
    assert uuid.UUID(task["taskUUID"]).version == 4


# --- response parsing ---


def test_returns_internal_result_type():
    assert _provider(FakeOpener(SUCCESS)).try_on(_input()) == FittingResult(RESULT_URL)


# --- 오류 처리 ---


@pytest.mark.parametrize("code", [400, 401, 429, 500, 503])
def test_http_error_becomes_model_error(code):
    with pytest.raises(FittingModelError) as exc_info:
        _provider(FakeOpener(error=_http_error(code))).try_on(_input())
    assert str(code) in str(exc_info.value)
    assert exc_info.value.status_code == 502


def test_http_504_becomes_timeout_error():
    with pytest.raises(FittingTimeoutError):
        _provider(FakeOpener(error=_http_error(504))).try_on(_input())


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), TimeoutError(), URLError(TimeoutError("timed out"))],
)
def test_timeout_becomes_timeout_error(error):
    with pytest.raises(FittingTimeoutError):
        _provider(FakeOpener(error=error)).try_on(_input())


@pytest.mark.parametrize(
    "error", [URLError("name resolution failed"), ConnectionResetError("reset")]
)
def test_network_failure_becomes_model_error(error):
    with pytest.raises(FittingModelError):
        _provider(FakeOpener(error=error)).try_on(_input())


@pytest.mark.parametrize("body", [b"not json", b"\xff\xfe", b"[]", b'"text"', b"null"])
def test_malformed_response_becomes_model_error(body):
    with pytest.raises(FittingModelError):
        _provider(FakeOpener(body)).try_on(_input())


def test_errors_in_body_become_model_error_even_with_http_200():
    payload = {"errors": [{"code": "invalidImage", "message": "bad image"}]}
    with pytest.raises(FittingModelError) as exc_info:
        _provider(FakeOpener(payload)).try_on(_input())
    assert "bad image" in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": []},
        {"data": None},
        {"data": [{}]},
        {"data": ["x"]},
        {"data": [{"imageURL": ""}]},
        {"data": [{"imageURL": None}]},
        {"data": [{"imageURL": 1}]},
        {"data": [{"imageURL": "not-a-url"}]},
    ],
)
def test_missing_result_image_becomes_model_error(payload):
    with pytest.raises(FittingModelError):
        _provider(FakeOpener(payload)).try_on(_input())


def test_api_key_is_not_leaked_in_error_messages():
    with pytest.raises(FittingModelError) as exc_info:
        _provider(FakeOpener(error=_http_error(401))).try_on(_input())
    assert API_KEY not in str(exc_info.value)


# --- API Key / 인터페이스 ---


def test_reads_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", f"  {API_KEY}  ")
    opener = FakeOpener(SUCCESS)

    RunwarePrunaProvider(opener=opener).try_on(_input())

    [(request, _)] = opener.calls
    assert request.get_header("Authorization") == f"Bearer {API_KEY}"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_api_key_raises_config_error(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("RUNWARE_VTON_API_KEY", raising=False)
    else:
        monkeypatch.setenv("RUNWARE_VTON_API_KEY", value)
    with pytest.raises(RunwareConfigError):
        RunwarePrunaProvider()


def test_satisfies_fitting_provider_interface():
    assert isinstance(_provider(FakeOpener(SUCCESS)), FittingProvider)


def test_uses_only_the_vton_key_and_ignores_the_llm_key(monkeypatch):
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "vton-key-value")
    monkeypatch.setenv("RUNWARE_LLM_API_KEY", "llm-key-value")
    opener = FakeOpener(SUCCESS)

    RunwarePrunaProvider(opener=opener).try_on(_input())

    [(request, _)] = opener.calls
    assert request.get_header("Authorization") == "Bearer vton-key-value"
    sent = request.data.decode() + str(dict(request.header_items()))
    assert "llm-key-value" not in sent


def test_llm_key_alone_is_not_enough_for_the_vton_provider(monkeypatch):
    monkeypatch.delenv("RUNWARE_VTON_API_KEY", raising=False)
    monkeypatch.setenv("RUNWARE_LLM_API_KEY", "llm-key-value")

    with pytest.raises(RunwareConfigError) as exc_info:
        RunwarePrunaProvider()

    assert str(exc_info.value) == "RUNWARE_VTON_API_KEY 환경변수가 설정되지 않았습니다."


def test_legacy_generic_key_is_no_longer_used(monkeypatch):
    monkeypatch.delenv("RUNWARE_VTON_API_KEY", raising=False)
    monkeypatch.setenv("RUNWARE_API_KEY", "legacy-key-value")

    with pytest.raises(RunwareConfigError):
        RunwarePrunaProvider()
