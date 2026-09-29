"""실제 Runware 를 호출하지 않고 fake opener 로 RunwareCommentProvider 를 검증한다."""

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from app.virtual_fitting.exceptions import (
    FittingModelError,
    FittingPostprocessError,
    FittingTimeoutError,
)
from app.virtual_fitting.providers.runware import RunwareConfigError
from app.virtual_fitting.providers.runware_comment import (
    COMMENT_SYSTEM_PROMPT,
    LLM_ENDPOINT,
    LLM_MODEL,
    TITLE_SYSTEM_PROMPT,
    RunwareCommentProvider,
    build_comment_request,
    build_title_request,
)

LLM_KEY = "test-llm-key-value"
VTON_KEY = "test-vton-key-value"
RUNWARE_URL = "https://im.runware.ai/image/os/a09dlim3/ws/3/ii/result.jpg"
SUMMARIES = ["네이비 컬러의 루즈핏 스웨트셔츠", "중청 워싱의 슬림 데님 팬츠"]
COMMENT = "네이비 스웨트셔츠와 중청 데님이 어우러진 캐주얼한 코디입니다."
NATIVE_FIELDS = {"taskType", "taskUUID", "deliveryMethod", "inputs", "settings"}


def _ok(text, finish_reason="stop"):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }


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
    def __init__(self, *results, error=None):
        self._results = list(results)
        self._error = error
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        if self._error is not None:
            raise self._error
        return FakeResponse(self._results.pop(0))


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("외부 HTTP 요청이 시도되었다")

    monkeypatch.setattr("app.virtual_fitting.providers.runware_comment.urlopen", forbidden)


def _provider(opener, **kwargs):
    return RunwareCommentProvider(LLM_KEY, opener=opener, **kwargs)


def _sent(opener, index=0):
    request, _ = opener.calls[index]
    return request, json.loads(request.data)


def _http_error(code, body=b'{"error": {"message": "boom"}}'):
    return HTTPError(LLM_ENDPOINT, code, "err", {}, io.BytesIO(body))


def _user_content(payload):
    [system, user] = payload["messages"]
    assert system["role"] == "system"
    assert user["role"] == "user"
    return user["content"]


# --- API Key ---


def test_uses_the_llm_key_and_ignores_the_vton_key(monkeypatch):
    monkeypatch.setenv("RUNWARE_LLM_API_KEY", LLM_KEY)
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", VTON_KEY)
    opener = FakeOpener(_ok(COMMENT))

    RunwareCommentProvider(opener=opener).generate_comment(RUNWARE_URL, SUMMARIES)

    request, _ = _sent(opener)
    assert request.get_header("Authorization") == f"Bearer {LLM_KEY}"
    assert VTON_KEY not in str(dict(request.header_items())) + request.data.decode()


def test_llm_key_value_is_not_in_the_payload():
    opener = FakeOpener(_ok(COMMENT))

    _provider(opener).generate_comment(RUNWARE_URL, SUMMARIES)

    assert LLM_KEY not in _sent(opener)[0].data.decode()


def test_vton_key_alone_is_not_enough(monkeypatch):
    monkeypatch.delenv("RUNWARE_LLM_API_KEY", raising=False)
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", VTON_KEY)

    with pytest.raises(RunwareConfigError) as exc_info:
        RunwareCommentProvider()

    assert str(exc_info.value) == "RUNWARE_LLM_API_KEY 환경변수가 설정되지 않았습니다."


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_llm_key_is_rejected(monkeypatch, value):
    monkeypatch.setenv("RUNWARE_LLM_API_KEY", value)

    with pytest.raises(RunwareConfigError):
        RunwareCommentProvider()


def test_llm_key_is_not_leaked_in_errors():
    with pytest.raises(FittingPostprocessError) as exc_info:
        _provider(FakeOpener(error=_http_error(401))).generate_comment(RUNWARE_URL, SUMMARIES)

    assert LLM_KEY not in str(exc_info.value) + repr(exc_info.value)


# --- comment request ---


def test_comment_request_is_a_non_streaming_chat_completion_for_luna():
    opener = FakeOpener(_ok(COMMENT))

    _provider(opener, timeout=12.5).generate_comment(RUNWARE_URL, SUMMARIES)

    request, payload = _sent(opener)
    assert request.full_url == "https://api.runware.ai/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert opener.calls[0][1] == 12.5
    assert isinstance(payload, dict)  # native task 배열이 아니다
    assert payload["model"] == LLM_MODEL == "openai:gpt@5.6-luna"
    assert payload["stream"] is False
    assert payload["max_completion_tokens"] > 0
    assert not NATIVE_FIELDS & set(payload)


@pytest.mark.parametrize(
    "url", [RUNWARE_URL, "https://example-bucket.s3.ap-northeast-2.amazonaws.com/r.jpg"]
)
def test_result_image_url_is_sent_as_an_image_url_content_part_verbatim(url):
    opener = FakeOpener(_ok(COMMENT))

    _provider(opener).generate_comment(url, SUMMARIES)

    content = _user_content(_sent(opener)[1])
    assert {"type": "image_url", "image_url": {"url": url}} in content
    assert "inputs" not in _sent(opener)[1]


def test_description_summaries_are_in_the_text_content_part_in_order():
    opener = FakeOpener(_ok(COMMENT))

    _provider(opener).generate_comment(RUNWARE_URL, SUMMARIES)

    [text_part] = [
        part for part in _user_content(_sent(opener)[1]) if part["type"] == "text"
    ]
    text = text_part["text"]
    assert text.index(SUMMARIES[0]) < text.index(SUMMARIES[1])
    assert RUNWARE_URL not in text


def test_comment_system_prompt_is_sent_as_the_system_message():
    payload = build_comment_request(RUNWARE_URL, SUMMARIES)

    assert payload["messages"][0] == {"role": "system", "content": COMMENT_SYSTEM_PROMPT}
    assert "이미지" in COMMENT_SYSTEM_PROMPT and "보조 근거" in COMMENT_SYSTEM_PROMPT
    assert "확인되지 않는" in COMMENT_SYSTEM_PROMPT and "과장" in COMMENT_SYSTEM_PROMPT
    assert "1~2문장" in COMMENT_SYSTEM_PROMPT


# --- title request ---


def test_title_request_contains_only_the_comment():
    opener = FakeOpener(_ok("네이비 데일리 캐주얼"))

    _provider(opener).generate_title(COMMENT)

    request, payload = _sent(opener)
    assert request.full_url == LLM_ENDPOINT
    assert payload["messages"] == [
        {"role": "system", "content": TITLE_SYSTEM_PROMPT},
        {"role": "user", "content": f"코디 설명: {COMMENT}"},
    ]
    sent = request.data.decode()
    assert RUNWARE_URL not in sent
    assert "image_url" not in sent
    assert not any(summary in sent for summary in SUMMARIES)


def test_title_request_never_carries_images_or_product_data():
    payload = build_title_request(COMMENT)

    assert set(payload) == {"model", "messages", "max_completion_tokens", "stream"}
    assert payload["model"] == LLM_MODEL
    assert payload["stream"] is False
    assert all(isinstance(message["content"], str) for message in payload["messages"])


def test_title_is_built_from_the_generated_comment():
    opener = FakeOpener(_ok(COMMENT), _ok("네이비 데일리 캐주얼"))
    provider = _provider(opener)

    comment = provider.generate_comment(RUNWARE_URL, SUMMARIES)
    title = provider.generate_title(comment)

    assert (comment, title) == (COMMENT, "네이비 데일리 캐주얼")
    _, title_payload = _sent(opener, 1)
    assert COMMENT in title_payload["messages"][1]["content"]


# --- 호출 횟수 ---


def test_each_generation_is_exactly_one_request():
    opener = FakeOpener(_ok(COMMENT), _ok("제목"))
    provider = _provider(opener)

    provider.generate_comment(RUNWARE_URL, SUMMARIES)
    assert len(opener.calls) == 1
    provider.generate_title(COMMENT)
    assert len(opener.calls) == 2


def test_a_failed_request_is_not_retried():
    opener = FakeOpener(error=_http_error(500))

    with pytest.raises(FittingPostprocessError):
        _provider(opener).generate_comment(RUNWARE_URL, SUMMARIES)

    assert len(opener.calls) == 1


# --- parsing ---


def test_text_is_stripped():
    assert _provider(FakeOpener(_ok(f"  {COMMENT}\n"))).generate_title(COMMENT) == COMMENT


def _choice(**fields):
    return {"choices": [fields]}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": None},
        {"choices": []},
        {"choices": "x"},
        {"choices": [None]},
        {"choices": ["x"]},
        _choice(finish_reason="stop"),
        _choice(message=None),
        _choice(message="x"),
        _choice(message={"role": "assistant"}),
        _choice(message={"content": None}),
        _choice(message={"content": ""}),
        _choice(message={"content": "   \n"}),
        _choice(message={"content": 123}),
        _choice(message={"content": [{"type": "text", "text": "x"}]}),
        _ok("잘림", finish_reason="length"),
        _ok("차단", finish_reason="content_filter"),
        {"error": {"message": "bad request", "type": "invalid_request_error"}},
        {"errors": [{"code": "invalidModel", "message": "bad model"}]},
    ],
    ids=[
        "no-choices", "null-choices", "empty-choices", "str-choices", "null-choice",
        "str-choice", "no-message", "null-message", "str-message", "no-content",
        "null-content", "empty-content", "whitespace-content", "non-str-content",
        "list-content", "truncated", "content-filter", "error", "errors",
    ],
)
def test_unusable_response_is_a_postprocess_error(payload):
    with pytest.raises(FittingPostprocessError):
        _provider(FakeOpener(payload)).generate_comment(RUNWARE_URL, SUMMARIES)


@pytest.mark.parametrize("body", [b"not json", b"\xff\xfe", b"[]", b'"text"', b"null"])
def test_malformed_json_is_a_postprocess_error(body):
    with pytest.raises(FittingPostprocessError):
        _provider(FakeOpener(body)).generate_title(COMMENT)


# --- HTTP failure ---


def _fails(opener):
    with pytest.raises(FittingPostprocessError) as exc_info:
        _provider(opener).generate_comment(RUNWARE_URL, SUMMARIES)
    return exc_info.value


@pytest.mark.parametrize("code", [400, 401, 429, 500, 503, 504])
def test_http_errors_are_postprocess_errors(code):
    error = _fails(FakeOpener(error=_http_error(code)))

    assert error.status_code == 500
    assert error.code == "FITTING_POSTPROCESS_FAILED"


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), TimeoutError(), URLError(TimeoutError("timed out"))],
)
def test_timeout_is_a_postprocess_error(error):
    assert _fails(FakeOpener(error=error)).code == "FITTING_POSTPROCESS_FAILED"


@pytest.mark.parametrize(
    "error", [URLError("name resolution failed"), ConnectionResetError("reset")]
)
def test_network_failure_is_a_postprocess_error(error):
    assert _fails(FakeOpener(error=error)).code == "FITTING_POSTPROCESS_FAILED"


def test_llm_failures_are_distinguishable_from_vton_failures():
    error = _fails(FakeOpener(error=_http_error(500)))

    assert not isinstance(error, (FittingModelError, FittingTimeoutError))


# --- 입력 검증: 가짜 값으로 조용히 넘어가지 않는다 ---


@pytest.mark.parametrize("summaries", [[], [""], ["   "], ["ok", ""]])
def test_missing_description_summaries_are_rejected_before_any_request(summaries):
    opener = FakeOpener(_ok(COMMENT))

    with pytest.raises(ValueError, match="description_summaries"):
        _provider(opener).generate_comment(RUNWARE_URL, summaries)

    assert opener.calls == []


@pytest.mark.parametrize("url", ["", "not-a-url", "ftp://x/y.jpg"])
def test_invalid_result_image_url_is_rejected_before_any_request(url):
    opener = FakeOpener(_ok(COMMENT))

    with pytest.raises(ValueError, match="result_image_url"):
        _provider(opener).generate_comment(url, SUMMARIES)

    assert opener.calls == []


@pytest.mark.parametrize("comment", ["", "   "])
def test_blank_comment_is_rejected_before_any_request(comment):
    opener = FakeOpener(_ok("제목"))

    with pytest.raises(ValueError, match="comment"):
        _provider(opener).generate_title(comment)

    assert opener.calls == []
