"""실제 Runware API 를 호출하지 않고 fake opener 로 V2 비동기 접수 Provider 를 검증한다."""

import io
import json
import logging
import uuid
from http.client import IncompleteRead, RemoteDisconnected
from urllib.error import HTTPError, URLError

import pytest

from app.virtual_fitting.providers.base import FittingInput
from app.virtual_fitting.providers.runware import (
    RUNWARE_ENDPOINT,
    RUNWARE_MODEL,
    RunwareConfigError,
)
from app.virtual_fitting.providers.runware_async import (
    DEFAULT_SUBMIT_TIMEOUT,
    AsyncSubmission,
    RunwarePrunaAsyncProvider,
    SubmissionOutcome,
    build_async_payload,
)

API_KEY = "test-secret-key"
TASK_UUID = "a770f077-f413-47de-9dac-be0b26a35da6"
OTHER_UUID = "0b8f8a3e-2c55-4c1e-9d5e-7c0f3b0a9e11"
PERSON_URL = "https://bucket.s3.amazonaws.com/users/1/body.jpg?X-Amz-Signature=PERSONSIG"
WEBHOOK_URL = "https://ai.example.com/api/v2/webhooks/runware?token=WEBHOOKTOKEN"
UPLOAD_URL = (
    "https://bucket.s3.amazonaws.com/virtual-fitting/results/3/pruna-async.jpg"
    "?X-Amz-Signature=UPLOADSIG&X-Amz-Expires=3600"
)
SECRETS = (API_KEY, "PERSONSIG", "WEBHOOKTOKEN", "UPLOADSIG")
ACK = {"data": [{"taskType": "imageInference", "taskUUID": TASK_UUID, "status": "processing"}]}


class FakeResponse:
    def __init__(self, payload, status=200, read_error=None):
        self._body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.status = status
        self._read_error = read_error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        if self._read_error is not None:
            raise self._read_error
        return self._body


class FakeOpener:
    def __init__(self, result=None, status=200, error=None, read_error=None):
        self._result = result
        self._status = status
        self._error = error
        self._read_error = read_error
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        if self._error is not None:
            raise self._error
        return FakeResponse(self._result, self._status, self._read_error)


def _input(garments=("https://img/top.jpg", "https://img/bottom.jpg"), prompt="Dress the person."):
    return FittingInput(
        person_image_url=PERSON_URL, garment_image_urls=list(garments), prompt=prompt
    )


def _submit(opener, **overrides):
    kwargs = {"task_uuid": TASK_UUID, "webhook_url": WEBHOOK_URL, "upload_endpoint": UPLOAD_URL}
    kwargs.update(overrides)
    return RunwarePrunaAsyncProvider(API_KEY, opener=opener).submit(_input(), **kwargs)


def _http_error(code, body=b'{"errors": [{"code": "boom", "message": "boom"}]}'):
    return HTTPError(RUNWARE_ENDPOINT, code, "err", {}, io.BytesIO(body))


def _sent_task(opener):
    request, _ = opener.calls[0]
    (task,) = json.loads(request.data)
    return task


# --- 1. 비동기 요청 JSON ---


def test_async_payload_matches_design():
    (task,) = build_async_payload(
        _input(), task_uuid=TASK_UUID, webhook_url=WEBHOOK_URL, upload_endpoint=UPLOAD_URL
    )

    assert task == {
        "taskType": "imageInference",
        "taskUUID": TASK_UUID,
        "model": RUNWARE_MODEL,
        "inputs": {
            "referenceImages": [
                {"image": PERSON_URL, "role": "person"},
                {"image": "https://img/top.jpg", "role": "garment"},
                {"image": "https://img/bottom.jpg", "role": "garment"},
            ]
        },
        "outputType": "URL",
        "outputFormat": "JPG",
        "numberResults": 1,
        "includeCost": True,
        "deliveryMethod": "async",
        "webhookURL": WEBHOOK_URL,
        "uploadEndpoint": UPLOAD_URL,
        "positivePrompt": "Dress the person.",
    }


def test_empty_prompt_is_not_sent():
    (task,) = build_async_payload(
        _input(prompt=""), task_uuid=TASK_UUID, webhook_url=WEBHOOK_URL, upload_endpoint=UPLOAD_URL
    )
    assert "positivePrompt" not in task


def test_submit_sends_one_post_with_external_task_uuid_and_bearer():
    opener = FakeOpener(ACK)

    _submit(opener)

    assert len(opener.calls) == 1
    request, timeout = opener.calls[0]
    assert request.full_url == RUNWARE_ENDPOINT
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == f"Bearer {API_KEY}"
    assert request.get_header("Content-type") == "application/json"
    assert timeout == DEFAULT_SUBMIT_TIMEOUT
    assert _sent_task(opener)["taskUUID"] == TASK_UUID  # Provider 가 새로 만들지 않는다


def test_provider_does_not_generate_task_uuid(monkeypatch):
    def fail():
        raise AssertionError("Provider 가 uuid4 를 만들면 안 된다")

    monkeypatch.setattr(uuid, "uuid4", fail)
    assert _submit(FakeOpener(ACK)).accepted


def test_uppercase_uuid_is_normalized_and_compared_lowercase():
    opener = FakeOpener(ACK)
    result = _submit(opener, task_uuid=TASK_UUID.upper())

    assert _sent_task(opener)["taskUUID"] == TASK_UUID
    assert result.accepted and result.task_uuid == TASK_UUID


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "",
        "not-a-uuid",
        str(uuid.uuid1()),
        "a770f077f41347de9dacbe0b26a35da6",
        f"{{{TASK_UUID}}}",
    ],
)
def test_invalid_task_uuid_is_rejected_before_sending(bad):
    opener = FakeOpener(ACK)
    with pytest.raises(ValueError):
        _submit(opener, task_uuid=bad)
    assert opener.calls == []


@pytest.mark.parametrize("field", ["webhook_url", "upload_endpoint"])
@pytest.mark.parametrize("value", [None, "", "http://insecure.example.com/x"])
def test_webhook_and_upload_endpoint_are_required(field, value):
    opener = FakeOpener(ACK)
    with pytest.raises(ValueError):
        _submit(opener, **{field: value})
    assert opener.calls == []


@pytest.mark.parametrize("garments", [(), tuple(f"https://img/{i}.jpg" for i in range(12))])
def test_garment_count_is_checked_before_sending(garments):
    opener = FakeOpener(ACK)
    provider = RunwarePrunaAsyncProvider(API_KEY, opener=opener)
    with pytest.raises(ValueError):
        provider.submit(
            _input(garments=garments),
            task_uuid=TASK_UUID,
            webhook_url=WEBHOOK_URL,
            upload_endpoint=UPLOAD_URL,
        )
    assert opener.calls == []


# --- 2. 정상 접수 ---


def test_ack_with_matching_task_uuid_is_accepted():
    result = _submit(FakeOpener(ACK))

    assert result == AsyncSubmission(
        SubmissionOutcome.ACCEPTED, TASK_UUID, "accepted", http_status=200, task_status="processing"
    )
    assert result.accepted


def test_ack_without_status_is_still_accepted_when_task_uuid_matches():
    result = _submit(FakeOpener({"data": [{"taskUUID": TASK_UUID}]}))
    assert result.outcome is SubmissionOutcome.ACCEPTED and result.task_status is None


# --- 3. 접수 근거가 불명확한 2xx → UNKNOWN ---


@pytest.mark.parametrize(
    "body, reason",
    [
        (b"not json", "invalid_response"),
        (b"", "invalid_response"),
        (b"\xff\xfe", "invalid_response"),
        ([ACK], "invalid_response"),
        ({}, "missing_task_uuid"),
        ({"data": []}, "missing_task_uuid"),
        ({"data": [{"taskType": "imageInference", "status": "processing"}]}, "missing_task_uuid"),
        ({"data": "processing"}, "missing_task_uuid"),
        ({"data": [{"taskUUID": OTHER_UUID, "status": "processing"}]}, "task_uuid_mismatch"),
    ],
)
def test_unclear_2xx_is_never_accepted(body, reason):
    result = _submit(FakeOpener(body))

    assert result.outcome is SubmissionOutcome.UNKNOWN
    assert result.reason == reason
    assert result.http_status == 200
    assert not result.accepted


def test_202_ack_is_accepted():
    assert _submit(FakeOpener(ACK, status=202)).accepted


# --- 4. 오류 응답 ---


def test_errors_in_2xx_body_are_rejected():
    body = {
        "errors": [
            {"code": "invalidReferenceImages", "message": "bad image", "taskUUID": TASK_UUID}
        ],
        "data": [{"taskUUID": TASK_UUID}],
    }
    result = _submit(FakeOpener(body))

    assert result.outcome is SubmissionOutcome.REJECTED
    assert result.reason == "error_response"
    assert result.error_codes == ("invalidReferenceImages",)
    assert not result.retryable


def test_retryable_error_code_and_retry_after_are_reported_not_retried():
    body = {
        "errors": [{"code": "providerRateLimitExceeded", "message": "slow down", "retryAfter": 5}]
    }
    opener = FakeOpener(body)
    result = _submit(opener)

    assert result.outcome is SubmissionOutcome.REJECTED
    assert result.retryable and result.retry_after == 5
    assert len(opener.calls) == 1  # 자동 재시도 없음


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_definitive_4xx_is_rejected(code):
    result = _submit(FakeOpener(error=_http_error(code)))

    assert result.outcome is SubmissionOutcome.REJECTED
    assert result.reason == "client_error"
    assert result.http_status == code
    assert result.error_codes == ("boom",)


def test_402_is_rejected_as_balance_exhausted():
    body = b'{"errors": [{"code": "insufficientCredits", "message": "no money"}]}'
    result = _submit(FakeOpener(error=_http_error(402, body)))

    assert (result.outcome, result.reason, result.http_status) == (
        SubmissionOutcome.REJECTED,
        "balance_exhausted",
        402,
    )
    assert result.error_codes == ("insufficientCredits",)


@pytest.mark.parametrize(
    "code, reason, retryable",
    [
        (429, "rate_limited", True),
        (500, "server_error", True),
        (502, "server_error", True),
        (503, "server_error", True),
        (504, "server_error", True),
        (408, "unexpected_status", False),
        (409, "unexpected_status", False),
        (302, "unexpected_status", False),
    ],
)
def test_ambiguous_status_is_unknown(code, reason, retryable):
    opener = FakeOpener(error=_http_error(code))
    result = _submit(opener)

    assert result.outcome is SubmissionOutcome.UNKNOWN
    assert (result.reason, result.http_status, result.retryable) == (reason, code, retryable)
    assert len(opener.calls) == 1


def test_error_body_that_is_not_json_still_classifies_by_status():
    result = _submit(FakeOpener(error=_http_error(400, b"<html>bad</html>")))
    assert result.outcome is SubmissionOutcome.REJECTED and result.error_codes == ()


def test_retry_after_on_429_is_kept():
    body = b'{"errors": [{"code": "rateLimitExceeded", "message": "x", "retryAfter": 2}]}'
    result = _submit(FakeOpener(error=_http_error(429, body)))
    assert result.retry_after == 2 and result.outcome is SubmissionOutcome.UNKNOWN


# --- 5. timeout·네트워크 오류 → UNKNOWN, 재시도 없음 ---


@pytest.mark.parametrize(
    "error, reason",
    [
        (TimeoutError("timed out"), "timeout"),
        (URLError(TimeoutError("timed out")), "timeout"),
        (URLError(ConnectionRefusedError("refused")), "network_error"),
        (URLError("dns failure"), "network_error"),
        (ConnectionResetError("reset"), "network_error"),
        (RemoteDisconnected("closed"), "network_error"),
    ],
)
def test_transport_failures_are_unknown_and_not_retried(error, reason):
    opener = FakeOpener(error=error)
    result = _submit(opener)

    assert result.outcome is SubmissionOutcome.UNKNOWN
    assert result.reason == reason
    assert result.http_status is None
    assert len(opener.calls) == 1


@pytest.mark.parametrize(
    "read_error, reason",
    [(TimeoutError("timed out"), "timeout"), (IncompleteRead(b"{"), "invalid_response")],
)
def test_failure_while_reading_2xx_body_is_unknown(read_error, reason):
    result = _submit(FakeOpener(ACK, read_error=read_error))

    assert result.outcome is SubmissionOutcome.UNKNOWN
    assert result.reason == reason
    assert result.http_status == 200


# --- 비밀값 ---


def test_secrets_never_appear_in_result_or_logs(caplog):
    body = {
        "errors": [
            {"code": "invalidWebhookURL", "message": f"cannot reach {WEBHOOK_URL} / {UPLOAD_URL}"}
        ]
    }
    with caplog.at_level(logging.DEBUG, logger="app.virtual_fitting.providers.runware_async"):
        results = [
            _submit(FakeOpener(ACK)),
            _submit(FakeOpener(body)),
            _submit(FakeOpener(error=_http_error(400, json.dumps(body).encode()))),
            _submit(FakeOpener(error=URLError("dns failure"))),
        ]

    assert f"task_uuid={TASK_UUID}" in caplog.text
    for secret in SECRETS:
        assert secret not in caplog.text
        for result in results:
            assert secret not in repr(result)


def test_api_key_is_read_from_vton_env(monkeypatch):
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "  env-key  ")
    opener = FakeOpener(ACK)

    RunwarePrunaAsyncProvider(opener=opener).submit(
        _input(), task_uuid=TASK_UUID, webhook_url=WEBHOOK_URL, upload_endpoint=UPLOAD_URL
    )

    assert opener.calls[0][0].get_header("Authorization") == "Bearer env-key"


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("RUNWARE_VTON_API_KEY", raising=False)
    with pytest.raises(RunwareConfigError):
        RunwarePrunaAsyncProvider()
