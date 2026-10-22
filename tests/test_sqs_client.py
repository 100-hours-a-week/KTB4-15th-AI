"""V2 비동기 가상피팅 SQS 공통 client.

실제 SQS 에 요청하지 않는다. 응답은 botocore Stubber(가짜 자격증명) 나 fake client 로 흉내 낸다.
"""

import json
import logging

import boto3
import pytest
from botocore.exceptions import (
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    ParamValidationError,
    ProxyConnectionError,
    ReadTimeoutError,
)
from botocore.stub import Stubber

from app.clients import sqs as sqs_module
from app.clients.s3 import get_s3_client
from app.clients.sqs import (
    FittingQueue,
    ReceivedMessage,
    SqsConfigError,
    SqsOperation,
    SqsOutcome,
    SqsQueueClient,
    SqsResult,
    encode_json_body,
    get_queue_url,
    get_sqs_client,
)
from app.config import settings

REGION = "ap-northeast-2"
ACCOUNT = "123456789012"
URLS = {
    FittingQueue.BE_FITTING_REQUEST: f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT}/be-fitting-request",
    FittingQueue.AI_FITTING_RESPONSE: f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT}/ai-fitting-response",
    FittingQueue.BE_FITTING_RESPONSE: f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT}/be-fitting-response",
}
ACCESS_KEY = "AKIAUNITTESTACCESSKEY"
SECRET_KEY = "unit-test-secret-access-key"
RECEIPT = "AQEBsecret-receipt-handle=="
PAYLOAD = {
    "fitting_job_id": 3,
    "user_image_key": "users/3.jpg",
    "products": [{"product_code": "1"}],
}


@pytest.fixture(autouse=True)
def queue_settings(monkeypatch):
    for name in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "AWS_REGION", REGION)
    for queue, url in URLS.items():
        monkeypatch.setattr(settings, queue.value, url)


def _boto_client():
    from botocore.config import Config

    return boto3.client(
        "sqs",
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(retries={"mode": "standard", "total_max_attempts": 1}),
    )


def _stubbed():
    client = _boto_client()
    return SqsQueueClient(client), client


class RaisingClient:
    """지정한 오류를 던지고, 몇 번 불렸는지 센다(자동 재시도가 없는지 확인)."""

    def __init__(self, error):
        self.error = error
        self.calls = []

    def _fail(self, name, kwargs):
        self.calls.append((name, kwargs))
        raise self.error

    def send_message(self, **kwargs):
        return self._fail("send_message", kwargs)

    def receive_message(self, **kwargs):
        return self._fail("receive_message", kwargs)

    def delete_message(self, **kwargs):
        return self._fail("delete_message", kwargs)

    def change_message_visibility(self, **kwargs):
        return self._fail("change_message_visibility", kwargs)


class ReturningClient(RaisingClient):
    def __init__(self, response):
        super().__init__(None)
        self.response = response

    def _fail(self, name, kwargs):
        self.calls.append((name, kwargs))
        return self.response


def _raw_message(body, receive_count="1", message_id="m-1"):
    return {
        "MessageId": message_id,
        "ReceiptHandle": RECEIPT,
        "MD5OfBody": "x" * 32,
        "Body": body,
        "Attributes": {"ApproximateReceiveCount": receive_count},
    }


# --- 설정: 큐 URL·리전·client ---


@pytest.mark.parametrize("queue", list(FittingQueue))
def test_queue_urls_come_from_settings(queue):
    assert get_queue_url(queue) == URLS[queue]


def test_queue_setting_names():
    assert [queue.value for queue in FittingQueue] == [
        "BE_FITTING_REQUEST_QUEUE_URL",
        "AI_FITTING_RESPONSE_QUEUE_URL",
        "BE_FITTING_RESPONSE_QUEUE_URL",
    ]
    for queue in FittingQueue:
        assert hasattr(settings, queue.value)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "http://sqs.ap-northeast-2.amazonaws.com/123456789012/be-fitting-request",
        "https://sqs.us-east-1.amazonaws.com/123456789012/be-fitting-request",
        "https://sqs.ap-northeast-2.amazonaws.com/123/be-fitting-request",
        "https://sqs.ap-northeast-2.amazonaws.com/123456789012/",
        "https://example.com/123456789012/be-fitting-request",
        "be-fitting-request",
        "https://sqs.ap-northeast-2.amazonaws.com/123456789012/be-fitting-request.fifo",
    ],
)
def test_bad_queue_url_is_a_config_error_before_any_call(monkeypatch, url):
    monkeypatch.setattr(settings, "AI_FITTING_RESPONSE_QUEUE_URL", url)
    client = RaisingClient(AssertionError("SQS 를 부르면 안 된다"))

    with pytest.raises(SqsConfigError):
        SqsQueueClient(client).send_json(FittingQueue.AI_FITTING_RESPONSE, PAYLOAD)
    assert client.calls == []


def test_fifo_queue_message_is_specific(monkeypatch):
    monkeypatch.setattr(
        settings,
        "BE_FITTING_RESPONSE_QUEUE_URL",
        f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT}/be-fitting-response.fifo",
    )
    with pytest.raises(SqsConfigError, match="Standard"):
        get_queue_url(FittingQueue.BE_FITTING_RESPONSE)


@pytest.fixture
def fresh_clients():
    get_sqs_client.cache_clear()
    get_s3_client.cache_clear()
    yield
    get_sqs_client.cache_clear()
    get_s3_client.cache_clear()


def test_sqs_client_uses_region_default_chain_and_no_sdk_retry(monkeypatch, fresh_clients):
    calls = []
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: calls.append((a, kw)) or object())

    get_sqs_client()
    get_sqs_client()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("sqs",) and kwargs["region_name"] == REGION
    assert not {"aws_access_key_id", "aws_secret_access_key", "aws_session_token"} & kwargs.keys()
    assert kwargs["config"].retries == {"mode": "standard", "total_max_attempts": 1}


def test_s3_client_settings_are_unchanged(monkeypatch, fresh_clients):
    calls = []
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: calls.append((a, kw)) or object())

    get_sqs_client()
    get_s3_client()

    assert calls[1] == (("s3",), {"region_name": REGION})


def test_real_boto3_client_gets_the_retry_config(monkeypatch, fresh_clients):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", ACCESS_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET_KEY)
    client = get_sqs_client()

    assert client.meta.region_name == REGION
    assert client.meta.config.retries == {"mode": "standard", "total_max_attempts": 1}


@pytest.fixture
def fresh_webhook_client():
    sqs_module.get_webhook_sqs_client.cache_clear()
    yield
    sqs_module.get_webhook_sqs_client.cache_clear()


def test_webhook_client_uses_short_timeouts_and_one_attempt(
    monkeypatch, fresh_clients, fresh_webhook_client
):
    from app.virtual_fitting.webhook_router import WEBHOOK_DEADLINE_SECONDS

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", ACCESS_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET_KEY)
    webhook = sqs_module.get_webhook_sqs_client()
    general = get_sqs_client()

    assert webhook.meta.region_name == REGION
    assert (webhook.meta.config.connect_timeout, webhook.meta.config.read_timeout) == (1.0, 2.0)
    assert webhook.meta.config.retries == {"mode": "standard", "total_max_attempts": 1}
    # 연결 + 응답 대기 상한이 Webhook 전체 처리 제한보다 짧다(전체 상한은 Router 가 따로 건다)
    assert 1.0 + 2.0 < WEBHOOK_DEADLINE_SECONDS == 4.5
    # 일반 Worker·결과 발행용 client 는 그대로(botocore 기본 60초, 재시도 없음)
    assert (general.meta.config.connect_timeout, general.meta.config.read_timeout) == (60, 60)
    assert general.meta.config.retries == {"mode": "standard", "total_max_attempts": 1}


def test_webhook_client_read_timeout_is_unknown_and_sent_once(monkeypatch, fresh_webhook_client):
    """실제 Webhook client(SDK 재시도 설정 포함)로: 응답 timeout 은 UNKNOWN 이고 SDK 도 다시 보내지 않는다."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", ACCESS_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET_KEY)
    client = sqs_module.get_webhook_sqs_client()
    attempts = []

    def no_response(request, **kwargs):  # 네트워크로 나가기 직전에 응답 timeout 을 흉내 낸다
        attempts.append(request.url)
        raise ReadTimeoutError(endpoint_url=request.url)

    client.meta.events.register("before-send.sqs.SendMessage", no_response)
    result = SqsQueueClient(client).send_json(FittingQueue.AI_FITTING_RESPONSE, PAYLOAD)

    assert (result.outcome, result.reason) == (SqsOutcome.UNKNOWN, "timeout")
    assert len(attempts) == 1


# --- 큐별 역할: AI 가 하지 않는 호출은 막는다 ---


@pytest.mark.parametrize(
    ("queue", "call"),
    [
        (FittingQueue.BE_FITTING_REQUEST, lambda c, q: c.send_json(q, PAYLOAD)),
        (FittingQueue.BE_FITTING_RESPONSE, lambda c, q: c.receive(q)),
        (FittingQueue.BE_FITTING_RESPONSE, lambda c, q: c.delete(q, RECEIPT)),
        (FittingQueue.BE_FITTING_RESPONSE, lambda c, q: c.change_visibility(q, RECEIPT, 30)),
    ],
)
def test_operations_outside_ai_role_are_rejected(queue, call):
    client = RaisingClient(AssertionError("SQS 를 부르면 안 된다"))
    with pytest.raises(ValueError):
        call(SqsQueueClient(client), queue)
    assert client.calls == []


def test_queue_must_be_a_fitting_queue():
    with pytest.raises(TypeError):
        SqsQueueClient(RaisingClient(None)).send_json("ai-fitting-response", PAYLOAD)


# --- 전송 ---


@pytest.mark.parametrize(
    "queue", [FittingQueue.AI_FITTING_RESPONSE, FittingQueue.BE_FITTING_RESPONSE]
)
def test_send_json_success(queue):
    queue_client, client = _stubbed()
    body = encode_json_body(PAYLOAD)
    with Stubber(client) as stubber:
        stubber.add_response(
            "send_message",
            {"MessageId": "msg-1", "MD5OfMessageBody": "x" * 32},
            expected_params={"QueueUrl": URLS[queue], "MessageBody": body},
        )
        result = queue_client.send_json(queue, PAYLOAD)
        stubber.assert_no_pending_responses()

    assert result == SqsResult(SqsOperation.SEND, queue, SqsOutcome.SUCCEEDED, message_id="msg-1")
    assert result.succeeded
    assert json.loads(body) == PAYLOAD


def test_body_keeps_korean_and_is_compact():
    assert (
        encode_json_body({"llm_title": "올블랙 셔츠룩", "a": 1})
        == '{"llm_title":"올블랙 셔츠룩","a":1}'
    )


@pytest.mark.parametrize(
    "payload",
    [
        [1, 2],
        "text",
        None,
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": object()},
        {"x": "\ud800"},
    ],
)
def test_invalid_payload_is_rejected_before_sending(payload):
    client = RaisingClient(AssertionError("SQS 를 부르면 안 된다"))
    with pytest.raises((TypeError, ValueError)):
        SqsQueueClient(client).send_json(FittingQueue.AI_FITTING_RESPONSE, payload)
    assert client.calls == []


def test_send_without_message_id_is_unknown():
    client = ReturningClient({})
    result = SqsQueueClient(client).send_json(FittingQueue.AI_FITTING_RESPONSE, PAYLOAD)
    assert (result.outcome, result.reason) == (SqsOutcome.UNKNOWN, "missing_message_id")


@pytest.mark.parametrize(
    ("code", "status", "outcome", "reason", "retryable"),
    [
        ("QueueDoesNotExist", 400, SqsOutcome.FAILED, "queue_not_found", False),
        (
            "AWS.SimpleQueueService.NonExistentQueue",
            400,
            SqsOutcome.FAILED,
            "queue_not_found",
            False,
        ),
        ("AccessDenied", 403, SqsOutcome.FAILED, "access_denied", False),
        ("KmsAccessDenied", 400, SqsOutcome.FAILED, "access_denied", False),
        ("InvalidMessageContents", 400, SqsOutcome.FAILED, "invalid_message", False),
        ("InvalidParameterValue", 400, SqsOutcome.FAILED, "client_error", False),
        ("RequestThrottled", 400, SqsOutcome.FAILED, "throttled", True),
        ("ThrottlingException", 400, SqsOutcome.FAILED, "throttled", True),
        ("InternalError", 500, SqsOutcome.UNKNOWN, "server_error", True),
        ("ServiceUnavailable", 503, SqsOutcome.UNKNOWN, "server_error", True),
    ],
)
def test_send_client_errors_are_classified(code, status, outcome, reason, retryable):
    queue_client, client = _stubbed()
    with Stubber(client) as stubber:
        stubber.add_client_error("send_message", service_error_code=code, http_status_code=status)
        result = queue_client.send_json(FittingQueue.BE_FITTING_RESPONSE, PAYLOAD)
        stubber.assert_no_pending_responses()  # 응답 하나만 소비: 재시도 없음

    assert (result.outcome, result.reason, result.retryable) == (outcome, reason, retryable)
    assert (result.error_code, result.http_status) == (code, status)
    assert result.message_id is None


@pytest.mark.parametrize(
    ("error", "outcome", "reason"),
    [
        (EndpointConnectionError(endpoint_url="https://sqs"), SqsOutcome.FAILED, "not_connected"),
        (ConnectTimeoutError(endpoint_url="https://sqs"), SqsOutcome.FAILED, "not_connected"),
        (ReadTimeoutError(endpoint_url="https://sqs"), SqsOutcome.UNKNOWN, "timeout"),
        (
            ConnectionClosedError(endpoint_url="https://sqs"),
            SqsOutcome.UNKNOWN,
            "connection_closed",
        ),
        (NoCredentialsError(), SqsOutcome.FAILED, "credentials"),
        (ParamValidationError(report="bad"), SqsOutcome.FAILED, "invalid_request"),
        (ProxyConnectionError(proxy_url="http://proxy"), SqsOutcome.UNKNOWN, "unknown_error"),
    ],
)
def test_send_transport_errors_are_returned_once_without_retry(error, outcome, reason):
    client = RaisingClient(error)
    result = SqsQueueClient(client).send_json(FittingQueue.AI_FITTING_RESPONSE, PAYLOAD)

    assert (result.outcome, result.reason) == (outcome, reason)
    assert result.error_code == type(error).__name__
    assert len(client.calls) == 1


# --- 수신 ---


def test_receive_returns_body_parse_result_ids_and_receive_count():
    queue_client, client = _stubbed()
    raw = [
        _raw_message(json.dumps(PAYLOAD), "1", "m-1"),
        _raw_message("not json", "2", "m-2"),
        _raw_message("[1, 2]", "3", "m-3"),
    ]
    with Stubber(client) as stubber:
        stubber.add_response(
            "receive_message",
            {"Messages": raw},
            expected_params={
                "QueueUrl": URLS[FittingQueue.BE_FITTING_REQUEST],
                "MaxNumberOfMessages": 3,
                "WaitTimeSeconds": 20,
                "MessageSystemAttributeNames": ["ApproximateReceiveCount"],
            },
        )
        result = queue_client.receive(FittingQueue.BE_FITTING_REQUEST, max_messages=3)
        stubber.assert_no_pending_responses()

    assert result.succeeded
    first, second, third = result.messages
    assert first == ReceivedMessage(
        message_id="m-1",
        receipt_handle=RECEIPT,
        body=json.dumps(PAYLOAD),
        payload=PAYLOAD,
        parse_error=None,
        approximate_receive_count=1,
    )
    assert (second.payload, second.parse_error, second.approximate_receive_count) == (
        None,
        "invalid_json",
        2,
    )
    assert (third.payload, third.parse_error, third.approximate_receive_count) == (
        None,
        "not_object",
        3,
    )
    assert second.body == "not json"  # 원문은 그대로 돌려준다


def test_receive_with_visibility_override_and_empty_queue():
    queue_client, client = _stubbed()
    with Stubber(client) as stubber:
        stubber.add_response(
            "receive_message",
            {},
            expected_params={
                "QueueUrl": URLS[FittingQueue.AI_FITTING_RESPONSE],
                "MaxNumberOfMessages": 10,
                "WaitTimeSeconds": 0,
                "MessageSystemAttributeNames": ["ApproximateReceiveCount"],
                "VisibilityTimeout": 300,
            },
        )
        result = queue_client.receive(
            FittingQueue.AI_FITTING_RESPONSE,
            max_messages=10,
            wait_time_seconds=0,
            visibility_timeout=300,
        )

    assert result.succeeded and result.messages == ()


def test_missing_or_bad_receive_count_is_none():
    client = ReturningClient(
        {"Messages": [{"MessageId": "m", "ReceiptHandle": RECEIPT, "Body": "{}"}]}
    )
    (message,) = SqsQueueClient(client).receive(FittingQueue.BE_FITTING_REQUEST).messages
    assert message.approximate_receive_count is None and message.payload == {}

    client = ReturningClient({"Messages": [_raw_message("{}", receive_count="abc")]})
    (message,) = SqsQueueClient(client).receive(FittingQueue.BE_FITTING_REQUEST).messages
    assert message.approximate_receive_count is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_messages": 0},
        {"max_messages": 11},
        {"max_messages": True},
        {"wait_time_seconds": -1},
        {"wait_time_seconds": 21},
        {"wait_time_seconds": 1.5},
        {"visibility_timeout": -1},
        {"visibility_timeout": 43201},
    ],
)
def test_receive_arguments_are_validated(kwargs):
    client = RaisingClient(AssertionError("SQS 를 부르면 안 된다"))
    with pytest.raises(ValueError):
        SqsQueueClient(client).receive(FittingQueue.BE_FITTING_REQUEST, **kwargs)
    assert client.calls == []


def test_receive_errors_are_returned():
    queue_client, client = _stubbed()
    with Stubber(client) as stubber:
        stubber.add_client_error(
            "receive_message", service_error_code="AccessDenied", http_status_code=403
        )
        result = queue_client.receive(FittingQueue.BE_FITTING_REQUEST)

    assert (result.outcome, result.reason, result.messages) == (
        SqsOutcome.FAILED,
        "access_denied",
        (),
    )

    timeout = SqsQueueClient(RaisingClient(ReadTimeoutError(endpoint_url="x"))).receive(
        FittingQueue.BE_FITTING_REQUEST
    )
    assert (timeout.outcome, timeout.reason) == (SqsOutcome.UNKNOWN, "timeout")


# --- 삭제·Visibility 연장 ---


def test_delete_success():
    queue_client, client = _stubbed()
    with Stubber(client) as stubber:
        stubber.add_response(
            "delete_message",
            {},
            expected_params={
                "QueueUrl": URLS[FittingQueue.BE_FITTING_REQUEST],
                "ReceiptHandle": RECEIPT,
            },
        )
        result = queue_client.delete(FittingQueue.BE_FITTING_REQUEST, RECEIPT)

    assert result == SqsResult(
        SqsOperation.DELETE, FittingQueue.BE_FITTING_REQUEST, SqsOutcome.SUCCEEDED
    )


def test_change_visibility_success():
    queue_client, client = _stubbed()
    with Stubber(client) as stubber:
        stubber.add_response(
            "change_message_visibility",
            {},
            expected_params={
                "QueueUrl": URLS[FittingQueue.AI_FITTING_RESPONSE],
                "ReceiptHandle": RECEIPT,
                "VisibilityTimeout": 300,
            },
        )
        result = queue_client.change_visibility(FittingQueue.AI_FITTING_RESPONSE, RECEIPT, 300)

    assert result.succeeded and result.operation is SqsOperation.CHANGE_VISIBILITY


@pytest.mark.parametrize(
    ("operation", "code", "reason"),
    [
        ("delete_message", "ReceiptHandleIsInvalid", "receipt_handle_invalid"),
        ("delete_message", "InvalidIdFormat", "receipt_handle_invalid"),
        ("change_message_visibility", "ReceiptHandleIsInvalid", "receipt_handle_invalid"),
        ("change_message_visibility", "MessageNotInflight", "message_not_inflight"),
        (
            "change_message_visibility",
            "AWS.SimpleQueueService.MessageNotInflight",
            "message_not_inflight",
        ),
    ],
)
def test_delete_and_visibility_errors(operation, code, reason):
    queue_client, client = _stubbed()
    with Stubber(client) as stubber:
        stubber.add_client_error(operation, service_error_code=code, http_status_code=400)
        if operation == "delete_message":
            result = queue_client.delete(FittingQueue.BE_FITTING_REQUEST, RECEIPT)
        else:
            result = queue_client.change_visibility(FittingQueue.BE_FITTING_REQUEST, RECEIPT, 120)

    assert (result.outcome, result.reason) == (SqsOutcome.FAILED, reason)


def test_delete_timeout_is_unknown_and_not_retried():
    client = RaisingClient(ReadTimeoutError(endpoint_url="x"))
    result = SqsQueueClient(client).delete(FittingQueue.AI_FITTING_RESPONSE, RECEIPT)

    assert (result.outcome, result.reason, result.retryable) == (
        SqsOutcome.UNKNOWN,
        "timeout",
        True,
    )
    assert len(client.calls) == 1


@pytest.mark.parametrize("handle", ["", "   ", None, 123])
def test_receipt_handle_is_required(handle):
    client = RaisingClient(AssertionError("SQS 를 부르면 안 된다"))
    with pytest.raises(ValueError):
        SqsQueueClient(client).delete(FittingQueue.BE_FITTING_REQUEST, handle)
    with pytest.raises(ValueError):
        SqsQueueClient(client).change_visibility(FittingQueue.BE_FITTING_REQUEST, handle, 60)
    assert client.calls == []


@pytest.mark.parametrize("seconds", [-1, 43201, 1.5, True, None])
def test_visibility_timeout_range(seconds):
    client = RaisingClient(AssertionError("SQS 를 부르면 안 된다"))
    with pytest.raises(ValueError):
        SqsQueueClient(client).change_visibility(FittingQueue.BE_FITTING_REQUEST, RECEIPT, seconds)
    assert client.calls == []


# --- 로그·repr 에 비밀값이 남지 않는다 ---


def test_logs_and_reprs_hide_urls_receipts_bodies_and_credentials(caplog):
    queue_client, client = _stubbed()
    with caplog.at_level(logging.DEBUG, logger="app.clients.sqs"), Stubber(client) as stubber:
        stubber.add_response("send_message", {"MessageId": "msg-1"})
        stubber.add_response("receive_message", {"Messages": [_raw_message(json.dumps(PAYLOAD))]})
        stubber.add_client_error(
            "delete_message", service_error_code="ReceiptHandleIsInvalid", http_status_code=400
        )
        results = [
            queue_client.send_json(FittingQueue.AI_FITTING_RESPONSE, PAYLOAD),
            queue_client.receive(FittingQueue.BE_FITTING_REQUEST),
            queue_client.delete(FittingQueue.BE_FITTING_REQUEST, RECEIPT),
        ]

    assert "queue=ai-fitting-response outcome=succeeded" in caplog.text
    assert "queue=be-fitting-request" in caplog.text
    for secret in (ACCOUNT, RECEIPT, "users/3.jpg", ACCESS_KEY, SECRET_KEY, "https://sqs"):
        assert secret not in caplog.text
        for result in results:
            assert secret not in repr(result)


def test_non_botocore_bug_is_not_hidden():
    client = RaisingClient(RuntimeError("bug"))
    with pytest.raises(RuntimeError):
        SqsQueueClient(client).send_json(FittingQueue.AI_FITTING_RESPONSE, PAYLOAD)
    assert len(client.calls) == 1


def test_module_does_not_create_queues_or_touch_iam():
    with open(sqs_module.__file__, encoding="utf-8") as file:
        source = file.read()
    for forbidden in ("create_queue", "set_queue_attributes", "delete_queue", "purge_queue", "iam"):
        assert forbidden not in source
