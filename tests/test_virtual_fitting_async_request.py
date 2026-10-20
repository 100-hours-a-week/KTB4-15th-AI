"""가상피팅 V2 be-fitting-request 메시지 검증과 한 번의 수신.

실제 SQS 는 부르지 않는다. 수신은 실제 SqsQueueClient 에 botocore Stubber(가짜 자격증명)를 붙여 확인하므로,
예상하지 않은 SQS 호출(삭제·Visibility 연장)이 있으면 Stubber 가 실패시킨다.
"""

import json
import logging
from datetime import UTC, datetime

import boto3
import pytest
from botocore.exceptions import ReadTimeoutError
from botocore.stub import Stubber

from app.clients import sqs as sqs_module
from app.clients.sqs import FittingQueue, SqsConfigError, SqsOutcome, SqsQueueClient
from app.config import settings
from app.virtual_fitting.async_request import (
    FittingRequest,
    InvalidFittingRequest,
    InvalidRequestMessage,
    RequestErrorKind,
    ValidRequestMessage,
    parse_fitting_request,
    receive_fitting_requests,
)
from app.virtual_fitting.repositories import async_job_repository, async_job_state

REGION = "ap-northeast-2"
QUEUE_URL = f"https://sqs.{REGION}.amazonaws.com/123456789012/be-fitting-request"
RECEIPT = "AQEBsecret-receipt-handle=="
IMAGE_KEY = "users/10/body.png"
VALID = {
    "fitting_job_id": 123,
    "requested_at": "2026-10-17T03:00:00Z",
    "user_image_key": IMAGE_KEY,
    "products": [{"product_code": "123456"}, {"product_code": "789012"}],
}


def with_(**changes):
    payload = {**VALID, **changes}
    return {key: value for key, value in payload.items() if value is not ...}


# --- 메시지 형식 ---


@pytest.mark.parametrize(
    "products",
    [[{"product_code": "123456"}], [{"product_code": "123456"}, {"product_code": "789012"}]],
    ids=["one", "two"],
)
def test_valid_request_with_one_or_two_products(products):
    request = parse_fitting_request(with_(products=products))

    assert request.fitting_job_id == 123
    assert request.requested_at == datetime(2026, 10, 17, 3, 0, tzinfo=UTC)
    assert request.user_image_key == IMAGE_KEY
    assert request.product_codes == tuple(p["product_code"] for p in products)


def test_product_codes_stay_strings_with_leading_zeros_and_order():
    request = parse_fitting_request(
        with_(products=[{"product_code": "001234"}, {"product_code": "000010"}])
    )
    assert request.product_codes == ("001234", "000010")


def test_product_codes_fit_the_repository_input():
    request = parse_fitting_request(VALID)
    job = async_job_repository.NewAsyncFittingJob(
        fitting_job_id=request.fitting_job_id,
        request_fingerprint="fp",
        user_image_key=request.user_image_key,
        products=request.product_codes,
        requested_at=request.requested_at,
    )
    # 저장 전 검증(상품 1~2개, 문자열 코드)을 통과하는 형태다. DB 에는 접근하지 않는다.
    assert async_job_repository._check_new_job(job)[3] == ["123456", "789012"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-10-17T03:00:00Z", datetime(2026, 10, 17, 3, 0, tzinfo=UTC)),
        ("2026-10-17T12:00:00+09:00", datetime(2026, 10, 17, 3, 0, tzinfo=UTC)),
        ("2026-10-16T22:30:00-04:30", datetime(2026, 10, 17, 3, 0, tzinfo=UTC)),
        ("2026-10-17T03:00:00.123456+00:00", datetime(2026, 10, 17, 3, 0, 0, 123456, tzinfo=UTC)),
    ],
)
def test_requested_at_is_normalized_to_utc(value, expected):
    requested_at = parse_fitting_request(with_(requested_at=value)).requested_at
    assert requested_at == expected and requested_at.tzinfo is UTC


def test_requested_at_is_the_be_time_not_the_receive_time_or_created_at():
    request = parse_fitting_request(with_(created_at="2030-01-01T00:00:00Z"))
    assert request.requested_at == datetime(2026, 10, 17, 3, 0, tzinfo=UTC)
    with pytest.raises(InvalidFittingRequest) as exc_info:
        parse_fitting_request(with_(requested_at=..., created_at="2026-10-17T03:00:00Z"))
    assert (exc_info.value.kind, exc_info.value.field) == (
        RequestErrorKind.MISSING_FIELD,
        "requested_at",
    )


@pytest.mark.parametrize("field", ["fitting_job_id", "requested_at", "user_image_key", "products"])
def test_missing_required_field(field):
    with pytest.raises(InvalidFittingRequest) as exc_info:
        parse_fitting_request(with_(**{field: ...}))
    assert (exc_info.value.kind, exc_info.value.field) == (RequestErrorKind.MISSING_FIELD, field)


@pytest.mark.parametrize("job_id", [0, -1, 2**63, True, False, "123", 123.0, 1.5, None, [123]])
def test_invalid_job_id(job_id):
    with pytest.raises(InvalidFittingRequest) as exc_info:
        parse_fitting_request(with_(fitting_job_id=job_id))
    assert (exc_info.value.kind, exc_info.value.field) == (
        RequestErrorKind.INVALID_FIELD,
        "fitting_job_id",
    )


def test_max_bigint_job_id_is_accepted():
    assert parse_fitting_request(with_(fitting_job_id=2**63 - 1)).fitting_job_id == 2**63 - 1


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-17T03:00:00",  # 시간대 없음
        "2026-10-17",
        "17/10/2026 03:00",
        "",
        1760670000,
        None,
    ],
)
def test_invalid_requested_at(value):
    with pytest.raises(InvalidFittingRequest) as exc_info:
        parse_fitting_request(with_(requested_at=value))
    assert exc_info.value.field == "requested_at"


@pytest.mark.parametrize(
    "key",
    [
        "",
        " users/10/body.png",
        "/users/10/body.png",
        "users//body.png",
        "../secret.png",
        "https://bucket.s3.amazonaws.com/users/10/body.png",
        None,
        10,
    ],
)
def test_invalid_user_image_key(key):
    with pytest.raises(InvalidFittingRequest) as exc_info:
        parse_fitting_request(with_(user_image_key=key))
    assert exc_info.value.field == "user_image_key"
    assert str(key) not in str(exc_info.value) or not key  # 원본 Key 를 오류 메시지에 싣지 않는다


@pytest.mark.parametrize(
    ("products", "field"),
    [
        ([], "products"),
        ([{"product_code": str(i)} for i in range(3)], "products"),
        ([{"product_code": str(i)} for i in range(5)], "products"),
        ({"product_code": "123456"}, "products"),
        ("123456", "products"),
        (["123456"], "products[0].product_code"),
        ([{"code": "123456"}], "products[0].product_code"),
        ([{"product_code": 123456}], "products[0].product_code"),
        ([{"product_code": "   "}], "products[0].product_code"),
        ([{"product_code": "1"}, {"product_code": None}], "products[1].product_code"),
    ],
)
def test_invalid_products(products, field):
    with pytest.raises(InvalidFittingRequest) as exc_info:
        parse_fitting_request(with_(products=products))
    assert exc_info.value.field == field


@pytest.mark.parametrize("payload", [[VALID], "text", 123, None])
def test_non_object_payload(payload):
    with pytest.raises(InvalidFittingRequest) as exc_info:
        parse_fitting_request(payload)
    assert exc_info.value.kind is RequestErrorKind.NOT_OBJECT


def test_repr_hides_the_image_key():
    request = parse_fitting_request(VALID)
    assert IMAGE_KEY not in repr(request)
    assert isinstance(request, FittingRequest)


# --- 한 번의 수신 ---


@pytest.fixture(autouse=True)
def queue_settings(monkeypatch):
    for name in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "AWS_REGION", REGION)
    monkeypatch.setattr(settings, "BE_FITTING_REQUEST_QUEUE_URL", QUEUE_URL)


@pytest.fixture
def no_db_or_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("요청 검증·수신만으로 DB·모델을 부르면 안 된다")

    monkeypatch.setattr(async_job_repository.AsyncJobRepository, "create", forbidden)
    monkeypatch.setattr(async_job_state.AsyncJobStateRepository, "_run", forbidden)
    monkeypatch.setattr("app.config.database.get_connection_pool", forbidden)
    monkeypatch.setattr(
        "app.virtual_fitting.providers.runware_async.RunwarePrunaAsyncProvider.submit", forbidden
    )


def stubbed():
    client = boto3.client(
        "sqs",
        region_name=REGION,
        aws_access_key_id="AKIAUNITTEST",
        aws_secret_access_key="unit-test-secret",
    )
    return SqsQueueClient(client), client


def raw(body, message_id, receive_count="1"):
    return {
        "MessageId": message_id,
        "ReceiptHandle": f"{RECEIPT}-{message_id}",
        "MD5OfBody": "x" * 32,
        "Body": body,
        "Attributes": {"ApproximateReceiveCount": receive_count},
    }


def test_mixed_batch_validates_each_message_and_keeps_delivery_info(no_db_or_model, caplog):
    sqs, client = stubbed()
    messages = [
        raw(json.dumps(VALID), "m-valid", "1"),
        raw("not json", "m-json", "2"),
        raw("[1, 2]", "m-array", "1"),
        raw(json.dumps(with_(products=[])), "m-products", "3"),
        raw(json.dumps(with_(fitting_job_id="123")), "m-job-id", "1"),
        raw(json.dumps(with_(products=[{"product_code": "001234"}])), "m-valid-2", "1"),
    ]
    with Stubber(client) as stubber, caplog.at_level(
        logging.INFO, logger="app.virtual_fitting.diagnostics"
    ):
        stubber.add_response(
            "receive_message",
            {"Messages": messages},
            expected_params={
                "QueueUrl": QUEUE_URL,
                "MaxNumberOfMessages": 10,
                "WaitTimeSeconds": 0,
                "MessageSystemAttributeNames": ["ApproximateReceiveCount"],
            },
        )
        batch = receive_fitting_requests(sqs, max_messages=10, wait_time_seconds=0)
        stubber.assert_no_pending_responses()  # receive 한 번뿐. 삭제·Visibility 연장 호출 없음

    assert batch.received
    assert [m.message_id for m in batch.messages] == [
        "m-valid",
        "m-json",
        "m-array",
        "m-products",
        "m-job-id",
        "m-valid-2",
    ]
    assert [m.message_id for m in batch.valid] == ["m-valid", "m-valid-2"]
    first, json_error, array, products, job_id, second = batch.messages

    assert isinstance(first, ValidRequestMessage)
    assert first.receipt_handle == f"{RECEIPT}-m-valid"
    assert first.approximate_receive_count == 1
    assert first.request.product_codes == ("123456", "789012")
    assert second.request.product_codes == ("001234",)

    assert isinstance(json_error, InvalidRequestMessage)
    assert (json_error.error.kind, json_error.fitting_job_id) == (
        RequestErrorKind.INVALID_JSON,
        None,
    )
    assert json_error.approximate_receive_count == 2
    assert json_error.receipt_handle == f"{RECEIPT}-m-json"
    assert array.error.kind is RequestErrorKind.NOT_OBJECT
    # 형식은 틀렸지만 유효한 작업 번호는 참고용으로 전달한다. 실패 결과를 저장·발행하지는 않는다.
    assert (products.error.field, products.fitting_job_id) == ("products", 123)
    assert products.approximate_receive_count == 3
    # 잘못된 작업 번호는 만들어 내지 않는다.
    assert (job_id.error.field, job_id.fitting_job_id) == ("fitting_job_id", None)

    def diagnostic_events(event_name):
        parsed = []
        for record in caplog.records:
            if record.name != "app.virtual_fitting.diagnostics":
                continue
            fields = dict(
                part.split("=", 1)
                for part in record.getMessage().removeprefix("vf_diag ").split()
                if "=" in part
            )
            if fields.get("event") == event_name:
                parsed.append(fields)
        return parsed

    received = diagnostic_events("message_received")
    validated = diagnostic_events("message_validated")
    validation_failed = diagnostic_events("message_validation_failed")
    expected_ids = {message["MessageId"] for message in messages}
    assert len(received) == 6 and {event["message_id"] for event in received} == expected_ids
    assert len(validated) == 2
    assert {event["message_id"] for event in validated} == {"m-valid", "m-valid-2"}
    assert len(validation_failed) == 4
    assert {event["message_id"] for event in validation_failed} == {
        "m-json",
        "m-array",
        "m-products",
        "m-job-id",
    }
    valid_by_id = {event["message_id"]: event for event in validated}
    assert valid_by_id["m-valid"]["fitting_job_id"] == "123"
    assert valid_by_id["m-valid"]["product_codes"] == "123456,789012"
    assert valid_by_id["m-valid-2"]["fitting_job_id"] == "123"
    assert valid_by_id["m-valid-2"]["product_codes"] == "001234"
    invalid_by_id = {event["message_id"]: event for event in validation_failed}
    assert invalid_by_id["m-json"]["error_kind"] == "invalid_json"
    assert invalid_by_id["m-array"]["error_kind"] == "not_object"
    assert invalid_by_id["m-products"]["error_kind"] == "invalid_field"
    assert invalid_by_id["m-products"]["field"] == "products"
    assert invalid_by_id["m-job-id"]["error_kind"] == "invalid_field"
    assert invalid_by_id["m-job-id"]["field"] == "fitting_job_id"
    for secret in (RECEIPT, IMAGE_KEY, "not json"):
        assert secret not in caplog.text
        assert all(secret not in repr(item) for item in batch.messages)


def test_empty_queue(no_db_or_model):
    sqs, client = stubbed()
    with Stubber(client) as stubber:
        stubber.add_response("receive_message", {})
        batch = receive_fitting_requests(sqs)
    assert batch.received and batch.messages == () and batch.valid == () and batch.invalid == ()


@pytest.mark.parametrize(
    ("status", "code", "outcome"),
    [(403, "AccessDenied", SqsOutcome.FAILED), (500, "InternalError", SqsOutcome.UNKNOWN)],
)
def test_receive_error_is_not_a_validation_error(no_db_or_model, status, code, outcome):
    sqs, client = stubbed()
    with Stubber(client) as stubber:
        stubber.add_client_error(
            "receive_message", service_error_code=code, http_status_code=status
        )
        batch = receive_fitting_requests(sqs)
        stubber.assert_no_pending_responses()

    assert not batch.received
    assert batch.messages == () and batch.invalid == ()
    assert (batch.receive_result.outcome, batch.receive_result.error_code) == (outcome, code)


def test_receive_timeout_is_returned_once_without_retry(no_db_or_model):
    calls = []

    class TimingOut:
        def receive_message(self, **kwargs):
            calls.append(kwargs)
            raise ReadTimeoutError(endpoint_url="https://sqs")

    batch = receive_fitting_requests(SqsQueueClient(TimingOut()))

    assert not batch.received
    assert (batch.receive_result.outcome, batch.receive_result.reason) == (
        SqsOutcome.UNKNOWN,
        "timeout",
    )
    assert len(calls) == 1


def test_missing_queue_url_is_a_config_error(monkeypatch, no_db_or_model):
    monkeypatch.setattr(settings, "BE_FITTING_REQUEST_QUEUE_URL", "")

    class NeverCalled:
        def receive_message(self, **kwargs):
            raise AssertionError("설정 오류면 SQS 를 부르지 않는다")

    with pytest.raises(SqsConfigError):
        receive_fitting_requests(SqsQueueClient(NeverCalled()))


def test_uses_the_sqs_client_without_sdk_retry():
    # 이 기능은 SQS 전용 client(SDK 재시도 total_max_attempts=1)를 그대로 쓴다.
    sqs_module.get_sqs_client.cache_clear()
    try:
        assert sqs_module.get_sqs_client().meta.config.retries == {
            "mode": "standard",
            "total_max_attempts": 1,
        }
    finally:
        sqs_module.get_sqs_client.cache_clear()


def test_only_the_request_queue_is_read():
    seen = []

    class Recording:
        def receive_message(self, **kwargs):
            seen.append(kwargs["QueueUrl"])
            return {}

    receive_fitting_requests(SqsQueueClient(Recording()))
    assert seen == [QUEUE_URL]
    assert FittingQueue.BE_FITTING_REQUEST.value == "BE_FITTING_REQUEST_QUEUE_URL"
