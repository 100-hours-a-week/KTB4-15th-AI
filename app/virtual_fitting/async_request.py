"""가상피팅 V2: be-fitting-request 메시지 형식 검증과 한 번의 수신.

BE 가 be-fitting-request 에 보내는 메시지(모든 필드 필수):
    {"fitting_job_id": 123, "requested_at": "2026-10-17T03:00:00Z",
     "user_image_key": "users/10/body.png",
     "products": [{"product_code": "123456"}, {"product_code": "789012"}]}

이 모듈은 형식만 본다. 사진·상품 DB 조회, 상품 조합·분류·예산·잔액 확인, AI DB 등록, 중복·입력 비교는 이후
요청 Worker 가 한다. 메시지는 삭제하지 않는다(처리·복구 기록을 남긴 뒤 요청 Worker 가 삭제한다).
수신은 app.clients.sqs.SqsQueueClient.receive 를 한 번 부른다(SDK 재시도 없음, 이 모듈도 다시 부르지 않음).
나중에 FastAPI 와 함께 시작하는 요청 Worker 가 이 함수를 반복해서 부른다.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from app.clients.s3 import InvalidS3KeyError, validate_s3_key
from app.clients.sqs import FittingQueue, ReceivedMessage, SqsQueueClient, SqsResult
from app.virtual_fitting import diagnostics as diag
from app.virtual_fitting.repositories.async_job_repository import MAX_PRODUCTS, _check_job_id

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("fitting_job_id", "requested_at", "user_image_key", "products")


class RequestErrorKind(str, Enum):
    INVALID_JSON = "invalid_json"
    NOT_OBJECT = "not_object"
    MISSING_FIELD = "missing_field"
    INVALID_FIELD = "invalid_field"


class InvalidFittingRequest(ValueError):
    """메시지 형식 오류. 원본 값(이미지 Key 등)은 담지 않는다."""

    def __init__(self, kind: RequestErrorKind, field: str | None, detail: str) -> None:
        super().__init__(f"{kind.value}: {field or '-'}: {detail}")
        self.kind = kind
        self.field = field
        self.detail = detail


@dataclass(frozen=True)
class FittingRequest:
    """검증된 BE 요청. requested_at 은 BE 가 보낸 시각을 UTC 로 바꾼 값이다(AI 수신 시각이 아님)."""

    fitting_job_id: int
    requested_at: datetime
    user_image_key: str
    # JSON 의 products 객체 배열에서 꺼낸 상품 코드. 문자열 그대로(앞자리 0 보존), 순서 유지.
    # AsyncJobRepository 의 NewAsyncFittingJob.products 에 그대로 넘길 수 있는 형태다.
    product_codes: tuple[str, ...]

    def __repr__(self) -> str:  # 원본 이미지 Key 는 repr·로그에 남기지 않는다.
        return (
            f"FittingRequest(fitting_job_id={self.fitting_job_id!r}, "
            f"requested_at={self.requested_at.isoformat()!r}, products={len(self.product_codes)})"
        )


def _invalid(field: str, detail: str) -> InvalidFittingRequest:
    return InvalidFittingRequest(RequestErrorKind.INVALID_FIELD, field, detail)


def _job_id(value: Any) -> int:
    try:
        return _check_job_id(value)  # bool·문자열·실수는 변환하지 않고 거절한다.
    except ValueError as error:
        raise _invalid("fitting_job_id", "BIGINT 범위의 양의 정수여야 합니다.") from error


def _requested_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise _invalid("requested_at", "ISO 8601 문자열이어야 합니다.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise _invalid("requested_at", "ISO 8601 형식이 아닙니다.") from error
    if parsed.utcoffset() is None:
        raise _invalid("requested_at", "시간대가 있어야 합니다.")
    return parsed.astimezone(UTC)


def _user_image_key(value: Any) -> str:
    try:
        return validate_s3_key(value)
    except InvalidS3KeyError as error:
        raise _invalid("user_image_key", "S3 Key 형식이 아닙니다.") from error


def _product_codes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _invalid("products", "상품 객체 배열이어야 합니다.")
    if not 1 <= len(value) <= MAX_PRODUCTS:
        raise _invalid("products", f"상품은 1~{MAX_PRODUCTS}개여야 합니다.")
    codes = []
    for index, product in enumerate(value):
        field = f"products[{index}].product_code"
        if not isinstance(product, Mapping) or "product_code" not in product:
            raise InvalidFittingRequest(
                RequestErrorKind.MISSING_FIELD, field, "product_code 가 없습니다."
            )
        code = product["product_code"]
        if not isinstance(code, str) or not code.strip():
            raise _invalid(field, "비어 있지 않은 문자열이어야 합니다.")
        codes.append(code)
    return tuple(codes)


def parse_fitting_request(payload: Any) -> FittingRequest:
    """JSON 객체(dict)를 검증한다. 합의에 없는 추가 필드는 쓰지 않고 무시한다(created_at 포함)."""
    if not isinstance(payload, Mapping):
        raise InvalidFittingRequest(RequestErrorKind.NOT_OBJECT, None, "JSON 객체가 아닙니다.")
    for field in REQUIRED_FIELDS:
        if field not in payload:
            raise InvalidFittingRequest(
                RequestErrorKind.MISSING_FIELD, field, "필수 필드가 없습니다."
            )
    return FittingRequest(
        fitting_job_id=_job_id(payload["fitting_job_id"]),
        requested_at=_requested_at(payload["requested_at"]),
        user_image_key=_user_image_key(payload["user_image_key"]),
        product_codes=_product_codes(payload["products"]),
    )


def readable_job_id(payload: Any) -> int | None:
    """형식 오류인 메시지에서도 유효한 fitting_job_id 를 읽을 수 있으면 돌려준다(없거나 잘못됐으면 None)."""
    if not isinstance(payload, Mapping):
        return None
    try:
        return _check_job_id(payload.get("fitting_job_id"))
    except ValueError:
        return None


@dataclass(frozen=True)
class _Delivery:
    """SQS 가 건넨 메시지의 수신 정보. 삭제·Visibility 연장은 호출부가 receipt_handle 로 한다."""

    message_id: str
    receipt_handle: str
    # SQS 가 이 메시지를 건넨 횟수(기록용). 실행 횟수·전송 횟수·최종 성공/실패 판단에 쓰지 않는다.
    approximate_receive_count: int | None


@dataclass(frozen=True, repr=False)
class ValidRequestMessage(_Delivery):
    request: FittingRequest

    def __repr__(self) -> str:
        return (
            f"ValidRequestMessage(message_id={self.message_id!r}, request={self.request!r}, "
            f"approximate_receive_count={self.approximate_receive_count!r})"
        )


@dataclass(frozen=True, repr=False)
class InvalidRequestMessage(_Delivery):
    error: InvalidFittingRequest
    # 읽을 수 있었던 유효한 작업 번호(참고용). 없으면 None. 이 번호로 바로 실패 결과를 저장·발행하지 않는다.
    fitting_job_id: int | None

    def __repr__(self) -> str:
        return (
            f"InvalidRequestMessage(message_id={self.message_id!r}, kind={self.error.kind.value!r}, "
            f"field={self.error.field!r}, fitting_job_id={self.fitting_job_id!r}, "
            f"approximate_receive_count={self.approximate_receive_count!r})"
        )


@dataclass(frozen=True)
class RequestBatch:
    """한 번의 수신 결과. receive_result 가 실패면 messages 는 비어 있다(수신 실패 ≠ 본문 검증 실패)."""

    receive_result: SqsResult
    # 받은 순서 그대로. 각 항목은 ValidRequestMessage 또는 InvalidRequestMessage.
    messages: tuple[ValidRequestMessage | InvalidRequestMessage, ...] = ()

    @property
    def received(self) -> bool:
        return self.receive_result.succeeded

    @property
    def valid(self) -> tuple[ValidRequestMessage, ...]:
        return tuple(m for m in self.messages if isinstance(m, ValidRequestMessage))

    @property
    def invalid(self) -> tuple[InvalidRequestMessage, ...]:
        return tuple(m for m in self.messages if isinstance(m, InvalidRequestMessage))


def _classify(message: ReceivedMessage) -> ValidRequestMessage | InvalidRequestMessage:
    delivery = {
        "message_id": message.message_id,
        "receipt_handle": message.receipt_handle,
        "approximate_receive_count": message.approximate_receive_count,
    }
    if message.parse_error is not None:
        kind = (
            RequestErrorKind.NOT_OBJECT
            if message.parse_error == "not_object"
            else RequestErrorKind.INVALID_JSON
        )
        error = InvalidFittingRequest(kind, None, "본문이 JSON 객체가 아닙니다.")
        return InvalidRequestMessage(**delivery, error=error, fitting_job_id=None)
    try:
        return ValidRequestMessage(**delivery, request=parse_fitting_request(message.payload))
    except InvalidFittingRequest as error:
        return InvalidRequestMessage(
            **delivery, error=error, fitting_job_id=readable_job_id(message.payload)
        )


def receive_fitting_requests(
    sqs: SqsQueueClient,
    *,
    max_messages: int = 1,
    wait_time_seconds: int = 20,
    visibility_timeout: int | None = None,
) -> RequestBatch:
    """be-fitting-request 를 한 번 받아 메시지마다 검증한다. 삭제·Visibility 연장·DB 기록은 하지 않는다.

    수신 실패는 SqsQueueClient 의 결과(FAILED/UNKNOWN)를 그대로 receive_result 에 담아 돌려준다. 큐 URL 설정
    오류(SqsConfigError)는 SqsQueueClient 와 같이 예외로 올라간다.
    """
    result = sqs.receive(
        FittingQueue.BE_FITTING_REQUEST,
        max_messages=max_messages,
        wait_time_seconds=wait_time_seconds,
        visibility_timeout=visibility_timeout,
    )
    if not result.succeeded:
        return RequestBatch(receive_result=result)
    messages = tuple(_classify(message) for message in result.messages)
    for item in messages:
        # 수신 직후(작업 저장 전). 수신 횟수는 SQS 의 ApproximateReceiveCount, 없으면 unknown.
        count = item.approximate_receive_count
        diag.event(
            "request_received",
            component="request_worker",
            message_id=item.message_id,
            receive_count=count if count is not None else "unknown",
        )
        if isinstance(item, ValidRequestMessage):
            diag.event(
                "request_validated",
                component="request_worker",
                message_id=item.message_id,
                fitting_job_id=item.request.fitting_job_id,
                product_codes=diag.codes(item.request.product_codes),
            )
        else:
            diag.event(
                "request_invalid",
                logging.WARNING,
                component="request_worker",
                message_id=item.message_id,
                fitting_job_id=item.fitting_job_id,
                error_kind=item.error.kind.value,
                field=item.error.field,
            )
        if isinstance(item, InvalidRequestMessage):
            logger.warning(
                "be-fitting-request invalid message_id=%s fitting_job_id=%s kind=%s field=%s "
                "receive_count=%s",
                item.message_id,
                item.fitting_job_id,
                item.error.kind.value,
                item.error.field,
                item.approximate_receive_count,
            )
    if messages:
        logger.info(
            "be-fitting-request received=%s valid=%s invalid=%s",
            len(messages),
            sum(isinstance(item, ValidRequestMessage) for item in messages),
            sum(isinstance(item, InvalidRequestMessage) for item in messages),
        )
    return RequestBatch(receive_result=result, messages=messages)
