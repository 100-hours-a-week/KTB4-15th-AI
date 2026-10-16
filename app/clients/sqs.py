"""V2 비동기 가상피팅이 쓰는 SQS 공통 client. JSON 메시지 전송·수신·삭제·Visibility Timeout 연장.

boto3 의 표준 credential provider chain 과 settings.AWS_REGION 을 S3 와 같이 쓴다. 큐·DLQ 생성과
IAM 설정은 Cloud 담당이라 여기서 하지 않는다.

- SDK 재시도를 끈다(total_max_attempts=1). 이 모듈도 다시 보내지 않는다. 결과를 그대로 돌려주고
  재전송 여부·횟수(예: 결과 메시지 최대 3회)는 호출하는 쪽 정책이 정한다.
- 결과는 SUCCEEDED / FAILED(확실히 처리되지 않음) / UNKNOWN(처리됐는지 알 수 없음) 으로 나눈다.
  Standard 큐라 같은 메시지가 두 번 갈 수 있으므로 받는 쪽은 중복을 견뎌야 한다.
- 큐별 업무 메시지 형식은 검증하지 않는다(이후 단계). 본문이 JSON 객체인지만 본다.
- 로그에는 큐 이름·결과만 남긴다. 큐 URL(계정 ID), ReceiptHandle, 본문은 남기지 않는다.
"""

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# SQS API 한도.
MAX_RECEIVE_MESSAGES = 10
MAX_WAIT_TIME_SECONDS = 20
MAX_VISIBILITY_TIMEOUT_SECONDS = 12 * 3600

_QUEUE_URL_PATTERN = re.compile(
    r"^https://sqs\.(?P<region>[a-z0-9-]+)\.amazonaws\.com/\d{12}/(?P<name>[A-Za-z0-9_-]{1,75}(?:\.fifo)?)$"
)
# JSON 프로토콜이지만 query 호환 모드라 예전 코드(AWS.SimpleQueueService.*)로 올 수도 있다.
_THROTTLE_CODES = frozenset(
    {"RequestThrottled", "ThrottlingException", "Throttling", "KmsThrottled"}
)
_REASON_BY_CODE = {
    "QueueDoesNotExist": "queue_not_found",
    "AWS.SimpleQueueService.NonExistentQueue": "queue_not_found",
    "ReceiptHandleIsInvalid": "receipt_handle_invalid",
    "InvalidIdFormat": "receipt_handle_invalid",
    "MessageNotInflight": "message_not_inflight",
    "AWS.SimpleQueueService.MessageNotInflight": "message_not_inflight",
    "InvalidMessageContents": "invalid_message",
    "AccessDenied": "access_denied",
    "AccessDeniedException": "access_denied",
    "InvalidSecurity": "access_denied",
    "KmsAccessDenied": "access_denied",
}


class SqsConfigError(RuntimeError):
    """큐 URL 설정이 없거나 형식·리전이 맞지 않을 때. SQS 에 요청하기 전에 난다."""

    code = "SQS_CONFIG_ERROR"


class FittingQueue(str, Enum):
    """AI 가 쓰는 큐. 값은 큐 URL 을 담은 settings 이름이다."""

    BE_FITTING_REQUEST = "BE_FITTING_REQUEST_QUEUE_URL"
    AI_FITTING_RESPONSE = "AI_FITTING_RESPONSE_QUEUE_URL"
    BE_FITTING_RESPONSE = "BE_FITTING_RESPONSE_QUEUE_URL"


class SqsOperation(str, Enum):
    SEND = "send"
    RECEIVE = "receive"
    DELETE = "delete"
    CHANGE_VISIBILITY = "change_visibility"


_CONSUME = frozenset({SqsOperation.RECEIVE, SqsOperation.DELETE, SqsOperation.CHANGE_VISIBILITY})
# AI 의 역할 밖 호출을 막는다. be-fitting-request 는 BE 가 보내고, be-fitting-response 는 BE 가 소비한다.
_ALLOWED_OPERATIONS = {
    FittingQueue.BE_FITTING_REQUEST: _CONSUME,
    FittingQueue.AI_FITTING_RESPONSE: _CONSUME | {SqsOperation.SEND},
    FittingQueue.BE_FITTING_RESPONSE: frozenset({SqsOperation.SEND}),
}


class SqsOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    # 요청이 처리되지 않았다(4xx 거절, 연결 전 실패, 자격증명 없음).
    FAILED = "failed"
    # 처리됐는지 알 수 없다(응답 timeout, 연결 끊김, 5xx). 전송이면 큐에 들어갔을 수 있다.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReceivedMessage:
    message_id: str
    receipt_handle: str = ""
    body: str = ""
    # 본문이 JSON 객체면 그 dict, 아니면 None 이고 parse_error 에 이유가 있다.
    payload: dict[str, Any] | None = None
    parse_error: str | None = None
    # SQS 가 이 메시지를 건넨 횟수(기록용). 모델 실행 횟수나 최종 실패 판단에 쓰지 않는다.
    approximate_receive_count: int | None = None

    def __repr__(self) -> str:  # ReceiptHandle·본문은 repr 에도 남기지 않는다.
        return (
            f"ReceivedMessage(message_id={self.message_id!r}, parse_error={self.parse_error!r}, "
            f"approximate_receive_count={self.approximate_receive_count!r})"
        )


@dataclass(frozen=True)
class SqsResult:
    """SQS 호출 한 번의 결과. URL·자격증명·ReceiptHandle 은 담지 않는다.

    reason 값:
      ok                                   — 성공
      queue_not_found / access_denied      — 설정·권한 문제 (FAILED)
      invalid_message                      — 본문 거절 (FAILED)
      receipt_handle_invalid               — ReceiptHandle 이 틀렸거나 만료 (FAILED)
      message_not_inflight                 — Visibility 연장 대상이 처리 중이 아님 (FAILED)
      throttled                            — 요청 한도 초과로 거절 (FAILED, retryable)
      client_error                         — 그 밖의 4xx (FAILED)
      not_connected                        — 연결 전에 실패해 보내지 못함 (FAILED, retryable)
      credentials                          — 자격증명 없음 등 SDK 설정 문제 (FAILED)
      invalid_request                      — SDK 가 요청 인자를 거절해 보내지 않음 (FAILED)
      server_error                         — 5xx (UNKNOWN, retryable)
      timeout / connection_closed          — 응답을 받지 못함 (UNKNOWN, retryable)
      missing_message_id                   — 전송 응답에 MessageId 가 없음 (UNKNOWN)
      unknown_error                        — 분류하지 못한 botocore 오류 (UNKNOWN)
    botocore 밖의 예외(코드 버그)는 결과로 바꾸지 않고 그대로 올라간다. 전송 중이었다면 호출하는 쪽은
    UNKNOWN 과 같게 다뤄야 한다.
    """

    operation: SqsOperation
    queue: FittingQueue
    outcome: SqsOutcome
    reason: str = "ok"
    error_code: str | None = None
    http_status: int | None = None
    # 같은 요청을 다시 해볼 만한지에 대한 힌트. 자동 재시도는 하지 않는다.
    retryable: bool = False
    message_id: str | None = None
    messages: tuple[ReceivedMessage, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.outcome is SqsOutcome.SUCCEEDED


def queue_name(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def get_queue_url(queue: FittingQueue) -> str:
    """settings 의 큐 URL. 비었거나, 형식이 틀렸거나, AWS_REGION 과 리전이 다르거나, FIFO 면 설정 오류."""
    url = str(getattr(settings, queue.value, "") or "").strip()
    if not url:
        raise SqsConfigError(f"{queue.value} 환경변수가 설정되지 않았습니다.")
    match = _QUEUE_URL_PATTERN.match(url)
    if match is None:
        raise SqsConfigError(f"{queue.value} 가 SQS 큐 URL 형식이 아닙니다.")
    if match["region"] != settings.AWS_REGION:
        raise SqsConfigError(f"{queue.value} 의 리전이 AWS_REGION 과 다릅니다.")
    if match["name"].endswith(".fifo"):
        raise SqsConfigError(f"{queue.value} 는 Standard 큐여야 합니다.")
    return url


@lru_cache(maxsize=1)
def get_sqs_client() -> Any:
    """SQS 전용 client. S3 client 설정과 별개이며 SDK 재시도를 하지 않는다."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "sqs",
        region_name=settings.AWS_REGION,
        config=Config(retries={"mode": "standard", "total_max_attempts": 1}),
    )


def _check_int(name: str, value: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} 는 {low}~{high} 사이 정수여야 합니다.")
    return value


def _check_receipt_handle(receipt_handle: str) -> str:
    if not isinstance(receipt_handle, str) or not receipt_handle.strip():
        raise ValueError("receipt_handle 이 비어 있습니다.")
    return receipt_handle


def encode_json_body(payload: Mapping[str, Any]) -> str:
    """JSON 객체를 SQS 본문 문자열로 만든다. NaN·Infinity·직렬화할 수 없는 값·깨진 문자는 거절한다."""
    if not isinstance(payload, Mapping):
        raise TypeError("SQS 메시지 본문은 JSON 객체(dict)여야 합니다.")
    try:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        body.encode("utf-8")  # 짝 없는 surrogate 같은 SQS 가 받지 않는 문자
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("SQS 메시지 본문을 JSON 으로 만들 수 없습니다.") from error
    return body


def _parse_body(body: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "not_object"
    return data, None


def _to_message(raw: Mapping[str, Any]) -> ReceivedMessage:
    body = raw.get("Body")
    body = body if isinstance(body, str) else ""
    payload, parse_error = _parse_body(body)
    count = (raw.get("Attributes") or {}).get("ApproximateReceiveCount")
    try:
        receive_count = int(count) if count is not None else None
    except (TypeError, ValueError):
        receive_count = None
    return ReceivedMessage(
        message_id=str(raw.get("MessageId") or ""),
        receipt_handle=str(raw.get("ReceiptHandle") or ""),
        body=body,
        payload=payload,
        parse_error=parse_error,
        approximate_receive_count=receive_count,
    )


def _classify_error(error: BaseException) -> dict[str, Any]:
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
        ConnectionClosedError,
        ConnectTimeoutError,
        EndpointConnectionError,
        ReadTimeoutError,
    )

    if isinstance(error, ClientError):
        response = error.response or {}
        code = str(response.get("Error", {}).get("Code", "")) or None
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        info = {"error_code": code, "http_status": status}
        if isinstance(status, int) and status >= 500:
            return {
                **info,
                "outcome": SqsOutcome.UNKNOWN,
                "reason": "server_error",
                "retryable": True,
            }
        if code in _THROTTLE_CODES or status == 429:
            return {**info, "outcome": SqsOutcome.FAILED, "reason": "throttled", "retryable": True}
        reason = _REASON_BY_CODE.get(code or "", "client_error")
        return {**info, "outcome": SqsOutcome.FAILED, "reason": reason}

    name = type(error).__name__
    # 연결을 맺지 못했다 → 요청이 SQS 에 닿지 않았다.
    if isinstance(error, (ConnectTimeoutError, EndpointConnectionError)):
        return {
            "error_code": name,
            "outcome": SqsOutcome.FAILED,
            "reason": "not_connected",
            "retryable": True,
        }
    # 보낸 뒤 응답을 받지 못했다 → 처리됐을 수 있다.
    if isinstance(error, ReadTimeoutError):
        return {
            "error_code": name,
            "outcome": SqsOutcome.UNKNOWN,
            "reason": "timeout",
            "retryable": True,
        }
    if isinstance(error, ConnectionClosedError):
        return {
            "error_code": name,
            "outcome": SqsOutcome.UNKNOWN,
            "reason": "connection_closed",
            "retryable": True,
        }
    if isinstance(error, BotoCoreError) and name in {
        "NoCredentialsError",
        "PartialCredentialsError",
        "NoRegionError",
    }:
        return {"error_code": name, "outcome": SqsOutcome.FAILED, "reason": "credentials"}
    if isinstance(error, BotoCoreError) and name == "ParamValidationError":
        # SDK 가 요청을 만들기 전에 거절했다(보내지 않음).
        return {"error_code": name, "outcome": SqsOutcome.FAILED, "reason": "invalid_request"}
    return {"error_code": name, "outcome": SqsOutcome.UNKNOWN, "reason": "unknown_error"}


class SqsQueueClient:
    """세 큐에 대한 JSON 메시지 전송·수신·삭제·Visibility Timeout 연장. 각 호출은 SQS 요청 한 번이다."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client if client is not None else get_sqs_client()

    def send_json(self, queue: FittingQueue, payload: Mapping[str, Any]) -> SqsResult:
        """JSON 객체를 한 번 보낸다. SUCCEEDED 는 MessageId 를 받았을 때만이다."""
        url = self._url(queue, SqsOperation.SEND)
        body = encode_json_body(payload)
        result = self._call(
            SqsOperation.SEND,
            queue,
            lambda: self._client.send_message(QueueUrl=url, MessageBody=body),
        )
        if isinstance(result, SqsResult):
            return self._log(result, url)
        message_id = result.get("MessageId") if isinstance(result, Mapping) else None
        if not message_id:
            # 2xx 인데 MessageId 가 없으면 등록 근거가 없다.
            outcome = SqsResult(SqsOperation.SEND, queue, SqsOutcome.UNKNOWN, "missing_message_id")
        else:
            outcome = SqsResult(
                SqsOperation.SEND, queue, SqsOutcome.SUCCEEDED, message_id=message_id
            )
        return self._log(outcome, url)

    def receive(
        self,
        queue: FittingQueue,
        *,
        max_messages: int = 1,
        wait_time_seconds: int = MAX_WAIT_TIME_SECONDS,
        visibility_timeout: int | None = None,
    ) -> SqsResult:
        """long polling 으로 받는다. 메시지가 없으면 SUCCEEDED 에 messages=() 다.

        visibility_timeout 을 주지 않으면 큐 설정값(Cloud)을 쓴다.
        """
        url = self._url(queue, SqsOperation.RECEIVE)
        params: dict[str, Any] = {
            "QueueUrl": url,
            "MaxNumberOfMessages": _check_int(
                "max_messages", max_messages, 1, MAX_RECEIVE_MESSAGES
            ),
            "WaitTimeSeconds": _check_int(
                "wait_time_seconds", wait_time_seconds, 0, MAX_WAIT_TIME_SECONDS
            ),
            "MessageSystemAttributeNames": ["ApproximateReceiveCount"],
        }
        if visibility_timeout is not None:
            params["VisibilityTimeout"] = _check_int(
                "visibility_timeout", visibility_timeout, 0, MAX_VISIBILITY_TIMEOUT_SECONDS
            )
        result = self._call(
            SqsOperation.RECEIVE, queue, lambda: self._client.receive_message(**params)
        )
        if isinstance(result, SqsResult):
            return self._log(result, url)
        raw_messages = result.get("Messages") if isinstance(result, Mapping) else None
        messages = tuple(
            _to_message(raw) for raw in (raw_messages or []) if isinstance(raw, Mapping)
        )
        received = SqsResult(SqsOperation.RECEIVE, queue, SqsOutcome.SUCCEEDED, messages=messages)
        if messages:
            self._log(received, url)
        return received

    def delete(self, queue: FittingQueue, receipt_handle: str) -> SqsResult:
        url = self._url(queue, SqsOperation.DELETE)
        handle = _check_receipt_handle(receipt_handle)
        result = self._call(
            SqsOperation.DELETE,
            queue,
            lambda: self._client.delete_message(QueueUrl=url, ReceiptHandle=handle),
        )
        if not isinstance(result, SqsResult):
            result = SqsResult(SqsOperation.DELETE, queue, SqsOutcome.SUCCEEDED)
        return self._log(result, url)

    def change_visibility(
        self, queue: FittingQueue, receipt_handle: str, visibility_timeout: int
    ) -> SqsResult:
        """지금부터 visibility_timeout 초 동안 다른 Worker 에게 보이지 않게 한다(연장)."""
        url = self._url(queue, SqsOperation.CHANGE_VISIBILITY)
        handle = _check_receipt_handle(receipt_handle)
        seconds = _check_int(
            "visibility_timeout", visibility_timeout, 0, MAX_VISIBILITY_TIMEOUT_SECONDS
        )
        result = self._call(
            SqsOperation.CHANGE_VISIBILITY,
            queue,
            lambda: self._client.change_message_visibility(
                QueueUrl=url, ReceiptHandle=handle, VisibilityTimeout=seconds
            ),
        )
        if not isinstance(result, SqsResult):
            result = SqsResult(SqsOperation.CHANGE_VISIBILITY, queue, SqsOutcome.SUCCEEDED)
        return self._log(result, url)

    @staticmethod
    def _url(queue: FittingQueue, operation: SqsOperation) -> str:
        if not isinstance(queue, FittingQueue):
            raise TypeError("queue 는 FittingQueue 여야 합니다.")
        if operation not in _ALLOWED_OPERATIONS[queue]:
            raise ValueError(f"AI 는 {queue.name} 큐에 {operation.value} 를 하지 않습니다.")
        return get_queue_url(queue)

    @staticmethod
    def _call(operation: SqsOperation, queue: FittingQueue, request: Any) -> Any:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            return request()
        except (ClientError, BotoCoreError) as error:  # SQS·네트워크 오류는 결과로 돌려준다.
            return SqsResult(operation, queue, **_classify_error(error))

    @staticmethod
    def _log(result: SqsResult, url: str) -> SqsResult:
        level = logging.INFO if result.succeeded else logging.WARNING
        logger.log(
            level,
            "sqs %s queue=%s outcome=%s reason=%s error_code=%s http_status=%s "
            "message_id=%s received=%s",
            result.operation.value,
            queue_name(url),
            result.outcome.value,
            result.reason,
            result.error_code,
            result.http_status,
            result.message_id,
            len(result.messages),
        )
        return result
