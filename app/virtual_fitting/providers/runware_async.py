"""V2: Runware 플랫폼 경유 Pruna p-image-try-on 비동기 가상피팅 접수 Provider.

POST https://api.runware.ai/v1 (V1 과 같은 endpoint·모델·VTON 키)
  본문: [{"taskType": "imageInference", "taskUUID": <AI 요청 Worker 가 만든 UUID v4>,
          "model": "prunaai:p-image@try-on",
          "inputs": {"referenceImages": [{"image", "role": "person" | "garment"}]},
          "outputType": "URL", "outputFormat": "JPG", "numberResults": 1, "includeCost": true,
          "deliveryMethod": "async", "webhookURL": ..., "uploadEndpoint": ...}]
  접수(설계 문서 예시, 실제 형식은 연동 테스트에서 확정):
        {"data": [{"taskType": "imageInference", "taskUUID": "...", "status": "processing"}]}

이 Provider 는 "접수 요청 한 번"만 한다. 결과 이미지는 Runware 가 uploadEndpoint 로 PUT 하고
완료는 webhookURL 로 알린다. 응답은 세 가지로 나눠 돌려준다.

- ACCEPTED: HTTP 2xx, JSON 객체, errors 없음, 요청한 taskUUID 가 응답 data 에 있다.
- REJECTED: Runware 가 이 요청을 받지 않았다고 볼 근거가 있다(4xx, 또는 errors 가 담긴 응답).
- UNKNOWN : 접수됐는지 알 수 없다. timeout·연결 끊김·네트워크 오류, 429·5xx, JSON 이 아닌 2xx,
            taskUUID 가 없거나 다른 2xx 등. 이미 접수됐을 수 있으므로 같은 taskUUID 로 다시 보내지 않고
            상위 Worker / Watchdog(getResponse) 이 확인한다.

대기 안내(retry_after): 본문 errors[].retryAfter 와 HTTP Retry-After 헤더를 함께 본다. 둘 다 Runware 계약상
초 단위다(https://runware.ai/docs/models-api/rate-limits). 단위를 추측해 바꾸지 않는다.
- 잘못된 값(안내 아님): 숫자가 아닌 값(본문의 문자열 포함)·음수·NaN·Infinity 표기·HTTP-date·지수 표기 헤더.
- 유효한 값: 0 이상의 숫자. 크기 제한 없이 그대로 쓰고, 둘 다 유효하면 더 긴 쪽을 쓴다(줄이지 않음).
- 유효하지만 float 로 표현할 수 없는 큰 값: retry_after 대신 retry_after_unrepresentable=True 로 알린다.
  이때는 다른 쪽의 짧은 값을 채택하지 않는다. 호출하는 쪽이 새 생성 요청을 보류한다.

자동 재시도는 하지 않는다. taskUUID 생성·S3 Presigned URL 생성·AI DB 기록은 호출하는 쪽의 일이다.
webhookURL(token)·uploadEndpoint(서명)·person 이미지 URL(서명)·API 키는 로그와 결과에 남기지 않는다.
"""

import json
import logging
import math
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.virtual_fitting.providers.base import FittingInput
from app.virtual_fitting.providers.http import _is_timeout
from app.virtual_fitting.providers.runware import (
    RUNWARE_ENDPOINT,
    RUNWARE_MODEL,
    get_runware_vton_api_key,
)

logger = logging.getLogger(__name__)

# 접수 응답만 기다린다(이미지 생성은 기다리지 않는다). 실제 접수 지연을 보고 조정한다.
DEFAULT_SUBMIT_TIMEOUT = 30.0
OUTPUT_FORMAT = "JPG"
# Pruna try-on 모델 제약: person 1장, garment 1~11장.
MAX_GARMENT_IMAGES = 11
# Runware 문서상 다시 보내도 될 수 있는 오류 코드. 판단은 상위 Worker 가 한다.
RETRYABLE_ERROR_CODES = frozenset({"timeoutProvider", "providerRateLimitExceeded"})
_DETAIL_LIMIT = 200
_URL_PATTERN = re.compile(r"https?://\S+")


class SubmissionOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AsyncSubmission:
    """접수 요청 결과. URL·토큰·키는 담지 않는다.

    reason 값:
      accepted                       — 접수 근거 확인
      error_response                 — 2xx 인데 errors 가 있다
      balance_exhausted              — HTTP 402
      client_error                   — 그 밖의 4xx (400·401·403·404 등)
      rate_limited / server_error    — HTTP 429 / 5xx (UNKNOWN)
      unexpected_status              — 3xx·408·409 등 판단할 수 없는 상태 (UNKNOWN)
      timeout / network_error        — 응답을 받지 못했다 (UNKNOWN)
      invalid_response               — 2xx 인데 JSON 객체가 아니거나 본문을 다 읽지 못했다 (UNKNOWN)
      missing_task_uuid              — 2xx 인데 응답에 taskUUID 가 없다 (UNKNOWN)
      task_uuid_mismatch             — 2xx 인데 다른 taskUUID 만 있다 (UNKNOWN)
    """

    outcome: SubmissionOutcome
    task_uuid: str
    reason: str
    http_status: int | None = None
    # 응답 data 항목의 status(예: "processing"). 접수 응답에 없으면 None.
    task_status: str | None = None
    error_codes: tuple[str, ...] = ()
    # Runware 가 다시 보내도 된다고 알려준 경우(429·5xx·재시도 가능 코드·retryAfter). 자동 재시도는 하지 않는다.
    retryable: bool = False
    # 유효한 대기 안내(초). 본문 errors[].retryAfter 와 HTTP Retry-After 중 더 긴 값. 없거나 잘못됐으면 None.
    retry_after: float | None = None
    # 유효한 안내지만 너무 커서 초(float)로 표현할 수 없다. True 면 retry_after 는 None 이고 새 생성 요청을 보류해야 한다.
    retry_after_unrepresentable: bool = False
    detail: str = ""

    @property
    def accepted(self) -> bool:
        return self.outcome is SubmissionOutcome.ACCEPTED


def validate_task_uuid(task_uuid: str) -> str:
    """AI 요청 Worker 가 만든 UUID v4 여야 한다. Provider 는 새로 만들지 않는다."""
    try:
        parsed = uuid.UUID(task_uuid)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("task_uuid 는 UUID v4 문자열이어야 합니다.") from error
    if parsed.version != 4 or str(parsed) != task_uuid.lower():
        raise ValueError("task_uuid 는 UUID v4 문자열이어야 합니다.")
    return str(parsed)


def _require_https(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.startswith("https://"):
        raise ValueError(f"{name} 는 https URL 이어야 합니다.")


def build_async_payload(
    fitting_input: FittingInput, *, task_uuid: str, webhook_url: str, upload_endpoint: str
) -> list:
    """V2 비동기 요청 본문. person 이 먼저, garment 는 입력 순서를 유지한다."""
    task_uuid = validate_task_uuid(task_uuid)
    _require_https("webhook_url", webhook_url)
    _require_https("upload_endpoint", upload_endpoint)
    if not fitting_input.person_image_url:
        raise ValueError("person_image_url 이 비어 있습니다.")
    garments = list(fitting_input.garment_image_urls)
    if not 1 <= len(garments) <= MAX_GARMENT_IMAGES:
        raise ValueError(f"garment 이미지는 1~{MAX_GARMENT_IMAGES}장이어야 합니다.")

    reference_images = [{"image": fitting_input.person_image_url, "role": "person"}]
    reference_images.extend({"image": url, "role": "garment"} for url in garments)
    task: dict[str, Any] = {
        "taskType": "imageInference",
        "taskUUID": task_uuid,
        "model": RUNWARE_MODEL,
        "inputs": {"referenceImages": reference_images},
        "outputType": "URL",
        "outputFormat": OUTPUT_FORMAT,
        "numberResults": 1,
        "includeCost": True,
        "deliveryMethod": "async",
        "webhookURL": webhook_url,
        "uploadEndpoint": upload_endpoint,
    }
    # positivePrompt 는 모델에서 선택 항목이다. 비어 있으면 보내지 않는다.
    if fitting_input.prompt:
        task["positivePrompt"] = fitting_input.prompt
    return [task]


def _redact(text: str) -> str:
    return _URL_PATTERN.sub("<url>", text)[:_DETAIL_LIMIT]


def _error_list(data: Any) -> list[dict]:
    errors = data.get("errors") if isinstance(data, dict) else None
    if not isinstance(errors, list):
        return []
    return [error for error in errors if isinstance(error, dict)]


class _NonFinite:
    """본문의 NaN·Infinity 표기. 숫자로 보지 않는다(잘못된 값)."""

    def __init__(self, text: str) -> None:
        self.text = text


@dataclass(frozen=True)
class RetryAfter:
    """대기 안내 해석 결과. seconds 와 unrepresentable 이 동시에 참이 되지 않는다."""

    seconds: float | None = None
    unrepresentable: bool = False


_TOO_LARGE = object()


def _seconds(value: Any, *, header: bool = False) -> Any:
    """유효한 안내면 초(float), 유효하지만 float 로 표현할 수 없으면 _TOO_LARGE, 잘못된 값이면 None.

    본문 retryAfter 는 JSON 숫자만(정확한 Decimal 로 읽음), 헤더 Retry-After 는 숫자로만 된 문자열만 받는다.
    """
    if isinstance(value, bool):
        return None
    if header:
        if not isinstance(value, str):
            return None
        text = value.strip()
        # 헤더는 RFC 9110 delay-seconds(숫자만). HTTP-date 는 Runware 계약(초)이 아니므로 쓰지 않는다.
        if not text or not all(c in "0123456789." for c in text):
            return None
        try:
            value = Decimal(text)
        except InvalidOperation:
            return None
    if isinstance(value, int | float):
        value = Decimal(value) if math.isfinite(value) else None
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        return None
    seconds = float(value)  # 범위를 넘으면 inf 가 된다(값을 줄이지 않는다)
    return seconds if math.isfinite(seconds) else _TOO_LARGE


def parse_retry_after(body_values: Sequence[Any], header_value: Any = None) -> RetryAfter:
    """본문 retryAfter 들과 헤더 Retry-After 를 함께 해석한다.

    하나라도 표현할 수 없게 크면 unrepresentable(다른 쪽의 짧은 값을 쓰지 않음). 아니면 유효한 값의 최댓값.
    """
    candidates = [_seconds(v) for v in body_values] + [_seconds(header_value, header=True)]
    if any(c is _TOO_LARGE for c in candidates):
        return RetryAfter(unrepresentable=True)
    valid = [c for c in candidates if c is not None]
    return RetryAfter(seconds=max(valid) if valid else None)


def _error_fields(errors: Sequence[dict]) -> dict[str, Any]:
    codes = tuple(str(error["code"]) for error in errors if error.get("code"))
    retry_after = parse_retry_after(
        [error["retryAfter"] for error in errors if error.get("retryAfter") is not None]
    )
    messages = "; ".join(
        f"{error.get('code', '')}: {error.get('message', '')}".strip(": ") for error in errors
    )
    return {
        "error_codes": codes,
        "retry_after": retry_after.seconds,
        "retry_after_unrepresentable": retry_after.unrepresentable,
        "retryable": retry_after != RetryAfter() or any(c in RETRYABLE_ERROR_CODES for c in codes),
        "detail": _redact(messages),
    }


def _parse_json(body: bytes) -> Any:
    try:
        # 숫자는 정확한 Decimal 로 읽는다(큰 정수의 자릿수 제한·float 반올림 없이 retryAfter 를 해석하려고).
        return json.loads(body, parse_float=Decimal, parse_int=Decimal, parse_constant=_NonFinite)
    except (ValueError, UnicodeDecodeError):
        return None


def classify_response(
    status: int, body: bytes, task_uuid: str, *, retry_after_header: Any = None
) -> AsyncSubmission:
    """HTTP status 와 본문(·Retry-After 헤더)으로 접수 결과를 나눈다. 근거가 부족하면 ACCEPTED 로 보지 않는다."""
    data = _parse_json(body)
    errors = _error_list(data)
    fields = _error_fields(errors) if errors else {}
    if retry_after_header is not None:
        body_values = [
            error["retryAfter"] for error in errors if error.get("retryAfter") is not None
        ]
        combined = parse_retry_after(body_values, retry_after_header)
        if combined != RetryAfter():
            fields = {
                **fields,
                "retry_after": combined.seconds,
                "retry_after_unrepresentable": combined.unrepresentable,
                "retryable": True,
            }

    def result(outcome: SubmissionOutcome, reason: str, **extra: Any) -> AsyncSubmission:
        return AsyncSubmission(
            outcome, task_uuid, reason, http_status=status, **{**fields, **extra}
        )

    if 200 <= status < 300:
        if not isinstance(data, dict):
            return result(SubmissionOutcome.UNKNOWN, "invalid_response")
        if data.get("errors"):
            return result(SubmissionOutcome.REJECTED, "error_response")
        items = data.get("data")
        items = (
            [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        )
        uuids = [item.get("taskUUID") for item in items if item.get("taskUUID")]
        if not uuids:
            return result(SubmissionOutcome.UNKNOWN, "missing_task_uuid")
        matched = next((item for item in items if item.get("taskUUID") == task_uuid), None)
        if matched is None:
            return result(SubmissionOutcome.UNKNOWN, "task_uuid_mismatch")
        task_status = matched.get("status")
        return result(
            SubmissionOutcome.ACCEPTED,
            "accepted",
            task_status=task_status if isinstance(task_status, str) else None,
        )

    if status == 429:
        return result(SubmissionOutcome.UNKNOWN, "rate_limited", retryable=True)
    if status >= 500:
        return result(SubmissionOutcome.UNKNOWN, "server_error", retryable=True)
    if status == 402:
        return result(SubmissionOutcome.REJECTED, "balance_exhausted")
    # 408(요청 timeout)·409(중복 taskUUID 가능성)·3xx 는 접수 여부를 단정할 수 없다.
    if 400 <= status < 500 and status not in (408, 409):
        return result(SubmissionOutcome.REJECTED, "client_error")
    return result(SubmissionOutcome.UNKNOWN, "unexpected_status")


class RunwarePrunaAsyncProvider:
    """V2 비동기 접수 Provider. V1 RunwarePrunaProvider 와 별개이며 V1 동작은 바꾸지 않는다."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = RUNWARE_ENDPOINT,
        timeout: float = DEFAULT_SUBMIT_TIMEOUT,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else get_runware_vton_api_key()
        self.endpoint = endpoint
        self.timeout = timeout
        self._opener = opener or urlopen

    def submit(
        self,
        fitting_input: FittingInput,
        *,
        task_uuid: str,
        webhook_url: str,
        upload_endpoint: str,
    ) -> AsyncSubmission:
        """접수 요청을 한 번 보낸다. 입력이 잘못되면 ValueError(전송 전), 그 밖에는 예외 없이 결과를 돌려준다."""
        payload = build_async_payload(
            fitting_input,
            task_uuid=task_uuid,
            webhook_url=webhook_url,
            upload_endpoint=upload_endpoint,
        )
        task_uuid = payload[0]["taskUUID"]
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        submission = self._send(request, task_uuid)
        logger.info(
            "runware async submit task_uuid=%s outcome=%s reason=%s http_status=%s "
            "task_status=%s error_codes=%s retryable=%s retry_after=%s "
            "retry_after_unrepresentable=%s",
            submission.task_uuid,
            submission.outcome.value,
            submission.reason,
            submission.http_status,
            submission.task_status,
            ",".join(submission.error_codes),
            submission.retryable,
            submission.retry_after,
            submission.retry_after_unrepresentable,
        )
        return submission

    def _send(self, request: Request, task_uuid: str) -> AsyncSubmission:
        status: int | None = None
        try:
            with self._opener(request, timeout=self.timeout) as response:
                status = response.status
                body = response.read()
        except HTTPError as error:
            try:
                body = error.read()
            except (OSError, HTTPException):
                body = b""
            headers = error.headers
            header = headers.get("Retry-After") if headers is not None else None
            return classify_response(error.code, body, task_uuid, retry_after_header=header)
        except URLError as error:
            reason = "timeout" if _is_timeout(error.reason) else "network_error"
            return self._unknown(task_uuid, reason, status)
        except TimeoutError:
            return self._unknown(task_uuid, "timeout", status)
        except (OSError, HTTPException):
            # 응답 status 를 받은 뒤 본문을 읽다 끊겼으면 status 가 있다. 어느 쪽이든 접수 여부는 모른다.
            return self._unknown(
                task_uuid, "network_error" if status is None else "invalid_response", status
            )
        return classify_response(status, body, task_uuid)

    @staticmethod
    def _unknown(task_uuid: str, reason: str, status: int | None) -> AsyncSubmission:
        return AsyncSubmission(SubmissionOutcome.UNKNOWN, task_uuid, reason, http_status=status)

    def _headers(self) -> Mapping[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
