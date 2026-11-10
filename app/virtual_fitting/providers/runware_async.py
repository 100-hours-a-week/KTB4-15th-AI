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
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from app.virtual_fitting.providers.base import FittingInput
from app.virtual_fitting.providers.http import _is_timeout
from app.virtual_fitting.providers.runware import (
    RUNWARE_ENDPOINT,
    RUNWARE_MODEL,
    get_runware_vton_api_key,
)
from app.virtual_fitting.providers.runware_errors import TaskErrorKind, classify_task_error

logger = logging.getLogger(__name__)

# 접수 응답만 기다린다(이미지 생성은 기다리지 않는다). 실제 접수 지연을 보고 조정한다.
DEFAULT_SUBMIT_TIMEOUT = 30.0
OUTPUT_FORMAT = "WEBP"
# Pruna try-on 모델 제약: person 1장, garment 1~11장.
MAX_GARMENT_IMAGES = 11
_DETAIL_LIMIT = 200
_URL_PATTERN = re.compile(r"https?://\S+")


class SubmissionOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AsyncSubmission:
    """접수 요청 결과. URL·토큰·키는 담지 않는다.

    reason 값(오류 항목은 요청한 taskUUID 에 연결된 것만 근거로 쓴다):
      accepted                       — 접수 근거 확인
      not_accepted                   — 이 task 가 처리되지 않았다는 문서 근거 코드(serviceOverloaded·inferenceRateLimitExceeded)만
                                       있다. REJECTED·retryable=True(재요청 후보, 자동 재요청은 아직 연결 안 됨)
      generation_fallback            — 이 task 의 생성 종료 실패 근거(timeoutProvider)만 있다 (REJECTED, fallback 조건 평가)
      generation_failed              — 이 task 의 확정 생성 실패 근거 코드만 있다 (REJECTED, 최종 FITTING_MODEL_FAILED)
      auth_error                     — HTTP 401·403 또는 이 task 의 invalidApiKey (REJECTED)
      balance_exhausted              — HTTP 402 (REJECTED)
      client_error                   — 그 밖의 4xx: 요청이 거절됐지만 전용 정책이 없다 (REJECTED, 보류·운영 확인)
      error_response_unclear         — 2xx 인데 이 task 의 다른 오류 코드가 있다(접수·생성 단계 근거 부족, UNKNOWN)
      error_response_unmatched       — 2xx 인데 errors 가 이 task 에 연결되지 않는다 (UNKNOWN)
      rate_limited / server_error    — 근거 코드 없는 HTTP 429 / 5xx (UNKNOWN, 미접수로 단정하지 않음)
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
    # 이 task 가 처리되지 않았다는 문서 근거(NOT_ACCEPTED_CODES)가 있어 다시 보내도 두 번 실행되지 않는다.
    # 429·5xx·retryAfter 만으로는 True 가 아니다. 자동 재요청은 하지 않는다(다음 단계에서 연결).
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
    fitting_input: FittingInput, *, task_uuid: str, webhook_url: str, upload_endpoint: str | None,
    output_format: str = OUTPUT_FORMAT, ai_stores_result: bool = False,
) -> list:
    """V2 비동기 요청 본문. person 이 먼저, garment 는 입력 순서를 유지한다.

    ai_stores_result 는 호출자가 DB 에 확정된 저장 방식을 그대로 넘긴다(endpoint 유무로 추측하지 않음).
    False(기본, 기존 계약): upload_endpoint 가 필수다. True: upload_endpoint 를 주면 안 되고(충돌은 전송 전 거절) WEBP 만.
    """
    if output_format not in {"JPG", "WEBP"}:
        raise ValueError("지원하지 않는 비동기 출력 형식입니다.")
    if not isinstance(ai_stores_result, bool):
        # ValueError: 호출부(Submitter)는 ValueError 를 "보내기 전 거절"로 처리한다.
        raise ValueError("ai_stores_result 는 bool 이어야 합니다.")  # noqa: TRY004
    task_uuid = validate_task_uuid(task_uuid)
    _require_https("webhook_url", webhook_url)
    if ai_stores_result:
        if upload_endpoint is not None:
            raise ValueError("AI 저장 작업에는 upload_endpoint 를 주지 않습니다.")
        if output_format != "WEBP":
            raise ValueError("AI 저장 작업의 출력 형식은 WEBP 여야 합니다.")
    else:
        _require_https("upload_endpoint", upload_endpoint)
        path = unquote(urlparse(upload_endpoint).path).lower()
        expected = "WEBP" if path.endswith(".webp") else "JPG" if path.endswith((".jpg", ".jpeg")) else None
        if expected is not None and output_format != expected:
            raise ValueError("출력 형식과 업로드 Key 확장자가 일치하지 않습니다.")
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
        "outputFormat": output_format,
        "numberResults": 1,
        "includeCost": True,
        "deliveryMethod": "async",
        "webhookURL": webhook_url,
    }
    if not ai_stores_result:
        task["uploadEndpoint"] = upload_endpoint
    if output_format == "WEBP":
        task["outputQuality"] = 90
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
    """대기 안내·코드·진단 문구. 대기 안내(retryAfter)는 공유 대기에만 쓰고 접수·재요청 판단 근거로 쓰지 않는다."""
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
        "detail": _redact(messages),
    }


def _same_task(error: dict, task_uuid: str) -> bool:
    value = error.get("taskUUID")
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == task_uuid
    except ValueError:
        return False


def _task_error_kind(codes: Sequence[Any]) -> TaskErrorKind | None:
    """이 task 에 연결된 제출 응답 오류 코드들의 분류(Webhook·getResponse 와 같은 classify_task_error).

    코드가 모두 같은 분류일 때만 그 분류를 쓴다. 섞여 있으면 인증 코드가 있을 때 AUTH(운영 확인), 아니면 UNCLEAR.
    """
    kinds = {classify_task_error(code if isinstance(code, str) else None) for code in codes}
    if len(kinds) == 1:
        return kinds.pop()
    if TaskErrorKind.AUTH in kinds:
        return TaskErrorKind.AUTH
    return TaskErrorKind.UNCLEAR if kinds else None


def _parse_json(body: bytes) -> Any:
    try:
        # 숫자는 정확한 Decimal 로 읽는다(큰 정수의 자릿수 제한·float 반올림 없이 retryAfter 를 해석하려고).
        return json.loads(body, parse_float=Decimal, parse_int=Decimal, parse_constant=_NonFinite)
    except (ValueError, UnicodeDecodeError):
        return None


def classify_response(
    status: int, body: bytes, task_uuid: str, *, retry_after_header: Any = None
) -> AsyncSubmission:
    """HTTP status 와 본문(·Retry-After 헤더)으로 접수 결과를 나눈다. 근거가 부족하면 ACCEPTED 로 보지 않는다.

    확실한 미접수(not_accepted)는 요청한 taskUUID 에 연결된 오류가 모두 문서상 미처리 코드일 때만이다. HTTP 429·5xx·
    retryAfter·연결되지 않은 오류만으로는 정하지 않는다. 생성 종료 실패(fallback·최종 실패)도 이 task 에 연결된 근거 코드로만
    정한다(Webhook·getResponse 와 같은 classify_task_error). 알 수 없는 코드는 생성 실패로 보지 않는다.
    """
    data = _parse_json(body)
    all_errors = _error_list(data)
    # 진단용 코드·대기 안내(공유 대기)는 기존처럼 모든 오류에서 읽는다. 분류 근거는 이 task 에 연결된 오류의 코드만 쓴다.
    fields = _error_fields(all_errors) if all_errors else {}
    errors = [error for error in all_errors if _same_task(error, task_uuid)]
    # 진단용 필터는 빈 코드를 생략한다. 판단에는 항목 전체를 써야 근거 없는 오류를 버리고 재생성을 허용하지 않는다.
    kind = _task_error_kind([error.get("code") for error in errors])
    if retry_after_header is not None:
        body_values = [
            error["retryAfter"] for error in all_errors if error.get("retryAfter") is not None
        ]
        combined = parse_retry_after(body_values, retry_after_header)
        if combined != RetryAfter():
            fields = {
                **fields,
                "retry_after": combined.seconds,
                "retry_after_unrepresentable": combined.unrepresentable,
            }

    def result(outcome: SubmissionOutcome, reason: str, **extra: Any) -> AsyncSubmission:
        return AsyncSubmission(
            outcome, task_uuid, reason, http_status=status, **{**fields, **extra}
        )

    items = data.get("data") if isinstance(data, dict) else None
    if errors and isinstance(items, list) and any(
        isinstance(item, dict)
        and _same_task(item, task_uuid)
        and item.get("status") in ("processing", "success")
        for item in items
    ):
        # 같은 task 의 접수·성공 근거와 오류가 모순된다. 재요청·fallback 대신 같은 task 조회로 확인한다.
        return result(SubmissionOutcome.UNKNOWN, "error_response_unclear")
    if kind is TaskErrorKind.NOT_ACCEPTED:
        # 이 task 의 오류가 모두 문서상 미처리 코드다(HTTP 상태와 무관하게 그 근거를 따른다).
        return result(
            SubmissionOutcome.REJECTED, "not_accepted", retryable=True,
            error_codes=tuple(error["code"] for error in errors),
        )
    if kind is TaskErrorKind.GENERATION_FALLBACK:
        # 이 task 의 생성이 종료 실패했다는 근거(timeoutProvider). Webhook·getResponse 와 같은 fallback 조건으로 넘긴다.
        return result(SubmissionOutcome.REJECTED, "generation_fallback")
    if kind is TaskErrorKind.GENERATION_FAILED:
        return result(SubmissionOutcome.REJECTED, "generation_failed")
    if status in (401, 403) or (kind is TaskErrorKind.AUTH and status < 500):
        return result(SubmissionOutcome.REJECTED, "auth_error")

    if 200 <= status < 300:
        if not isinstance(data, dict):
            return result(SubmissionOutcome.UNKNOWN, "invalid_response")
        if data.get("errors"):
            # 이 task 의 다른 오류이거나 연결되지 않은 오류: 접수·생성 단계를 단정할 근거가 없다(조회로 확인).
            return result(
                SubmissionOutcome.UNKNOWN,
                "error_response_unclear" if errors else "error_response_unmatched",
            )
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
        return result(SubmissionOutcome.UNKNOWN, "rate_limited")
    if status >= 500:
        return result(SubmissionOutcome.UNKNOWN, "server_error")
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
        upload_endpoint: str | None,
        output_format: str = OUTPUT_FORMAT,
        ai_stores_result: bool = False,
    ) -> AsyncSubmission:
        """접수 요청을 한 번 보낸다. 입력이 잘못되면 ValueError(전송 전), 그 밖에는 예외 없이 결과를 돌려준다."""
        payload = build_async_payload(
            fitting_input,
            task_uuid=task_uuid,
            webhook_url=webhook_url,
            upload_endpoint=upload_endpoint,
            output_format=output_format,
            ai_stores_result=ai_stores_result,
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
