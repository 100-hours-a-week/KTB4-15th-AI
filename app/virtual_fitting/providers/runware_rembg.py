"""Runware RemBG v1.4 배경 제거 Provider (V2 후처리: Pruna 결과의 검은 배경 → 투명).

POST https://api.runware.ai/v1
  헤더: Authorization: Bearer <RUNWARE_VTON_API_KEY>
  본문: [{"taskType": "removeBackground", "taskUUID", "model": "runware:109@1",
          "inputs": {"image": <Presigned GET URL>}, "settings": {...},
          "outputType": "URL", "outputFormat": "WEBP", "deliveryMethod": "sync", "includeCost": true}]
  성공: {"data": [{"taskType": "removeBackground", "taskUUID", "imageUUID", "imageURL", "cost"}]}

settings 는 단독 테스트에서 성공한 값이다(RGBA, Alpha 0~255, 투명·반투명 79.14%, $0.0006).
이미지 생성 계열 호출이라 VTON 키를 쓴다(LLM 키와 같은 계정). 요청은 한 번만 보낸다(작업당 1회, 재시도 없음).
실제 응답의 taskType 은 "imageBackgroundRemoval" 이라 taskType 이 아니라 taskUUID 로 결과 항목을 찾는다.
"""

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.virtual_fitting.exceptions import (
    FittingBalanceExhaustedError,
    FittingModelError,
    FittingTimeoutError,
)
from app.virtual_fitting.providers.http import send_request
from app.virtual_fitting.providers.runware import RUNWARE_ENDPOINT, get_runware_vton_api_key
from app.virtual_fitting.providers.runware_errors import NOT_ACCEPTED_CODES
from app.virtual_fitting.repositories.async_job_repository import (
    check_runware_image_url,
    check_runware_image_uuid,
)

REMBG_MODEL = "runware:109@1"
REMBG_TASK_TYPE = "removeBackground"
REMBG_OUTPUT_FORMAT = "WEBP"
REMBG_SETTINGS = {
    "alphaMatting": True,
    "postProcessMask": True,
    "returnOnlyMask": False,
    "rgba": [255, 255, 255, 0],
}
# 단독 측정 약 5초. Runware 문서의 평균보다 넉넉히 둔다(후처리 lease 300초 안).
DEFAULT_TIMEOUT = 30.0


class RembgError(RuntimeError):
    """배경 제거 호출 실패. reason 은 안전한 원인 코드다(URL·키·응답 본문 없음)."""

    def __init__(
        self,
        reason: str,
        *,
        retryable: bool,
        http_status: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(f"Runware 배경 제거에 실패했습니다: {reason}")
        self.reason = reason
        self.retryable = retryable
        self.http_status = http_status
        # Runware errors[].code (있을 때만, 영문 코드라 그대로 남긴다)
        self.code = code

    @property
    def error_code(self) -> str:
        """최종 실패 원인으로 남길 안전한 코드(호출 표시 객체·로그)."""
        if self.code:
            return self.code
        return f"http_{self.http_status}" if self.http_status else self.reason


@dataclass(frozen=True)
class RembgResult:
    task_uuid: str
    image_uuid: str
    # im.runware.ai 결과 URL(token 포함 가능). repr·로그에 남기지 않는다.
    image_url: str = field(repr=False)
    cost: Decimal | None


def build_payload(image_url: str, task_uuid: str) -> list:
    return [
        {
            "taskType": REMBG_TASK_TYPE,
            "taskUUID": task_uuid,
            "model": REMBG_MODEL,
            "inputs": {"image": image_url},
            "settings": dict(REMBG_SETTINGS),
            "outputType": "URL",
            "outputFormat": REMBG_OUTPUT_FORMAT,
            "deliveryMethod": "sync",
            "includeCost": True,
        }
    ]


def _cost(item: Mapping[str, Any]) -> Decimal | None:
    """비용은 기록용이다. 해석하지 못해도 이미 만든 결과를 버리지 않는다(None, 0 으로 보지 않음)."""
    value = item.get("cost")
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except ArithmeticError:
        return None


def parse_result(body: bytes, task_uuid: str) -> RembgResult:
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise RembgError("invalid_response", retryable=False) from None
    if not isinstance(data, dict):
        raise RembgError("invalid_response", retryable=False)
    errors = data.get("errors")
    if errors:
        first = errors[0] if isinstance(errors, list) and errors else None
        code = first.get("code") if isinstance(first, dict) else None
        code = code if isinstance(code, str) and code.isascii() and len(code) <= 64 else None
        # 처리되지 않았다고 확인된 코드(과부하·rate limit)만 다시 보내 볼 수 있다.
        raise RembgError("api_error", retryable=code in NOT_ACCEPTED_CODES, code=code)
    items = data.get("data")
    item = next(
        (
            entry
            for entry in (items if isinstance(items, list) else [])
            if isinstance(entry, dict) and entry.get("taskUUID") == task_uuid
        ),
        None,
    )
    if item is None:
        raise RembgError("invalid_response", retryable=False)
    try:
        url = check_runware_image_url(item.get("imageURL"))
        image_uuid = check_runware_image_uuid(item.get("imageUUID"))
    except ValueError:
        # 허용 호스트(im.runware.ai)가 아니거나 형식이 다르다. 내려받지 않는다.
        raise RembgError("invalid_result_ref", retryable=False) from None
    return RembgResult(task_uuid, image_uuid, url, _cost(item))


def _transport_error(error: Exception) -> RembgError:
    if isinstance(error, FittingTimeoutError):
        return RembgError("timeout", retryable=True)
    if isinstance(error, FittingBalanceExhaustedError):
        return RembgError("balance_exhausted", retryable=False, http_status=402)
    cause = error.__cause__
    if isinstance(cause, HTTPError):
        status = cause.code
        return RembgError(
            "http_error",
            retryable=status >= 500 or status == 429,
            http_status=status,
        )
    if isinstance(cause, (URLError, OSError, HTTPException)):
        return RembgError("connection_failed", retryable=True)
    return RembgError("request_failed", retryable=False)


class RunwareRembgProvider:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = RUNWARE_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        opener: Callable[..., Any] | None = None,
        new_task_uuid: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self._api_key = api_key if api_key is not None else get_runware_vton_api_key()
        self.endpoint = endpoint
        self.timeout = timeout
        self._opener = opener or urlopen
        self._new_task_uuid = new_task_uuid

    def remove_background(self, image_url: str, *, task_uuid: str | None = None) -> RembgResult:
        """동기 배경 제거를 **한 번** 요청한다(자동 재시도 없음). 실패는 RembgError(reason·retryable·http_status).

        task_uuid 를 주면 그 값으로 보낸다(V2: 작업·원본 task 로 정한 값). 없으면 새로 만든다.
        """
        if not isinstance(image_url, str) or not image_url.startswith("https://"):
            raise ValueError("image_url 은 https URL 이어야 합니다.")
        task = str(uuid.UUID(task_uuid)) if task_uuid is not None else self._new_task_uuid()
        request = Request(
            self.endpoint,
            data=json.dumps(build_payload(image_url, task)).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            body = send_request(
                request, opener=self._opener, timeout=self.timeout, service="Runware RemBG"
            )
        except (FittingTimeoutError, FittingBalanceExhaustedError, FittingModelError) as error:
            raise _transport_error(error) from None
        return parse_result(body, task)
