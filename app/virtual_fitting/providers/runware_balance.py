"""Runware 계정 잔액 사전 확인 (accountManagement / getDetails). V2 Pruna 제출 전에만 쓴다.

POST https://api.runware.ai/v1
  본문: [{"taskType": "accountManagement", "taskUUID": <조회용 UUID>, "operation": "getDetails"}]
  성공(공식 문서 예시): {"data": [{..., "balance": {"amount": 2450.75, "freeBalance": 120.00, "currency": "USD"}, ...}]}

- 공식 문서는 amount 를 "Current balance amount", freeBalance 를 "Available free credit balance" 로만 설명하고 둘의
  관계(포함 여부)를 밝히지 않는다. 그래서 두 값을 더하지 않고 amount 만 사용 가능한 잔액으로 본다(실제보다 적게
  잡는 쪽). 실제 계정으로 확인할 항목이다.
- currency 가 USD 가 아니거나 amount 가 0 이상의 유한한 숫자가 아니면 잔액을 확정할 수 없다(FittingBalanceUnknownError).
- 지원 형식은 두 가지뿐이다(그 밖의 필드를 찾아다니지 않는다).
  1) 객체 {"amount", "freeBalance"?, "currency"} — 공식 문서(models-api·platform account-management)·공식 TypeScript SDK 타입.
  2) 숫자 하나 — 공식 문서·SDK 에는 없다. 우리 계정의 실제 getDetails 응답에서 관찰됐고, 같은 계정의 대시보드 달러 잔액과
     표시 정밀도 기준으로 일치함을 확인해(2026-10-18, 그 계정에서 관찰한 근거) USD 금액으로 본다. 같은 항목에 currency 가
     함께 오면 USD 일 때만 받고 다르면 거절한다(상충). 다른 계정·API 버전에서도 같다는 보장은 아니다.
- HTTP 200 이어도 errors 가 있으면 잔액으로 보지 않는다. 금액은 JSON 을 읽을 때부터 Decimal 로 읽는다.
- 응답 전문(조직·팀·API Key·사용량)은 저장·로그하지 않고 amount·currency 만 꺼낸다.
- 조회용 taskUUID 는 매번 새로 만들고, 이미지 생성 task UUID 와 섞지 않는다.
- 재시도하지 않는다(send_request 한 번). 실패는 FittingBalanceUnknownError 로 올린다.
- 하루 사용액 상한(FittingBudget, FITTING_DAILY_BUDGET_USD)과는 별개다.
"""

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from urllib.request import Request, urlopen

from app.virtual_fitting.exceptions import (
    FittingBalanceExhaustedError,
    FittingBalanceUnknownError,
    VirtualFittingError,
)
from app.virtual_fitting.providers.http import send_request
from app.virtual_fitting.providers.runware import RUNWARE_ENDPOINT, get_runware_vton_api_key

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
CURRENCY = "USD"


@dataclass(frozen=True)
class BalanceReading:
    amount: Decimal
    currency: str
    # 조회 응답을 받은 시각(UTC). 생성 요청이 늦어지면 오래된 값을 쓰지 않도록 호출부가 본다.
    checked_at: datetime
    # 응답 형식("object": 문서 형식, "number": 계정에서 관찰한 숫자 형식). 진단용.
    format: str = "object"


def build_details_payload(lookup_uuid: str) -> list:
    return [{"taskType": "accountManagement", "taskUUID": lookup_uuid, "operation": "getDetails"}]


class _NonFinite:
    """JSON 의 NaN·Infinity 표기. 숫자로 보지 않는다."""


def _is_number(value: Any) -> bool:
    return isinstance(value, Decimal) and not isinstance(value, bool)


def parse_balance(
    body: bytes, *, now: Callable[[], datetime] = lambda: datetime.now(UTC)
) -> BalanceReading:
    try:
        # 정수·소수 모두 Decimal 로(부동소수 반올림 없이), NaN·Infinity 는 숫자가 아닌 표식으로 읽는다.
        data = json.loads(
            body, parse_float=Decimal, parse_int=Decimal, parse_constant=lambda _: _NonFinite()
        )
    except (ValueError, UnicodeDecodeError) as error:
        raise FittingBalanceUnknownError("Runware 잔액 응답이 JSON 이 아닙니다.") from error
    if not isinstance(data, dict):
        raise FittingBalanceUnknownError("Runware 잔액 응답 형식이 객체가 아닙니다.")
    if data.get("errors"):
        raise FittingBalanceUnknownError("Runware 잔액 조회가 오류를 돌려줬습니다.")
    items = data.get("data")
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        raise FittingBalanceUnknownError("Runware 잔액 응답에 data[0] 이 없습니다.")
    if "balance" not in items[0]:
        raise FittingBalanceUnknownError("Runware 잔액 응답에서 balance 를 찾을 수 없습니다.")
    balance = items[0]["balance"]
    if _is_number(balance):
        # 계정에서 관찰한 숫자 형식(대시보드 달러 잔액과 일치 확인) → USD 금액. 통화 정보가 같이 오면 USD 만.
        if "currency" in items[0] and items[0]["currency"] != CURRENCY:
            raise FittingBalanceUnknownError("Runware 잔액 통화가 USD 가 아닙니다.")
        if not balance.is_finite() or balance < 0:
            raise FittingBalanceUnknownError("Runware 잔액이 0 이상의 유한한 숫자가 아닙니다.")
        return BalanceReading(amount=balance, currency=CURRENCY, checked_at=now(), format="number")
    if not isinstance(balance, dict):
        raise FittingBalanceUnknownError("Runware 잔액 balance 형식을 지원하지 않습니다.")
    currency = balance.get("currency")
    if currency != CURRENCY:
        raise FittingBalanceUnknownError("Runware 잔액 통화가 USD 가 아닙니다.")
    amount = balance.get("amount")
    if not _is_number(amount):
        raise FittingBalanceUnknownError("Runware 잔액 amount 가 숫자가 아닙니다.")
    if not amount.is_finite() or amount < 0:
        raise FittingBalanceUnknownError("Runware 잔액 amount 가 0 이상의 유한한 숫자가 아닙니다.")
    return BalanceReading(amount=amount, currency=currency, checked_at=now())


class RunwareBalanceClient:
    """VTON 키로 계정 잔액을 한 번 조회한다."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = RUNWARE_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else get_runware_vton_api_key()
        self.endpoint = endpoint
        self.timeout = timeout
        self._opener = opener or urlopen

    def read(self) -> BalanceReading:
        request = Request(
            self.endpoint,
            data=json.dumps(build_details_payload(str(uuid.uuid4()))).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            body = send_request(
                request, opener=self._opener, timeout=self.timeout, service="Runware balance"
            )
        except VirtualFittingError as error:
            # timeout·권한·5xx·402 모두 "잔액을 확인하지 못함"이다. 조회 응답만으로 잔액 부족을 확정하지 않는다.
            raise FittingBalanceUnknownError("Runware 잔액을 조회하지 못했습니다.") from error
        return parse_balance(body)


class MinimumBalanceCheck:
    """BalanceCheck 구현. 사용 가능한 잔액(amount, USD)이 minimum 이하이면 FittingBalanceExhaustedError.

    잔액이 기준을 넘는다고 이번 생성 비용까지 충분하다고 보장하지는 않는다(생성 요청의 402 는 따로 처리).
    """

    def __init__(self, client: RunwareBalanceClient, minimum: Decimal) -> None:
        if not isinstance(minimum, Decimal) or not minimum.is_finite() or minimum < 0:
            raise ValueError("minimum 은 0 이상의 유한한 Decimal 이어야 합니다.")
        self._client = client
        self._minimum = minimum

    def ensure_available(self) -> None:
        reading = self._client.read()
        if reading.amount <= self._minimum:
            logger.warning(
                "runware balance at or below the minimum; fitting submissions are blocked"
            )
            raise FittingBalanceExhaustedError("Runware 잔액이 최소 기준 이하입니다.")
