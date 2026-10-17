"""Runware 계정 잔액 사전 확인 (accountManagement / getDetails). V2 Pruna 제출 전에만 쓴다.

POST https://api.runware.ai/v1
  본문: [{"taskType": "accountManagement", "taskUUID": <조회용 UUID>, "operation": "getDetails"}]
  성공(공식 문서 예시): {"data": [{..., "balance": {"amount": 2450.75, "freeBalance": 120.00, "currency": "USD"}, ...}]}

- 공식 문서는 amount 를 "Current balance amount", freeBalance 를 "Available free credit balance" 로만 설명하고 둘의
  관계(포함 여부)를 밝히지 않는다. 그래서 두 값을 더하지 않고 amount 만 사용 가능한 잔액으로 본다(실제보다 적게
  잡는 쪽). 실제 계정으로 확인할 항목이다.
- currency 가 USD 가 아니거나 amount 가 숫자가 아니면 잔액을 확정할 수 없다(FittingBalanceUnknownError).
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
from decimal import Decimal, InvalidOperation
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


def build_details_payload(lookup_uuid: str) -> list:
    return [{"taskType": "accountManagement", "taskUUID": lookup_uuid, "operation": "getDetails"}]


def parse_balance(
    body: bytes, *, now: Callable[[], datetime] = lambda: datetime.now(UTC)
) -> BalanceReading:
    try:
        data = json.loads(body, parse_float=Decimal)
        item = data["data"][0]
        balance = item["balance"]
        amount, currency = balance["amount"], balance["currency"]
    except (ValueError, UnicodeDecodeError, LookupError, TypeError) as error:
        raise FittingBalanceUnknownError(
            "Runware 잔액 응답에서 balance 를 찾을 수 없습니다."
        ) from error
    if isinstance(data, dict) and data.get("errors"):
        raise FittingBalanceUnknownError("Runware 잔액 조회가 오류를 돌려줬습니다.")
    if currency != CURRENCY:
        raise FittingBalanceUnknownError("Runware 잔액 통화가 USD 가 아닙니다.")
    if isinstance(amount, bool) or not isinstance(amount, Decimal | int):
        raise FittingBalanceUnknownError("Runware 잔액 amount 가 숫자가 아닙니다.")
    try:
        value = Decimal(str(amount))
    except InvalidOperation as error:
        raise FittingBalanceUnknownError("Runware 잔액 amount 가 숫자가 아닙니다.") from error
    if not value.is_finite():
        raise FittingBalanceUnknownError("Runware 잔액 amount 가 유한한 숫자가 아닙니다.")
    return BalanceReading(amount=value, currency=currency, checked_at=now())


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
