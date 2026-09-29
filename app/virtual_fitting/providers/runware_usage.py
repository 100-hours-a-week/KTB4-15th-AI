"""Runware 계정 사용량 조회 (accountManagement / getUsageActivity).

하루 사용액 보정에만 쓴다. Runware 집계는 늦게 반영되므로 이 값으로 피팅을 막지 않고,
우리 합계와 비교해 큰 쪽을 남기는 데만 쓴다 (FittingBudget).

POST https://api.runware.ai/v1
  본문: [{"taskType": "accountManagement", "operation": "getUsageActivity",
          "startDate", "endDate", "models": [...], "groupBy": ["date"], "timezone": "UTC"}]
  성공: {"data": [{"usage": {"timeseries": {"meta": {"totalSpend": 1.23}}}}]}
"""

import json
import uuid
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any
from urllib.request import Request, urlopen

from app.virtual_fitting.exceptions import FittingModelError
from app.virtual_fitting.providers.http import send_request
from app.virtual_fitting.providers.runware import (
    RUNWARE_ENDPOINT,
    RUNWARE_MODEL,
    get_runware_vton_api_key,
)

DEFAULT_TIMEOUT = 10.0


def build_usage_payload(day: date, task_uuid: str) -> list:
    """하루치, 가상피팅 모델만. models 로 거르면 같은 계정의 채팅 LLM 사용액이 섞이지 않는다."""
    return [
        {
            "taskType": "accountManagement",
            "taskUUID": task_uuid,
            "operation": "getUsageActivity",
            "startDate": day.isoformat(),
            "endDate": day.isoformat(),
            "models": [RUNWARE_MODEL],
            "groupBy": ["date"],
            "timezone": "UTC",
        }
    ]


def parse_total_spend(body: bytes) -> Decimal:
    try:
        data = json.loads(body, parse_float=Decimal)
        total = data["data"][0]["usage"]["timeseries"]["meta"]["totalSpend"]
    except (ValueError, UnicodeDecodeError, LookupError, TypeError) as error:
        raise FittingModelError(
            "Runware 사용량 응답에서 totalSpend 를 찾을 수 없습니다."
        ) from error
    return Decimal(str(total))


class RunwareUsageClient:
    """VTON 키로 조회한다. 사용량은 계정 단위라 어느 키로 물어도 같은 계정의 값이 온다."""

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

    def spend_on(self, day: date) -> Decimal:
        request = Request(
            self.endpoint,
            data=json.dumps(build_usage_payload(day, str(uuid.uuid4()))).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = send_request(
            request, opener=self._opener, timeout=self.timeout, service="Runware usage"
        )
        return parse_total_spend(body)
