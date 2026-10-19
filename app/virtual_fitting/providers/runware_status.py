"""Runware 비동기 작업 상태 조회(getResponse) — 15단계 Watchdog 전용, 조회만 한다(생성 요청 아님).

공식 문서(models-api/task-polling): 요청 {"taskType": "getResponse", "taskUUID": <UUID v4>}. 응답은 data(processing·success)와
errors 배열. 상태는 "processing"(progress 가능)·"success"(결과·cost)·"error". **비동기(deliveryMethod: async) 작업 전용**이다.
동기 요청(V1·fallback)의 결과를 이 조회로 다시 얻을 수 있다는 근거는 문서에 없으므로 fallback taskUUID 에는 쓰지 않는다.
문서에 없는 것: 작업 없음 오류 코드, 결과 보관 기간, 조회 한도 수치 — 추측하지 않는다(실제 계정 검증 전).

- 한 번 조회에 HTTP 요청 1번(urllib, 자동 재시도 없음), timeout 10초. 새 이미지를 만들지 않는다.
- 응답 항목의 taskUUID 가 조회 대상과 같을 때만 그 상태로 본다. 다르거나 없으면 NOT_FOUND.
- errors 의 timeoutProvider 등 코드는 FAILED 로 돌려주되, 확정 실패로 쓸지는 호출부가 기존 분류(FALLBACK_ERROR_CODES)로 정한다.
- 401·403 → AUTH_ERROR, 429 → RATE_LIMITED(본문·헤더 대기 안내), timeout·연결 오류·5xx·그 밖 → UNKNOWN.
- 응답 원문·이미지 URL·API Key 는 로그·결과에 남기지 않는다(cost·오류 코드만).
"""

import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.virtual_fitting.providers.runware import RUNWARE_ENDPOINT, get_runware_vton_api_key
from app.virtual_fitting.providers.runware_async import parse_retry_after

logger = logging.getLogger(__name__)

DEFAULT_STATUS_TIMEOUT = 10.0
_CODE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class TaskStatusKind(str, Enum):
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    AUTH_ERROR = "auth_error"
    RATE_LIMITED = "rate_limited"
    NOT_FOUND = "not_found"  # 응답에 이 taskUUID 항목이 없다(작업 없음과 같다고 단정하지 않음)
    UNKNOWN = "unknown"  # timeout·연결 오류·5xx·해석 불가


@dataclass(frozen=True)
class TaskStatus:
    kind: TaskStatusKind
    task_uuid: str
    cost: Decimal | None = None
    error_code: str | None = None
    retry_after: float | None = None
    http_status: int | None = None


class _NonFinite:
    def __init__(self, text: str) -> None:
        self.text = text


def _cost(item: dict) -> Decimal | None:
    value = item.get("cost")
    if isinstance(value, bool) or not isinstance(value, Decimal | int):
        return None
    cost = Decimal(value)
    return cost if cost.is_finite() and cost >= 0 else None


def parse_status(body: bytes, task_uuid: str, http_status: int | None = None) -> TaskStatus:
    try:
        data = json.loads(body, parse_float=Decimal, parse_int=Decimal, parse_constant=_NonFinite)
    except (ValueError, UnicodeDecodeError):
        return TaskStatus(TaskStatusKind.UNKNOWN, task_uuid, http_status=http_status)
    if not isinstance(data, dict):
        return TaskStatus(TaskStatusKind.UNKNOWN, task_uuid, http_status=http_status)
    items = [i for i in data.get("data") or [] if isinstance(i, dict)]
    errors = [i for i in data.get("errors") or [] if isinstance(i, dict)]
    if "status" in data and "taskUUID" in data:  # 단일 객체 형태(문서 예시)
        items.append(data)

    def same(item: dict) -> bool:
        value = item.get("taskUUID")
        try:
            return isinstance(value, str) and str(uuid.UUID(value)) == task_uuid
        except ValueError:
            return False

    for item in items:
        if not same(item):
            continue
        status = item.get("status")
        if status == "processing":
            return TaskStatus(TaskStatusKind.PROCESSING, task_uuid, http_status=http_status)
        if status == "success":
            return TaskStatus(
                TaskStatusKind.SUCCEEDED, task_uuid, _cost(item), http_status=http_status
            )
        if status == "error":
            errors.append(item)
    for item in errors:
        if not same(item):
            continue
        code = item.get("code")
        safe = code if isinstance(code, str) and _CODE.match(code) else None
        return TaskStatus(
            TaskStatusKind.FAILED, task_uuid, _cost(item), error_code=safe, http_status=http_status
        )
    return TaskStatus(TaskStatusKind.NOT_FOUND, task_uuid, http_status=http_status)


class RunwareTaskStatusClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = RUNWARE_ENDPOINT,
        timeout: float = DEFAULT_STATUS_TIMEOUT,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else get_runware_vton_api_key()
        self.endpoint, self.timeout = endpoint, timeout
        self._opener = opener or urlopen

    def query(self, task_uuid: str) -> TaskStatus:
        task = str(uuid.UUID(task_uuid))
        request = Request(
            self.endpoint,
            data=json.dumps([{"taskType": "getResponse", "taskUUID": task}]).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                result = parse_status(response.read(), task, getattr(response, "status", None))
        except HTTPError as error:
            result = self._http_error(error, task)
        except (URLError, TimeoutError, OSError, HTTPException):
            result = TaskStatus(TaskStatusKind.UNKNOWN, task)
        logger.info(
            "runware status task_uuid=%s kind=%s http_status=%s error_code=%s",
            task,
            result.kind.value,
            result.http_status,
            result.error_code,
        )
        return result

    @staticmethod
    def _http_error(error: HTTPError, task: str) -> TaskStatus:
        if error.code in (401, 403):
            return TaskStatus(TaskStatusKind.AUTH_ERROR, task, http_status=error.code)
        if error.code == 429:
            try:
                body = json.loads(error.read() or b"{}")
            except (ValueError, OSError, HTTPException):
                body = {}
            values = [
                e.get("retryAfter")
                for e in (body.get("errors") or [] if isinstance(body, dict) else [])
                if isinstance(e, dict) and e.get("retryAfter") is not None
            ]
            header = error.headers.get("Retry-After") if error.headers is not None else None
            retry = parse_retry_after(values, header)
            return TaskStatus(
                TaskStatusKind.RATE_LIMITED, task, retry_after=retry.seconds, http_status=429
            )
        return TaskStatus(TaskStatusKind.UNKNOWN, task, http_status=error.code)
