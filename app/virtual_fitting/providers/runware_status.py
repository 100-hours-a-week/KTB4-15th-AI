import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.virtual_fitting.providers.runware import RUNWARE_ENDPOINT, get_runware_vton_api_key
from app.virtual_fitting.providers.runware_async import parse_retry_after
from app.virtual_fitting.repositories.async_job_repository import (
    check_runware_image_url,
    check_runware_image_uuid,
)

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
    # 조회 요청 자체의 대기 안내(HTTP 429 의 Retry-After·본문). 조회 간격에만 쓴다(생성 재요청 근거 아님).
    retry_after: float | None = None
    http_status: int | None = None
    # FAILED 일 때 그 생성 task 오류 항목의 대기 안내(retryAfter). 확실한 미접수의 다음 재요청 예약에만 쓴다.
    task_retry_after: float | None = None
    task_retry_after_unrepresentable: bool = False
    # success 항목의 결과 참조(imageURL·imageUUID). 저장할 수 없는 값이면 None 이고 image_ref_invalid=True.
    image_url: str | None = field(default=None, repr=False)
    image_uuid: str | None = None
    image_ref_invalid: bool = False


class _NonFinite:
    def __init__(self, text: str) -> None:
        self.text = text


def image_ref(item: dict) -> tuple[str | None, str | None, bool]:
    """Runware 성공 항목(Webhook·getResponse 공통)의 결과 참조 (image_url, image_uuid, invalid).

    없거나 빈 값은 None. 있지만 저장할 수 없는 값(https 아님·허용 호스트 아님·UUID 아님 등)도 None 으로 버리고 invalid=True 로
    알린다. 성공 판정은 바꾸지 않는다(호출부의 기존 기준 그대로). 값은 로그에 남기지 않는다.
    """
    invalid = False
    url, image_uuid = item.get("imageURL"), item.get("imageUUID")
    if url is None or (isinstance(url, str) and not url.strip()):
        url = None
    else:
        try:
            url = check_runware_image_url(url)
        except ValueError:
            url, invalid = None, True
    if image_uuid is None or (isinstance(image_uuid, str) and not image_uuid.strip()):
        image_uuid = None
    else:
        try:
            image_uuid = check_runware_image_uuid(image_uuid)
        except ValueError:
            image_uuid, invalid = None, True
    return url, image_uuid, invalid


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

    matched_errors = [item for item in errors if same(item)]
    matched_errors += [item for item in items if same(item) and item.get("status") == "error"]
    if matched_errors and any(same(i) and i.get("status") == "success" for i in items):
        # 같은 task 의 성공과 실패가 함께 왔다. 어느 쪽도 고르지 않는다(상태를 바꾸지 않음).
        return TaskStatus(TaskStatusKind.UNKNOWN, task_uuid, http_status=http_status)
    for item in items:
        if not same(item):
            continue
        status = item.get("status")
        if status == "processing":
            return TaskStatus(TaskStatusKind.PROCESSING, task_uuid, http_status=http_status)
        if status == "success":
            url, image_uuid, invalid = image_ref(item)
            others = {
                image_ref(other)[:2]
                for other in items
                if same(other) and other.get("status") == "success"
            }
            if len(others) > 1:
                # 같은 task 의 결과가 여럿이고 서로 다르다(요청은 1개). 첫 항목을 고르지 않고 참조를 쓰지 않는다.
                url, image_uuid, invalid = None, None, True
            return TaskStatus(
                TaskStatusKind.SUCCEEDED,
                task_uuid,
                _cost(item),
                http_status=http_status,
                image_url=url,
                image_uuid=image_uuid,
                image_ref_invalid=invalid,
            )
        if status == "error":
            errors.append(item)
    for item in errors:
        if not same(item):
            continue
        codes = {
            code if isinstance(code, str) and _CODE.match(code) else None
            for code in (other.get("code") for other in errors if same(other))
        }
        # 같은 task 의 실패 코드가 서로 다르면 어느 코드도 고르지 않는다(분류하지 않고 알림만).
        safe = next(iter(codes)) if len(codes) == 1 else None
        retry = parse_retry_after(
            [other["retryAfter"] for other in errors if same(other) and other.get("retryAfter") is not None]
        )
        return TaskStatus(
            TaskStatusKind.FAILED,
            task_uuid,
            _cost(item),
            error_code=safe,
            http_status=http_status,
            task_retry_after=retry.seconds,
            task_retry_after_unrepresentable=retry.unrepresentable,
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
