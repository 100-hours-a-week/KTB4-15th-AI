"""가상피팅 V2 비동기 작업 (scripts/db/virtual_fitting_async_jobs.sql 의 virtual_fitting_async_jobs).

이번 단계는 저장과 조회만 한다. 상태 전환·실행 권한(lease)·횟수 증가는 5단계 이후에 추가한다.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import psycopg

from app.clients.s3 import validate_s3_key
from app.virtual_fitting.exceptions import FittingDatabaseError

TABLE = "virtual_fitting_async_jobs"
_MAX_BIGINT = 2**63 - 1


class AsyncJobStatus(str, Enum):
    RECEIVED = "RECEIVED"
    WAITING_PRUNA = "WAITING_PRUNA"
    POSTPROCESSING = "POSTPROCESSING"
    FALLBACK_RUNNING = "FALLBACK_RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class NextAction(str, Enum):
    SUBMIT_PRUNA = "SUBMIT_PRUNA"
    POSTPROCESS = "POSTPROCESS"
    RUN_FALLBACK = "RUN_FALLBACK"
    PUBLISH_RESULT = "PUBLISH_RESULT"


class ResultMessageStatus(str, Enum):
    PENDING = "PENDING"
    PUBLISHED = "PUBLISHED"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"


@dataclass(frozen=True)
class NewAsyncFittingJob:
    """처음 받은 BE 요청에서 저장할 값. 나머지 컬럼은 초기값(상태 RECEIVED, 횟수 0, 나머지 NULL)이다."""

    fitting_job_id: int
    request_fingerprint: str
    user_image_key: str
    products: Sequence[str]
    # BE 가 보낸 최초 요청 시각. timezone 이 있어야 한다.
    requested_at: datetime


@dataclass(frozen=True)
class AsyncFittingJob:
    """테이블 한 줄. 시각은 모두 UTC 다. 필드 순서는 SELECT 컬럼 순서와 같다."""

    fitting_job_id: int
    request_fingerprint: str
    user_image_key: str | None
    products: tuple[str, ...] | None
    task_uuid: str | None
    async_image_key: str | None
    result_image_key: str | None
    status: AsyncJobStatus
    requested_at: datetime
    created_at: datetime
    updated_at: datetime
    pruna_submit_started_at: datetime | None
    runware_accepted_at: datetime | None
    next_action: NextAction | None
    verified_image_key: str | None
    image_verified_at: datetime | None
    result_payload: dict[str, Any] | None
    result_message_status: ResultMessageStatus
    result_message_attempt_count: int
    llm_generation_attempt_count: int
    fallback_attempt_count: int
    s3_check_attempt_count: int
    worker_lease_token: str | None
    worker_lease_until: datetime | None


@dataclass(frozen=True)
class AsyncJobCreateResult:
    job: AsyncFittingJob
    # False 면 같은 fitting_job_id 가 이미 있었고, job 은 바꾸지 않은 기존 줄이다.
    created: bool


COLUMNS = tuple(field.name for field in fields(AsyncFittingJob))
_SELECT_LIST = ", ".join(COLUMNS)

INSERT_SQL = f"""
INSERT INTO {TABLE}
    (fitting_job_id, request_fingerprint, user_image_key, products, requested_at,
     status, next_action, result_message_status)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (fitting_job_id) DO NOTHING
RETURNING {_SELECT_LIST}
"""
SELECT_BY_JOB_ID_SQL = f"SELECT {_SELECT_LIST} FROM {TABLE} WHERE fitting_job_id = %s"
SELECT_BY_TASK_UUID_SQL = f"SELECT {_SELECT_LIST} FROM {TABLE} WHERE task_uuid = %s"


def _check_job_id(fitting_job_id: int) -> int:
    if (
        isinstance(fitting_job_id, bool)
        or not isinstance(fitting_job_id, int)
        or not 1 <= fitting_job_id <= _MAX_BIGINT
    ):
        raise ValueError("fitting_job_id 는 BIGINT 범위의 양의 정수여야 합니다.")
    return fitting_job_id


def _check_new_job(job: NewAsyncFittingJob) -> tuple:
    _check_job_id(job.fitting_job_id)
    if not isinstance(job.request_fingerprint, str) or not job.request_fingerprint.strip():
        raise ValueError("request_fingerprint 가 비어 있습니다.")
    validate_s3_key(job.user_image_key)
    if isinstance(job.products, (str, bytes)) or not isinstance(job.products, Sequence):
        raise TypeError("products 는 상품 코드 문자열 목록이어야 합니다.")
    products = list(job.products)
    if not products or any(not isinstance(code, str) or not code.strip() for code in products):
        raise ValueError("products 는 비어 있지 않은 상품 코드 문자열 목록이어야 합니다.")
    if not isinstance(job.requested_at, datetime) or job.requested_at.utcoffset() is None:
        raise ValueError("requested_at 은 timezone 이 있는 datetime 이어야 합니다.")
    return (
        job.fitting_job_id,
        job.request_fingerprint,
        job.user_image_key,
        products,
        job.requested_at.astimezone(UTC),
        AsyncJobStatus.RECEIVED.value,
        NextAction.SUBMIT_PRUNA.value,
        ResultMessageStatus.PENDING.value,
    )


def _utc(value: datetime | None) -> datetime | None:
    return value.astimezone(UTC) if value is not None else None


def _row_to_job(row: Sequence[Any]) -> AsyncFittingJob:
    data: dict[str, Any] = dict(zip(COLUMNS, row, strict=True))
    for name in (
        "requested_at",
        "created_at",
        "updated_at",
        "pruna_submit_started_at",
        "runware_accepted_at",
        "image_verified_at",
        "worker_lease_until",
    ):
        data[name] = _utc(data[name])
    data["products"] = tuple(data["products"]) if data["products"] is not None else None
    data["task_uuid"] = str(data["task_uuid"]) if data["task_uuid"] is not None else None
    data["status"] = AsyncJobStatus(data["status"])
    data["next_action"] = NextAction(data["next_action"]) if data["next_action"] else None
    data["result_message_status"] = ResultMessageStatus(data["result_message_status"])
    payload = data["result_payload"]
    data["result_payload"] = dict(payload) if isinstance(payload, Mapping) else payload
    return AsyncFittingJob(**data)


class AsyncJobRepository:
    """쓸 때만 pool 에서 connection 을 빌리고 즉시 반환한다. 없는 작업은 None 이다."""

    def __init__(self, connection_pool: Any):
        self._connection_pool = connection_pool

    def create(self, new_job: NewAsyncFittingJob) -> AsyncJobCreateResult:
        """새 작업을 저장한다. 같은 fitting_job_id 가 있으면 아무것도 바꾸지 않고 기존 줄을 돌려준다.

        입력 비교(request_fingerprint 가 같은지)와 중복 처리 방법은 호출하는 쪽이 정한다.
        """
        params = _check_new_job(new_job)
        try:
            with self._connection_pool.connection() as connection:
                connection.autocommit = True
                # INSERT 와 기존 줄 조회를 한 트랜잭션으로 묶는다. 실패하면 둘 다 되돌린다.
                with connection.transaction(), connection.cursor() as cursor:
                    cursor.execute(INSERT_SQL, params)
                    row = cursor.fetchone()
                    created = row is not None
                    if not created:
                        cursor.execute(SELECT_BY_JOB_ID_SQL, (new_job.fitting_job_id,))
                        row = cursor.fetchone()
        except psycopg.Error as error:
            raise FittingDatabaseError("AI PostgreSQL 비동기 작업 저장에 실패했습니다.") from error
        if row is None:  # 같은 번호와 충돌했는데 그 줄이 없다. 지금 구조에서는 지우는 경로가 없다.
            raise FittingDatabaseError("AI PostgreSQL 비동기 작업 저장 결과를 확인하지 못했습니다.")
        return AsyncJobCreateResult(job=_row_to_job(row), created=created)

    def find_by_fitting_job_id(self, fitting_job_id: int) -> AsyncFittingJob | None:
        row = self._fetch_one(SELECT_BY_JOB_ID_SQL, (_check_job_id(fitting_job_id),))
        return _row_to_job(row) if row is not None else None

    def find_by_task_uuid(self, task_uuid: str) -> AsyncFittingJob | None:
        try:
            parsed = uuid.UUID(task_uuid)
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("task_uuid 는 UUID 문자열이어야 합니다.") from error
        row = self._fetch_one(SELECT_BY_TASK_UUID_SQL, (parsed,))
        return _row_to_job(row) if row is not None else None

    def _fetch_one(self, sql: str, params: tuple) -> Any:
        try:
            with self._connection_pool.connection() as connection:
                connection.autocommit = True
                with connection.cursor() as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchone()
        except psycopg.Error as error:
            raise FittingDatabaseError("AI PostgreSQL 비동기 작업 조회에 실패했습니다.") from error
