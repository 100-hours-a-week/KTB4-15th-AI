import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

import psycopg

from app.clients.s3 import validate_s3_key
from app.virtual_fitting.exceptions import FittingDatabaseError

TABLE = "virtual_fitting_async_jobs"
ATTEMPTS = "virtual_fitting_pruna_attempts"
_MAX_BIGINT = 2**63 - 1
# 한 작업의 상품 수. 상의·하의 각 1개까지라 최대 2개다.
MAX_PRODUCTS = 2


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
    # fallback(13단계, scripts/db/virtual_fitting_fallback.sql). 비동기 task_uuid 와 별개인 생성 시도 기록.
    fallback_task_uuid: str | None
    fallback_started_at: datetime | None
    fallback_image_key: str | None
    # Runware 인증 오류 보류(15단계 보완, scripts/db/virtual_fitting_auth_hold.sql). 거절된 비동기 생성 시도와 안전한 원인 코드.
    auth_hold_task_uuid: str | None = None
    auth_hold_reason: str | None = None
    auth_hold_at: datetime | None = None
    # Runware 가 이 비동기 task_uuid 의 성공 결과로 알려 준 결과 참조(scripts/db/virtual_fitting_runware_image_ref.sql).
    # 추후 다운로드 복구용 내부 기록이다. 성공 전환과 같은 UPDATE 로 처음 한 번만 저장한다. URL 은 repr·로그에 남기지 않는다.
    runware_image_url: str | None = field(default=None, repr=False)
    runware_image_uuid: str | None = None
    # 결과 저장 방식(scripts/db/virtual_fitting_ai_result_storage.sql). True: AI 가 결과 URL 에서 내려받아 저장.
    # False(기존 행·현재 요청): Runware 가 uploadEndpoint 로 직접 업로드. 요청 준비 때 한 번 정하고 바꾸지 않는다.
    ai_stores_result: bool = False
    # 확실한 미접수 재요청(scripts/db/virtual_fitting_pruna_retry.sql). 현재 task 의 시도를 미접수로 기록한 시각과 다음 재요청
    # 가능 시각. 미접수 기록이 있고 다음 시각이 없으면 자동 재요청 보류(운영 확인)다.
    pruna_not_accepted_at: datetime | None = None
    pruna_retry_at: datetime | None = None


@dataclass(frozen=True)
class PrunaAttempt:
    """비동기 Pruna 전송 시도 한 줄(virtual_fitting_pruna_attempts). 이전 시도의 늦은 알림을 찾을 때 쓴다."""

    fitting_job_id: int
    attempt_no: int
    task_uuid: str
    started_at: datetime
    not_accepted_code: str | None
    not_accepted_at: datetime | None
    conflict_at: datetime | None


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


MAX_RUNWARE_IMAGE_URL_LENGTH = 2048
# Runware 결과 이미지 서버. 공식 문서의 이미지 출력 예제와 실제 실험(webp-ai-upload, HTTPS·리다이렉트 없음)에서 확인한
# 호스트만 정확히 일치로 허용한다(하위 도메인·와일드카드 없음). 우리 업로드용 Presigned PUT(S3 버킷 호스트)은 여기 없다.
RUNWARE_IMAGE_HOSTS = frozenset({"im.runware.ai"})


def check_runware_image_url(url: str) -> str:
    """저장할 수 있는 Runware 결과 URL 인지 확인하고 원본 문자열 그대로 돌려준다.

    https://(DB CHECK 와 같은 소문자 접두사), 허용 호스트와 정확히 일치(포트는 없음 또는 443), 사용자 정보·공백·
    제어문자 없음, 2048자 이하. 쿼리(token·서명·만료 등)는 신뢰 판단에 쓰지 않으며 지우거나 바꾸지 않는다.
    오류 메시지에는 URL 을 넣지 않는다.
    """
    if not isinstance(url, str) or not 0 < len(url) <= MAX_RUNWARE_IMAGE_URL_LENGTH:
        raise ValueError("runware_image_url 은 비어 있지 않은 2048자 이하 문자열이어야 합니다.")
    if any(ch.isspace() or not ch.isprintable() for ch in url):
        raise ValueError("runware_image_url 에 공백·제어문자가 있습니다.")
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError as error:
        raise ValueError("runware_image_url 형식이 올바르지 않습니다.") from error
    if not url.startswith("https://") or parts.username is not None or parts.password is not None:
        raise ValueError("runware_image_url 은 사용자 정보 없는 https URL 이어야 합니다.")
    if host not in RUNWARE_IMAGE_HOSTS or port not in (None, 443):
        raise ValueError("runware_image_url 이 허용된 Runware 이미지 서버 주소가 아닙니다.")
    return url


def check_runware_image_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("runware_image_uuid 는 UUID 문자열이어야 합니다.") from error


def _check_new_job(job: NewAsyncFittingJob) -> tuple:
    _check_job_id(job.fitting_job_id)
    if not isinstance(job.request_fingerprint, str) or not job.request_fingerprint.strip():
        raise ValueError("request_fingerprint 가 비어 있습니다.")
    validate_s3_key(job.user_image_key)
    if isinstance(job.products, (str, bytes)) or not isinstance(job.products, Sequence):
        raise TypeError("products 는 상품 코드 문자열 목록이어야 합니다.")
    products = list(job.products)
    if not 1 <= len(products) <= MAX_PRODUCTS:
        raise ValueError(f"products 는 1~{MAX_PRODUCTS}개여야 합니다.")
    if any(not isinstance(code, str) or not code.strip() for code in products):
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
        "fallback_started_at",
        "auth_hold_at",
        "pruna_not_accepted_at",
        "pruna_retry_at",
    ):
        data[name] = _utc(data[name])
    data["products"] = tuple(data["products"]) if data["products"] is not None else None
    data["task_uuid"] = str(data["task_uuid"]) if data["task_uuid"] is not None else None
    fallback_task = data["fallback_task_uuid"]
    data["fallback_task_uuid"] = str(fallback_task) if fallback_task is not None else None
    hold_task = data["auth_hold_task_uuid"]
    data["auth_hold_task_uuid"] = str(hold_task) if hold_task is not None else None
    image_uuid = data["runware_image_uuid"]
    data["runware_image_uuid"] = str(image_uuid) if image_uuid is not None else None
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

    def find_attempt(self, task_uuid: str) -> PrunaAttempt | None:
        """이 taskUUID 의 전송 시도 기록. 현재 시도인지는 작업의 task_uuid 와 비교해 호출부가 정한다."""
        try:
            parsed = uuid.UUID(task_uuid)
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("task_uuid 는 UUID 문자열이어야 합니다.") from error
        row = self._fetch_one(
            f"SELECT fitting_job_id, attempt_no, task_uuid, started_at, not_accepted_code, "
            f"not_accepted_at, conflict_at FROM {ATTEMPTS} WHERE task_uuid = %s",
            (parsed,),
        )
        if row is None:
            return None
        job_id, attempt_no, task, started, code, not_accepted, conflict = row
        return PrunaAttempt(
            job_id, attempt_no, str(task), _utc(started), code, _utc(not_accepted), _utc(conflict)
        )

    def find_result_publish_due(self, limit: int) -> list[int]:
        """결과 발행(14단계)을 기다리는 작업 번호. 결과가 저장돼 있고(PUBLISH_RESULT·PENDING) 종료되지 않았으며 다른
        실행 주체의 유효한 실행 권한이 없는 것만, 오래된 것부터. 선점은 호출부가 acquire_lease 로 다시 확인한다.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit 는 1~100 정수여야 합니다.")
        try:
            with self._connection_pool.connection() as connection:
                connection.autocommit = True
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        SELECT fitting_job_id FROM {TABLE}
                        WHERE next_action = 'PUBLISH_RESULT' AND result_payload IS NOT NULL
                          AND result_message_status = 'PENDING'
                          AND status NOT IN ('COMPLETED', 'FAILED')
                          AND (worker_lease_token IS NULL OR worker_lease_until <= statement_timestamp())
                        ORDER BY updated_at, fitting_job_id
                        LIMIT %s
                        """,
                        (limit,),
                    )
                    return [row[0] for row in cursor.fetchall()]
        except psycopg.Error as error:
            raise FittingDatabaseError(
                "AI PostgreSQL 결과 발행 대상 조회에 실패했습니다."
            ) from error

    def _fetch_one(self, sql: str, params: tuple) -> Any:
        try:
            with self._connection_pool.connection() as connection:
                connection.autocommit = True
                with connection.cursor() as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchone()
        except psycopg.Error as error:
            raise FittingDatabaseError("AI PostgreSQL 비동기 작업 조회에 실패했습니다.") from error
