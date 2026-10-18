"""가상피팅 V2 비동기 작업 테이블과 Repository.

앞부분은 DB 없이 입력 검증과 SQL 모양만 본다. 뒷부분은 실제 PostgreSQL 에서 마이그레이션을 적용해
제약조건·JSONB·timestamptz·트랜잭션을 확인한다. pytest 는 .env 를 읽지 않는다.
    TEST_DATABASE_URL=postgresql://... (없거나 연결할 수 없으면 PostgreSQL 테스트는 건너뛴다)
테스트마다 고유한 schema 를 만들어 그 안에만 테이블을 만들고 끝나면 지운다. 다른 테이블은 건드리지 않는다.
"""

import os
import re
import uuid
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pytest
from psycopg.types.json import Jsonb

from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.repositories import async_job_repository as repo_module
from app.virtual_fitting.repositories.async_job_repository import (
    COLUMNS,
    INSERT_SQL,
    AsyncFittingJob,
    AsyncJobRepository,
    AsyncJobStatus,
    NewAsyncFittingJob,
    NextAction,
    ResultMessageStatus,
)

MIGRATION = (
    Path(__file__).resolve().parents[1] / "scripts" / "db" / "virtual_fitting_async_jobs.sql"
)
# 13단계 변경 SQL(fallback 진행 기록 컬럼). 신규 생성 SQL 뒤에 적용한다(배포 순서와 같음).
FALLBACK_MIGRATION = MIGRATION.with_name("virtual_fitting_fallback.sql")
KST = timezone(timedelta(hours=9))
REQUESTED_AT = datetime(2026, 10, 13, 9, 30, tzinfo=KST)  # = 2026-10-13 00:30 UTC
TASK_UUID = "a770f077-f413-47de-9dac-be0b26a35da6"


def _new_job(fitting_job_id=3, **overrides):
    values = {
        "fitting_job_id": fitting_job_id,
        "request_fingerprint": "fp-v1:users/3.jpg|123456,789012",
        "user_image_key": "users/3.jpg",
        "products": ["123456", "789012"],
        "requested_at": REQUESTED_AT,
    }
    values.update(overrides)
    return NewAsyncFittingJob(**values)


class UnusablePool:
    def connection(self):
        raise AssertionError("검증에 실패한 입력으로 DB 에 접근하면 안 된다")


# --- DB 없이: 입력 검증·SQL ---


@pytest.mark.parametrize(
    "overrides",
    [
        {"fitting_job_id": 0},
        {"fitting_job_id": -1},
        {"fitting_job_id": 2**63},
        {"fitting_job_id": True},
        {"fitting_job_id": "3"},
        {"request_fingerprint": ""},
        {"request_fingerprint": "   "},
        {"request_fingerprint": None},
        {"user_image_key": ""},
        {"user_image_key": "https://bucket.s3.amazonaws.com/users/3.jpg?X-Amz-Signature=x"},
        {"user_image_key": "/users/3.jpg"},
        {"products": []},
        {"products": ["1", "2", "3"]},
        {"products": ["1", "2", "3", "4", "5"]},
        {"products": "123456"},
        {"products": [123456]},
        {"products": ["123456", ""]},
        {"products": None},
        {"requested_at": datetime(2026, 10, 13, 9, 30)},  # noqa: DTZ001 (timezone 없음을 확인)
        {"requested_at": "2026-10-13T09:30:00+09:00"},
    ],
)
def test_invalid_new_job_is_rejected_before_db(overrides):
    with pytest.raises((TypeError, ValueError)):
        AsyncJobRepository(UnusablePool()).create(_new_job(**overrides))


@pytest.mark.parametrize("products", [["123456"], ["123456", "789012"]])
def test_one_or_two_products_pass_validation(products):
    params = repo_module._check_new_job(_new_job(products=products))
    assert params[3] == products


@pytest.mark.parametrize("count", [0, 3, 4])
def test_wrong_product_count_never_reaches_the_db(count):
    products = [str(100000 + i) for i in range(count)]
    with pytest.raises(ValueError, match="1~2"):
        AsyncJobRepository(UnusablePool()).create(_new_job(products=products))


@pytest.mark.parametrize("value", [None, "", "not-a-uuid", 123])
def test_invalid_task_uuid_lookup_is_rejected_before_db(value):
    with pytest.raises(ValueError):
        AsyncJobRepository(UnusablePool()).find_by_task_uuid(value)


def test_insert_never_overwrites_an_existing_job():
    sql = " ".join(INSERT_SQL.split())
    assert "ON CONFLICT (fitting_job_id) DO NOTHING" in sql
    assert re.search(r"\bUPDATE\b", sql, re.IGNORECASE) is None  # DO UPDATE 로 덮어쓰지 않는다


def _migration_columns():
    """신규 생성 SQL 의 컬럼 뒤에 변경 SQL(13단계 fallback)이 ADD COLUMN 으로 붙이는 컬럼을 순서대로 잇는다."""
    text = MIGRATION.read_text(encoding="utf-8")
    body = text[text.index("CREATE TABLE") :]
    created = [
        match.group(1)
        for match in re.finditer(
            r"^\s{4}([a-z0-9_]+)\s+(?:bigint|text|uuid|timestamptz|jsonb|integer)",
            body,
            re.MULTILINE,
        )
    ]
    added = re.findall(
        r"ADD COLUMN IF NOT EXISTS ([a-z0-9_]+)\s", FALLBACK_MIGRATION.read_text(encoding="utf-8")
    )
    return created + added


def test_model_columns_match_migration_file():
    assert tuple(_migration_columns()) == COLUMNS


# --- PostgreSQL ---


@pytest.fixture
def pg():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL 이 없다")
    try:
        admin = psycopg.connect(url, autocommit=True, connect_timeout=3)
    except psycopg.Error as error:
        pytest.skip(f"테스트 DB 에 연결할 수 없다: {error}")

    from psycopg_pool import ConnectionPool

    schema = f"vf_async_test_{uuid.uuid4().hex[:12]}"
    admin.execute(f"CREATE SCHEMA {schema}")
    # 서버와 다른 session TimeZone 으로 연결해, 읽은 시각이 UTC 로 맞춰지는지 함께 본다.
    options = f"-c search_path={schema} -c TimeZone=America/Los_Angeles"
    with psycopg.connect(url, autocommit=True, options=options) as connection:
        connection.execute(MIGRATION.read_text(encoding="utf-8"))
        connection.execute(FALLBACK_MIGRATION.read_text(encoding="utf-8"))
    pool = ConnectionPool(url, min_size=1, max_size=2, kwargs={"options": options}, open=True)

    def sql(statement, params=None):
        with psycopg.connect(url, autocommit=True, options=options) as connection:
            cursor = connection.execute(statement, params)
            return cursor.fetchall() if cursor.description else None

    try:
        yield {"url": url, "schema": schema, "pool": pool, "sql": sql, "options": options}
    finally:
        pool.close()
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()


def test_first_save_has_initial_values_and_nulls(pg):
    result = AsyncJobRepository(pg["pool"]).create(_new_job())

    assert result.created is True
    job = result.job
    assert job.fitting_job_id == 3
    assert job.request_fingerprint == "fp-v1:users/3.jpg|123456,789012"
    assert job.user_image_key == "users/3.jpg"
    assert job.products == ("123456", "789012")
    assert job.status is AsyncJobStatus.RECEIVED
    assert job.result_message_status is ResultMessageStatus.PENDING
    assert job.next_action is NextAction.SUBMIT_PRUNA
    assert (
        job.result_message_attempt_count,
        job.llm_generation_attempt_count,
        job.fallback_attempt_count,
        job.s3_check_attempt_count,
    ) == (0, 0, 0, 0)
    for name in (
        "task_uuid",
        "async_image_key",
        "result_image_key",
        "pruna_submit_started_at",
        "runware_accepted_at",
        "verified_image_key",
        "image_verified_at",
        "result_payload",
        "worker_lease_token",
        "worker_lease_until",
    ):
        assert getattr(job, name) is None, name
    # DB 에 저장된 상품 코드는 문자열 배열이다.
    assert pg["sql"](
        "SELECT products, pg_typeof(products)::text FROM virtual_fitting_async_jobs"
    ) == [(["123456", "789012"], "text[]")]


def test_find_by_job_id_and_task_uuid(pg):
    repository = AsyncJobRepository(pg["pool"])
    created = repository.create(_new_job()).job

    assert repository.find_by_fitting_job_id(3) == created
    assert repository.find_by_fitting_job_id(4) is None
    assert repository.find_by_task_uuid(TASK_UUID) is None

    pg["sql"](
        "UPDATE virtual_fitting_async_jobs SET task_uuid = %s WHERE fitting_job_id = 3",
        (TASK_UUID,),
    )
    found = repository.find_by_task_uuid(TASK_UUID.upper())  # 대소문자와 무관하게 같은 UUID
    assert found is not None and found.fitting_job_id == 3 and found.task_uuid == TASK_UUID
    assert repository.find_by_task_uuid(str(uuid.uuid4())) is None


def test_duplicate_job_id_keeps_existing_input_and_result(pg):
    repository = AsyncJobRepository(pg["pool"])
    original = repository.create(_new_job()).job
    pg["sql"](
        "UPDATE virtual_fitting_async_jobs SET status = 'COMPLETED', result_payload = %s, "
        "result_image_key = 'virtual-fitting/results/3/pruna-async.jpg' WHERE fitting_job_id = 3",
        (
            Jsonb(
                {
                    "fitting_job_id": 3,
                    "status": "COMPLETED",
                    "result_image_key": "virtual-fitting/results/3/pruna-async.jpg",
                    "llm_title": "차분한 데일리룩",
                    "llm_comment": "선택하신 상품으로 완성한 가상 피팅 결과입니다.",
                }
            ),
        ),
    )
    before = repository.find_by_fitting_job_id(3)

    again = repository.create(
        _new_job(
            request_fingerprint="different",
            user_image_key="users/other.jpg",
            products=["999999"],
            requested_at=REQUESTED_AT + timedelta(hours=1),
        )
    )

    assert again.created is False
    assert again.job == before  # 입력·결과·상태·시각 모두 그대로
    assert again.job.created_at == original.created_at
    assert again.job.requested_at == REQUESTED_AT  # 재수신으로 바뀌지 않는다
    assert pg["sql"]("SELECT count(*) FROM virtual_fitting_async_jobs") == [(1,)]


@pytest.mark.parametrize("products", [["123456"], ["123456", "789012"]])
def test_one_or_two_products_are_stored(pg, products):
    job = AsyncJobRepository(pg["pool"]).create(_new_job(products=products)).job
    assert job.products == tuple(products)


@pytest.mark.parametrize("count", [0, 3])
def test_wrong_product_count_is_not_saved(pg, count):
    with pytest.raises(ValueError):
        AsyncJobRepository(pg["pool"]).create(_new_job(products=[str(i + 1) for i in range(count)]))
    assert pg["sql"]("SELECT count(*) FROM virtual_fitting_async_jobs") == [(0,)]


def test_max_bigint_job_id_is_stored(pg):
    job_id = 2**63 - 1
    assert AsyncJobRepository(pg["pool"]).create(_new_job(job_id)).job.fitting_job_id == job_id


def test_task_uuid_allows_many_nulls_but_no_duplicate_values(pg):
    repository = AsyncJobRepository(pg["pool"])
    repository.create(_new_job(1))
    repository.create(_new_job(2))
    assert pg["sql"]("SELECT count(*) FROM virtual_fitting_async_jobs WHERE task_uuid IS NULL") == [
        (2,)
    ]

    pg["sql"](
        "UPDATE virtual_fitting_async_jobs SET task_uuid = %s WHERE fitting_job_id = 1",
        (TASK_UUID,),
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        pg["sql"](
            "UPDATE virtual_fitting_async_jobs SET task_uuid = %s WHERE fitting_job_id = 2",
            (TASK_UUID,),
        )
    with pytest.raises(psycopg.errors.InvalidTextRepresentation):
        pg["sql"](
            "UPDATE virtual_fitting_async_jobs SET task_uuid = 'nope' WHERE fitting_job_id = 2"
        )


def test_requested_at_is_be_time_and_created_at_is_ai_time(pg):
    job = AsyncJobRepository(pg["pool"]).create(_new_job()).job
    (db_now,) = pg["sql"]("SELECT now()")[0]

    assert job.requested_at == REQUESTED_AT
    assert job.created_at != job.requested_at
    assert abs(job.created_at - db_now) < timedelta(minutes=1)
    assert job.updated_at == job.created_at


def test_times_are_timestamptz_and_read_back_as_utc(pg):
    job = AsyncJobRepository(pg["pool"]).create(_new_job()).job

    for value in (job.requested_at, job.created_at, job.updated_at):
        assert value.tzinfo is UTC
    assert job.requested_at == datetime(2026, 10, 13, 0, 30, tzinfo=UTC)
    # session TimeZone 과 무관하게 같은 순간으로 저장됐다.
    assert pg["sql"](
        "SELECT to_char(requested_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI') FROM virtual_fitting_async_jobs"
    ) == [("2026-10-13 00:30",)]

    later = datetime(2026, 10, 13, 0, 31, 5, 123456, tzinfo=UTC)
    pg["sql"](
        "UPDATE virtual_fitting_async_jobs SET pruna_submit_started_at = %s, runware_accepted_at = %s, "
        "image_verified_at = %s, worker_lease_until = %s",
        (later, later, later, later),
    )
    reread = AsyncJobRepository(pg["pool"]).find_by_fitting_job_id(3)
    for name in (
        "pruna_submit_started_at",
        "runware_accepted_at",
        "image_verified_at",
        "worker_lease_until",
    ):
        value = getattr(reread, name)
        assert value == later and value.tzinfo is UTC, name


def test_result_payload_jsonb_round_trip(pg):
    repository = AsyncJobRepository(pg["pool"])
    repository.create(_new_job())
    # V2 결과 형식(최상위 status). 한글과 줄바꿈이 그대로 남는지도 본다.
    payload = {
        "fitting_job_id": 3,
        "status": "COMPLETED",
        "result_image_key": "virtual-fitting/results/3/pruna-async.jpg",
        "llm_title": "차분한 올블랙 셔츠룩",
        "llm_comment": "가슴 포켓이 돋보입니다.\n블랙 팬츠와 맞췄습니다.",
    }
    pg["sql"]("UPDATE virtual_fitting_async_jobs SET result_payload = %s", (Jsonb(payload),))

    assert repository.find_by_fitting_job_id(3).result_payload == payload
    assert pg["sql"](
        "SELECT pg_typeof(result_payload)::text, result_payload->>'llm_title' "
        "FROM virtual_fitting_async_jobs"
    ) == [("jsonb", "차분한 올블랙 셔츠룩")]
    with pytest.raises(psycopg.errors.CheckViolation):
        pg["sql"]("UPDATE virtual_fitting_async_jobs SET result_payload = %s", (Jsonb([1, 2]),))


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE virtual_fitting_async_jobs SET result_message_attempt_count = -1",
        "UPDATE virtual_fitting_async_jobs SET llm_generation_attempt_count = -1",
        "UPDATE virtual_fitting_async_jobs SET fallback_attempt_count = -1",
        "UPDATE virtual_fitting_async_jobs SET s3_check_attempt_count = -1",
        "UPDATE virtual_fitting_async_jobs SET status = 'ACCEPTED'",
        "UPDATE virtual_fitting_async_jobs SET next_action = 'RETRY'",
        "UPDATE virtual_fitting_async_jobs SET result_message_status = 'SENT'",
        "UPDATE virtual_fitting_async_jobs SET products = ARRAY['1', NULL]",
        "UPDATE virtual_fitting_async_jobs SET products = ARRAY[]::text[]",
        "UPDATE virtual_fitting_async_jobs SET request_fingerprint = ''",
    ],
)
def test_check_constraints_reject_bad_values(pg, statement):
    AsyncJobRepository(pg["pool"]).create(_new_job())
    with pytest.raises(psycopg.errors.CheckViolation):
        pg["sql"](statement)


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE virtual_fitting_async_jobs SET status = NULL",
        "UPDATE virtual_fitting_async_jobs SET requested_at = NULL",
        "UPDATE virtual_fitting_async_jobs SET request_fingerprint = NULL",
        "UPDATE virtual_fitting_async_jobs SET result_message_status = NULL",
        "UPDATE virtual_fitting_async_jobs SET fallback_attempt_count = NULL",
    ],
)
def test_not_null_columns(pg, statement):
    AsyncJobRepository(pg["pool"]).create(_new_job())
    with pytest.raises(psycopg.errors.NotNullViolation):
        pg["sql"](statement)


def test_enum_values_match_db_checks_and_cleanup_nulls_are_allowed(pg):
    AsyncJobRepository(pg["pool"]).create(_new_job())
    for status in AsyncJobStatus:
        pg["sql"]("UPDATE virtual_fitting_async_jobs SET status = %s", (status.value,))
    for action in [*NextAction, None]:
        pg["sql"](
            "UPDATE virtual_fitting_async_jobs SET next_action = %s",
            (action.value if action else None,),
        )
    for message_status in ResultMessageStatus:
        pg["sql"](
            "UPDATE virtual_fitting_async_jobs SET result_message_status = %s",
            (message_status.value,),
        )
    # 완료 후 원본 입력 정리
    pg["sql"]("UPDATE virtual_fitting_async_jobs SET user_image_key = NULL, products = NULL")
    job = AsyncJobRepository(pg["pool"]).find_by_fitting_job_id(3)
    assert job.user_image_key is None and job.products is None and job.next_action is None


def test_job_id_check_in_db(pg):
    with pytest.raises(psycopg.errors.CheckViolation):
        pg["sql"](
            "INSERT INTO virtual_fitting_async_jobs (fitting_job_id, request_fingerprint, requested_at) "
            "VALUES (0, 'x', now())"
        )


# --- 트랜잭션 ---


class _FailAfterInsertCursor:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cursor.__exit__(*exc)

    def execute(self, *args, **kwargs):
        return self._cursor.execute(*args, **kwargs)

    def fetchone(self):
        # INSERT 는 이미 실행됐다. 결과를 읽는 순간 연결 문제가 난 것처럼 실패시킨다.
        raise psycopg.OperationalError("simulated failure after insert")


class _FailAfterInsertPool:
    def __init__(self, pool):
        self._pool = pool

    def connection(self):
        pool = self._pool

        class _Context:
            def __enter__(self_inner):
                self_inner._cm = pool.connection()
                connection = self_inner._cm.__enter__()
                original_cursor = connection.cursor
                connection.cursor = lambda *a, **kw: _FailAfterInsertCursor(
                    original_cursor(*a, **kw)
                )
                self_inner._connection, self_inner._original = connection, original_cursor
                return connection

            def __exit__(self_inner, *exc):
                self_inner._connection.cursor = self_inner._original
                return self_inner._cm.__exit__(*exc)

        return _Context()


def test_failure_inside_create_rolls_back_the_insert(pg):
    with pytest.raises(FittingDatabaseError) as exc_info:
        AsyncJobRepository(_FailAfterInsertPool(pg["pool"])).create(_new_job())

    assert "simulated" not in str(exc_info.value)  # DB 상세를 메시지에 싣지 않는다
    assert pg["sql"]("SELECT count(*) FROM virtual_fitting_async_jobs") == [(0,)]
    # 같은 pool 의 connection 이 깨끗하게 돌아와 다음 저장이 된다.
    assert AsyncJobRepository(pg["pool"]).create(_new_job()).created is True


def test_db_constraint_error_is_wrapped_and_nothing_is_saved(pg, monkeypatch):
    original = repo_module._check_new_job

    def bypass_validation(job):
        params = list(original(job))
        params[3] = ["123456", None]  # products 에 NULL — DB CHECK 위반
        return tuple(params)

    monkeypatch.setattr(repo_module, "_check_new_job", bypass_validation)
    with pytest.raises(FittingDatabaseError) as exc_info:
        AsyncJobRepository(pg["pool"]).create(_new_job())

    assert isinstance(exc_info.value.__cause__, psycopg.errors.CheckViolation)
    assert pg["sql"]("SELECT count(*) FROM virtual_fitting_async_jobs") == [(0,)]


def test_connection_failure_is_wrapped():
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        "postgresql://nobody@127.0.0.1:1/none",
        min_size=0,
        max_size=1,
        timeout=0.5,
        open=True,
        kwargs={"connect_timeout": 1},
    )
    try:
        with pytest.raises(FittingDatabaseError):
            AsyncJobRepository(pool).find_by_fitting_job_id(3)
    finally:
        pool.close()


# --- 마이그레이션 구조와 모델 ---


EXPECTED_COLUMNS = {
    "fitting_job_id": ("bigint", "NO"),
    "request_fingerprint": ("text", "NO"),
    "user_image_key": ("text", "YES"),
    "products": ("ARRAY", "YES"),
    "task_uuid": ("uuid", "YES"),
    "async_image_key": ("text", "YES"),
    "result_image_key": ("text", "YES"),
    "status": ("text", "NO"),
    "requested_at": ("timestamp with time zone", "NO"),
    "created_at": ("timestamp with time zone", "NO"),
    "updated_at": ("timestamp with time zone", "NO"),
    "pruna_submit_started_at": ("timestamp with time zone", "YES"),
    "runware_accepted_at": ("timestamp with time zone", "YES"),
    "next_action": ("text", "YES"),
    "verified_image_key": ("text", "YES"),
    "image_verified_at": ("timestamp with time zone", "YES"),
    "result_payload": ("jsonb", "YES"),
    "result_message_status": ("text", "NO"),
    "result_message_attempt_count": ("integer", "NO"),
    "llm_generation_attempt_count": ("integer", "NO"),
    "fallback_attempt_count": ("integer", "NO"),
    "s3_check_attempt_count": ("integer", "NO"),
    "worker_lease_token": ("text", "YES"),
    "worker_lease_until": ("timestamp with time zone", "YES"),
    "fallback_task_uuid": ("uuid", "YES"),
    "fallback_started_at": ("timestamp with time zone", "YES"),
    "fallback_image_key": ("text", "YES"),
}


def test_migrated_table_matches_the_model(pg):
    rows = pg["sql"](
        "SELECT column_name, data_type, is_nullable, udt_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = 'virtual_fitting_async_jobs' ORDER BY ordinal_position",
        (pg["schema"],),
    )

    assert tuple(row[0] for row in rows) == COLUMNS
    assert {row[0]: (row[1], row[2]) for row in rows} == EXPECTED_COLUMNS
    assert {row[0]: row[3] for row in rows}["products"] == "_text"
    assert set(EXPECTED_COLUMNS) == {field for field in AsyncFittingJob.__dataclass_fields__}


def test_keys_and_no_foreign_keys(pg):
    constraints = pg["sql"](
        "SELECT tc.constraint_type, kcu.column_name FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu "
        "  ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema "
        "WHERE tc.table_schema = %s AND tc.table_name = 'virtual_fitting_async_jobs' "
        "ORDER BY 1, 2",
        (pg["schema"],),
    )
    assert constraints == [("PRIMARY KEY", "fitting_job_id"), ("UNIQUE", "task_uuid")]
    assert ("FOREIGN KEY",) not in [(kind,) for kind, _ in constraints]


def test_migration_can_run_again(pg):
    AsyncJobRepository(pg["pool"]).create(_new_job())
    with psycopg.connect(pg["url"], autocommit=True, options=pg["options"]) as connection:
        connection.execute(MIGRATION.read_text(encoding="utf-8"))
        connection.execute(FALLBACK_MIGRATION.read_text(encoding="utf-8"))
    assert pg["sql"]("SELECT count(*) FROM virtual_fitting_async_jobs") == [(1,)]
