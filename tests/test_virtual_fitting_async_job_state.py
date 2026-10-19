"""가상피팅 V2 작업 상태 관리(조건부 변경·lease·시도 횟수·진행 기록)를 실제 PostgreSQL 에서 검증한다.

TEST_DATABASE_URL 이 없거나 연결할 수 없으면 PostgreSQL 테스트는 건너뛴다(pytest 는 .env 를 읽지 않는다).
테스트마다 고유한 schema 에 마이그레이션을 적용하고 끝나면 지운다.

동시성 테스트는 순차 호출이 아니다. 별도 connection 이 행 잠금(또는 커밋하지 않은 INSERT)을 쥔 상태에서
N 개 thread 가 각자 pool 의 다른 connection 으로 같은 UPDATE/INSERT 를 보내고, pg_stat_activity 로 N 개 backend 가
모두 그 잠금을 기다리는 것을 확인한 뒤에 잠금을 푼다. 그래서 실제로 겹친 트랜잭션들의 경쟁을 본다.
"""

import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pytest

from app.clients.s3 import S3ObjectState
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobRepository,
    AsyncJobStatus,
    NewAsyncFittingJob,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import (
    FALLBACK_LIMIT,
    LLM_GENERATION_LIMIT,
    RESULT_MESSAGE_LIMIT,
    S3_CHECK_LIMIT,
    AsyncJobStateRepository,
    ChangeReason,
)

MIGRATION = (
    Path(__file__).resolve().parents[1] / "scripts" / "db" / "virtual_fitting_async_jobs.sql"
)
# 13단계 변경 SQL(fallback 진행 기록 컬럼). 신규 생성 SQL 뒤에 적용한다(배포 순서와 같음).
FALLBACK_MIGRATION = MIGRATION.with_name("virtual_fitting_fallback.sql")
AUTH_HOLD_MIGRATION = MIGRATION.with_name("virtual_fitting_auth_hold.sql")
KST = timezone(timedelta(hours=9))
JOB = 3
TASK = "a770f077-f413-47de-9dac-be0b26a35da6"
ASYNC_KEY = "virtual-fitting/results/3/pruna-async.jpg"
# V2 결과 형식(result_payload 와 be-fitting-response 메시지). 최상위 status 로 성공·실패를 나눈다.
SUCCESS_PAYLOAD = {
    "fitting_job_id": JOB,
    "status": "COMPLETED",
    "result_image_key": ASYNC_KEY,
    "llm_title": "차분한 데일리룩",
    "llm_comment": "선택하신 상품으로 완성한 가상 피팅 결과입니다.",
}
OTHER_SUCCESS_PAYLOAD = {**SUCCESS_PAYLOAD, "llm_title": "다른 제목", "llm_comment": "다른 코멘트"}
FAILED_PAYLOAD = {
    "fitting_job_id": JOB,
    "status": "FAILED",
    "error_code": "FITTING_MODEL_FAILED",
    "error_message": "가상 피팅 이미지 생성에 실패했습니다.",
}
LEASE = 60


def _new_job(fitting_job_id=JOB, fingerprint="fp-1"):
    return NewAsyncFittingJob(
        fitting_job_id=fitting_job_id,
        request_fingerprint=fingerprint,
        user_image_key="users/3.jpg",
        products=["123456", "789012"],
        requested_at=datetime(2026, 10, 13, 9, 30, tzinfo=KST),
    )


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

    schema = f"vf_state_test_{uuid.uuid4().hex[:12]}"
    admin.execute(f"CREATE SCHEMA {schema}")
    options = f"-c search_path={schema}"
    with psycopg.connect(url, autocommit=True, options=options) as connection:
        connection.execute(MIGRATION.read_text(encoding="utf-8"))
        connection.execute(FALLBACK_MIGRATION.read_text(encoding="utf-8"))
        connection.execute(AUTH_HOLD_MIGRATION.read_text(encoding="utf-8"))
    pool = ConnectionPool(url, min_size=1, max_size=20, kwargs={"options": options}, open=True)

    def sql(statement, params=None):
        with psycopg.connect(url, autocommit=True, options=options) as connection:
            cursor = connection.execute(statement, params)
            return cursor.fetchall() if cursor.description else None

    def connect():
        return psycopg.connect(url, options=options)

    try:
        yield {
            "pool": pool,
            "sql": sql,
            "connect": connect,
            "jobs": AsyncJobRepository(pool),
            "state": AsyncJobStateRepository(pool),
        }
    finally:
        pool.close()
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()


def created(pg, job_id=JOB):
    return pg["jobs"].create(_new_job(job_id)).job


def arrange(pg, job_id=JOB, **columns):
    """준비용 직접 UPDATE(테스트 상태를 만들 때만 쓴다)."""
    assignments = ", ".join(f"{name} = %({name})s" for name in columns)
    pg["sql"](
        f"UPDATE virtual_fitting_async_jobs SET {assignments} WHERE fitting_job_id = %(job_id)s",
        {**columns, "job_id": job_id},
    )


def expire_lease(pg, job_id=JOB):
    pg["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second' "
        "WHERE fitting_job_id = %s",
        (job_id,),
    )


def row(pg, job_id=JOB):
    return pg["jobs"].find_by_fitting_job_id(job_id)


def lease(pg, action, job_id=JOB):
    grant = pg["state"].acquire_lease(job_id, action, LEASE)
    assert grant.acquired, grant.reason
    return grant.token


def submitted(pg):
    """RECEIVED 작업에 task_uuid 를 준비하고 호출 시작까지 기록한 상태."""
    created(pg)
    token = lease(pg, NextAction.SUBMIT_PRUNA)
    assert pg["state"].prepare_pruna_submission(JOB, token, TASK, ASYNC_KEY).applied
    assert pg["state"].mark_pruna_submit_started(JOB, token).applied
    return token


def postprocessing(pg):
    """Webhook 성공까지 받은 POSTPROCESSING / POSTPROCESS 상태와 후처리 Worker 의 토큰."""
    token = submitted(pg)
    assert pg["state"].release_lease(JOB, token).applied
    assert pg["state"].record_pruna_succeeded(TASK).applied
    return lease(pg, NextAction.POSTPROCESS)


def lock_waiters(pg):
    return pg["sql"](
        "SELECT count(*) FROM pg_stat_activity "
        "WHERE datname = current_database() AND wait_event_type = 'Lock'"
    )[0][0]


def run_contended(pg, calls, hold):
    """hold() 가 잠금을 쥔 동안 calls 를 각자 다른 thread·connection 에서 동시에 보내고, 모두 잠금을
    기다리는 것을 확인한 뒤 풀어 준다. 결과 목록을 돌려준다."""
    results = [None] * len(calls)
    errors = []

    def worker(index, call):
        try:
            results[index] = call()
        except Exception as error:  # noqa: BLE001 (thread 안 예외를 테스트로 전달)
            errors.append(error)

    with hold():
        threads = [threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(calls)]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 15
        while lock_waiters(pg) < len(calls):
            assert time.monotonic() < deadline, (
                "모든 요청이 같은 잠금을 기다리는 상태가 되지 않았다"
            )
            time.sleep(0.02)
    for thread in threads:
        thread.join(20)
    assert not errors, errors
    return results


def row_lock(pg, job_id=JOB):
    @contextmanager
    def hold():
        connection = pg["connect"]()
        try:
            connection.execute(
                "SELECT 1 FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s FOR UPDATE",
                (job_id,),
            )
            yield
        finally:
            connection.rollback()
            connection.close()

    return hold


# --- 실행 권한(lease) ---


def test_acquire_gives_a_new_token_and_db_time_expiry(pg):
    created(pg)
    grant = pg["state"].acquire_lease(JOB, NextAction.SUBMIT_PRUNA, LEASE)

    assert grant.acquired and grant.reason is ChangeReason.APPLIED
    assert uuid.UUID(grant.token).version == 4
    (db_now,) = pg["sql"]("SELECT now()")[0]
    assert grant.job.worker_lease_token == grant.token
    assert timedelta(seconds=50) < grant.job.worker_lease_until - db_now <= timedelta(seconds=60)


def test_acquire_requires_the_expected_action_and_a_live_job(pg):
    assert (
        pg["state"].acquire_lease(JOB, NextAction.SUBMIT_PRUNA, LEASE).reason
        is ChangeReason.NOT_FOUND
    )
    created(pg)
    wrong = pg["state"].acquire_lease(JOB, NextAction.POSTPROCESS, LEASE)
    assert (wrong.acquired, wrong.reason, wrong.token) == (False, ChangeReason.STATE_MISMATCH, None)
    arrange(pg, status="COMPLETED", next_action="SUBMIT_PRUNA")
    assert (
        pg["state"].acquire_lease(JOB, NextAction.SUBMIT_PRUNA, LEASE).reason
        is ChangeReason.TERMINAL
    )


def test_live_lease_blocks_others_and_expiry_gives_a_new_token(pg):
    created(pg)
    first = lease(pg, NextAction.SUBMIT_PRUNA)

    blocked = pg["state"].acquire_lease(JOB, NextAction.SUBMIT_PRUNA, LEASE)
    assert (blocked.acquired, blocked.reason) == (False, ChangeReason.LEASE_HELD)
    assert row(pg).worker_lease_token == first

    expire_lease(pg)
    second = lease(pg, NextAction.SUBMIT_PRUNA)
    assert second != first
    for change in (
        pg["state"].renew_lease(JOB, first, LEASE),
        pg["state"].release_lease(JOB, first),
        pg["state"].prepare_pruna_submission(JOB, first, TASK, ASYNC_KEY),
    ):
        assert (change.applied, change.reason) == (False, ChangeReason.LEASE_INVALID)
    assert row(pg).worker_lease_token == second and row(pg).task_uuid is None


def test_renew_and_release_with_a_valid_token(pg):
    created(pg)
    token = lease(pg, NextAction.SUBMIT_PRUNA)
    before = row(pg).worker_lease_until

    renewed = pg["state"].renew_lease(JOB, token, 300)
    assert renewed.applied and renewed.job.worker_lease_until > before
    assert pg["state"].holds_lease(JOB, token)

    released = pg["state"].release_lease(JOB, token)
    assert released.applied
    assert (released.job.worker_lease_token, released.job.worker_lease_until) == (None, None)
    assert not pg["state"].holds_lease(JOB, token)


def test_wrong_or_expired_token_changes_nothing(pg):
    created(pg)
    token = lease(pg, NextAction.SUBMIT_PRUNA)
    snapshot = row(pg)

    wrong = pg["state"].renew_lease(JOB, str(uuid.uuid4()), LEASE)
    assert (wrong.applied, wrong.reason) == (False, ChangeReason.LEASE_INVALID)
    expire_lease(pg)
    expired_snapshot = row(pg)
    for change in (
        pg["state"].renew_lease(JOB, token, LEASE),
        pg["state"].release_lease(JOB, token),
        pg["state"].prepare_pruna_submission(JOB, token, TASK, ASYNC_KEY),
    ):
        assert (change.applied, change.reason) == (False, ChangeReason.LEASE_INVALID)
    assert not pg["state"].holds_lease(JOB, token)
    assert row(pg) == expired_snapshot
    assert snapshot.task_uuid is None


@pytest.mark.parametrize("bad", [0, -1, 3601, True, "60"])
def test_lease_seconds_are_validated(pg, bad):
    created(pg)
    with pytest.raises(ValueError):
        pg["state"].acquire_lease(JOB, NextAction.SUBMIT_PRUNA, bad)


def test_concurrent_acquire_has_exactly_one_winner(pg):
    created(pg)
    calls = [
        lambda: pg["state"].acquire_lease(JOB, NextAction.SUBMIT_PRUNA, LEASE) for _ in range(12)
    ]

    grants = run_contended(pg, calls, row_lock(pg))

    winners = [grant for grant in grants if grant.acquired]
    assert len(winners) == 1
    assert {grant.reason for grant in grants if not grant.acquired} == {ChangeReason.LEASE_HELD}
    assert row(pg).worker_lease_token == winners[0].token


# --- Pruna 요청 기록 ---


def test_prepare_then_submit_started_once(pg):
    created(pg)
    token = lease(pg, NextAction.SUBMIT_PRUNA)

    prepared = pg["state"].prepare_pruna_submission(JOB, token, TASK, ASYNC_KEY)
    assert prepared.applied and (prepared.job.task_uuid, prepared.job.async_image_key) == (
        TASK,
        ASYNC_KEY,
    )
    again = pg["state"].prepare_pruna_submission(JOB, token, str(uuid.uuid4()), "other/key.jpg")
    assert (again.applied, again.reason, again.job.task_uuid) == (
        False,
        ChangeReason.ALREADY_RECORDED,
        TASK,
    )

    started = pg["state"].mark_pruna_submit_started(JOB, token)
    assert started.applied and started.job.pruna_submit_started_at is not None
    repeat = pg["state"].mark_pruna_submit_started(JOB, token)
    assert (repeat.applied, repeat.reason) == (False, ChangeReason.ALREADY_RECORDED)
    assert repeat.job.pruna_submit_started_at == started.job.pruna_submit_started_at


def test_submit_cannot_start_before_preparation(pg):
    created(pg)
    token = lease(pg, NextAction.SUBMIT_PRUNA)
    change = pg["state"].mark_pruna_submit_started(JOB, token)
    assert (change.applied, change.reason) == (False, ChangeReason.STATE_MISMATCH)
    assert row(pg).pruna_submit_started_at is None


def test_lease_expiry_does_not_allow_a_second_pruna_call(pg):
    first = submitted(pg)
    expire_lease(pg)
    second = lease(pg, NextAction.SUBMIT_PRUNA)

    # 접수 여부를 모르는 채로 호출 시작 기록이 남아 있다. 새 Worker 가 다시 시작할 수 없다.
    change = pg["state"].mark_pruna_submit_started(JOB, second)
    assert (change.applied, change.reason) == (False, ChangeReason.ALREADY_RECORDED)
    assert second != first


def test_worker_acceptance_moves_to_waiting_and_keeps_the_first_time(pg):
    token = submitted(pg)

    first = pg["state"].record_runware_accepted(JOB, token, TASK)
    assert first.applied
    assert (first.job.status, first.job.next_action) == (AsyncJobStatus.WAITING_PRUNA, None)
    accepted_at = first.job.runware_accepted_at
    assert accepted_at is not None

    again = pg["state"].record_runware_accepted(JOB, token, TASK)
    assert again.applied and again.job.runware_accepted_at == accepted_at


def test_acceptance_needs_the_registered_task_uuid(pg):
    token = submitted(pg)
    change = pg["state"].record_runware_accepted(JOB, token, str(uuid.uuid4()))
    assert (change.applied, change.reason) == (False, ChangeReason.STATE_MISMATCH)
    assert row(pg).status is AsyncJobStatus.RECEIVED


def test_late_acceptance_does_not_move_back_to_waiting(pg):
    token = submitted(pg)
    webhook = pg["state"].record_pruna_succeeded(TASK)  # Webhook 이 접수 처리보다 먼저 왔다
    assert webhook.applied and webhook.job.status is AsyncJobStatus.POSTPROCESSING
    first_accepted = webhook.job.runware_accepted_at

    late = pg["state"].record_runware_accepted(JOB, token, TASK)

    assert late.applied
    assert (late.job.status, late.job.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert late.job.runware_accepted_at == first_accepted


# --- 외부 이벤트(Webhook) 경로 ---


def test_event_path_requires_registered_task_and_submit_record(pg):
    assert pg["state"].record_pruna_succeeded(TASK).reason is ChangeReason.NOT_FOUND
    created(pg)
    token = lease(pg, NextAction.SUBMIT_PRUNA)
    assert pg["state"].prepare_pruna_submission(JOB, token, TASK, ASYNC_KEY).applied

    # task_uuid 는 있지만 호출 시작 기록이 없다 → 검증 없는 갱신을 받지 않는다.
    change = pg["state"].record_pruna_succeeded(TASK)
    assert (change.applied, change.reason) == (False, ChangeReason.STATE_MISMATCH)
    assert row(pg).status is AsyncJobStatus.RECEIVED


def test_duplicate_success_event_changes_nothing(pg):
    submitted(pg)
    assert pg["state"].record_pruna_succeeded(TASK).applied
    before = row(pg)
    duplicate = pg["state"].record_pruna_succeeded(TASK)
    assert (duplicate.applied, duplicate.reason) == (False, ChangeReason.STATE_MISMATCH)
    assert row(pg) == before


def test_failure_events(pg):
    submitted(pg)
    recoverable = pg["state"].record_pruna_failed(TASK, recoverable=True)
    assert recoverable.applied
    assert (recoverable.job.status, recoverable.job.next_action) == (
        AsyncJobStatus.FALLBACK_RUNNING,
        NextAction.RUN_FALLBACK,
    )


def test_unrecoverable_failure_event_saves_the_failure_result(pg):
    submitted(pg)
    payload = FAILED_PAYLOAD
    change = pg["state"].record_pruna_failed(TASK, recoverable=False, failure_payload=payload)
    assert change.applied
    assert (change.job.result_payload, change.job.next_action) == (
        payload,
        NextAction.PUBLISH_RESULT,
    )
    assert change.job.status is AsyncJobStatus.RECEIVED  # 발행 성공 뒤에 FAILED 로 확정한다
    with pytest.raises(ValueError):
        pg["state"].record_pruna_failed(TASK, recoverable=False)


# --- 단계별 시도 횟수 ---


@pytest.mark.parametrize(
    ("claim", "column", "limit", "prepare"),
    [
        ("claim_s3_check_attempt", "s3_check_attempt_count", S3_CHECK_LIMIT, None),
        (
            "claim_llm_generation_attempt",
            "llm_generation_attempt_count",
            LLM_GENERATION_LIMIT,
            "verify",
        ),
    ],
)
def test_postprocess_counters_stop_at_their_limit(pg, claim, column, limit, prepare):
    token = postprocessing(pg)
    if prepare == "verify":
        assert pg["state"].record_image_verified(JOB, token, ASYNC_KEY).applied

    results = [getattr(pg["state"], claim)(JOB, token) for _ in range(limit + 2)]

    assert [r.applied for r in results] == [True] * limit + [False, False]
    assert {r.reason for r in results[limit:]} == {ChangeReason.LIMIT_REACHED}
    assert getattr(row(pg), column) == limit


def test_s3_check_stops_once_the_image_is_verified(pg):
    token = postprocessing(pg)
    assert pg["state"].claim_s3_check_attempt(JOB, token).applied
    verified = pg["state"].record_image_verified(JOB, token, ASYNC_KEY)
    assert verified.applied and verified.job.image_verified_at is not None

    again = pg["state"].claim_s3_check_attempt(JOB, token)
    assert (again.applied, again.reason) == (False, ChangeReason.ALREADY_RECORDED)
    other = pg["state"].record_image_verified(JOB, token, "virtual-fitting/results/3/other.jpg")
    assert (other.applied, other.reason) == (False, ChangeReason.ALREADY_RECORDED)
    assert row(pg).verified_image_key == ASYNC_KEY  # 확인한 이미지는 재사용을 위해 보존
    assert row(pg).s3_check_attempt_count == 1


def test_llm_generation_needs_a_verified_image(pg):
    token = postprocessing(pg)
    change = pg["state"].claim_llm_generation_attempt(JOB, token)
    assert (change.applied, change.reason) == (False, ChangeReason.STATE_MISMATCH)
    assert row(pg).llm_generation_attempt_count == 0


def s3_checks_used_up(pg, token):
    for _ in range(S3_CHECK_LIMIT):
        assert pg["state"].claim_s3_check_attempt(JOB, token).applied


def to_fallback(pg, token):
    return pg["state"].move_to_fallback(JOB, token, final_s3_check=S3ObjectState.NOT_FOUND)


def test_fallback_runs_at_most_once(pg):
    token = postprocessing(pg)
    s3_checks_used_up(pg, token)
    moved = to_fallback(pg, token)
    assert moved.applied
    assert (moved.job.status, moved.job.next_action) == (
        AsyncJobStatus.FALLBACK_RUNNING,
        NextAction.RUN_FALLBACK,
    )

    first = pg["state"].claim_fallback_attempt(JOB, token)
    second = pg["state"].claim_fallback_attempt(JOB, token)
    assert first.applied and (second.applied, second.reason) == (False, ChangeReason.LIMIT_REACHED)
    assert row(pg).fallback_attempt_count == FALLBACK_LIMIT

    # lease 가 만료돼 다른 Worker 가 잡아도 V1 을 다시 부를 수 없다.
    expire_lease(pg)
    other = lease(pg, NextAction.RUN_FALLBACK)
    retry = pg["state"].claim_fallback_attempt(JOB, other)
    assert (retry.applied, retry.reason) == (False, ChangeReason.LIMIT_REACHED)


def test_move_to_fallback_is_refused_after_the_image_is_verified(pg):
    token = postprocessing(pg)
    s3_checks_used_up(pg, token)  # 횟수 조건은 충족시켜, 확인한 이미지가 막는지만 본다
    assert pg["state"].record_image_verified(JOB, token, ASYNC_KEY).applied
    change = to_fallback(pg, token)
    assert (change.applied, change.reason) == (False, ChangeReason.STATE_MISMATCH)
    assert row(pg).status is AsyncJobStatus.POSTPROCESSING
    assert row(pg).verified_image_key == ASYNC_KEY  # 확인한 이미지는 보존


def test_counts_carry_over_to_the_next_worker(pg):
    first = postprocessing(pg)
    assert pg["state"].record_image_verified(JOB, first, ASYNC_KEY).applied
    assert pg["state"].claim_llm_generation_attempt(JOB, first).applied

    expire_lease(pg)
    second = lease(pg, NextAction.POSTPROCESS)
    stale = pg["state"].claim_llm_generation_attempt(JOB, first)
    assert (stale.applied, stale.reason) == (False, ChangeReason.LEASE_INVALID)

    assert (
        pg["state"].claim_llm_generation_attempt(JOB, second).job.llm_generation_attempt_count == 2
    )
    assert (
        pg["state"].claim_llm_generation_attempt(JOB, second).job.llm_generation_attempt_count == 3
    )
    assert (
        pg["state"].claim_llm_generation_attempt(JOB, second).reason is ChangeReason.LIMIT_REACHED
    )


def test_re_registration_does_not_reset_counts(pg):
    token = postprocessing(pg)
    assert pg["state"].claim_s3_check_attempt(JOB, token).applied

    again = pg["jobs"].create(_new_job(fingerprint="fp-other"))

    assert again.created is False
    assert again.job.s3_check_attempt_count == 1
    assert again.job.request_fingerprint == "fp-1"
    assert again.job.status is AsyncJobStatus.POSTPROCESSING


@pytest.mark.parametrize(
    ("claim", "column", "limit"),
    [
        ("claim_s3_check_attempt", "s3_check_attempt_count", S3_CHECK_LIMIT),
        ("claim_llm_generation_attempt", "llm_generation_attempt_count", LLM_GENERATION_LIMIT),
        ("claim_result_message_attempt", "result_message_attempt_count", RESULT_MESSAGE_LIMIT),
    ],
)
def test_concurrent_claims_never_exceed_the_limit(pg, claim, column, limit):
    token = postprocessing(pg)
    if claim == "claim_llm_generation_attempt":
        assert pg["state"].record_image_verified(JOB, token, ASYNC_KEY).applied
    if claim == "claim_result_message_attempt":
        assert pg["state"].save_final_result(JOB, token, SUCCESS_PAYLOAD, ASYNC_KEY).applied
    calls = [lambda: getattr(pg["state"], claim)(JOB, token) for _ in range(10)]

    results = run_contended(pg, calls, row_lock(pg))

    assert sum(r.applied for r in results) == limit
    assert {r.reason for r in results if not r.applied} == {ChangeReason.LIMIT_REACHED}
    assert getattr(row(pg), column) == limit


def test_concurrent_fallback_claim_runs_once(pg):
    token = postprocessing(pg)
    s3_checks_used_up(pg, token)
    assert to_fallback(pg, token).applied
    calls = [lambda: pg["state"].claim_fallback_attempt(JOB, token) for _ in range(8)]

    results = run_contended(pg, calls, row_lock(pg))

    assert sum(r.applied for r in results) == 1
    assert row(pg).fallback_attempt_count == 1


# --- 같은 작업 동시 등록 ---


def test_concurrent_create_makes_one_row_and_keeps_it(pg):
    @contextmanager
    def pending_insert():
        # 같은 번호의 INSERT 를 커밋하지 않고 쥐고 있으면 다른 INSERT 들이 그 결과를 기다린다. 롤백하면
        # 기다리던 요청 중 하나가 행을 만들고, 나머지는 그 행과 충돌해 기존 행을 받는다.
        connection = pg["connect"]()
        try:
            connection.execute(
                "INSERT INTO virtual_fitting_async_jobs (fitting_job_id, request_fingerprint, requested_at) "
                "VALUES (%s, 'gate', now())",
                (JOB,),
            )
            yield
        finally:
            connection.rollback()
            connection.close()

    calls = [(lambda i=i: pg["jobs"].create(_new_job(fingerprint=f"fp-{i}"))) for i in range(8)]

    results = run_contended(pg, calls, pending_insert)

    winners = [r for r in results if r.created]
    assert len(winners) == 1
    assert pg["sql"]("SELECT count(*) FROM virtual_fitting_async_jobs") == [(1,)]
    stored = row(pg)
    assert stored.request_fingerprint == winners[0].job.request_fingerprint
    assert all(r.job.request_fingerprint == stored.request_fingerprint for r in results)


# --- 최종 결과와 발행 ---


def test_final_result_is_saved_once_with_publish_action(pg):
    token = postprocessing(pg)
    saved = pg["state"].save_final_result(JOB, token, SUCCESS_PAYLOAD, ASYNC_KEY)
    assert saved.applied
    assert (saved.job.result_payload, saved.job.result_image_key, saved.job.next_action) == (
        SUCCESS_PAYLOAD,
        ASYNC_KEY,
        NextAction.PUBLISH_RESULT,
    )

    other = pg["state"].save_final_result(JOB, token, OTHER_SUCCESS_PAYLOAD, ASYNC_KEY)
    assert (other.applied, other.reason) == (False, ChangeReason.ALREADY_RECORDED)
    assert row(pg).result_payload == SUCCESS_PAYLOAD and row(pg).result_image_key == ASYNC_KEY


def test_stale_worker_cannot_overwrite_after_reacquire(pg):
    old = postprocessing(pg)
    expire_lease(pg)
    new = lease(pg, NextAction.POSTPROCESS)
    assert pg["state"].save_final_result(JOB, new, SUCCESS_PAYLOAD, ASYNC_KEY).applied

    late = pg["state"].save_final_result(JOB, old, OTHER_SUCCESS_PAYLOAD, ASYNC_KEY)
    assert (late.applied, late.reason) == (False, ChangeReason.LEASE_INVALID)
    assert row(pg).result_payload == SUCCESS_PAYLOAD


def test_success_on_the_last_allowed_attempt_is_recorded_consistently(pg):
    token = postprocessing(pg)
    assert pg["state"].save_final_result(JOB, token, SUCCESS_PAYLOAD, ASYNC_KEY).applied
    for _ in range(RESULT_MESSAGE_LIMIT):
        assert pg["state"].claim_result_message_attempt(JOB, token).applied

    published = pg["state"].mark_result_published(JOB, token, AsyncJobStatus.COMPLETED)

    assert published.applied
    job = published.job
    assert job.result_message_attempt_count == RESULT_MESSAGE_LIMIT
    assert (job.status, job.result_message_status, job.next_action) == (
        AsyncJobStatus.COMPLETED,
        ResultMessageStatus.PUBLISHED,
        None,
    )
    assert (job.worker_lease_token, job.worker_lease_until) == (None, None)
    assert job.result_payload == SUCCESS_PAYLOAD  # 결과는 추적을 위해 남는다


def test_publish_needs_a_claimed_attempt(pg):
    token = postprocessing(pg)
    assert pg["state"].save_final_result(JOB, token, SUCCESS_PAYLOAD, ASYNC_KEY).applied
    change = pg["state"].mark_result_published(JOB, token, AsyncJobStatus.COMPLETED)
    assert (change.applied, change.reason) == (False, ChangeReason.STATE_MISMATCH)
    with pytest.raises(ValueError):
        pg["state"].mark_result_published(JOB, token, AsyncJobStatus.POSTPROCESSING)


def test_retry_exhausted_keeps_the_result_for_recovery(pg):
    token = postprocessing(pg)
    assert pg["state"].save_final_result(JOB, token, SUCCESS_PAYLOAD, ASYNC_KEY).applied
    assert pg["state"].claim_result_message_attempt(JOB, token).applied
    early = pg["state"].mark_result_retry_exhausted(JOB, token)
    assert (early.applied, early.reason) == (False, ChangeReason.STATE_MISMATCH)
    for _ in range(RESULT_MESSAGE_LIMIT - 1):
        assert pg["state"].claim_result_message_attempt(JOB, token).applied
    assert pg["state"].claim_result_message_attempt(JOB, token).reason is ChangeReason.LIMIT_REACHED

    exhausted = pg["state"].mark_result_retry_exhausted(JOB, token)

    assert exhausted.applied
    job = exhausted.job
    assert job.result_message_status is ResultMessageStatus.RETRY_EXHAUSTED
    assert job.status is AsyncJobStatus.POSTPROCESSING  # 종료 상태로 확정하지 않는다
    assert (job.next_action, job.result_payload) == (NextAction.PUBLISH_RESULT, SUCCESS_PAYLOAD)
    assert (job.worker_lease_token, job.worker_lease_until) == (None, None)
    assert (job.user_image_key, job.products) == ("users/3.jpg", ("123456", "789012"))


def test_terminal_job_is_never_changed(pg):
    token = postprocessing(pg)
    assert pg["state"].save_final_result(JOB, token, SUCCESS_PAYLOAD, ASYNC_KEY).applied
    assert pg["state"].claim_result_message_attempt(JOB, token).applied
    assert pg["state"].mark_result_published(JOB, token, AsyncJobStatus.COMPLETED).applied
    final = row(pg)

    changes = [
        pg["state"].record_pruna_succeeded(TASK),
        pg["state"].record_pruna_failed(TASK, recoverable=True),
        pg["state"].acquire_lease(JOB, NextAction.POSTPROCESS, LEASE),
        pg["state"].save_final_result(JOB, token, OTHER_SUCCESS_PAYLOAD, ASYNC_KEY),
        pg["state"].claim_result_message_attempt(JOB, token),
    ]

    assert all(
        not getattr(change, "applied", getattr(change, "acquired", False)) for change in changes
    )
    assert changes[0].reason is ChangeReason.TERMINAL and changes[3].reason is ChangeReason.TERMINAL
    assert row(pg) == final


def test_updated_at_moves_only_when_something_changed(pg):
    created(pg)
    before = row(pg).updated_at
    time.sleep(0.01)
    token = lease(pg, NextAction.SUBMIT_PRUNA)
    assert row(pg).updated_at > before
    after_lease = row(pg).updated_at
    assert not pg["state"].mark_pruna_submit_started(JOB, token).applied
    assert row(pg).updated_at == after_lease


def test_returned_times_are_utc(pg):
    token = submitted(pg)
    job = pg["state"].record_runware_accepted(JOB, token, TASK).job
    for value in (job.runware_accepted_at, job.pruna_submit_started_at, job.worker_lease_until):
        assert value.tzinfo is UTC


# --- 트랜잭션 ---


def test_failure_after_the_update_rolls_it_back(pg):
    token = postprocessing(pg)
    before = row(pg)

    class FailAfterUpdate:
        def __init__(self, pool):
            self._pool = pool

        @contextmanager
        def connection(self):
            with self._pool.connection() as connection:
                original = connection.cursor

                @contextmanager
                def cursor(*args, **kwargs):
                    with original(*args, **kwargs) as real:

                        class Cursor:
                            updated = False

                            def execute(self, sql, *a, **k):
                                self.updated = self.updated or sql.lstrip().startswith("UPDATE")
                                return real.execute(sql, *a, **k)

                            def fetchone(self):
                                if self.updated:
                                    # UPDATE 는 이미 실행됐다. 결과를 읽는 순간 연결 문제가 난 것처럼 실패시킨다.
                                    raise psycopg.OperationalError("simulated failure after update")
                                return real.fetchone()

                        yield Cursor()

                connection.cursor = cursor
                try:
                    yield connection
                finally:
                    connection.cursor = original

    with pytest.raises(FittingDatabaseError) as exc_info:
        AsyncJobStateRepository(FailAfterUpdate(pg["pool"])).claim_s3_check_attempt(JOB, token)

    assert "simulated" not in str(exc_info.value)
    assert row(pg) == before  # 횟수·updated_at 모두 그대로
    assert pg["state"].claim_s3_check_attempt(JOB, token).job.s3_check_attempt_count == 1


def test_invalid_inputs_are_rejected_before_the_db(pg):
    created(pg)
    token = lease(pg, NextAction.SUBMIT_PRUNA)
    with pytest.raises(ValueError):
        pg["state"].prepare_pruna_submission(JOB, token, "not-a-uuid", ASYNC_KEY)
    with pytest.raises(ValueError):
        pg["state"].prepare_pruna_submission(JOB, token, TASK, "/bad/key")
    with pytest.raises(ValueError):
        pg["state"].renew_lease(JOB, "", LEASE)
    with pytest.raises(ValueError):
        pg["state"].save_final_result(JOB, token, {}, None)
    assert row(pg).task_uuid is None


# =====================================================================================
# 5단계 보완 점검: 잠금 대기 중 lease 만료, V1 fallback 전환 조건, 저장된 결과와 발행 상태 일치.
# (보완 전에는 xfail(strict=True) 로 실제 DB 에서 문제를 재현했던 테스트들이다.)
# =====================================================================================


def db_clock_passed_lease(pg, job_id=JOB):
    # 일반 SELECT 는 FOR UPDATE 잠금에 막히지 않는다. clock_timestamp() 는 실제 현재 시각이다.
    return pg["sql"](
        "SELECT clock_timestamp() > worker_lease_until FROM virtual_fitting_async_jobs "
        "WHERE fitting_job_id = %s",
        (job_id,),
    )[0][0]


def set_lease_ttl(pg, seconds, job_id=JOB):
    pg["sql"](
        "UPDATE virtual_fitting_async_jobs "
        "SET worker_lease_until = clock_timestamp() + make_interval(secs => %s) "
        "WHERE fitting_job_id = %s",
        (seconds, job_id),
    )


def call_while_row_locked(pg, call, *, release_after, gate_mode, job_id=JOB):
    """별도 connection 이 행을 잠근 동안 call 을 다른 thread·connection 에서 보내 잠금을 기다리게 하고,
    release_after() 가 참이 될 때까지(DB 실제 시각 기준) 기다린 뒤 잠금을 푼다.

    gate_mode: "lock_only" 는 FOR UPDATE 후 rollback(행 버전 그대로), "update" 는 같은 행을 UPDATE 하고 commit
    (기다리던 UPDATE 가 새 버전에서 WHERE 를 다시 평가하는 경우)."""
    gate = pg["connect"]()
    result, errors = {}, []
    if gate_mode == "update":
        gate.execute(
            "UPDATE virtual_fitting_async_jobs SET updated_at = updated_at WHERE fitting_job_id = %s",
            (job_id,),
        )
    else:
        gate.execute(
            "SELECT 1 FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s FOR UPDATE",
            (job_id,),
        )

    def worker():
        try:
            result["value"] = call()
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=worker)
    try:
        thread.start()
        deadline = time.monotonic() + 15
        while lock_waiters(pg) < 1:
            assert time.monotonic() < deadline, "요청이 잠금을 기다리지 않았다"
            time.sleep(0.02)
        while not release_after():
            assert time.monotonic() < deadline, "DB 시각 기준 조건이 오지 않았다"
            time.sleep(0.05)
    finally:
        if gate_mode == "update":
            gate.commit()
        else:
            gate.rollback()
        gate.close()
    thread.join(15)
    assert not errors, errors
    return result["value"]


def _verified_postprocessing(pg):
    token = postprocessing(pg)
    assert pg["state"].record_image_verified(JOB, token, ASYNC_KEY).applied
    return token


LOCK_WAIT_OPERATIONS = {
    "renew": lambda pg, token: pg["state"].renew_lease(JOB, token, LEASE),
    "claim_llm": lambda pg, token: pg["state"].claim_llm_generation_attempt(JOB, token),
    "save_result": lambda pg, token: pg["state"].save_final_result(
        JOB, token, SUCCESS_PAYLOAD, ASYNC_KEY
    ),
}


def test_control_lease_valid_at_call_time_but_expired_before_lock_release_is_a_real_race(pg):
    # 아래 재현 테스트가 "호출 전에 이미 만료된 lease" 테스트와 다르다는 것을 확인하는 대조 실험이다.
    token = _verified_postprocessing(pg)
    set_lease_ttl(pg, 1.5)
    assert pg["state"].holds_lease(JOB, token)  # 호출 시점에는 유효하다
    assert not db_clock_passed_lease(pg)

    def call():
        return pg["state"].claim_llm_generation_attempt(JOB, token)

    call_while_row_locked(
        pg, call, release_after=lambda: db_clock_passed_lease(pg), gate_mode="lock_only"
    )
    assert db_clock_passed_lease(pg)  # 잠금을 푼 시점에는 DB 실제 시각으로 만료돼 있었다


@pytest.mark.parametrize("gate_mode", ["lock_only", "update"])
@pytest.mark.parametrize("operation", sorted(LOCK_WAIT_OPERATIONS))
def test_lease_expiring_during_the_lock_wait_must_not_change_anything(pg, operation, gate_mode):
    token = _verified_postprocessing(pg)
    set_lease_ttl(pg, 1.5)
    assert pg["state"].holds_lease(JOB, token)
    before = row(pg)

    change = call_while_row_locked(
        pg,
        lambda: LOCK_WAIT_OPERATIONS[operation](pg, token),
        release_after=lambda: db_clock_passed_lease(pg),
        gate_mode=gate_mode,
    )

    after = row(pg)
    assert (change.applied, change.reason) == (False, ChangeReason.LEASE_INVALID)
    assert after.llm_generation_attempt_count == before.llm_generation_attempt_count
    assert after.result_payload == before.result_payload
    assert after.worker_lease_until == before.worker_lease_until
    assert (after.status, after.next_action) == (before.status, before.next_action)


def test_lease_granted_after_a_long_lock_wait_is_not_already_expired(pg):
    created(pg)
    waited = {}

    def call():
        return pg["state"].acquire_lease(JOB, NextAction.SUBMIT_PRUNA, 1)

    def two_seconds_passed():
        now = pg["sql"]("SELECT clock_timestamp()")[0][0]
        waited.setdefault("start", now)
        return now - waited["start"] > timedelta(seconds=2)

    grant = call_while_row_locked(pg, call, release_after=two_seconds_passed, gate_mode="lock_only")

    assert grant.acquired
    # 1초짜리 lease 를 받았는데, 받은 직후 DB 실제 시각으로 이미 만료돼 있으면 안 된다.
    assert not db_clock_passed_lease(pg)


# --- V1 fallback 전환 조건 ---


def test_no_fallback_while_pruna_is_running(pg):
    token = submitted(pg)
    assert (
        pg["state"].record_runware_accepted(JOB, token, TASK).applied
    )  # WAITING_PRUNA, Pruna 실행 중
    change = to_fallback(pg, token)
    assert not change.applied
    assert row(pg).status is AsyncJobStatus.WAITING_PRUNA


def test_no_fallback_when_acceptance_is_unknown_even_after_lease_expiry(pg):
    submitted(pg)  # 호출 시작 기록은 있고 접수 결과는 모른다
    expire_lease(pg)
    other = lease(pg, NextAction.SUBMIT_PRUNA)  # lease 만료 → 다른 Worker 가 잡았다
    change = to_fallback(pg, other)
    assert not change.applied
    assert row(pg).status is AsyncJobStatus.RECEIVED and row(pg).fallback_attempt_count == 0


def test_no_fallback_before_the_s3_check_batches_are_used_up(pg):
    token = postprocessing(pg)
    assert pg["state"].claim_s3_check_attempt(JOB, token).applied  # 1묶음만 썼다
    change = to_fallback(pg, token)
    assert not change.applied
    assert row(pg).status is AsyncJobStatus.POSTPROCESSING


def test_fallback_after_both_s3_batches_failed(pg):
    token = postprocessing(pg)
    s3_checks_used_up(pg, token)
    change = to_fallback(pg, token)
    assert change.applied
    assert (change.job.status, change.job.next_action) == (
        AsyncJobStatus.FALLBACK_RUNNING,
        NextAction.RUN_FALLBACK,
    )
    assert pg["state"].claim_fallback_attempt(JOB, token).applied
    assert pg["state"].claim_fallback_attempt(JOB, token).reason is ChangeReason.LIMIT_REACHED


def test_confirmed_recoverable_failure_event_moves_to_fallback_once(pg):
    request_token = submitted(pg)
    assert (
        pg["state"].release_lease(JOB, request_token).applied
    )  # 요청 Worker 는 접수 처리 후 반납했다
    assert pg["state"].record_pruna_failed(TASK, recoverable=True).applied
    again = pg["state"].record_pruna_failed(TASK, recoverable=True)
    assert (again.applied, again.reason) == (False, ChangeReason.STATE_MISMATCH)
    token = lease(pg, NextAction.RUN_FALLBACK)
    assert pg["state"].claim_fallback_attempt(JOB, token).applied
    assert pg["state"].claim_fallback_attempt(JOB, token).reason is ChangeReason.LIMIT_REACHED


def test_no_fallback_once_a_result_exists(pg):
    token = postprocessing(pg)
    s3_checks_used_up(pg, token)
    assert pg["state"].save_final_result(JOB, token, FAILED_PAYLOAD, None).applied
    change = to_fallback(pg, token)
    assert (change.applied, change.reason) == (False, ChangeReason.STATE_MISMATCH)
    submitted_failure = pg["state"].record_pruna_failed(TASK, recoverable=True)
    assert not submitted_failure.applied
    assert row(pg).status is AsyncJobStatus.POSTPROCESSING


# --- 저장된 결과와 발행 완료 상태 ---


def _ready_to_publish(pg, payload, attempts=1):
    token = postprocessing(pg)
    image_key = payload.get("result_image_key")
    assert pg["state"].save_final_result(JOB, token, payload, image_key).applied
    for _ in range(attempts):
        assert pg["state"].claim_result_message_attempt(JOB, token).applied
    return token


@pytest.mark.parametrize(
    ("payload", "wrong_status"),
    [(SUCCESS_PAYLOAD, AsyncJobStatus.FAILED), (FAILED_PAYLOAD, AsyncJobStatus.COMPLETED)],
    ids=["completed-payload-as-failed", "failed-payload-as-completed"],
)
def test_publish_status_must_match_the_saved_result(pg, payload, wrong_status):
    token = _ready_to_publish(pg, payload)
    before = row(pg)

    change = pg["state"].mark_result_published(JOB, token, wrong_status)

    assert not change.applied
    after = row(pg)
    assert (after.status, after.result_message_status, after.next_action) == (
        before.status,
        before.result_message_status,
        before.next_action,
    )
    assert after.result_payload == payload
    assert after.worker_lease_token == token


@pytest.mark.parametrize(
    ("payload", "final_status"),
    [(SUCCESS_PAYLOAD, AsyncJobStatus.COMPLETED), (FAILED_PAYLOAD, AsyncJobStatus.FAILED)],
)
@pytest.mark.parametrize("attempts", [1, RESULT_MESSAGE_LIMIT])
def test_matching_publish_status_is_recorded_even_on_the_last_attempt(
    pg, payload, final_status, attempts
):
    token = _ready_to_publish(pg, payload, attempts)
    change = pg["state"].mark_result_published(JOB, token, final_status)
    assert change.applied
    assert (change.job.status, change.job.result_message_status, change.job.next_action) == (
        final_status,
        ResultMessageStatus.PUBLISHED,
        None,
    )
    assert change.job.worker_lease_token is None and change.job.result_payload == payload


# --- fallback 호출부 계약 ---


@pytest.mark.parametrize(
    "final_check",
    [
        S3ObjectState.ACCESS_DENIED,
        S3ObjectState.ERROR,
        S3ObjectState.TRANSIENT_ERROR,
        S3ObjectState.EXISTS,
    ],
)
def test_s3_permission_config_or_transient_errors_never_fall_back(pg, final_check):
    token = postprocessing(pg)
    s3_checks_used_up(pg, token)  # 횟수를 다 썼어도
    before = row(pg)
    with pytest.raises(ValueError):
        pg["state"].move_to_fallback(JOB, token, final_s3_check=final_check)
    assert row(pg) == before


def test_recoverable_failure_cannot_carry_a_failure_result(pg):
    submitted(pg)
    with pytest.raises(ValueError):
        pg["state"].record_pruna_failed(TASK, recoverable=True, failure_payload=FAILED_PAYLOAD)
    assert row(pg).status is AsyncJobStatus.RECEIVED


# --- V2 결과 형식 ---


@pytest.mark.parametrize(
    ("payload", "image_key"),
    [
        ({**SUCCESS_PAYLOAD, "fitting_job_id": JOB + 1}, ASYNC_KEY),  # 다른 작업 번호
        (SUCCESS_PAYLOAD, "virtual-fitting/results/3/pruna-sync.jpg"),  # 별도 Key 필드와 불일치
        (SUCCESS_PAYLOAD, None),
        (FAILED_PAYLOAD, ASYNC_KEY),  # 실패 결과에 이미지 위치
        ({**FAILED_PAYLOAD, "result_image_key": ASYNC_KEY}, None),
        ({**FAILED_PAYLOAD, "llm_title": "제목"}, None),
        ({k: v for k, v in SUCCESS_PAYLOAD.items() if k != "llm_comment"}, ASYNC_KEY),
        ({k: v for k, v in SUCCESS_PAYLOAD.items() if k != "status"}, ASYNC_KEY),
        ({**SUCCESS_PAYLOAD, "status": "ACCEPTED"}, ASYNC_KEY),
        ({**SUCCESS_PAYLOAD, "fitting_job_id": "3"}, ASYNC_KEY),
        ({**SUCCESS_PAYLOAD, "llm_title": ""}, ASYNC_KEY),
        (
            {"code": "FITTING_SUCCESS", "data": {"status": "COMPLETED"}, "message": "예전 형식"},
            ASYNC_KEY,
        ),
    ],
    ids=[
        "wrong-job-id",
        "key-mismatch",
        "success-without-key",
        "failure-with-key-arg",
        "failure-with-image-field",
        "failure-with-title",
        "success-missing-comment",
        "missing-status",
        "invalid-status",
        "job-id-not-int",
        "empty-title",
        "old-code-data-message",
    ],
)
def test_invalid_v2_result_is_rejected_before_the_db(pg, payload, image_key):
    token = postprocessing(pg)
    before = row(pg)
    with pytest.raises((TypeError, ValueError)):
        pg["state"].save_final_result(JOB, token, payload, image_key)
    assert row(pg) == before


@pytest.mark.parametrize("payload", [SUCCESS_PAYLOAD, FAILED_PAYLOAD], ids=["success", "failure"])
def test_v2_result_round_trips_unchanged(pg, payload):
    token = postprocessing(pg)
    image_key = payload.get("result_image_key")
    saved = pg["state"].save_final_result(JOB, token, payload, image_key)
    assert saved.applied and saved.job.result_payload == payload
    stored = row(pg)
    assert stored.result_payload == payload and stored.result_image_key == image_key
    stored_json = pg["sql"](
        "SELECT result_payload ->> 'status', result_payload ? 'error_code' "
        "FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s",
        (JOB,),
    )
    assert stored_json == [(payload["status"], payload["status"] == "FAILED")]


def test_failure_event_payload_must_belong_to_the_job(pg):
    submitted(pg)
    change = pg["state"].record_pruna_failed(
        TASK, recoverable=False, failure_payload={**FAILED_PAYLOAD, "fitting_job_id": JOB + 1}
    )
    assert (change.applied, change.reason) == (False, ChangeReason.STATE_MISMATCH)
    assert row(pg).result_payload is None


def test_stored_result_without_a_valid_status_is_never_published(pg):
    token = _ready_to_publish(pg, SUCCESS_PAYLOAD)
    # 형식 검증을 거치지 않고 들어온(예: 손상된) 결과를 가정한다.
    pg["sql"](
        "UPDATE virtual_fitting_async_jobs SET result_payload = result_payload - 'status' "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    before = row(pg)
    for final_status in (AsyncJobStatus.COMPLETED, AsyncJobStatus.FAILED):
        change = pg["state"].mark_result_published(JOB, token, final_status)
        assert not change.applied
    assert row(pg) == before
