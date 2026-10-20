"""가상피팅 V2 단계별 진단 로그(vf_diag). 실제 formatter 로 렌더링한 출력으로 확인한다.

외부 경계는 tests/v2_integration_harness 의 가짜(SQS·S3·Runware), DB 는 격리 PostgreSQL schema 다. 네트워크는 막는다.
"""

import asyncio
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from types import SimpleNamespace

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.clients.sqs import FittingQueue, SqsOperation, SqsOutcome, SqsResult
from app.config import settings
from app.config.logging_config import LOG_FORMAT, KstDefaultFormatter

# app.main 을 먼저 import 해 운영과 같은 logging 설정(dictConfig)을 적용한 뒤 출력을 잡는다.
from app.main import app as fastapi_app
from app.virtual_fitting import diagnostics as diag
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.repositories.async_job_repository import AsyncJobRepository
from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository
from app.virtual_fitting.request_worker import RequestOutcome, RequestWorker
from tests.v2_integration_harness import USER_IMAGE_KEY, Stack, close_env, make_env

JOB = 9701


@pytest.fixture
def rendered():
    """운영과 같은 default formatter(KstDefaultFormatter + LOG_FORMAT)로 렌더링한 app.* 로그."""
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(KstDefaultFormatter(fmt=LOG_FORMAT, use_colors=False))
    app_logger = logging.getLogger("app")
    previous = app_logger.level
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(handler)
    try:
        yield buffer
    finally:
        app_logger.removeHandler(handler)
        app_logger.setLevel(previous)


def events(buffer, name=None):
    lines = [line for line in buffer.getvalue().splitlines() if " vf_diag " in line]
    parsed = []
    for line in lines:
        fields = dict(re.findall(r"(\w+)=(\S+)", line.split(" vf_diag ", 1)[1]))
        parsed.append(fields | {"_line": line})
    return [e for e in parsed if name is None or e["event"] == name]


@pytest.fixture
def env(monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL 이 없다")
    admin = psycopg.connect(url, autocommit=True, connect_timeout=3)
    schema = f"vf_diag_{uuid.uuid4().hex[:10]}"
    admin.execute(f"CREATE SCHEMA {schema}")
    e = make_env(monkeypatch, url, schema)
    e.schema_name = schema
    monkeypatch.setattr(diag, "_db_target_logged", False)
    try:
        yield e
    finally:
        close_env(e)
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()


FLAGS = (
    "VIRTUAL_FITTING_REQUEST_WORKER_ENABLED",
    "VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED",
    "VIRTUAL_FITTING_RESULT_PUBLISHER_ENABLED",
    "VIRTUAL_FITTING_WATCHDOG_ENABLED",
)


# 1·2. 설정 해석과 실제 루프 시작을 구분한다


def test_all_disabled_still_logs_the_parsed_flags_and_makes_no_db_connection(
    env, rendered, monkeypatch
):
    opened = []
    monkeypatch.setattr(diag, "log_db_target", lambda *a, **k: opened.append(1))
    with TestClient(fastapi_app):
        pass
    (start,) = events(rendered, "app_start")
    assert {
        start[f"{n}_enabled"]
        for n in ("request_worker", "postprocess_worker", "result_publisher", "watchdog")
    } == {"false"}
    assert (
        start["image_commit"] == "unknown"
        and start["component"] == "app"
        and start["run_id"] == diag.RUN_ID
    )
    assert events(rendered, "component_started") == [] and opened == []  # 진단만을 위한 연결 없음


def test_enabled_flags_and_actual_loop_starts_are_separate_events(env, rendered, monkeypatch):
    for name in FLAGS:
        monkeypatch.setattr(settings, name, True)
    monkeypatch.setenv("APP_IMAGE_COMMIT", "abc1234\nfake=1")
    with TestClient(fastapi_app):
        pass
    (start,) = events(rendered, "app_start")
    assert start["request_worker_enabled"] == "true" and start["image_commit"] == "abc1234_fake_1"
    started = {e["component"] for e in events(rendered, "component_started")}
    assert started == {"request_worker", "postprocess_worker", "result_publisher", "watchdog"}
    stopped = {e["component"] for e in events(rendered, "component_stopped")}
    assert stopped == started
    (target,) = events(rendered, "db_target")
    assert target["database"] == "stage4_test" and target["db_user"] == "stage4"
    assert target["jobs_table_schema"] == env.schema_name == target["products_table_schema"]
    assert target["search_path_schemas"] == env.schema_name
    # 테스트 계정은 pg_control_system() 을 읽을 수 있다(권한 없는 경우는 db_target_without_fingerprint 테스트)
    assert re.fullmatch(r"[0-9a-f]{16}", target["server_fp"])
    assert target["server_fp_error"] == "none" and target["server_identity"] == "cluster_lineage"
    assert re.fullmatch(r"[0-9a-f]{16}", target["endpoint_fp"])


# 3·4·5·6. 요청 수신 → 검증 → 작업 저장 → 상품 조회 → 실패 저장


def test_normal_request_is_traceable_by_one_job_id(env, rendered):
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    stack.request_once()
    names = [e["event"] for e in events(rendered) if e.get("fitting_job_id") == str(JOB)]
    assert names[:5] == [
        "request_validated",
        "job_save_started",
        "job_saved",
        "product_lookup",
        "request_decision",
    ]
    (received,) = events(rendered, "request_received")
    (validated,) = events(rendered, "request_validated")
    assert received["message_id"] == validated["message_id"] and received["receive_count"] == "1"
    (lookup,) = events(rendered, "product_lookup")
    assert (
        lookup["result"] == "found"
        and lookup["requested"] == "123456,789012"
        and lookup["missing"] == "empty"
    )
    assert set(lookup["found"].split(",")) == {"123456", "789012"}


def test_missing_product_logs_requested_found_missing_and_the_saved_failure(env, rendered):
    stack = Stack(env).mark_running()
    env.send_request(JOB, codes=("123456", "999999"))
    stack.request_once()
    (lookup,) = events(rendered, "product_lookup")
    assert (lookup["result"], lookup["requested"], lookup["found"], lookup["missing"]) == (
        "missing",
        "123456,999999",
        "123456",
        "999999",
    )
    (saved,) = events(rendered, "failure_result_saved")
    assert saved["error_code"] == "PRODUCT_NOT_FOUND" and saved["change"] == "applied"
    assert events(rendered, "product_lookup_failed") == []
    (decision,) = events(rendered, "request_decision")
    assert decision["action"] == "failure_saved_PRODUCT_NOT_FOUND"


def test_db_error_on_products_is_not_logged_as_a_missing_product(env, rendered):
    env.sql("ALTER TABLE products RENAME TO products_unavailable")
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    stack.request_once()
    (failed,) = events(rendered, "product_lookup_failed")
    assert failed["result"] == "db_error" and failed["error_code"] == "DATABASE_ERROR"
    assert " ERROR " in failed["_line"]
    assert (
        events(rendered, "product_lookup") == [] and events(rendered, "failure_result_saved") == []
    )


def test_job_save_failure_never_logs_job_saved(env, rendered, monkeypatch):
    def broken(self, new_job):
        raise FittingDatabaseError("injected")

    monkeypatch.setattr(AsyncJobRepository, "create", broken)
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    stack.request_once()
    assert [e["event"] for e in events(rendered) if e["event"].startswith("job_")] == [
        "job_save_started",
        "job_save_failed",
    ]


def test_failure_save_error_never_logs_failure_saved(env, rendered, monkeypatch):
    def broken(self, *args, **kwargs):
        raise FittingDatabaseError("injected")

    monkeypatch.setattr(AsyncJobStateRepository, "save_final_result", broken)
    stack = Stack(env).mark_running()
    env.send_request(JOB, codes=("999999",))
    stack.request_once()
    (failed,) = events(rendered, "failure_result_save_failed")
    assert (
        failed["error_code"] == "PRODUCT_NOT_FOUND"
        and failed["error_type"] == "FittingDatabaseError"
    )
    assert events(rendered, "failure_result_saved") == []


# 7·8. 결과 발행: SQS MessageId 연결, 불명확·기록 실패 구분


def _saved_failure(env):
    stack = Stack(env).mark_running()
    env.send_request(JOB, codes=("999999",))
    stack.request_once()
    env.sql(
        "UPDATE virtual_fitting_async_jobs SET worker_lease_token = NULL, worker_lease_until = NULL"
    )
    return stack


def test_published_result_links_the_real_sdk_message_id(env, rendered):
    stack = _saved_failure(env)
    stack.publish_once()
    sdk_ids = [
        mid for op, queue, mid in env.sqs.log if op == "send" and queue == "be-fitting-response"
    ]
    (sent,) = events(rendered, "result_sent")
    assert sent["sqs_message_id"] == sdk_ids[0] and sent["fitting_job_id"] == str(JOB)
    assert sent["result_status"] == "FAILED" and sent["attempt"] == "1"
    (recorded,) = events(rendered, "result_publish_recorded")
    assert recorded["sqs_message_id"] == sdk_ids[0]


def test_unknown_send_is_not_logged_as_sent(env, rendered):
    stack = _saved_failure(env)
    env.sqs.fault("be-fitting-response", "send", "stored_timeout")
    stack.publish_once()
    (failed,) = events(rendered, "result_send_failed")
    assert failed["sqs_result"] == "unknown" and events(rendered, "result_sent") == []


def test_sent_but_not_recorded_is_separate(env, rendered, monkeypatch):
    stack = _saved_failure(env)

    def broken(self, *args, **kwargs):
        raise FittingDatabaseError("injected")

    monkeypatch.setattr(AsyncJobStateRepository, "mark_result_published", broken)
    with pytest.raises(FittingDatabaseError):
        stack.publish_once()
    (sent,) = events(rendered, "result_sent")
    (not_recorded,) = events(rendered, "result_sent_not_recorded")
    assert not_recorded["sqs_message_id"] == sent["sqs_message_id"]
    assert events(rendered, "result_publish_recorded") == []


# 9. 삭제 판단과 실제 DeleteMessage 결과


@pytest.mark.parametrize("decision", ["delete_ok", "delete_failed", "kept"])
def test_delete_decision_and_api_result_are_separate(rendered, decision):
    item = SimpleNamespace(message_id="m-1", receipt_handle="RECEIPT-SECRET-HANDLE")

    def receive():
        return SimpleNamespace(received=True, messages=(item,), receive_result=None)

    worker_box = {}

    def process(message):
        worker_box["w"]._stop.set()
        return RequestOutcome("m-1", JOB, "x_action", decision != "kept")

    def delete(receipt):
        outcome = SqsOutcome.SUCCEEDED if decision == "delete_ok" else SqsOutcome.FAILED
        return SqsResult(
            SqsOperation.DELETE, FittingQueue.BE_FITTING_REQUEST, outcome, "receipt_handle_invalid"
        )

    async def run():
        worker_box["w"] = RequestWorker(receive, process, delete)
        await asyncio.wait_for(worker_box["w"].start(), 5)

    asyncio.run(run())
    names = [e["event"] for e in events(rendered) if e.get("message_id") == "m-1"]
    expected = {
        "delete_ok": ["message_deleted"],
        "delete_failed": ["message_delete_failed"],
        "kept": ["message_kept"],
    }
    assert names == expected[decision]
    assert "RECEIPT-SECRET-HANDLE" not in rendered.getvalue()


# 10·11. 실제 formatter 출력·민감정보


def test_rendered_line_format_and_injection_safety(rendered):
    diag.event(
        "request_received",
        component="request_worker",
        message_id="a b\nevent=forged",
        receive_count=None,
    )
    (line,) = [line for line in rendered.getvalue().splitlines() if "vf_diag" in line]
    assert re.match(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+09:00 INFO app\.virtual_fitting\.diagnostics - vf_diag ",
        line,
    )
    assert "message_id=a_b_event_forged" in line and line.count("event=") == 1
    assert "receive_count=none" in line


SECRETS = ("SECRETPW123", "SIGSECRET987", "TOKENSECRET555", "AKIAABCDEFGHIJKLMNOP", "WEBHOOKTOK777")


def test_secrets_never_reach_the_final_output_even_in_tracebacks(rendered):
    log = logging.getLogger("app.virtual_fitting.request_worker")
    try:
        try:
            raise OSError(
                "connection to postgresql://lookddak:SECRETPW123@db.internal:5432/lookddak failed "
                "url=https://bucket.s3.amazonaws.com/users/1/body.png?X-Amz-Signature=SIGSECRET987 "
                "Authorization: Bearer TOKENSECRET555 password=SECRETPW123 key AKIAABCDEFGHIJKLMNOP "
                "webhook https://ai.example.test/api/v2/webhooks/runware?token=WEBHOOKTOK777"
            )
        except OSError as cause:
            raise FittingDatabaseError("AI PostgreSQL 연결에 실패했습니다.") from cause
    except FittingDatabaseError:
        log.exception("request worker failed fitting_job_id=%s", JOB)
    text = rendered.getvalue()
    assert (
        "Traceback" in text and "OSError" in text and "FittingDatabaseError" in text
    )  # 스택·예외 종류는 남는다
    for secret in SECRETS:
        assert secret not in text, secret


def test_real_job_save_db_error_does_not_leak_the_dsn(env, rendered, monkeypatch):
    def broken(self, new_job):
        try:
            raise psycopg.OperationalError(
                'connection to "postgresql://u:SECRETPW123@h:5432/d" failed'
            )
        except psycopg.Error as error:
            raise FittingDatabaseError("AI PostgreSQL 비동기 작업 저장에 실패했습니다.") from error

    monkeypatch.setattr(AsyncJobRepository, "create", broken)
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    stack.request_once()
    text = rendered.getvalue()
    assert "request worker failed" in text and "SECRETPW123" not in text
    assert USER_IMAGE_KEY not in text  # 사용자 사진 Key 를 남기지 않는다


# 12. 로그 추가로 업무 동작이 바뀌지 않음(대표 경로; 전체는 기존 회귀 테스트)


def test_logging_does_not_change_counts_or_deletion(env, rendered):
    stack = Stack(env).mark_running()
    env.send_request(JOB, codes=("999999",))
    (outcome,) = stack.request_once()
    assert (outcome.action, outcome.delete) == ("failure_saved_PRODUCT_NOT_FOUND", False)
    stack.publish_once()
    current = env.job(JOB)
    assert (
        current.result_message_attempt_count == 1
        and current.result_message_status.value == "PUBLISHED"
    )
    assert (
        env.runware.count("async_submit") == 0 and len(env.sqs.bodies("be-fitting-response")) == 1
    )


# 13. 지문(pg_control_system) 권한이 없을 때: unverified 로 남기고, 실패한 트랜잭션을 롤백해 같은 연결을 계속 쓴다
#
# 함수 실행 권한(PUBLIC)은 DB 단위라 스키마·DB 이름만 분리해서는 부족하다. 외부에서 받은 DB(TEST_DATABASE_URL 등)에는
# 권한 변경을 하지 않고, 이 테스트가 initdb 로 직접 만든 임시 클러스터에서만 실행한다. 준비할 수 없으면 skip 한다.


def _pg_bin(name):
    found = shutil.which(name)
    if found:
        return found
    pg_config = shutil.which("pg_config")
    if pg_config:
        bindir = subprocess.run(
            [pg_config, "--bindir"], capture_output=True, text=True, check=False
        ).stdout.strip()
        candidate = os.path.join(bindir, name)
        if bindir and os.access(candidate, os.X_OK):
            return candidate
    return None


@pytest.fixture
def private_cluster():
    """이 테스트 전용 임시 PostgreSQL 클러스터(TCP 없이 자기 소켓 디렉터리만). 성공·실패와 관계없이 멈추고 지운다."""
    initdb, pg_ctl = _pg_bin("initdb"), _pg_bin("pg_ctl")
    if not (initdb and pg_ctl):
        pytest.skip(
            "initdb·pg_ctl 이 없어 전용 임시 클러스터를 만들 수 없다(기존 DB 로 대체하지 않는다)"
        )
    root = tempfile.mkdtemp(prefix="vfpg_")
    data, sock, owner = os.path.join(root, "data"), os.path.join(root, "s"), "vf_owner"
    os.mkdir(sock)
    started = False
    try:
        made = subprocess.run(
            [initdb, "-D", data, "-U", owner, "-A", "trust", "-E", "UTF8", "--no-sync"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if made.returncode != 0:
            pytest.skip(f"initdb 실패로 전용 클러스터를 만들 수 없다: {made.stderr.strip()[:200]}")
        run = subprocess.run(
            [
                pg_ctl,
                "-D",
                data,
                "-l",
                os.path.join(root, "log"),
                "-w",
                "-t",
                "60",
                "start",
                "-o",
                f"-k {sock} -c listen_addresses='' -p 5432",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        started = run.returncode == 0
        if not started:
            pytest.skip(f"전용 클러스터 시작 실패: {run.stderr.strip()[:200]}")
        url = f"postgresql://{owner}@/postgres?host={sock}&port=5432"
        with psycopg.connect(url, autocommit=True) as check:
            (directory,) = check.execute("SHOW data_directory").fetchone()
        # 접속한 서버가 방금 만든 데이터 디렉터리의 서버인지 확인(다른 클러스터면 권한 변경 전에 멈춘다)
        assert os.path.realpath(directory) == os.path.realpath(data)
        yield url
    finally:
        if started:
            subprocess.run(
                [pg_ctl, "-D", data, "-m", "immediate", "-w", "stop"],
                capture_output=True,
                check=False,
                timeout=60,
            )
        shutil.rmtree(root, ignore_errors=True)


def test_db_target_without_fingerprint_is_unverified_and_rolled_back(
    private_cluster, rendered, monkeypatch
):
    from psycopg.pq import TransactionStatus
    from psycopg_pool import ConnectionPool

    from app.config import database

    url, schema, role = private_cluster, "vf_nofp", "vf_nofp_reader"
    admin = psycopg.connect(url, autocommit=True)
    env = pool = None
    try:
        admin.execute(f"CREATE SCHEMA {schema}")
        env = make_env(monkeypatch, url, schema)
        monkeypatch.setattr(diag, "_db_target_logged", False)
        admin.execute(f"CREATE ROLE {role} NOSUPERUSER")
        admin.execute(f"GRANT USAGE ON SCHEMA {schema} TO {role}")
        admin.execute(f"GRANT ALL ON ALL TABLES IN SCHEMA {schema} TO {role}")
        admin.execute(f"GRANT ALL ON ALL SEQUENCES IN SCHEMA {schema} TO {role}")
        admin.execute(
            "REVOKE EXECUTE ON FUNCTION pg_control_system() FROM PUBLIC"
        )  # 전용 클러스터에서만
        pool = ConnectionPool(
            url,
            min_size=1,
            max_size=1,  # 진단과 이후 업무가 같은 연결을 쓰게
            timeout=10,
            kwargs={"options": f"{env.options} -c role={role}"},  # 권한 검사는 이 계정으로
            open=True,
        )
        monkeypatch.setattr(database, "_pool", pool)
        with pool.connection() as connection:
            pid = connection.info.backend_pid
        rollbacks = []
        real_rollback = psycopg.Connection.rollback

        def spy(self):
            before = self.info.transaction_status
            real_rollback(self)
            rollbacks.append((before, self.info.transaction_status))

        monkeypatch.setattr(psycopg.Connection, "rollback", spy)

        diag.log_db_target(pool, "app")

        (target,) = events(rendered, "db_target")
        assert events(rendered, "db_target_failed") == []
        assert target["db_user"] == role and target["server_fp"] == "unavailable"
        assert target["server_fp_error"] == "InsufficientPrivilege"
        assert target["server_identity"] == "unverified"
        assert target["jobs_table_schema"] == schema  # 스키마는 남지만 같은 서버 판단 근거가 아니다
        # 권한 오류로 중단된 트랜잭션을 롤백했고, 정리 롤백 뒤 열린 트랜잭션이 없다
        assert rollbacks == [
            (TransactionStatus.INERROR, TransactionStatus.IDLE),
            (TransactionStatus.IDLE, TransactionStatus.IDLE),
        ]
        state = admin.execute(
            "SELECT state, xact_start IS NULL FROM pg_stat_activity WHERE pid = %s", (pid,)
        ).fetchone()
        assert state == ("idle", True)

        stack = Stack(env).mark_running()
        env.send_request(JOB)
        (outcome,) = stack.request_once()
        assert outcome.action == "submitted"
        (lookup,) = events(rendered, "product_lookup")
        assert lookup["result"] == "found" and len(events(rendered, "job_saved")) == 1
        assert env.job(JOB).task_uuid is not None
        with pool.connection() as connection:
            assert connection.info.backend_pid == pid  # 진단에 쓴 연결 그대로
    finally:
        if pool is not None:
            pool.close()
        if env is not None:
            close_env(env)
        admin.close()


# 14. 두 요청: request_validated 한 줄에 같은 요청의 message_id·fitting_job_id 가 함께 나오고 서로 섞이지 않는다


def test_two_requests_link_message_id_to_job_id_on_one_line(env, rendered):
    first, second = JOB, JOB + 1
    sent = {env.send_request(first): first, env.send_request(second): second}
    stack = Stack(env).mark_running()
    outcomes = stack.request_once() + stack.request_once()  # 한 번 수신에 한 건씩
    assert sorted(o.fitting_job_id for o in outcomes) == [first, second]
    received = events(rendered, "request_received")
    validated = events(rendered, "request_validated")
    assert {e["message_id"] for e in received} == set(sent)
    # 최종 출력 한 줄에 둘 다 있다(extra 가 아니라 렌더링된 문자열에서 읽음)
    assert {e["message_id"]: int(e["fitting_job_id"]) for e in validated} == sent
    for e in validated:
        assert f"message_id={e['message_id']}" in e["_line"]
        assert f"fitting_job_id={sent[e['message_id']]}" in e["_line"]
    for job_id in (first, second):
        saved = [e for e in events(rendered, "job_saved") if e["fitting_job_id"] == str(job_id)]
        (line,) = saved
        assert sent[line["message_id"]] == job_id  # job_saved 의 message_id 도 같은 요청
    text = rendered.getvalue()
    assert USER_IMAGE_KEY not in text and "receipt" not in text.lower()
    assert '"products"' not in text and "user_image_key" not in text  # 메시지 본문 없음
