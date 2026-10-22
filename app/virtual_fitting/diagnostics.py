import hashlib
import logging
import os
import re
import socket
import time
import uuid
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger("app.virtual_fitting.diagnostics")

RUN_ID = uuid.uuid4().hex[:12]
# 이미지 커밋은 배포가 넣어 줄 때만 안다(이미지에 Git 이 없다). 없으면 unknown.
COMMIT_ENV_NAMES = ("APP_IMAGE_COMMIT", "GIT_COMMIT_SHA", "IMAGE_TAG")

_UNSAFE = re.compile(r"[^A-Za-z0-9_.:,/@+\-]")
_MAX_VALUE = 120


def safe(value: Any, limit: int = _MAX_VALUE) -> str:
    """로그 값 한 개. 공백·줄바꿈·= 등은 _ 로 바꾸고 limit 자로 자른다."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = _UNSAFE.sub("_", str(value))
    return text[:limit] if text else "empty"


def codes(values: Iterable[Any]) -> str:
    """상품 코드 목록. 숫자 코드만 그대로, 그 밖은 invalid(입력 원문을 남기지 않음)."""
    items = [str(v) if str(v).isdigit() and len(str(v)) <= 20 else "invalid" for v in values]
    return safe(",".join(items) or "empty", 240)


def event(name: str, level: int = logging.INFO, **fields: Any) -> None:
    """진단 한 줄. 값 변환·출력이 실패해도 호출한 업무 처리로 예외를 올리지 않는다."""
    try:
        parts = [f"event={safe(name)}", f"run_id={RUN_ID}"]
        parts += [f"{key}={safe(value, 240)}" for key, value in fields.items()]
        logger.log(level, "vf_diag %s", " ".join(parts))
    except Exception:  # noqa: BLE001, S110 — 진단 실패가 업무 처리를 막지 않게(출력 자체가 실패한 경우라 남길 곳이 없다)
        pass


def elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def image_commit() -> str:
    for name in COMMIT_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return safe(value, 64)
    return "unknown"


def hostname() -> str:
    """이 프로세스의 호스트 이름(컨테이너면 task 마다 다름). 읽지 못하면 unknown."""
    try:
        return safe(socket.gethostname(), 64)
    except Exception:  # noqa: BLE001 — 진단 값만 포기
        return "unknown"


def process_fields() -> dict[str, Any]:
    """어느 프로세스인지(run_id 는 모든 줄에 이미 있다). 여러 task·컨테이너가 같은 큐를 읽을 때 구분용."""
    return {"hostname": hostname(), "pid": os.getpid()}


def app_start(flags: dict[str, bool]) -> None:
    """앱 시작: 이미지 커밋·프로세스 식별자·네 Worker 설정의 해석값(꺼져 있어도 남긴다)."""
    event(
        "app_start",
        component="app",
        image_commit=image_commit(),
        hostname=hostname(),
        pid=os.getpid(),
        **{f"{name}_enabled": value for name, value in flags.items()},
    )


# --- DB 접속 대상(프로세스당 한 번, 앱이 실제 쓰는 pool 의 연결로) ---

_db_target_logged = False


def log_db_target(pool: Any, component: str) -> None:
    """실제 연결에서 읽은 DB 이름·계정·스키마·테이블 위치. 실패해도 업무 처리를 멈추지 않는다(경고만).

    server_fp: sha256(pg_control_system().system_identifier | current_database())[:16].
        system_identifier 는 initdb 때 정해지는 클러스터 값이고, 물리 복제본(standby·read replica)과
        백업·스냅샷 복원본도 같은 값을 물려받는다.
        - 다르면: 다른 클러스터이거나 다른 DB 이름이다(같은 DB 가 아니다).
        - 같으면: 같은 클러스터 계보의 같은 이름 DB 다. primary·replica, 원본·복원본 중 어느 인스턴스인지는 모른다.
        - unavailable: 권한 없음 등으로 읽지 못함(server_fp_error 에 예외 종류). 이때 server_identity=unverified 이며,
          DB 이름·계정·스키마가 같아도 같은 서버라고 판단하지 않는다.
    endpoint_fp: sha256(connection.info.host | port | dbname)[:16]. 실제 연결의 libpq 접속 값이다(host 는 접속 설정에
        쓴 이름·IP, 소켓이면 디렉터리이며 DNS 해석 결과가 아니다. 계정·비밀번호는 넣지 않는다).
        - 같으면: 같은 접속 설정을 썼다. 같은 이름도 DNS 변경·failover·프록시로 다른 서버를 가리킬 수 있어
          같은 서버라는 근거가 아니다.
        - 다르면: 접속 설정이 다르다. 다른 이름(CNAME·프록시·IP)으로 같은 서버에 붙어도 달라지므로
          다른 서버라는 근거도 아니다.
        - 후보 host 를 알면 해시를 계산해 맞춰 볼 수 있다(값을 숨기는 용도가 아니라 설정 비교용).
    server_identity: cluster_lineage(server_fp 를 얻음) 또는 unverified(얻지 못함).
    """
    global _db_target_logged
    if _db_target_logged:
        return
    _db_target_logged = True
    try:
        with pool.connection() as connection:
            info = connection.info
            endpoint_fp = hashlib.sha256(
                f"{info.host}|{info.port}|{info.dbname}".encode()
            ).hexdigest()[:16]
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT current_database(), current_user, array_to_string(current_schemas(false), ','),
                           (SELECT n.nspname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                             WHERE c.oid = to_regclass('products')),
                           (SELECT n.nspname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                             WHERE c.oid = to_regclass('virtual_fitting_async_jobs'))
                    """
                )
                database, user, schemas, products_schema, jobs_schema = cursor.fetchone()
                try:
                    cursor.execute("SELECT system_identifier::text FROM pg_control_system()")
                    system_id = cursor.fetchone()[0]
                    server_fp = hashlib.sha256(f"{system_id}|{database}".encode()).hexdigest()[:16]
                    server_fp_error, server_identity = "none", "cluster_lineage"
                except Exception as error:  # noqa: BLE001 — 권한 없음 등, 진단 값만 포기
                    connection.rollback()
                    server_fp, server_fp_error = "unavailable", type(error).__name__
                    server_identity = "unverified"  # DB 이름·스키마만으로 같은 서버로 보지 않는다
            connection.rollback()  # 읽기만 했다. pool 로 돌려줄 때 트랜잭션을 남기지 않는다.
        event(
            "db_target",
            component=component,
            database=database,
            db_user=user,
            search_path_schemas=schemas,
            products_table_schema=products_schema or "not_found",
            jobs_table_schema=jobs_schema or "not_found",
            server_fp=server_fp,
            server_fp_error=server_fp_error,
            server_identity=server_identity,
            endpoint_fp=endpoint_fp,
        )
    except Exception as error:  # noqa: BLE001 — 진단 실패가 업무 처리를 막지 않게
        event(
            "db_target_failed",
            logging.WARNING,
            component=component,
            error_type=type(error).__name__,
        )
