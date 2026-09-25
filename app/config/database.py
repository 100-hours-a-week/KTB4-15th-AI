import os
import threading
from typing import Any

from app.config import settings


class DatabaseConfigError(RuntimeError):
    """DATABASE_URL이 설정되지 않았거나 형식이 올바르지 않을 때 발생한다."""


_pool: Any | None = None
_pool_lock = threading.Lock()


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url or not url.strip():
        raise DatabaseConfigError(
            "DATABASE_URL 환경변수가 설정되지 않았습니다. "
            "예: postgresql://USER:PASSWORD@localhost:5432/DB_NAME"
        )
    url = url.strip()
    if not url.startswith(("postgresql://", "postgres://")):
        raise DatabaseConfigError(
            "DATABASE_URL은 postgresql:// 또는 postgres:// 스킴이어야 합니다: "
            f"{url!r}"
        )
    return url


def get_connection_pool() -> Any:
    """프로세스에서 하나의 psycopg connection pool을 생성해 재사용한다."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            from psycopg_pool import ConnectionPool

            _pool = ConnectionPool(
                conninfo=get_database_url(),
                min_size=settings.DATABASE_POOL_MIN_SIZE,
                max_size=settings.DATABASE_POOL_MAX_SIZE,
                open=True,
            )
    return _pool


def close_connection_pool() -> None:
    """애플리케이션 종료 때 pool의 모든 connection을 닫는다."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
