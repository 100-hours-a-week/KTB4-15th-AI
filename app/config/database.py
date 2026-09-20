import os
from typing import Any

class DatabaseConfigError(RuntimeError):
    """DATABASE_URL이 설정되지 않았거나 형식이 올바르지 않을 때 발생한다."""

def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url or not url.strip():
        raise DatabaseConfigError(
            "DATABASE_URL 환경변수가 설정되지 않았습니다. "
            "예: postgresql://USER:PASSWORD@localhost:5432/DB_NAME"
        )
    url = url.strip()
    if not (url.startswith("postgresql://") or url.startswith("postgres://")):
        raise DatabaseConfigError(
            "DATABASE_URL은 postgresql:// 또는 postgres:// 스킴이어야 합니다: "
            f"{url!r}"
        )
    return url

def get_connection() -> Any:
    import psycopg
    return psycopg.connect(get_database_url())
