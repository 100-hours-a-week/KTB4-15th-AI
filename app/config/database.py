import logging
import os
import threading
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


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


# 검색 기반 추천 전용 async pool. 검색은 async 경로(SSE)에서 돌기 때문에 위의 동기 pool 을
# 쓰면 DB 를 기다리는 동안 이벤트 루프가 멈춘다. async pool 은 이벤트 루프가 돌고 있어야
# 열 수 있으므로 처음 쓸 때 만들지 않고 lifespan 에서 연다.
_async_pool: Any | None = None


async def open_async_pool() -> None:
    """lifespan 시작 때 한 번 연다. 연결은 뒤에서 맺고 서버 시작을 기다리게 하지 않는다.

    DATABASE_URL 이 없으면 열지 않고 넘어간다. 동기 pool 과 마찬가지로 서버는 뜨고,
    DB 를 쓰는 요청에서만 실패한다. 형식이 틀린 주소는 설정 실수이므로 그대로 예외를 낸다.
    """
    global _async_pool
    if not os.getenv("DATABASE_URL", "").strip():
        logger.warning("DATABASE_URL 이 없어 검색용 DB pool 을 열지 않는다. 검색 요청은 실패한다.")
        return
    from psycopg_pool import AsyncConnectionPool

    _async_pool = AsyncConnectionPool(
        conninfo=get_database_url(),
        min_size=settings.SEARCH_DB_POOL_MIN_SIZE,
        max_size=settings.SEARCH_DB_POOL_MAX_SIZE,
        open=False,
    )
    await _async_pool.open(wait=False)


# sabu: 열리지 않은 pool — lifespan 을 거치지 않고 검색을 부르면(스크립트, 테스트) 이 함수는
#       무엇을 돌려주고, search_products 의 except 는 그걸 잡는가? 사용자는 무엇을 보게 되나?
def get_async_pool() -> Any:
    return _async_pool


async def close_async_pool() -> None:
    """lifespan 종료 때 닫는다."""
    global _async_pool
    if _async_pool is not None:
        await _async_pool.close()
        _async_pool = None
