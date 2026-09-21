"""대화 상태 저장소.

대화 내역은 Backend 와 AI 양쪽이 보관하며(2026-09-17 결정), AI 쪽 보관은 LangGraph
checkpointer 가 맡는다. 저장소 연결과 그래프 인스턴스를 만드는 자리를 여기로 모은다.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Request
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings


@asynccontextmanager
async def checkpointer_scope() -> AsyncGenerator[BaseCheckpointSaver, None]:
    """요청을 받는 동안 열어두는 checkpointer.

    DSN 이 없으면 인메모리로 뜬다. 로컬 개발용이며, 프로세스가 죽으면 대화 상태가
    사라지고 워커가 여러 개면 서로 공유되지 않는다.
    """
    if not settings.CHECKPOINT_DSN:
        yield InMemorySaver()
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(settings.CHECKPOINT_DSN) as checkpointer:
        # checkpoint 테이블을 만든다. 여러 번 실행해도 안전하지만, 운영에서는
        # 배포 파이프라인의 마이그레이션 단계로 옮기는 편이 낫다.
        await checkpointer.setup()
        yield checkpointer


def get_graph(request: Request):
    """컴파일된 그래프. 앱이 뜰 때 한 번 만들어 재사용한다."""
    return request.app.state.graph


def get_checkpointer(request: Request) -> BaseCheckpointSaver:
    return request.app.state.checkpointer


__all__ = ["checkpointer_scope", "get_checkpointer", "get_graph"]
