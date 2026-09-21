"""AI Server 진입점.

컨테이너는 `uvicorn app.main:app` 으로 이 파일을 부른다 (Dockerfile CMD).
로컬에서는 `python -m app.main` 으로도 뜬다.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.chat import router as chat_router
from app.chat.graph import build_graph
from app.config import settings
from app.config.checkpointer import checkpointer_scope
from app.errors import register_error_handlers
from app.virtual_fitting import router as virtual_fitting_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with checkpointer_scope() as checkpointer:
        app.state.checkpointer = checkpointer
        app.state.graph = build_graph(checkpointer)
        yield


app = FastAPI(title="KTB4-15th AI Server", lifespan=lifespan)

register_error_handlers(app)
app.include_router(chat_router.router)
app.include_router(virtual_fitting_router.router)
# 도메인이 늘어나면 여기에 한 줄씩 추가한다


def run() -> None:
    """로컬 실행용. 컨테이너에서는 Dockerfile 의 CMD 가 uvicorn 을 직접 부른다."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )


if __name__ == "__main__":
    run()
