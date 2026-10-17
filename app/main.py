"""AI Server 진입점.

컨테이너는 `uvicorn app.main:app` 으로 이 파일을 부른다 (Dockerfile CMD).
로컬에서는 `python -m app.main` 으로도 뜬다.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from anyio import to_thread
from fastapi import FastAPI

from app import health as health_router
from app.body_image_validation import router as body_image_validation_router
from app.body_image_validation.runtime import runtime_manager
from app.chat import router as chat_router
from app.chat.graph import build_graph
from app.config import settings
from app.config.checkpointer import checkpointer_scope
from app.config.database import close_async_pool, close_connection_pool, open_async_pool
from app.config.logging_config import LOGGING_CONFIG, configure_logging
from app.errors import register_error_handlers
from app.virtual_fitting import router as virtual_fitting_router
from app.virtual_fitting import webhook_router as virtual_fitting_webhook_router
from app.virtual_fitting.concurrency import create_fitting_limiters
from app.virtual_fitting.postprocess_worker import (
    connected_postprocess_handlers,
    create_postprocess_worker,
)
from app.virtual_fitting.providers.runware import (
    get_runware_llm_api_key,
    get_runware_vton_api_key,
)
from app.virtual_fitting.request_worker import create_request_worker

# 컨테이너 CMD(`uvicorn app.main:app`)는 log_config 를 넘기지 않는다. uvicorn 은 이 모듈을
# import 한 뒤에 "Started server process" 를 찍으므로, 여기서 적용하면 서버 로그부터 포맷이 맞는다.
configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 가상피팅이 쓰는 Runware 키다. 빠진 채로 뜨면 첫 요청에서야 500 이 나므로 여기서 멈춘다.
    get_runware_vton_api_key()
    get_runware_llm_api_key()
    # 가상피팅 VTON / LLM 동시 실행 제한. 앱마다 새로 만들어 요청이 앱 상태로 꺼내 쓴다.
    app.state.fitting_limiters = create_fitting_limiters()
    runtime_manager.start()
    # V2 요청 Worker 는 켰을 때만 이 프로세스와 함께 돈다(기본 꺼짐: SQS·AWS 를 보지 않는다).
    app.state.request_worker = None
    # V2 후처리 Worker 도 켰을 때만 돈다. 핸들러(11·13단계)가 없으면 시작하지 않고 이유를 status 에 남긴다.
    app.state.postprocess_worker = None
    app.state.postprocess_worker_status = "disabled"
    try:
        await open_async_pool()
        if settings.VIRTUAL_FITTING_REQUEST_WORKER_ENABLED:
            # 조립은 blocking(DB pool·boto3)이라 thread 에서 한다.
            app.state.request_worker = await to_thread.run_sync(create_request_worker)
            app.state.request_worker.start()
        if settings.VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED:
            handlers = connected_postprocess_handlers()
            missing = handlers.missing()
            if missing:
                app.state.postprocess_worker_status = f"not_ready:{','.join(missing)}"
                logger.warning(
                    "postprocess worker not started: handlers not connected (%s)", ",".join(missing)
                )
            else:
                app.state.postprocess_worker = await to_thread.run_sync(
                    create_postprocess_worker, handlers
                )
                app.state.postprocess_worker.start()
                app.state.postprocess_worker_status = "running"
        async with checkpointer_scope() as checkpointer:
            app.state.checkpointer = checkpointer
            app.state.graph = build_graph(checkpointer)
            yield
    finally:
        # 두 Worker 가 함께 쓰는 DB pool 을 닫기 전에 두 Worker 를 모두 멈춘다(동시에 기다린다). 어느 Worker 도
        # 공유 자원(DB pool·boto3 client)을 스스로 닫지 않는다.
        stops = [
            worker.stop(timeout)
            for worker, timeout in (
                (
                    app.state.request_worker,
                    settings.VIRTUAL_FITTING_REQUEST_SHUTDOWN_TIMEOUT_SECONDS,
                ),
                (
                    app.state.postprocess_worker,
                    settings.VIRTUAL_FITTING_POSTPROCESS_SHUTDOWN_TIMEOUT_SECONDS,
                ),
            )
            if worker is not None
        ]
        if stops:
            for result in await asyncio.gather(*stops, return_exceptions=True):
                if isinstance(result, BaseException):
                    logger.error("worker stop failed", exc_info=result)
        runtime_manager.close()
        close_connection_pool()
        await close_async_pool()


app = FastAPI(title="KTB4-15th AI Server", lifespan=lifespan)

register_error_handlers(app)
app.include_router(health_router.router)
app.include_router(chat_router.router)
app.include_router(virtual_fitting_router.router)
# Runware Webhook 은 내부 API 키가 아니라 URL token 으로 인증한다(별도 Router).
app.include_router(virtual_fitting_webhook_router.router)
app.include_router(body_image_validation_router.router)
# 도메인이 늘어나면 여기에 한 줄씩 추가한다


def run() -> None:
    """로컬 실행용. 컨테이너에서는 Dockerfile 의 CMD 가 uvicorn 을 직접 부른다."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_config=LOGGING_CONFIG,
    )


if __name__ == "__main__":
    run()
