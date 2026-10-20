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
from app.virtual_fitting import diagnostics as vf_diag
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
from app.virtual_fitting.result_publisher import (
    create_result_publisher,
    register_result_publisher,
)
from app.virtual_fitting.watchdog import create_watchdog, register_watchdog

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
    # V2 결과 전송 루프(14단계)도 켰을 때만 돈다. 실제로 시작한 루프만 등록해 다른 경로의 PUBLISH_RESULT 준비 판단에 쓴다.
    app.state.result_publisher = None
    # V2 Watchdog(15단계)도 켰을 때만 돈다. 켜져 있어야 할 Worker·결과 전송 루프가 멈추면 알린다.
    app.state.watchdog = None
    # 설정상 활성화 여부(꺼져 있어도 남긴다). 실제 루프 시작은 각 루프의 component_started 로 따로 남는다.
    flags = {
        "request_worker": settings.VIRTUAL_FITTING_REQUEST_WORKER_ENABLED,
        "postprocess_worker": settings.VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED,
        "result_publisher": settings.VIRTUAL_FITTING_RESULT_PUBLISHER_ENABLED,
        "watchdog": settings.VIRTUAL_FITTING_WATCHDOG_ENABLED,
    }
    vf_diag.app_start(flags)
    try:
        await open_async_pool()
        if settings.VIRTUAL_FITTING_RESULT_PUBLISHER_ENABLED:
            app.state.result_publisher = await to_thread.run_sync(create_result_publisher)
            app.state.result_publisher.start()
            register_result_publisher(app.state.result_publisher)
        if settings.VIRTUAL_FITTING_WATCHDOG_ENABLED:

            def components() -> dict:
                return {
                    "request_worker": app.state.request_worker,
                    "postprocess_worker": app.state.postprocess_worker,
                    "result_publisher": app.state.result_publisher,
                }

            app.state.watchdog = await to_thread.run_sync(create_watchdog, components)
            app.state.watchdog.start()
            register_watchdog(app.state.watchdog)
        if settings.VIRTUAL_FITTING_REQUEST_WORKER_ENABLED:
            # 조립은 blocking(DB pool·boto3)이라 thread 에서 한다.
            app.state.request_worker = await to_thread.run_sync(create_request_worker)
            app.state.request_worker.start()
        if settings.VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED:
            handlers = connected_postprocess_handlers()
            missing = handlers.missing()
            if missing:
                app.state.postprocess_worker_status = f"not_ready:{','.join(missing)}"
                vf_diag.event(
                    "component_not_started",
                    logging.WARNING,
                    component="postprocess_worker",
                    reason="handlers_not_connected",
                )
                logger.warning(
                    "postprocess worker not started: handlers not connected (%s)", ",".join(missing)
                )
            else:
                app.state.postprocess_worker = await to_thread.run_sync(
                    create_postprocess_worker, handlers
                )
                app.state.postprocess_worker.start()
                app.state.postprocess_worker_status = "running"
        if any(flags.values()):
            # 켜진 루프가 쓰는 DB pool 의 실제 연결로 한 번만 확인한다(모두 꺼져 있으면 새 연결을 만들지 않는다).
            await to_thread.run_sync(_log_fitting_db_target)
        async with checkpointer_scope() as checkpointer:
            app.state.checkpointer = checkpointer
            app.state.graph = build_graph(checkpointer)
            yield
    finally:
        # Worker 들과 결과 전송 루프가 함께 쓰는 DB pool 을 닫기 전에 모두 멈춘다(동시에 기다린다). 어느 쪽도
        # 공유 자원(DB pool·boto3 client)을 스스로 닫지 않는다. 결과 전송 등록을 먼저 풀어 새 준비 판단에 쓰지 않게 한다.
        register_result_publisher(None)
        register_watchdog(None)
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
                (
                    app.state.result_publisher,
                    settings.VIRTUAL_FITTING_RESULT_PUBLISHER_SHUTDOWN_TIMEOUT_SECONDS,
                ),
                (app.state.watchdog, settings.VIRTUAL_FITTING_WATCHDOG_SHUTDOWN_TIMEOUT_SECONDS),
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


def _log_fitting_db_target() -> None:
    """진단만. 실패해도 시작·업무 처리를 멈추지 않는다."""
    try:
        from app.config.database import get_connection_pool

        vf_diag.log_db_target(get_connection_pool(), "app")
    except Exception as error:  # noqa: BLE001 — 진단 실패가 앱 시작을 막지 않게
        vf_diag.event(
            "db_target_failed", logging.WARNING, component="app", error_type=type(error).__name__
        )


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
