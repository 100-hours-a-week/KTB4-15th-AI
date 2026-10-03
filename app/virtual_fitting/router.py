"""V1 동기 가상피팅 (단계1 §V1 동기 가상 피팅 요청).

Backend 는 user_image_url 과 product_code 만 보낸다. 카테고리와 이미지 URL 은 AI DB 에서
조회한다.
"""

import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from datetime import timedelta

import psycopg
from anyio import to_thread
from fastapi import APIRouter, Depends, Request, status

from app.clients.s3 import S3ConfigError, S3ImageStorage
from app.config import settings
from app.config.database import DatabaseConfigError, get_connection_pool
from app.errors import ApiResponse, error_response
from app.security import verify_internal_key
from app.virtual_fitting import controller
from app.virtual_fitting.budget import FittingBudget
from app.virtual_fitting.exceptions import (
    FittingDatabaseError,
    FittingImageStorageError,
    VirtualFittingError,
)
from app.virtual_fitting.providers.runware import RunwarePrunaProvider
from app.virtual_fitting.providers.runware_comment import RunwareCommentProvider
from app.virtual_fitting.providers.runware_usage import RunwareUsageClient
from app.virtual_fitting.repositories.product_repository import ProductRepository
from app.virtual_fitting.repositories.usage_repository import UsageRepository
from app.virtual_fitting.schemas import SyncFittingRequest, SyncFittingResponse
from app.virtual_fitting.service import VirtualFittingService

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["virtual-fitting"],
    dependencies=[Depends(verify_internal_key)],
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ApiResponse},
        status.HTTP_401_UNAUTHORIZED: {"model": ApiResponse},
        status.HTTP_402_PAYMENT_REQUIRED: {"model": ApiResponse},
        status.HTTP_404_NOT_FOUND: {"model": ApiResponse},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ApiResponse},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ApiResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiResponse},
        status.HTTP_502_BAD_GATEWAY: {"model": ApiResponse},
        status.HTTP_504_GATEWAY_TIMEOUT: {"model": ApiResponse},
    },
)


@contextmanager
def open_virtual_fitting_service() -> Iterator[VirtualFittingService]:
    """요청 하나가 쓸 Service를 조립하고 DB connection을 pool에 즉시 반환한다.

    Depends 로 만들지 않는 이유: FastAPI 는 body 검증이 실패한 요청에서도 의존성을 먼저
    실행한다. 그러면 잘못된 요청마다 DB 연결을 열고, 설정이 빠진 서버는 400 대신 500 을 낸다.
    """
    try:
        connection_pool = get_connection_pool()
    except (DatabaseConfigError, psycopg.Error) as error:
        raise FittingDatabaseError("AI PostgreSQL 연결에 실패했습니다.") from error
    yield VirtualFittingService(
        repository=ProductRepository(connection_pool),
        fitting_provider=RunwarePrunaProvider(),
        comment_provider=RunwareCommentProvider(),
        image_storage=S3ImageStorage(),
        budget=FittingBudget(
            UsageRepository(connection_pool),
            RunwareUsageClient(),
            daily_limit=settings.FITTING_DAILY_BUDGET_USD,
            reconcile_interval=timedelta(seconds=settings.FITTING_USAGE_RECONCILE_SECONDS),
        ),
    )


@asynccontextmanager
async def _open_in_thread[T](context: AbstractContextManager[T]) -> AsyncIterator[T]:
    """동기 context manager 의 진입·종료를 worker thread 에서 한다.

    Service 조립은 DB pool 과 boto3 client 를 만들 수 있어 blocking 이다. 이벤트 루프에서 하지 않는다.
    """
    value = await to_thread.run_sync(context.__enter__)
    try:
        yield value
    except BaseException as error:
        if not await to_thread.run_sync(
            context.__exit__, type(error), error, error.__traceback__
        ):
            raise
    else:
        await to_thread.run_sync(context.__exit__, None, None, None)


# async def 다. 동기 단계(DB·urllib·boto3)는 Service 가 하나씩 worker thread 로 넘기고, VTON / LLM
# 슬롯 대기는 이벤트 루프에서 한다. 그래서 슬롯을 기다리는 요청은 worker thread 를 잡지 않는다.
# Backend 입장에서는 이전과 같은 동기 API 다 — 연결을 유지한 채 최종 결과를 같은 응답으로 받는다.
@router.post("/api/v1/sync-fitting", response_model=SyncFittingResponse)
async def sync_fitting(request: SyncFittingRequest, http_request: Request):
    try:
        async with _open_in_thread(open_virtual_fitting_service()) as service:
            return await controller.sync_fit(
                service, request, http_request.app.state.fitting_limiters
            )
    except VirtualFittingError as error:
        if error.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            logger.exception("sync-fitting failed: %s", error.code)
        elif error.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            logger.warning("sync-fitting rejected: %s", error.code)
        return error_response(error.status_code, error.code, error.message)
    except S3ConfigError:
        # S3ImageStorage() 는 Service 조립 단계에서 만들어지므로 외부 가상피팅 API 를 부르기 전에
        # 여기서 멈춘다. 일반 INTERNAL_SERVER_ERROR 와 구분하도록 전용 code 를 쓴다.
        logger.exception("sync-fitting S3 configuration error")
        return error_response(
            FittingImageStorageError.status_code,
            S3ConfigError.code,
            FittingImageStorageError.message,
        )
