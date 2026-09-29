"""V1 동기 가상피팅 (단계1 §V1 동기 가상 피팅 요청).

Backend 는 user_image_url 과 product_code 만 보낸다. 카테고리와 이미지 URL 은 AI DB 에서
조회한다.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta

import psycopg
from fastapi import APIRouter, Depends, status

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


# Service 가 동기 HTTP(urllib)와 동기 DB(psycopg)를 쓰므로 async def 가 아니라 def 로 둔다.
# FastAPI 가 threadpool 에서 실행해 최대 60초 걸리는 호출이 이벤트 루프(chat SSE)를 막지 않는다.
#
# sabu: 관측 — fitting_balance_exhausted(402) 는 아래 분기에서 로그가 남는가? message 를 나눈 목적이
#       "누가 대응해야 하는지 구분"이었다면, 잔액 소진을 사람이 알게 되는 경로는 어디인가?
@router.post("/api/v1/sync-fitting", response_model=SyncFittingResponse)
def sync_fitting(request: SyncFittingRequest):
    try:
        with open_virtual_fitting_service() as service:
            return controller.sync_fit(service, request)
    except VirtualFittingError as error:
        if error.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            logger.exception("sync-fitting failed: %s", error.code)
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
