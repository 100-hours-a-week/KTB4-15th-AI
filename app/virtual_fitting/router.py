"""V1 동기 가상피팅 (단계1 §V1 동기 가상 피팅 요청).

Backend 는 user_image_url 과 product_code 만 보낸다. 카테고리와 이미지 URL 은 AI DB 에서
조회한다.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from fastapi import APIRouter, Depends

from app.clients.s3 import S3ConfigError, S3ImageStorage
from app.config.database import DatabaseConfigError, get_connection_pool
from app.errors import ErrorResponse, error_response
from app.security import verify_internal_key
from app.virtual_fitting import controller
from app.virtual_fitting.exceptions import (
    FittingDatabaseError,
    FittingImageStorageError,
    VirtualFittingError,
)
from app.virtual_fitting.providers.comment import MockCommentProvider
from app.virtual_fitting.providers.runware import RunwarePrunaProvider
from app.virtual_fitting.repositories.product_repository import ProductRepository
from app.virtual_fitting.schemas import SyncFittingRequest, SyncFittingResponse
from app.virtual_fitting.service import VirtualFittingService

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["virtual-fitting"],
    dependencies=[Depends(verify_internal_key)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
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
        comment_provider=MockCommentProvider(),
        image_storage=S3ImageStorage(),
    )


# Service 가 동기 HTTP(urllib)와 동기 DB(psycopg)를 쓰므로 async def 가 아니라 def 로 둔다.
# FastAPI 가 threadpool 에서 실행해 최대 60초 걸리는 호출이 이벤트 루프(chat SSE)를 막지 않는다.
@router.post("/api/v1/sync-fitting", response_model=SyncFittingResponse)
def sync_fitting(request: SyncFittingRequest):
    try:
        with open_virtual_fitting_service() as service:
            return controller.sync_fit(service, request)
    except VirtualFittingError as error:
        if error.status_code >= 500:
            logger.exception("sync-fitting failed: %s", error.message)
        return error_response(error.status_code, error.message)
    except S3ConfigError:
        # S3ImageStorage() 는 Service 조립 단계에서 만들어지므로 외부 가상피팅 API 를 부르기 전에
        # 여기서 멈춘다. 일반 internal_server_error 와 구분하도록 reason_code 를 싣는다.
        logger.exception("sync-fitting S3 configuration error")
        return error_response(
            FittingImageStorageError.status_code,
            FittingImageStorageError.message,
            {"reason_code": S3ConfigError.reason_code},
        )
