"""V1 동기 가상피팅 (단계1 §V1 동기 가상 피팅 요청).

Backend 는 user_image_url 과 product_code 만 보낸다. 카테고리와 이미지 URL 은 AI DB 에서
조회한다.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from fastapi import APIRouter, Depends

from app.config.database import get_connection
from app.errors import ErrorResponse, error_response
from app.security import verify_internal_key
from app.virtual_fitting import controller
from app.virtual_fitting.exceptions import FittingDatabaseError, VirtualFittingError
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
    """요청 하나가 쓸 Service 를 조립하고, 끝나면 DB connection 을 닫는다.

    Depends 로 만들지 않는 이유: FastAPI 는 body 검증이 실패한 요청에서도 의존성을 먼저
    실행한다. 그러면 잘못된 요청마다 DB 연결을 열고, 설정이 빠진 서버는 400 대신 500 을 낸다.
    """
    try:
        connection = get_connection()
    except psycopg.Error as error:
        raise FittingDatabaseError("AI PostgreSQL 연결에 실패했습니다.") from error
    try:
        # SELECT 뒤에 트랜잭션을 연 채로 Runware 응답(최대 60초)을 기다리지 않게 한다.
        connection.autocommit = True
        yield VirtualFittingService(
            repository=ProductRepository(connection),
            fitting_provider=RunwarePrunaProvider(),
            comment_provider=MockCommentProvider(),
        )
    finally:
        connection.close()


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
