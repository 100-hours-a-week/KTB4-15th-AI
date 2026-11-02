"""POST /api/v1/validation/body-image."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.body_image_validation import deadline
from app.body_image_validation.diagnostics import REQUEST_ID_HEADER, parse_request_id
from app.body_image_validation.exceptions import SYSTEM_FAILURE_MESSAGE, BodyImageValidationError
from app.body_image_validation.remote import RemoteBodyImageValidationService
from app.body_image_validation.runtime import BodyImageRuntimeError, runtime_manager
from app.body_image_validation.schemas import (
    BodyImageValidationData,
    BodyImageValidationResponse,
)
from app.body_image_validation.service import BodyImageValidationService
from app.clients.s3 import S3ConfigError
from app.config import body_image_validation as config
from app.errors import ApiResponse, error_response
from app.security import verify_internal_key

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["body-image-validation"],
    dependencies=[Depends(verify_internal_key)],
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ApiResponse},
        status.HTTP_401_UNAUTHORIZED: {"model": ApiResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ApiResponse},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ApiResponse},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ApiResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiResponse},
        # remote 모드(AI Server → Modal worker)에서만: BODY_IMAGE_WORKER_UNAVAILABLE(502/503),
        # BODY_IMAGE_WORKER_TIMEOUT(504).
        status.HTTP_502_BAD_GATEWAY: {"model": ApiResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiResponse},
        status.HTTP_504_GATEWAY_TIMEOUT: {"model": ApiResponse},
    },
)


def get_service() -> BodyImageValidationService:
    return runtime_manager.get_service()


ImageFile = Annotated[UploadFile, File()]
@router.post("/api/v1/validation/body-image", response_model=BodyImageValidationResponse)
async def validate_body_image(request: Request, image: ImageFile):
    # event loop 에서 돈다. remote 는 그대로 await 하고, 동기 작업(local runtime 시작, 모델 추론, S3 업로드)은
    # threadpool 로 보낸다. run_in_threadpool 은 ContextVar(worker 의 deadline)를 그 thread 로 전한다.
    # limit 은 ASGI 진입에서 시작한 전체 제한시간이다. local 모드와 Modal worker 에는 없다.
    limit = request.scope.get(deadline.SCOPE_KEY)
    try:
        if limit is not None:
            limit.enter("service_lookup")
        service = await run_in_threadpool(get_service)
        if limit is not None:
            limit.enter("image_read")
        body = await image.read(config.MAX_IMAGE_BYTES + 1)
        if isinstance(service, RemoteBodyImageValidationService):
            result = await service.avalidate(body, limit)
        else:
            ai_request_id = parse_request_id(request.headers.get(REQUEST_ID_HEADER))
            result = await run_in_threadpool(service.validate, body, ai_request_id)
        return BodyImageValidationResponse(data=BodyImageValidationData(s3_key=result.s3_key))
    except BodyImageValidationError as error:
        if error.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            logger.exception("body image validation failed: %s", error.code)
        elif error.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            logger.warning("body image validation rejected: %s", error.code)
        return error_response(
            error.status_code, error.code, error.message or SYSTEM_FAILURE_MESSAGE
        )
    except S3ConfigError:
        logger.exception("body image validation S3 configuration error")
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            S3ConfigError.code,
            SYSTEM_FAILURE_MESSAGE,
        )
    except BodyImageRuntimeError:
        logger.exception("body image validation runtime unavailable")
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "BODY_IMAGE_RUNTIME_UNAVAILABLE",
            SYSTEM_FAILURE_MESSAGE,
        )
