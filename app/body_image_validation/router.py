"""POST /api/v1/validation/body-image."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.body_image_validation.exceptions import SYSTEM_FAILURE_MESSAGE, BodyImageValidationError
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
    },
)


def get_service() -> BodyImageValidationService:
    return runtime_manager.get_service()


ImageFile = Annotated[UploadFile, File()]
@router.post("/api/v1/validation/body-image", response_model=BodyImageValidationResponse)
def validate_body_image(image: ImageFile):
    try:
        service = get_service()
        body = image.file.read(config.MAX_IMAGE_BYTES + 1)
        result = service.validate(body)
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
