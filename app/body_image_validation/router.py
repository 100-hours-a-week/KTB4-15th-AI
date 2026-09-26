"""POST /api/v1/validation/body-image."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.body_image_validation.exceptions import BodyImageValidationError
from app.body_image_validation.runtime import BodyImageRuntimeError, runtime_manager
from app.body_image_validation.schemas import (
    BodyImageValidationData,
    BodyImageValidationResponse,
)
from app.body_image_validation.service import BodyImageValidationService
from app.clients.s3 import S3ConfigError
from app.config import body_image_validation as config
from app.errors import ErrorResponse, error_response
from app.security import verify_internal_key

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["body-image-validation"],
    dependencies=[Depends(verify_internal_key)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
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
        return BodyImageValidationResponse(
            data=BodyImageValidationData(s3_key=result.s3_key, warnings=result.warnings)
        )
    except BodyImageValidationError as error:
        if error.status_code >= 500:
            logger.exception("body image validation failed: %s", error.reason_code)
        data = {"reason_code": error.reason_code}
        if error.reason is not None:
            data["reason"] = error.reason
        return error_response(error.status_code, error.message, data)
    except S3ConfigError:
        logger.exception("body image validation S3 configuration error")
        return error_response(
            500,
            "body_image_validation_system_failed",
            {"reason_code": S3ConfigError.reason_code},
        )
    except BodyImageRuntimeError:
        logger.exception("body image validation runtime unavailable")
        return error_response(
            500,
            "body_image_validation_system_failed",
            {"reason_code": "BODY_IMAGE_RUNTIME_UNAVAILABLE"},
        )
