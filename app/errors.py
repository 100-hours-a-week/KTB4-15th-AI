"""공통 오류 응답 (단계1 §6).

모든 오류를 `{"code": ..., "message": ..., "data": ...}` 형식으로 맞춘다.
FastAPI 기본값은 `{"detail": ...}` 이고 스키마 위반을 422 로 내므로 그대로 두면
Backend 가 명세대로 읽지 못한다.
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """단계1 §6 공통 오류 응답 형식. 도메인과 무관하게 모든 API 가 이 형태로 낸다."""

    code: int
    message: str
    data: dict | None = None


def error_response(status_code: int, message: str, data: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(code=status_code, message=message, data=data).model_dump(),
    )


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """필수 필드 누락, 잘못된 타입, 허용되지 않은 enum 등 (단계1 §6).

    FastAPI 는 이것을 422 로 내지만 명세는 400 invalid_request 다.
    """
    return error_response(400, "invalid_request")


async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """라우터나 의존성이 던진 HTTPException.

    detail 에 담긴 문자열을 message 코드로 쓴다.
    """
    return error_response(exc.status_code, str(exc.detail))


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """예상하지 못한 내부 예외 (단계1 §6).

    원인이 분명한 오류는 각 Endpoint 의 특수 오류로 구분하고, 여기로는 정말
    예상하지 못한 것만 온다. 그래서 여기 로그가 쌓이면 그건 분류가 덜 된 것이다.
    """
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return error_response(500, "internal_server_error")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(HTTPException, handle_http_exception)
    app.add_exception_handler(Exception, handle_unexpected_error)
