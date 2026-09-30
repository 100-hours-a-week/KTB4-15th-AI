"""팀 공통 API 응답 형식.

성공/실패 모두 `{"code": str, "data": ..., "message": str}` 이다.
- HTTP status: 요청 처리의 큰 범주
- code: 로직 분기에 쓰는 애플리케이션 응답 코드(문자열)
- message: 사람이 읽는 설명. 분기에 쓰지 않는다.

FastAPI 기본값은 `{"detail": ...}` 이고 스키마 위반을 422 로 내므로, 여기서 공통 형식으로 바꾼다.
"""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

INVALID_REQUEST = "INVALID_REQUEST"
INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"

_HTTP_ERROR_MESSAGES = {
    status.HTTP_401_UNAUTHORIZED: "인증에 실패했습니다.",
    status.HTTP_403_FORBIDDEN: "요청 권한이 없습니다.",
    status.HTTP_404_NOT_FOUND: "요청한 리소스를 찾을 수 없습니다.",
    status.HTTP_405_METHOD_NOT_ALLOWED: "허용되지 않은 요청 메서드입니다.",
}


class ApiResponse(BaseModel):
    """모든 API 응답의 공통 형식. 도메인 응답은 data 타입만 좁혀서 이것을 상속한다."""

    code: str
    data: Any | None = None
    message: str


def error_response(
    status_code: int, code: str, message: str, data: Any | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ApiResponse(code=code, data=data, message=message).model_dump(),
    )


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """필수 필드 누락, 잘못된 타입, 허용되지 않은 enum 등. FastAPI 의 422 대신 400 으로 낸다."""
    return error_response(
        status.HTTP_400_BAD_REQUEST, INVALID_REQUEST, "입력값이 올바르지 않습니다."
    )


async def handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """인증 실패(401), 없는 경로(404), 허용되지 않은 메서드(405) 등.

    Starlette 의 HTTPException 으로 받아야 라우팅 단계의 404/405 까지 잡힌다.
    code 는 HTTP status 이름에서 만든다(예: 404 → NOT_FOUND).
    """
    code = HTTPStatus(exc.status_code).name
    message = _HTTP_ERROR_MESSAGES.get(exc.status_code, "요청을 처리할 수 없습니다.")
    return error_response(exc.status_code, code, message)


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """예상하지 못한 내부 예외. 원인은 로그에만 남기고 응답에는 싣지 않는다.

    원인이 분명한 오류는 각 Endpoint 의 특수 오류로 구분하고, 여기로는 정말
    예상하지 못한 것만 온다. 그래서 여기 로그가 쌓이면 그건 분류가 덜 된 것이다.
    """
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        INTERNAL_SERVER_ERROR,
        "서버 내부 오류가 발생했습니다.",
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(Exception, handle_unexpected_error)
