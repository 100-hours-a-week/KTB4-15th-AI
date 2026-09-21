"""Provider 들이 공유하는 동기 HTTP 전송. 전송 실패를 도메인 예외로 변환한다."""

import socket
from collections.abc import Callable
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

from app.virtual_fitting.exceptions import FittingModelError, FittingTimeoutError

_ERROR_BODY_LIMIT = 200


def _is_timeout(error: BaseException) -> bool:
    return isinstance(error, (socket.timeout, TimeoutError))


def _http_error(error: HTTPError, service: str) -> Exception:
    if error.code == 504:
        return FittingTimeoutError(f"{service} 동기 요청이 시간 내에 완료되지 않았습니다.")
    try:
        detail = error.read()[:_ERROR_BODY_LIMIT].decode("utf-8", "replace")
    except (OSError, HTTPException):
        detail = ""
    return FittingModelError(f"{service} HTTP 오류: {error.code} {detail}".strip())


def send_request(
    request: Request, *, opener: Callable[..., Any], timeout: float, service: str
) -> bytes:
    """응답 본문을 반환한다. HTTP 오류·timeout·네트워크 오류는 VirtualFittingError 로 던진다."""
    try:
        with opener(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as error:
        raise _http_error(error, service) from error
    except URLError as error:
        if _is_timeout(error.reason):
            raise FittingTimeoutError(f"{service} 요청 시간이 초과되었습니다.") from error
        raise FittingModelError(f"{service} 요청에 실패했습니다.") from error
    except TimeoutError as error:
        raise FittingTimeoutError(f"{service} 요청 시간이 초과되었습니다.") from error
    except OSError as error:
        raise FittingModelError(f"{service} 요청에 실패했습니다.") from error
