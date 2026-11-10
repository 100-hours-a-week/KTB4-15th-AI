import asyncio
import logging
import time
from collections.abc import Callable

from fastapi import status
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.body_image_validation.deadline import SCOPE_KEY, RequestDeadline
from app.body_image_validation.exceptions import SYSTEM_FAILURE_MESSAGE
from app.body_image_validation.remote import (
    ENDPOINT_PATH,
    WORKER_TIMEOUT,
    log_request_before_service,
)
from app.config import body_image_validation as config
from app.errors import error_response

logger = logging.getLogger(__name__)


class BodyImageDeadlineMiddleware:
    """remote 모드의 POST 전신 이미지 검증에 전체 제한시간을 건다. Multipart 파싱보다 먼저 시작한다.

    순수 ASGI 라 본문을 읽지 않고 receive 도 감싸지 않는다. 시각은 요청마다 새로 생기는 scope 에 둬서 섞이지 않는다.
    service 가 이어받기 전에 제한시간이 지나면 앱을 취소하고 504 를 보낸다. 그 뒤의 제한시간은 service 몫이다.

    앱을 별도 Task 로 돌리는 이유: 취소 정리 중인 앱을 기다리지 않고 응답하기 위해서다. 1MB 를 넘는 업로드의 close 는
    threadpool 토큰이 필요해 threadpool 이 포화되면 정리가 늦어진다.
    """

    def __init__(self, app: ASGIApp, clock: Callable[[], float] = time.monotonic) -> None:
        self.app = app
        self._clock = clock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_remote_validation(scope):
            await self.app(scope, receive, send)
            return
        limit = RequestDeadline(config.REMOTE_TOTAL_DEADLINE_SECONDS, self._clock)
        scope[SCOPE_KEY] = limit
        timed_out = response_started = False

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if timed_out:
                return
            response_started = response_started or message["type"] == "http.response.start"
            await send(message)

        call = asyncio.ensure_future(self.app(scope, receive, guarded_send))
        limit.guard = asyncio.get_running_loop().create_future()
        try:
            await asyncio.wait({call, limit.guard}, timeout=limit.total, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            if limit.guard is not None:
                log_request_before_service(limit, "cancelled")
            call.cancel()
            await asyncio.gather(call, return_exceptions=True)
            raise
        if call.done() or limit.guard is None:
            await call  # 끝났거나 service 가 이어받았다. 취소는 await 를 따라 call 로 전해진다.
            return
        timed_out = True
        call.cancel()
        log_request_before_service(limit, "deadline_exceeded", WORKER_TIMEOUT)
        logger.warning("body image validation failed: %s", WORKER_TIMEOUT)
        if not response_started:  # 이미 시작한 응답에는 status 를 다시 보낼 수 없다
            response = error_response(status.HTTP_504_GATEWAY_TIMEOUT, WORKER_TIMEOUT, SYSTEM_FAILURE_MESSAGE)
            await response(scope, receive, send)
        await asyncio.gather(call, return_exceptions=True)


def _is_remote_validation(scope: Scope) -> bool:
    return (
        scope["type"] == "http"
        and scope["method"] == "POST"
        and scope["path"] == ENDPOINT_PATH
        and config.VALIDATION_MODE == "remote"
    )
