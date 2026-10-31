import asyncio
import contextlib
import http.cookiejar
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import httpx
from anyio import to_thread

from app.body_image_validation import deadline as worker_deadline
from app.body_image_validation.async_concurrency import AsyncBodyValidationLimiter
from app.body_image_validation.concurrency import SlotWaitTimeout
from app.body_image_validation.exceptions import BodyImageValidationError, ServerBusyError
from app.body_image_validation.image_io import open_image
from app.body_image_validation.models import BodyImageValidationResult

logger = logging.getLogger(__name__)

ENDPOINT_PATH = "/api/v1/validation/body-image"
SUCCESS_CODE = "BODY_IMAGE_UPLOAD_SUCCESS"
WORKER_UNAVAILABLE = "BODY_IMAGE_WORKER_UNAVAILABLE"
WORKER_TIMEOUT = "BODY_IMAGE_WORKER_TIMEOUT"
# worker 가 공통 응답 형식으로 낸 이 status 들은 Backend 에 그대로 전한다. 401/403/404/405 는 AI Server 의
# 설정(키·URL) 문제라 사용자에게 그대로 넘기지 않는다.
_RELAYED_STATUSES = frozenset({400, 413, 422, 429, 500})
BAD_GATEWAY, SERVICE_UNAVAILABLE, GATEWAY_TIMEOUT = 502, 503, 504
_MAX_RESPONSE_BYTES = 64 * 1024
_CONTENT_TYPES = {"JPEG": ("image/jpeg", "jpg"), "PNG": ("image/png", "png")}
# Modal 이 303 을 보내기 전 요청을 붙잡아 두는 시간. deadline 이 이보다 길면 네트워크 timeout 도 이보다 커야 한다.
MODAL_REDIRECT_AFTER_SECONDS = 150
# 한 요청에서 따라갈 303 의 최대 횟수(Modal 결과 조회 polling 포함). deadline 이 먼저 끝나는 것이 보통이다.
MAX_REDIRECTS = 5
# 로그에 남기는 HTTP 실패 원인. 예외 원문에는 URL·헤더가 들어갈 수 있어 남기지 않고 이 이름만 남긴다.
# 뒤의 두 항목은 위 목록에 없는 httpx 예외를 받는다(type(error).__mro__ 순서로 찾는다).
_ERROR_TYPES = {
    httpx.ConnectTimeout: "connect_timeout",
    httpx.ReadTimeout: "read_timeout",
    httpx.WriteTimeout: "write_timeout",
    httpx.PoolTimeout: "pool_timeout",
    httpx.ConnectError: "connect_error",
    httpx.ReadError: "read_error",
    httpx.WriteError: "write_error",
    httpx.CloseError: "close_error",
    httpx.RemoteProtocolError: "remote_protocol_error",
    httpx.LocalProtocolError: "local_protocol_error",
    httpx.ProxyError: "proxy_error",
    httpx.UnsupportedProtocol: "unsupported_protocol",
    httpx.TimeoutException: "timeout",
    httpx.TransportError: "transport_error",
}
# 이 쪽에서 HTTP 호출을 끊은 결과. Modal 작업이 멈췄는지는 알 수 없다.
_HTTP_CANCELLED_OUTCOMES = frozenset({"deadline_exceeded", "cancelled", "network_timeout"})


class RemoteWorkerConfigError(ValueError):
    """remote 모드 설정이 빠졌거나 잘못됐다."""


class RelayedWorkerError(BodyImageValidationError):
    """status 를 지정하는 worker 연동 오류. worker 가 공통 형식으로 돌려준 오류(status·code·message 그대로)와
    worker 에 닿지 못했거나 응답이 비정상인 경우(502/503/504, message 없음 → 공통 시스템 오류 문구)에 쓴다."""

    def __init__(self, status_code: int, code: str, message: str | None) -> None:
        super().__init__(code, message)
        self.status_code = status_code


def _unavailable(status_code: int) -> RelayedWorkerError:
    return RelayedWorkerError(status_code, WORKER_UNAVAILABLE, None)


def _timeout() -> RelayedWorkerError:
    return RelayedWorkerError(GATEWAY_TIMEOUT, WORKER_TIMEOUT, None)


def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
    return url.scheme, url.host, url.port


def _error_type(error: httpx.TransportError) -> str:
    return next(_ERROR_TYPES[cls] for cls in type(error).__mro__ if cls in _ERROR_TYPES)


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}s"


@dataclass
class _RequestLog:
    """요청 하나의 요약 로그. 지나가지 않은 단계의 시간은 None 이다(로그에는 - 로 남는다)."""

    request_id: str
    outcome: str = "error"
    error: str | None = None
    code: str | None = None
    status: int | None = None
    redirects: int = 0
    input_check: float | None = None
    queue_wait: float | None = None
    http: float | None = None
    budget: float | None = None


def _image_format(image_body: bytes) -> str:
    """입력을 검사하고 형식을 돌려준다. 이미지는 여기서 닫는다(호출자가 취소돼 결과를 받지 않아도 남지 않는다)."""
    source = open_image(image_body)
    try:
        return source.format
    finally:
        source.close()


class RemoteBodyImageValidationService:
    def __init__(
        self,
        base_url: str,
        *,
        proxy_key: str,
        proxy_secret: str,
        api_key: str,
        timeout: float,
        total_deadline: float,
        limiter: AsyncBodyValidationLimiter,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        missing = [
            name
            for name, value in (
                ("BODY_IMAGE_REMOTE_URL", base_url),
                ("BODY_IMAGE_REMOTE_PROXY_KEY", proxy_key),
                ("BODY_IMAGE_REMOTE_PROXY_SECRET", proxy_secret),
                ("BODY_IMAGE_REMOTE_API_KEY", api_key),
            )
            if not value
        ]
        if missing:
            raise RemoteWorkerConfigError(f"remote 모드 설정이 없습니다: {missing}")
        if not base_url.startswith("https://"):
            # 인증 헤더가 평문으로 나가지 않게 https 만 받는다.
            raise RemoteWorkerConfigError("BODY_IMAGE_REMOTE_URL 은 https:// 로 시작해야 합니다.")
        if timeout <= 0:
            raise RemoteWorkerConfigError("REMOTE_TIMEOUT_SECONDS 는 0 보다 커야 합니다.")
        if total_deadline <= 0:
            raise RemoteWorkerConfigError("REMOTE_TOTAL_DEADLINE_SECONDS 는 0 보다 커야 합니다.")
        if total_deadline > MODAL_REDIRECT_AFTER_SECONDS and timeout <= MODAL_REDIRECT_AFTER_SECONDS:
            # 상수 설정 실수다. 이 상태에서는 Modal 이 150초에 보내는 303 을 받기 전에 네트워크 timeout 이 먼저 난다.
            logger.warning(
                "REMOTE_TIMEOUT_SECONDS(%s) <= %ss: requests longer than %ss end with 504 "
                "before Modal's 303 redirect arrives (deadline=%s)",
                timeout, MODAL_REDIRECT_AFTER_SECONDS, MODAL_REDIRECT_AFTER_SECONDS, total_deadline,
            )
        self.endpoint = base_url.rstrip("/") + ENDPOINT_PATH
        self.timeout = timeout
        self.total_deadline = total_deadline
        self.limiter = limiter
        self._origin = _origin(httpx.URL(self.endpoint))
        self._headers = {
            "Modal-Key": proxy_key,
            "Modal-Secret": proxy_secret,
            "Authorization": f"Bearer {api_key}",
        }
        self._transport = transport
        self._clock = clock
        self._wall_clock = wall_clock
        self._shared_client: httpx.AsyncClient | None = None
        self._shared_loop: asyncio.AbstractEventLoop | None = None
        self._shared_calls = 0
        self._shared_idle: asyncio.Event

    async def avalidate(self, image_body: bytes) -> BodyImageValidationResult:
        """서버 event loop 에서 돈다. 입력 검사만 threadpool 에서 한다.

        취소(asyncio.CancelledError)는 바꾸지 않고 그대로 전한다. 슬롯은 어떤 경우에도 반환한다.
        Backend 가 연결을 끊어도 endpoint 는 취소되지 않아 이 호출은 끝까지 가고 슬롯도 그때 돌아온다. deadline
        초과는 연결을 끊고 슬롯을 돌려주지만 이미 시작한 Modal 작업은 멈추지 않는다.
        """
        # 전체 deadline 은 요청마다 여기서 한 번만 만든다(endpoint 가 service 를 부른 시점, monotonic).
        # 입력 검사·threadpool 대기·대기열 대기·Modal 연결·실행·303·응답 수신이 모두 이 하나의 deadline 을 나눠 쓴다.
        received = self._clock()
        log = _RequestLog(request_id=uuid.uuid4().hex[:16])
        try:
            result = await self._validate(image_body, received, log)
            log.code = SUCCESS_CODE
            return result
        except asyncio.CancelledError:
            log.outcome = log.error = "cancelled"
            raise
        except BodyImageValidationError as error:
            log.code = error.code
            raise
        except Exception as error:
            log.error = type(error).__name__  # 예상 밖 오류는 원문 없이 종류만 남긴다
            raise
        finally:
            self._log_request(log, self._clock() - received)

    async def _validate(
        self, image_body: bytes, received: float, log: _RequestLog
    ) -> BodyImageValidationResult:
        deadline_at = received + self.total_deadline
        try:
            async with asyncio.timeout(self.total_deadline):
                image_format = await to_thread.run_sync(_image_format, image_body)
        except TimeoutError:
            raise self._deadline_exceeded(log) from None
        except BodyImageValidationError:
            log.outcome = "invalid_input"
            raise
        finally:
            log.input_check = self._clock() - received
        remaining = deadline_at - self._clock()
        if remaining <= 0:
            raise self._deadline_exceeded(log)
        queue_limit = self.limiter.max_wait_seconds
        async with contextlib.AsyncExitStack() as stack:
            queued = self._clock()
            try:
                # 대기열에서는 최대 대기시간과 남은 deadline 중 짧은 쪽까지만 기다린다.
                await stack.enter_async_context(self.limiter.slot(max_wait=remaining))
            except SlotWaitTimeout:
                if queue_limit is None or remaining <= queue_limit:
                    # deadline 이 먼저 끝났다. Modal 은 부르지 않는다.
                    raise self._deadline_exceeded(log) from None
                log.outcome = "queue_timeout"
                raise
            except ServerBusyError:
                log.outcome = "queue_full"
                raise
            finally:
                log.queue_wait = self._clock() - queued
            if deadline_at - self._clock() <= 0:
                raise self._deadline_exceeded(log)
            status, body = await self._call(image_body, image_format, deadline_at, log)
        return self._result(status, body)

    @staticmethod
    def _deadline_exceeded(log: _RequestLog) -> "RelayedWorkerError":
        log.outcome = log.error = "deadline_exceeded"
        return _timeout()

    def _log_request(self, log: _RequestLog, total: float) -> None:
        http_cancelled = log.http is not None and log.outcome in _HTTP_CANCELLED_OUTCOMES
        # http_cancelled: 이 쪽 연결을 끊었다. remote_cancel: Modal 작업 취소 API 는 부르지 않는다(HTTP 로는
        # 없음). remote_stopped: 알 수 없음 — worker 가 deadline header 를 보고 다음 단계 전에 스스로 멈춘다.
        logger.info(
            "body image request: request_id=%s outcome=%s error=%s code=%s status=%s redirects=%s "
            "input_check=%s queue_wait=%s http=%s total=%s budget=%s deadline=%ss http_cancelled=%s "
            "remote_cancel=not_requested remote_stopped=%s",
            log.request_id, log.outcome, log.error or "-", log.code or "-", log.status or "-", log.redirects,
            _seconds(log.input_check), _seconds(log.queue_wait), _seconds(log.http), _seconds(total),
            _seconds(log.budget), self.total_deadline, http_cancelled, "unknown" if http_cancelled else "n/a",
        )

    async def aopen(self) -> None:
        """공유 client 를 이 event loop 에서 만든다. 이미 열려 있으면 아무것도 하지 않는다."""
        self._check_shared_loop()
        if self._shared_client is not None:
            return
        client = self._build_client()
        # 공유 client 에는 cookie 를 저장하지 않는다. 요청 사이에 응답 상태가 이어지지 않게 한다(_exchange 참고).
        client.cookies.jar.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
        self._shared_client, self._shared_loop = client, asyncio.get_running_loop()
        self._shared_idle = asyncio.Event()
        self._shared_idle.set()

    async def aclose(self) -> None:
        """새 호출을 막고(503) 진행 중인 호출이 끝난 뒤 공유 client 를 닫는다. 여러 번 불러도 된다.

        진행 중인 호출은 취소하지 않는다. 각 호출은 자기 deadline 안에 끝나므로 기다리는 시간도 그 안이다.
        이 대기가 취소되면(lifespan 취소) 바로 닫는다. 남은 호출은 연결 실패와 같게 처리된다.
        """
        self._check_shared_loop()
        client, self._shared_client = self._shared_client, None
        if client is None:
            return
        in_flight, started = self._shared_calls, self._clock()
        try:
            await self._shared_idle.wait()
        finally:
            self._shared_loop = None
            logger.info(
                "body image worker client closing: in_flight=%s left=%s waited=%s",
                in_flight, self._shared_calls, _seconds(self._clock() - started),
            )
            await client.aclose()

    def _check_shared_loop(self) -> None:
        if self._shared_loop is not None and self._shared_loop is not asyncio.get_running_loop():
            raise RuntimeError("공유 HTTP client 는 만든 event loop 에서만 쓸 수 있습니다.")

    async def _call(
        self, image_body: bytes, image_format: str, deadline_at: float, log: _RequestLog
    ) -> tuple[int, bytes]:
        start = self._clock()
        remaining = log.budget = deadline_at - start
        # worker 에는 같은 deadline 을 절대 시각(epoch 초)으로 보낸다. 지금 wall clock + monotonic 으로 잰 남은
        # 시간이라 대기열에서 쓴 시간이 빠져 있고, 호출을 시작했다고 늘어나지 않는다. redirect 에도 같은 값.
        headers = {
            **self._headers,
            worker_deadline.HEADER: f"{self._wall_clock() + remaining:.3f}",
        }
        try:
            async with asyncio.timeout(remaining):
                status, body = await self._exchange(image_body, image_format, deadline_at, log, headers)
            log.status, log.outcome = status, "response"
            return status, body
        except TimeoutError:
            # asyncio.timeout 이 진행 중인 소켓 작업을 취소하고 TimeoutError 로 바꾼다.
            raise self._deadline_exceeded(log) from None
        # httpx 예외 원문은 router 의 traceback 로 나가지 않게 체인을 끊는다. 원인은 log.error 로 남는다.
        except httpx.TimeoutException as error:
            log.outcome, log.error = "network_timeout", _error_type(error)
            raise _timeout() from None
        except httpx.TransportError as error:  # 연결 실패·끊김. keep-alive 재사용 탓인지는 여기서 알 수 없다.
            log.outcome, log.error = "connection_failed", _error_type(error)
            raise _unavailable(SERVICE_UNAVAILABLE) from None
        finally:
            log.http = self._clock() - start

    async def _exchange(self, image_body, image_format, deadline_at, log, headers) -> tuple[int, bytes]:
        content_type, extension = _CONTENT_TYPES.get(image_format, ("application/octet-stream", "bin"))
        # 이 호출의 cookie. 303 을 따라갈 때만 쓰고 호출이 끝나면 버린다.
        cookies = httpx.Cookies()
        async with self._borrow_shared_client() as client:
            request = client.build_request(
                "POST",
                self.endpoint,
                headers=headers,
                files={"image": (f"body.{extension}", image_body, content_type)},
                timeout=self._network_timeout(deadline_at),
            )
            visited = {str(request.url)}
            while True:
                response = await client.send(request, stream=True)
                try:
                    if response.status_code != 303:
                        return response.status_code, await _read_limited(response)
                    location = response.headers.get("location")
                    cookies.extract_cookies(response)
                finally:
                    await response.aclose()
                log.redirects += 1
                log.status = 303
                target = self._redirect_target(request.url, location, visited, log.redirects)
                # 303 은 GET 으로 바꾼다. 이미지 본문과 Content-Type·Content-Length 는 보내지 않는다.
                request = client.build_request(
                    "GET", target, headers=headers, cookies=cookies, timeout=self._network_timeout(deadline_at)
                )

    def _build_client(self) -> httpx.AsyncClient:
        # trust_env=False: 환경변수 proxy·.netrc 인증을 끼워 넣지 않는다. redirect 는 _exchange 가 직접 판단한다.
        return httpx.AsyncClient(transport=self._transport, follow_redirects=False, trust_env=False)

    @contextlib.asynccontextmanager
    async def _borrow_shared_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """공유 client 를 빌려주고 닫지 않는다. aclose() 는 빌려간 호출이 끝나기를 기다린다."""
        self._check_shared_loop()
        client = self._shared_client
        if client is None:
            logger.error("body image worker client is not open")
            raise _unavailable(SERVICE_UNAVAILABLE)
        self._shared_calls += 1
        self._shared_idle.clear()
        try:
            yield client
        finally:
            self._shared_calls -= 1
            if self._shared_calls == 0:
                self._shared_idle.set()

    def _network_timeout(self, deadline_at: float) -> httpx.Timeout:
        remaining = deadline_at - self._clock()
        return httpx.Timeout(max(0.001, min(self.timeout, remaining)))

    def _redirect_target(self, current: httpx.URL, location, visited: set, count: int) -> httpx.URL:
        if not location:
            logger.error("body image worker sent 303 without a Location header")
            raise _unavailable(BAD_GATEWAY)
        target = current.join(location)
        if target.scheme != "https" or _origin(target) != self._origin:
            # 인증 헤더·요청을 다른 origin 으로 보내지 않는다. 대상 URL 은 로그에 남기지 않는다.
            logger.error("body image worker redirect to a different origin was blocked")
            raise _unavailable(BAD_GATEWAY)
        if count > MAX_REDIRECTS:
            logger.error("body image worker exceeded %s redirects", MAX_REDIRECTS)
            raise _unavailable(BAD_GATEWAY)
        key = str(target)
        # 같은 결과 조회 URL 로 다시 303 하는 것은 Modal 의 polling 이라 허용한다(횟수 제한 안에서).
        # 이미 거쳐 온 다른 URL 로 돌아가거나 query 없는 원래 endpoint 로 돌아가면 loop 로 본다.
        if (key in visited and key != str(current)) or key == self.endpoint:
            logger.error("body image worker redirect loop was blocked")
            raise _unavailable(BAD_GATEWAY)
        visited.add(key)
        return target

    @staticmethod
    def _result(status: int, body: bytes) -> BodyImageValidationResult:
        payload = _common_format(body)
        if status == 200:
            data = payload.get("data") if payload else None
            s3_key = data.get("s3_key") if isinstance(data, dict) else None
            if payload and payload["code"] == SUCCESS_CODE and isinstance(s3_key, str) and s3_key:
                return BodyImageValidationResult(s3_key=s3_key)
            logger.error("body image worker returned 200 without a valid success body")
            raise _unavailable(BAD_GATEWAY)
        if payload and status in _RELAYED_STATUSES and payload["code"] != "INVALID_REQUEST":
            raise RelayedWorkerError(status, payload["code"], payload["message"])
        code = payload["code"] if payload else None
        if status == SERVICE_UNAVAILABLE:
            logger.error("body image worker is unavailable: status=503 code=%s", code)
            raise _unavailable(SERVICE_UNAVAILABLE)
        if status == GATEWAY_TIMEOUT:
            logger.error("body image worker gateway timed out: status=504 code=%s", code)
            raise _timeout()
        if status in (401, 403):
            # 공통 형식이면 worker(app)의 Bearer 거절, 아니면 Modal proxy 의 Modal-Key/Secret 거절이다.
            layer = "worker bearer (BODY_IMAGE_REMOTE_API_KEY)" if payload else (
                "modal proxy (BODY_IMAGE_REMOTE_PROXY_KEY/SECRET)"
            )
            logger.error("body image worker rejected credentials: status=%s layer=%s", status, layer)
        elif 300 <= status < 400:
            # 303 외의 redirect(301/302/307/308 등)는 따라가지 않는다. 307/308 은 이미지를 다시 보내게 된다.
            logger.error("body image worker returned an unsupported redirect: status=%s", status)
        elif status in (404, 405):
            logger.error("body image worker endpoint mismatch: status=%s (check BODY_IMAGE_REMOTE_URL)", status)
        else:
            logger.error("body image worker returned an invalid response: status=%s code=%s", status, code)
        raise _unavailable(BAD_GATEWAY)


def _common_format(body: bytes) -> dict | None:
    """{"code": str, "data": ..., "message": str} 이면 dict, 아니면 None."""
    if len(body) > _MAX_RESPONSE_BYTES:
        return None
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError):
        return None
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("code"), str)
        and isinstance(payload.get("message"), str)
    ):
        return payload
    return None


async def _read_limited(response: httpx.Response) -> bytes:
    """본문을 최대 _MAX_RESPONSE_BYTES + 1 까지만 읽는다(넘으면 _common_format 이 비정상으로 본다)."""
    chunks, size = [], 0
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size > _MAX_RESPONSE_BYTES:
            break
    return b"".join(chunks)
