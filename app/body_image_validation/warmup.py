import asyncio
import enum
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from app.body_image_validation.remote import RemoteWorkerConfigError, _read_limited
from app.config import body_image_validation as config

logger = logging.getLogger(__name__)

HEALTH_PATH = "/health"


class WarmupOutcome(enum.StrEnum):
    SUCCESS = "success"
    CONNECTION_FAILED = "connection_failed"  # 연결 실패·끊김 등 전송 오류
    NETWORK_TIMEOUT = "network_timeout"  # httpx 단계별 timeout
    DEADLINE_EXCEEDED = "deadline_exceeded"  # 호출 전체 시간 초과(asyncio.timeout)
    AUTH_FAILED = "auth_failed"  # 401/403: Modal proxy 키 문제
    NOT_FOUND = "not_found"  # 404: BODY_IMAGE_REMOTE_URL 확인
    SERVER_ERROR = "server_error"  # 5xx
    REDIRECT = "redirect"  # 303 등 3xx. 따라가지 않는다
    UNEXPECTED_STATUS = "unexpected_status"
    INVALID_JSON = "invalid_json"  # 200 인데 JSON 이 아님
    UNEXPECTED_BODY = "unexpected_body"  # 200 인데 {"status": "ok"} 가 아님


@dataclass(frozen=True)
class WarmupResult:
    outcome: WarmupOutcome
    status_code: int | None
    elapsed: float  # 초

    @property
    def succeeded(self) -> bool:
        return self.outcome is WarmupOutcome.SUCCESS


class ModalWarmupClient:
    def __init__(
        self,
        base_url: str,
        *,
        proxy_key: str,
        proxy_secret: str,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        missing = [
            name
            for name, value in (
                ("BODY_IMAGE_REMOTE_URL", base_url),
                ("BODY_IMAGE_REMOTE_PROXY_KEY", proxy_key),
                ("BODY_IMAGE_REMOTE_PROXY_SECRET", proxy_secret),
            )
            if not value
        ]
        if missing:
            raise RemoteWorkerConfigError(f"warm-up 설정이 없습니다: {missing}")
        if not base_url.startswith("https://"):
            # 인증 헤더가 평문으로 나가지 않게 https 만 받는다.
            raise RemoteWorkerConfigError("BODY_IMAGE_REMOTE_URL 은 https:// 로 시작해야 합니다.")
        try:
            url = httpx.URL(base_url.rstrip("/") + HEALTH_PATH)
        except httpx.InvalidURL as error:
            raise RemoteWorkerConfigError(
                "BODY_IMAGE_REMOTE_URL 형식이 올바르지 않습니다."
            ) from error
        if not url.host or url.query or url.fragment:
            raise RemoteWorkerConfigError(
                "BODY_IMAGE_REMOTE_URL 은 query 없는 https base URL 이어야 합니다."
            )
        if timeout <= 0:
            raise RemoteWorkerConfigError("WARMUP_TIMEOUT_SECONDS 는 0 보다 커야 합니다.")
        self.endpoint = str(url)
        self.timeout = timeout
        self._headers = {"Modal-Key": proxy_key, "Modal-Secret": proxy_secret}
        self._transport = transport
        self._clock = clock

    async def warm_up(self) -> WarmupResult:
        """GET /health 를 한 번 보낸다. 실패해도 예외를 내지 않는다(취소는 그대로 전파된다)."""
        start = self._clock()
        status_code = None
        error_type = None
        try:
            async with asyncio.timeout(self.timeout):
                status_code, outcome = await self._exchange()
        except TimeoutError:
            # asyncio.timeout 이 진행 중인 연결·대기·읽기를 취소하고 TimeoutError 로 바꾼다.
            outcome = WarmupOutcome.DEADLINE_EXCEEDED
        except httpx.TimeoutException as error:
            outcome, error_type = WarmupOutcome.NETWORK_TIMEOUT, type(error).__name__
        except httpx.HTTPError as error:
            # 예외 메시지에 host 가 들어갈 수 있어 타입 이름만 남긴다.
            outcome, error_type = WarmupOutcome.CONNECTION_FAILED, type(error).__name__
        result = WarmupResult(outcome, status_code, self._clock() - start)
        log = logger.info if result.succeeded else logger.warning
        log(
            "body image worker warm-up: outcome=%s status=%s error=%s elapsed=%.2fs timeout=%ss",
            outcome,
            status_code,
            error_type,
            result.elapsed,
            self.timeout,
        )
        return result

    async def _exchange(self) -> tuple[int, WarmupOutcome]:
        # trust_env=False: 환경변수 proxy·.netrc 인증을 끼워 넣지 않는다. redirect 는 따라가지 않는다.
        async with httpx.AsyncClient(
            transport=self._transport, follow_redirects=False, trust_env=False
        ) as client:
            request = client.build_request(
                "GET", self.endpoint, headers=self._headers, timeout=httpx.Timeout(self.timeout)
            )
            response = await client.send(request, stream=True)
            try:
                if response.status_code != 200:
                    return response.status_code, _failure(response.status_code)
                return 200, _health_body(await _read_limited(response))
            finally:
                await response.aclose()


def _failure(status_code: int) -> WarmupOutcome:
    if 300 <= status_code < 400:
        return WarmupOutcome.REDIRECT
    if status_code in (401, 403):
        return WarmupOutcome.AUTH_FAILED
    if status_code == 404:
        return WarmupOutcome.NOT_FOUND
    if status_code >= 500:
        return WarmupOutcome.SERVER_ERROR
    return WarmupOutcome.UNEXPECTED_STATUS


def _health_body(body: bytes) -> WarmupOutcome:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError):
        return WarmupOutcome.INVALID_JSON
    if isinstance(payload, dict) and payload.get("status") == "ok":
        return WarmupOutcome.SUCCESS
    return WarmupOutcome.UNEXPECTED_BODY


def create_warmup_client(transport: httpx.AsyncBaseTransport | None = None) -> ModalWarmupClient:
    """BODY_IMAGE_REMOTE_URL·proxy 키와 코드 상수 timeout 으로 만든다. 설정이 잘못되면 RemoteWorkerConfigError."""
    from app.config import settings

    return ModalWarmupClient(
        config.REMOTE_URL,
        proxy_key=settings.BODY_IMAGE_REMOTE_PROXY_KEY,
        proxy_secret=settings.BODY_IMAGE_REMOTE_PROXY_SECRET,
        timeout=config.WARMUP_TIMEOUT_SECONDS,
        transport=transport,
    )
