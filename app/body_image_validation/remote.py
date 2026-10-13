"""AI Server 가 전신 이미지 검증을 Modal T4 worker 에 HTTP 로 맡기는 service.

router 는 local service 와 같은 `validate(bytes) -> BodyImageValidationResult` 만 부른다. 검증·배경 제거·
PNG·S3 업로드는 worker 가 기존 local service 로 하고, 여기서는 모델을 올리지 않는다.

- 요청 전에 헤더 검사(open_image)를 여기서도 한다. 잘못된 파일은 같은 400/413/422 로 바로 거절하고
  T4 를 깨우지 않는다. 동시 실행 제한도 local 과 같은 limiter 로 건다(넘치면 429).
- worker 가 공통 형식으로 돌려준 사용자 오류(400/413/422), 429, 500 은 code·message·status 를 그대로 전한다.
- worker 에 닿지 못하면(연결 실패, 503) 503 BODY_IMAGE_WORKER_UNAVAILABLE, 정상 worker 응답이 아니면
  (공통 형식이 아님, 200 인데 s3_key 없음, Modal proxy·worker 인증 실패, 404/405, 502, 허용하지 않는 redirect 등)
  502 BODY_IMAGE_WORKER_UNAVAILABLE, 시간 초과(504 포함)는 504 BODY_IMAGE_WORKER_TIMEOUT 이다. message 는 공통
  시스템 오류 문구다. 원인은 로그로 구분하고 키·Secret·URL query 는 남기지 않는다.
- 재시도하지 않는다. 업로드 요청을 다시 보내면 worker 가 같은 사진을 다시 처리하고 S3 에 한 번 더 올린다.
  시간 초과는 worker 가 처리를 끝내고 S3 에 올렸을 수도 있다는 뜻이다(응답만 받지 못함).

시간 제한은 두 가지다.
- 전체 deadline(BODY_IMAGE_REMOTE_TOTAL_DEADLINE_SECONDS): 첫 POST 부터 마지막 응답 본문을 다 읽을 때까지.
  redirect 를 따라가도 다시 시작하지 않는다. asyncio.timeout 이 넘는 순간 진행 중인 연결·전송·대기·읽기를 취소한다.
- 네트워크 timeout(BODY_IMAGE_REMOTE_TIMEOUT_SECONDS): 연결·전송·응답 대기·읽기 각 단계가 멈춰 있을 수 있는
  최대 시간. 남은 deadline 보다 길게 잡히지 않는다. Modal 은 150초가 지나야 303 을 보내므로, deadline 이 150초를
  넘으면 이 값도 150초보다 커야 303 을 받을 수 있다.

Modal web function 은 150초를 넘기면 같은 URL 에 결과 조회용 query 를 붙여 303 으로 돌려보낸다
(https://modal.com/docs/guide/webhook-timeouts). 303 은 같은 https origin 일 때만 GET(본문 없음)으로 따라가고,
301/302/307/308 등 다른 redirect 와 다른 origin 으로의 redirect 는 따라가지 않고 502 로 끝낸다(이미지·인증 헤더를
다른 곳으로 보내지 않는다).

FastAPI endpoint 는 sync(def)라 threadpool 의 worker thread 에서 실행된다. 그 thread 에서 asyncio.run 으로 이
요청 하나만을 위한 event loop 를 돌린다(서버의 event loop 와 무관). deadline 취소는 실제 소켓 작업을 멈춘다.
"""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable

import httpx

from app.body_image_validation import deadline as worker_deadline
from app.body_image_validation.concurrency import BodyValidationLimiter, SlotWaitTimeout
from app.body_image_validation.exceptions import BodyImageValidationError
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
        limiter: BodyValidationLimiter,
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
            raise RemoteWorkerConfigError("BODY_IMAGE_REMOTE_TIMEOUT_SECONDS 는 0 보다 커야 합니다.")
        if total_deadline <= 0:
            raise RemoteWorkerConfigError("BODY_IMAGE_REMOTE_TOTAL_DEADLINE_SECONDS 는 0 보다 커야 합니다.")
        if total_deadline > MODAL_REDIRECT_AFTER_SECONDS and timeout <= MODAL_REDIRECT_AFTER_SECONDS:
            # 설정 실수다. 이 상태에서는 Modal 이 150초에 보내는 303 을 받기 전에 네트워크 timeout 이 먼저 난다.
            logger.warning(
                "BODY_IMAGE_REMOTE_TIMEOUT_SECONDS(%s) <= %ss: requests longer than %ss end with 504 "
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

    def validate(self, image_body: bytes) -> BodyImageValidationResult:
        # 전체 deadline 은 요청마다 여기서 한 번만 만든다(endpoint 가 service 를 부른 시점, monotonic).
        # 대기열 대기·Modal 연결·스케줄링·실행·303·응답 수신이 모두 이 하나의 deadline 을 나눠 쓴다.
        received = self._clock()
        deadline_at = received + self.total_deadline
        # local 과 같은 입력 검사. 실패하면 worker 를 부르지 않는다.
        source = open_image(image_body)
        image_format = source.format
        source.close()
        remaining = deadline_at - self._clock()
        if remaining <= 0:
            raise _timeout()
        queue_limit = self.limiter.max_wait_seconds
        with contextlib.ExitStack() as stack:
            queued = self._clock()
            try:
                # 대기열에서는 최대 대기시간과 남은 deadline 중 짧은 쪽까지만 기다린다.
                stack.enter_context(self.limiter.slot(max_wait=remaining))
            except SlotWaitTimeout as error:
                waited = self._clock() - queued
                if queue_limit is None or remaining <= queue_limit:
                    # deadline 이 먼저 끝났다. Modal 은 부르지 않는다.
                    logger.info("body image request timed out in queue: waited=%.2fs deadline=%ss",
                                waited, self.total_deadline)
                    raise _timeout() from error
                logger.info("body image request gave up in queue: waited=%.2fs limit=%ss", waited, queue_limit)
                raise
            queue_wait = self._clock() - queued
            if deadline_at - self._clock() <= 0:
                logger.info("body image request has no time left after queue: waited=%.2fs", queue_wait)
                raise _timeout()
            # 이 thread(threadpool worker) 전용 event loop. 서버 event loop 를 막거나 쓰지 않는다.
            status, body = asyncio.run(self._call(image_body, image_format, deadline_at, queue_wait))
        return self._result(status, body)

    async def _call(
        self, image_body: bytes, image_format: str, deadline_at: float, queue_wait: float = 0.0
    ) -> tuple[int, bytes]:
        start = self._clock()
        remaining = deadline_at - start
        # worker 에는 같은 deadline 을 절대 시각(epoch 초)으로 보낸다. 지금 wall clock + monotonic 으로 잰 남은
        # 시간이라 대기열에서 쓴 시간이 빠져 있고, 호출을 시작했다고 늘어나지 않는다. redirect 에도 같은 값.
        headers = {
            **self._headers,
            worker_deadline.HEADER: f"{self._wall_clock() + remaining:.3f}",
        }
        state = {"redirects": 0, "status": None, "outcome": "error"}
        try:
            async with asyncio.timeout(remaining):
                status, body = await self._exchange(image_body, image_format, deadline_at, state, headers)
            state.update(status=status, outcome="response")
            return status, body
        except TimeoutError as error:
            # asyncio.timeout 이 진행 중인 소켓 작업을 취소하고 TimeoutError 로 바꾼다.
            state["outcome"] = "deadline_exceeded"
            raise _timeout() from error
        except httpx.TimeoutException as error:
            state["outcome"] = f"network_timeout:{type(error).__name__}"
            raise _timeout() from error
        except httpx.TransportError as error:  # 연결 실패·끊김
            state["outcome"] = f"connection_failed:{type(error).__name__}"
            raise _unavailable(SERVICE_UNAVAILABLE) from error
        finally:
            timed_out = state["outcome"] == "deadline_exceeded" or state["outcome"].startswith("network_timeout")
            # http_cancelled: 이 쪽 연결을 끊었다. remote_cancel: Modal 작업 취소 API 는 부르지 않는다(HTTP 로는
            # 없음). remote_stopped: 알 수 없음 — worker 가 deadline header 를 보고 다음 단계 전에 스스로 멈춘다.
            logger.info(
                "body image worker call: outcome=%s status=%s redirects=%s queue_wait=%.2fs elapsed=%.2fs "
                "budget=%.2fs deadline=%ss http_cancelled=%s remote_cancel=not_requested remote_stopped=%s",
                state["outcome"], state["status"], state["redirects"], queue_wait, self._clock() - start,
                remaining, self.total_deadline, timed_out, "unknown" if timed_out else "n/a",
            )

    async def _exchange(self, image_body, image_format, deadline_at, state, headers) -> tuple[int, bytes]:
        content_type, extension = _CONTENT_TYPES.get(image_format, ("application/octet-stream", "bin"))
        # trust_env=False: 환경변수 proxy·.netrc 인증을 끼워 넣지 않는다. redirect 는 아래에서 직접 판단한다.
        async with httpx.AsyncClient(
            transport=self._transport, follow_redirects=False, trust_env=False
        ) as client:
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
                finally:
                    await response.aclose()
                state["redirects"] += 1
                state["status"] = 303
                target = self._redirect_target(request.url, location, visited, state["redirects"])
                # 303 은 GET 으로 바꾼다. 이미지 본문과 Content-Type·Content-Length 는 보내지 않는다.
                request = client.build_request(
                    "GET", target, headers=headers, timeout=self._network_timeout(deadline_at)
                )

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
