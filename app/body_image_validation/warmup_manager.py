import asyncio
import collections
import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from app.body_image_validation.warmup import WarmupOutcome, WarmupResult, create_warmup_client
from app.config import body_image_validation as config

logger = logging.getLogger(__name__)

# 연결을 끊어도 Modal input 은 계속 진행될 수 있어 일반 실패보다 긴 간격을 쓴다.
_TIMEOUT_OUTCOMES = frozenset({WarmupOutcome.NETWORK_TIMEOUT, WarmupOutcome.DEADLINE_EXCEEDED})


class WarmupStatus(enum.StrEnum):
    STARTED = "STARTED"  # 새 호출 등록
    IN_PROGRESS = "IN_PROGRESS"  # 진행 중인 호출 재사용
    COOLDOWN = "COOLDOWN"  # 최근 성공으로 생략
    RATE_LIMITED = "RATE_LIMITED"  # 실패 후 재시도 간격 또는 호출 수 상한
    UNAVAILABLE = "UNAVAILABLE"  # 종료되어 접수 불가


@dataclass(frozen=True)
class WarmupAcceptance:
    """요청 접수 결과. 실제 Modal 호출 결과(WarmupResult)와 다르다."""

    status: WarmupStatus
    retry_after: float | None = (
        None  # COOLDOWN·RATE_LIMITED 에서 다시 시작할 수 있을 때까지 남은 초
    )

    @property
    def accepted(self) -> bool:
        return self.status in (
            WarmupStatus.STARTED,
            WarmupStatus.IN_PROGRESS,
            WarmupStatus.COOLDOWN,
        )


class WarmupClient(Protocol):
    async def warm_up(self) -> WarmupResult: ...


class BodyImageWarmupManager:
    def __init__(
        self,
        client: WarmupClient,
        *,
        cooldown: float,
        retry_interval: float,
        timeout_retry_interval: float,
        max_starts: int,
        start_window: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            min(cooldown, retry_interval, timeout_retry_interval, start_window) <= 0
            or max_starts < 1
        ):
            raise ValueError("warm-up 시간 값은 0 보다 크고 max_starts 는 1 이상이어야 합니다.")
        self._client = client
        self._cooldown = cooldown
        self._retry_interval = retry_interval
        self._timeout_retry_interval = timeout_retry_interval
        self._start_window = start_window
        self._clock = clock
        self._lock = asyncio.Lock()
        # 진행 중인 호출의 강한 참조. singleflight 라 많아야 하나다.
        self._task: asyncio.Task | None = None
        self._closed = False
        self._succeeded_at: float | None = None
        self._retry_not_before: float | None = None
        # 최근 시작 시각. 상한 개수만 들고 있어 메모리가 늘지 않는다.
        self._starts: collections.deque[float] = collections.deque(maxlen=max_starts)
        self.last_result: WarmupResult | None = None

    @property
    def in_progress(self) -> bool:
        return self._task is not None

    async def request(self) -> WarmupAcceptance:
        """접수 판단과 task 등록만 하고 바로 돌아온다(Modal 응답을 기다리지 않는다)."""
        async with self._lock:
            acceptance = self._decide()
            if acceptance.status is WarmupStatus.STARTED:
                self._starts.append(self._clock())
                self._task = asyncio.create_task(self._run(), name="body-image-warmup")
        if acceptance.status is WarmupStatus.RATE_LIMITED:
            # 요청마다 나므로 DEBUG 로 둔다. 429 는 access log 에, 원인이 된 호출 실패는 warmup 로그(WARNING)에 남는다.
            logger.debug(
                "body image warm-up rate limited: retry_after=%.1fs", acceptance.retry_after
            )
        return acceptance

    def _decide(self) -> WarmupAcceptance:
        if self._closed:
            return WarmupAcceptance(WarmupStatus.UNAVAILABLE)
        if self._task is not None:
            return WarmupAcceptance(WarmupStatus.IN_PROGRESS)
        now = self._clock()
        if self._succeeded_at is not None and now - self._succeeded_at < self._cooldown:
            return WarmupAcceptance(
                WarmupStatus.COOLDOWN, self._succeeded_at + self._cooldown - now
            )
        if self._retry_not_before is not None and now < self._retry_not_before:
            return WarmupAcceptance(WarmupStatus.RATE_LIMITED, self._retry_not_before - now)
        if len(self._starts) == self._starts.maxlen and now - self._starts[0] < self._start_window:
            return WarmupAcceptance(
                WarmupStatus.RATE_LIMITED, self._starts[0] + self._start_window - now
            )
        return WarmupAcceptance(WarmupStatus.STARTED)

    async def _run(self) -> None:
        try:
            result = await self._client.warm_up()
        except asyncio.CancelledError:
            # 종료 때의 취소. 성공·실패 어느 쪽으로도 기록하지 않는다.
            raise
        except Exception as error:  # noqa: BLE001 — client 는 실패를 결과로 돌려주므로 여기는 예상 밖 오류다
            # 메시지에 URL 이 들어갈 수 있어 타입 이름만 남긴다. 일반 실패처럼 재시도 간격을 둔다.
            logger.error(
                "body image warm-up task failed unexpectedly: error=%s", type(error).__name__
            )
            self._record(None)
        else:
            self._record(result)
        finally:
            self._task = None

    def _record(self, result: WarmupResult | None) -> None:
        now = self._clock()
        self.last_result = result
        if result is not None and result.succeeded:
            self._succeeded_at, self._retry_not_before = now, None
            return
        interval = self._retry_interval
        if result is not None and result.outcome in _TIMEOUT_OUTCOMES:
            interval = self._timeout_retry_interval
        self._succeeded_at, self._retry_not_before = None, now + interval

    async def aclose(self) -> None:
        """새 접수를 멈추고 진행 중인 호출을 취소해 끝날 때까지 기다린다. 여러 번 불러도 된다."""
        async with self._lock:
            self._closed = True
            task = self._task
        if task is None:
            return
        task.cancel()
        # 취소된 task 의 결과를 여기서 받아 미처리 예외로 남지 않게 한다.
        await asyncio.gather(task, return_exceptions=True)
        # 시작 전에 취소된 task 는 _run 의 finally 를 거치지 않는다.
        self._task = None


def create_warmup_manager(client: WarmupClient | None = None) -> BodyImageWarmupManager:
    """코드 상수로 만든다. client 를 넘기지 않으면 remote 설정으로 만든다(잘못되면 RemoteWorkerConfigError)."""
    return BodyImageWarmupManager(
        client if client is not None else create_warmup_client(),
        cooldown=config.WARMUP_COOLDOWN_SECONDS,
        retry_interval=config.WARMUP_MIN_RETRY_INTERVAL_SECONDS,
        timeout_retry_interval=config.WARMUP_TIMEOUT_RETRY_INTERVAL_SECONDS,
        max_starts=config.WARMUP_MAX_STARTS_PER_WINDOW,
        start_window=config.WARMUP_START_WINDOW_SECONDS,
    )
