"""가상피팅 외부 호출(VTON / LLM)의 프로세스 내 동시 실행 제한.

슬롯 대기는 이벤트 루프에서 async 로 한다. 슬롯을 얻은 뒤에야 동기 Provider 호출을 worker thread 로
넘기므로, 기다리는 요청은 worker thread(AnyIO 기본 40개)를 잡지 않는다. 전신 이미지 검증의
BodyValidationLimiter 는 thread 안에서 기다리므로 여기서 쓰지 않는다.

모든 상태 변경은 이벤트 루프 한 곳에서, await 없이 한 번에 일어난다. 그래서 lock 없이도
running / waiting 확인과 증가 사이에 다른 요청이 끼어들지 못한다.

메모리 limiter 라 프로세스마다 따로 센다. uvicorn worker 나 인스턴스가 늘면 전체 한도도 그만큼 는다.
"""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial

from app.config import settings
from app.virtual_fitting.exceptions import (
    FittingPostprocessError,
    FittingQueueTimeoutError,
    FittingServerBusyError,
)


class ConcurrencyLimiter:
    """실행 max_running 개, 대기 max_waiting 개. 대기 자리까지 차면 즉시 FittingServerBusyError.

    max_waiting / wait_timeout 이 None 이면 제한하지 않는다. wait_timeout 은 슬롯을 기다리는 시간에만
    걸리고, 넘기면 timeout_error() 를 던진다. 슬롯은 반환할 때 가장 오래 기다린 요청에 바로
    넘긴다(FIFO). 넘기는 동안 running 은 줄지 않으므로 새 요청이 끼어들지 못한다.
    """

    def __init__(
        self,
        max_running: int,
        *,
        max_waiting: int | None = None,
        wait_timeout: float | None = None,
        timeout_error: Callable[[], Exception] = FittingQueueTimeoutError,
    ) -> None:
        if max_running < 1:
            raise ValueError("max_running 은 1 이상이어야 합니다.")
        if max_waiting is not None and max_waiting < 0:
            raise ValueError("max_waiting 은 0 이상이어야 합니다.")
        if wait_timeout is not None and wait_timeout <= 0:
            raise ValueError("wait_timeout 은 0 보다 커야 합니다.")
        self._max_running = max_running
        self._max_waiting = max_waiting
        self._wait_timeout = wait_timeout
        self._timeout_error = timeout_error
        self._running = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def running(self) -> int:
        return self._running

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """실행 슬롯을 얻는다. 성공·실패·예외·취소와 관계없이 블록을 나가면 반드시 반환한다."""
        await self._acquire()
        try:
            yield
        finally:
            self._release()

    async def _acquire(self) -> None:
        if self._running < self._max_running:
            self._running += 1
            return
        if self._max_waiting is not None and len(self._waiters) >= self._max_waiting:
            raise FittingServerBusyError()

        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            async with asyncio.timeout(self._wait_timeout):
                await waiter
        except BaseException as error:
            if waiter.done() and not waiter.cancelled():
                # 슬롯을 넘겨받은 직후에 시간 초과·취소가 겹쳤다. 받은 슬롯을 그대로 다음에 넘긴다.
                self._release()
            elif waiter in self._waiters:
                self._waiters.remove(waiter)
            if isinstance(error, TimeoutError):
                raise self._timeout_error() from None
            raise

    def _release(self) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            # 취소된 대기자는 _acquire 의 except 가 정리 중이다. 건너뛴다.
            if not waiter.done():
                waiter.set_result(None)
                return
        self._running -= 1


@dataclass(frozen=True)
class FittingLimiters:
    """VTON 과 LLM 은 서로 다른 모델이라 슬롯을 나누지 않는다."""

    vton: ConcurrencyLimiter
    llm: ConcurrencyLimiter


def create_fitting_limiters() -> FittingLimiters:
    """lifespan 에서 한 번 만들어 app.state 에 둔다. 설정값이 잘못되면 여기서 ValueError 가 난다."""
    return FittingLimiters(
        vton=ConcurrencyLimiter(
            settings.VIRTUAL_FITTING_VTON_CONCURRENCY,
            max_waiting=settings.VIRTUAL_FITTING_VTON_QUEUE_SIZE,
            wait_timeout=settings.VIRTUAL_FITTING_VTON_QUEUE_TIMEOUT_SECONDS,
        ),
        # LLM 슬롯을 못 얻은 것은 후처리 실패와 같다. 이미 만든 피팅 결과는 fallback 문구로 돌려준다.
        llm=ConcurrencyLimiter(
            settings.VIRTUAL_FITTING_LLM_CONCURRENCY,
            wait_timeout=settings.VIRTUAL_FITTING_LLM_SLOT_WAIT_TIMEOUT_SECONDS,
            timeout_error=partial(
                FittingPostprocessError, "Runware LLM 슬롯 대기 시간이 초과되었습니다."
            ),
        ),
    )
