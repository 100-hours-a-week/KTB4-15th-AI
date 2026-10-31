import asyncio
import collections
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from app.body_image_validation.concurrency import SlotWaitTimeout
from app.body_image_validation.exceptions import ServerBusyError


class AsyncBodyValidationLimiter:
    def __init__(
        self,
        max_running: int,
        max_waiting: int,
        max_wait_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_running < 1 or max_waiting < 0:
            raise ValueError("max_running 은 1 이상, max_waiting 은 0 이상이어야 합니다.")
        if max_wait_seconds is not None and max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds 는 0 보다 커야 합니다.")
        self._max_running = max_running
        self._max_waiting = max_waiting
        self._max_wait_seconds = max_wait_seconds
        self._clock = clock
        self._running = 0
        self._queue: collections.deque[object] = collections.deque()
        self._condition = asyncio.Condition()
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def running(self) -> int:
        return self._running

    @property
    def waiting(self) -> int:
        return len(self._queue)

    @property
    def max_wait_seconds(self) -> float | None:
        return self._max_wait_seconds

    @asynccontextmanager
    async def slot(self, max_wait: float | None = None) -> AsyncIterator[None]:
        """실행 슬롯을 얻는다. 성공·실패·예외·취소와 관계없이 블록을 나가면 정확히 한 번 반환한다.

        대기열에서 기다리는 시간은 생성자의 max_wait_seconds 와 max_wait(호출자의 남은 deadline 등) 중 짧은
        쪽까지다. 둘 다 없으면 슬롯이 날 때까지 기다린다.
        """
        self._check_loop()
        await self._acquire(max_wait)
        try:
            yield
        finally:
            await self._release()

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("AsyncBodyValidationLimiter 는 처음 쓴 event loop 에서만 쓸 수 있습니다.")

    async def _acquire(self, max_wait: float | None) -> None:
        limits = [w for w in (self._max_wait_seconds, max_wait) if w is not None]
        wait_limit = min(limits) if limits else None
        async with self._condition:
            # 대기 중인 요청이 있으면 새 요청이 먼저 실행 슬롯을 가져가지 않게 한다.
            if self._running >= self._max_running or self._queue:
                if len(self._queue) >= self._max_waiting:
                    raise ServerBusyError()
                await self._wait_for_turn(wait_limit)
            self._running += 1

    async def _wait_for_turn(self, wait_limit: float | None) -> None:
        """lock 을 잡은 채로 부른다. 돌아올 때 맨 앞 차례이고 실행 슬롯이 비어 있다(lock 도 잡고 있다)."""
        ticket = object()
        self._queue.append(ticket)
        try:
            give_up_at = None if wait_limit is None else self._clock() + wait_limit
            while self._queue[0] is not ticket or self._running >= self._max_running:
                if give_up_at is None:
                    await self._condition.wait()
                    continue
                remaining = give_up_at - self._clock()
                if remaining <= 0:
                    raise SlotWaitTimeout()
                # 시간이 끝나도 바로 포기하지 않고 위 조건부터 다시 본다. 그 사이 슬롯이 났으면 가져간다.
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(remaining):
                        await self._condition.wait()
        finally:
            # 성공·시간 초과·취소 모두 여기서 나간다. 다음 요청이 맨 앞이 됐으므로 다시 보게 깨운다.
            self._queue.remove(ticket)
            self._condition.notify_all()

    async def _release(self) -> None:
        # 이 lock 을 기다리는 중에 취소돼도 슬롯은 반환해야 한다. 반환을 마친 뒤 취소를 다시 낸다.
        cancelled = None
        while True:
            try:
                await self._condition.acquire()
                break
            except asyncio.CancelledError as error:
                cancelled = error
        try:
            self._running -= 1
            self._condition.notify_all()
        finally:
            self._condition.release()
        if cancelled is not None:
            raise cancelled
