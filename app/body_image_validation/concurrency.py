"""무거운 전신 이미지 검증(모델 추론·rembg)의 프로세스 내 동시 실행 제한.

실행 max_running 개 + 대기 max_waiting 개까지만 받고, 둘 다 차면 기다리게 하지 않고 즉시
ServerBusyError(429)를 낸다. 무제한 대기열은 만들지 않는다. max_wait_seconds 를 주면 대기열에 들어간 요청도
그 시간 안에 실행 슬롯을 얻지 못하면 SlotWaitTimeout(같은 429)으로 나간다.

endpoint 가 def 라 요청마다 threadpool 의 별도 스레드에서 동기로 실행된다. 그래서 asyncio 가
아니라 threading.Condition 으로 막는다.
"""

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from app.body_image_validation.exceptions import ServerBusyError


class SlotWaitTimeout(ServerBusyError):
    """대기열에서 정해진 시간 안에 실행 슬롯을 얻지 못했다. 응답은 ServerBusyError 와 같은 429 SERVER_BUSY 다."""


class BodyValidationLimiter:
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
        self._waiting = 0
        self._condition = threading.Condition()

    @property
    def running(self) -> int:
        with self._condition:
            return self._running

    @property
    def waiting(self) -> int:
        with self._condition:
            return self._waiting

    @property
    def max_wait_seconds(self) -> float | None:
        return self._max_wait_seconds

    @contextmanager
    def slot(self, max_wait: float | None = None) -> Iterator[None]:
        """실행 슬롯을 얻는다. 성공·실패·예외와 관계없이 블록을 나가면 반드시 반환한다.

        대기열에서 기다리는 시간은 생성자의 max_wait_seconds 와 max_wait(호출자의 남은 deadline 등) 중 짧은
        쪽까지다. 둘 다 없으면 기존처럼 슬롯이 날 때까지 기다린다.
        """
        limits = [w for w in (self._max_wait_seconds, max_wait) if w is not None]
        wait_limit = min(limits) if limits else None
        with self._condition:
            # 대기 중인 요청이 있으면 새 요청이 먼저 실행 슬롯을 가져가지 않게 한다.
            # 그래야 앞 작업이 끝났을 때 대기하던 요청이 먼저 실행된다.
            if self._running >= self._max_running or self._waiting > 0:
                if self._waiting >= self._max_waiting:
                    raise ServerBusyError()
                self._waiting += 1
                try:
                    give_up_at = None if wait_limit is None else self._clock() + wait_limit
                    while self._running >= self._max_running:
                        if give_up_at is None:
                            self._condition.wait()
                            continue
                        remaining = give_up_at - self._clock()
                        if remaining <= 0:
                            # 이 요청에 온 깨움을 다른 대기 요청에 넘긴다(그 사이 슬롯이 났을 수 있다).
                            self._condition.notify()
                            raise SlotWaitTimeout()
                        self._condition.wait(remaining)
                finally:
                    self._waiting -= 1
            self._running += 1
        try:
            yield
        finally:
            with self._condition:
                self._running -= 1
                self._condition.notify()
