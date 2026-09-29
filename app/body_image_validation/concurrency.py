"""무거운 전신 이미지 검증(모델 추론·rembg)의 프로세스 내 동시 실행 제한.

실행 max_running 개 + 대기 max_waiting 개까지만 받고, 둘 다 차면 기다리게 하지 않고 즉시
ServerBusyError(429)를 낸다. 무제한 대기열은 만들지 않는다.

endpoint 가 def 라 요청마다 threadpool 의 별도 스레드에서 동기로 실행된다. 그래서 asyncio 가
아니라 threading.Condition 으로 막는다.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from app.body_image_validation.exceptions import ServerBusyError


class BodyValidationLimiter:
    def __init__(self, max_running: int, max_waiting: int) -> None:
        if max_running < 1 or max_waiting < 0:
            raise ValueError("max_running 은 1 이상, max_waiting 은 0 이상이어야 합니다.")
        self._max_running = max_running
        self._max_waiting = max_waiting
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

    @contextmanager
    def slot(self) -> Iterator[None]:
        """실행 슬롯을 얻는다. 성공·실패·예외와 관계없이 블록을 나가면 반드시 반환한다."""
        with self._condition:
            # 대기 중인 요청이 있으면 새 요청이 먼저 실행 슬롯을 가져가지 않게 한다.
            # 그래야 앞 작업이 끝났을 때 대기하던 요청이 먼저 실행된다.
            if self._running >= self._max_running or self._waiting > 0:
                if self._waiting >= self._max_waiting:
                    raise ServerBusyError()
                self._waiting += 1
                try:
                    while self._running >= self._max_running:
                        self._condition.wait()
                finally:
                    self._waiting -= 1
            self._running += 1
        try:
            yield
        finally:
            with self._condition:
                self._running -= 1
                self._condition.notify()
