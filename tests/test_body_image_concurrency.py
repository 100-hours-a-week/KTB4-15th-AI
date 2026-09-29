"""무거운 전신 이미지 검증의 동시 실행 제한(실행 1 + 대기 1, 초과 시 즉시 429).

sleep 타이밍에 기대지 않도록 Event 로 순서를 고정하고, 상태 변화는 wait_until 로 확인한다.
"""

import threading
import time

import pytest

from app.body_image_validation.concurrency import BodyValidationLimiter
from app.body_image_validation.exceptions import ServerBusyError

TIMEOUT = 5.0


def wait_until(predicate, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("상태가 기대값이 되지 않았다")
        time.sleep(0.001)


class HeldRequest:
    """슬롯을 얻은 뒤 release() 전까지 실행 중으로 머무는 요청."""

    def __init__(self, limiter, name, order, error=None):
        self.started = threading.Event()
        self._release = threading.Event()
        self._error = error
        self.rejected = None

        def run():
            try:
                with limiter.slot():
                    order.append(f"{name}-start")
                    self.started.set()
                    assert self._release.wait(TIMEOUT)
                    order.append(f"{name}-end")
                    if self._error is not None:
                        raise self._error
            except ServerBusyError as busy:
                self.rejected = busy
            except RuntimeError:
                pass

        self.thread = threading.Thread(target=run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def release(self):
        self._release.set()

    def join(self):
        self.thread.join(TIMEOUT)
        assert not self.thread.is_alive()


def test_case_a_first_request_runs_immediately():
    limiter = BodyValidationLimiter(1, 1)

    with limiter.slot():
        assert (limiter.running, limiter.waiting) == (1, 0)

    assert (limiter.running, limiter.waiting) == (0, 0)


def test_case_b_second_request_waits_and_runs_after_the_first_finishes():
    limiter = BodyValidationLimiter(1, 1)
    order = []

    first = HeldRequest(limiter, "first", order).start()
    assert first.started.wait(TIMEOUT)
    second = HeldRequest(limiter, "second", order).start()
    wait_until(lambda: limiter.waiting == 1)

    assert order == ["first-start"]  # 두 번째는 아직 실행되지 않았다
    assert (limiter.running, limiter.waiting) == (1, 1)

    first.release()
    assert second.started.wait(TIMEOUT)  # 앞 작업이 끝나면 대기하던 요청이 자동으로 시작된다
    assert (limiter.running, limiter.waiting) == (1, 0)
    second.release()
    first.join()
    second.join()

    assert order == ["first-start", "first-end", "second-start", "second-end"]


def test_case_c_third_request_is_rejected_immediately_while_one_runs_and_one_waits():
    limiter = BodyValidationLimiter(1, 1)
    order = []
    first = HeldRequest(limiter, "first", order).start()
    assert first.started.wait(TIMEOUT)
    second = HeldRequest(limiter, "second", order).start()
    wait_until(lambda: limiter.waiting == 1)

    with pytest.raises(ServerBusyError) as exc_info, limiter.slot():
        order.append("third-start")

    busy = exc_info.value
    assert (busy.status_code, busy.code) == (429, "SERVER_BUSY")
    assert busy.message == "현재 이미지 처리 요청이 많습니다. 잠시 후 다시 시도해주세요."
    assert "third-start" not in order
    assert (limiter.running, limiter.waiting) == (1, 1)  # 거절은 상태를 바꾸지 않는다

    first.release()
    assert second.started.wait(TIMEOUT)
    second.release()
    first.join()
    second.join()


def test_case_d_an_exception_in_the_running_request_still_lets_the_waiting_one_run():
    limiter = BodyValidationLimiter(1, 1)
    order = []
    first = HeldRequest(limiter, "first", order, error=RuntimeError("model crashed")).start()
    assert first.started.wait(TIMEOUT)
    second = HeldRequest(limiter, "second", order).start()
    wait_until(lambda: limiter.waiting == 1)

    first.release()
    assert second.started.wait(TIMEOUT)
    second.release()
    first.join()
    second.join()

    assert order == ["first-start", "first-end", "second-start", "second-end"]
    assert (limiter.running, limiter.waiting) == (0, 0)


def test_case_e_counts_return_to_zero_after_success_failure_and_rejection():
    limiter = BodyValidationLimiter(1, 1)

    with limiter.slot():
        pass
    with pytest.raises(ValueError), limiter.slot():
        raise ValueError("processing failed")
    assert (limiter.running, limiter.waiting) == (0, 0)

    order = []
    first = HeldRequest(limiter, "first", order).start()
    assert first.started.wait(TIMEOUT)
    second = HeldRequest(limiter, "second", order).start()
    wait_until(lambda: limiter.waiting == 1)
    with pytest.raises(ServerBusyError), limiter.slot():
        pass
    first.release()
    assert second.started.wait(TIMEOUT)
    second.release()
    first.join()
    second.join()

    assert (limiter.running, limiter.waiting) == (0, 0)
    with limiter.slot():  # 다시 바로 쓸 수 있다
        assert limiter.running == 1


def test_never_more_than_one_heavy_validation_runs_at_the_same_time():
    limiter = BodyValidationLimiter(1, 20)
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def work():
        with limiter.slot():
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.002)
            with lock:
                state["now"] -= 1

    threads = [threading.Thread(target=work) for _ in range(15)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)

    assert state["max"] == 1
    assert (limiter.running, limiter.waiting) == (0, 0)


@pytest.mark.parametrize(("running", "waiting"), [(0, 1), (1, -1)])
def test_invalid_limits_are_rejected(running, waiting):
    with pytest.raises(ValueError):
        BodyValidationLimiter(running, waiting)
