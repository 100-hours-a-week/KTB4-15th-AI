import asyncio
import random
import time

import pytest

from app.body_image_validation.async_concurrency import AsyncBodyValidationLimiter
from app.body_image_validation.concurrency import SlotWaitTimeout
from app.body_image_validation.exceptions import ServerBusyError

TIMEOUT = 5.0


def run(scenario):
    """새 event loop 에서 실행한다. 슬롯이 반환되지 않아도 멈추지 않고 실패하게 전체 시간을 제한한다."""

    async def bounded():
        async with asyncio.timeout(2 * TIMEOUT):
            await scenario()

    asyncio.run(bounded())


async def wait_until(predicate):
    async with asyncio.timeout(TIMEOUT):
        while not predicate():
            await asyncio.sleep(0)


class Held:
    """슬롯을 얻은 뒤 release() 전까지 실행 중으로 머무는 요청."""

    def __init__(self, limiter, name, order, max_wait=None, error=None):
        self.limiter, self.name, self.order = limiter, name, order
        self.max_wait, self.error = max_wait, error
        self.started = asyncio.Event()
        self._release = asyncio.Event()
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        async with self.limiter.slot(self.max_wait):
            self.order.append(f"{self.name}-start")
            self.started.set()
            await self._release.wait()
            self.order.append(f"{self.name}-end")
            if self.error is not None:
                raise self.error

    def release(self):
        self._release.set()

    async def started_within(self):
        async with asyncio.timeout(TIMEOUT):
            await self.started.wait()

    async def outcome(self):
        """끝날 때까지 기다려 결과(None 또는 예외)를 돌려준다."""
        async with asyncio.timeout(TIMEOUT):
            done = await asyncio.gather(self.task, return_exceptions=True)
        return done[0]


async def run_one(limiter, name, order):
    held = Held(limiter, name, order)
    await held.started_within()
    held.release()
    assert await held.outcome() is None


async def wait_for_queue(limiter, size):
    await wait_until(lambda: limiter.waiting == size)


async def run_and_queue(limiter, order, *names):
    """첫 요청은 실행, 나머지는 이 순서로 대기열에 넣는다."""
    requests = [Held(limiter, names[0], order)]
    await requests[0].started_within()
    for name in names[1:]:
        requests.append(Held(limiter, name, order))
        await wait_for_queue(limiter, len(requests) - 1)
    return requests


async def hold_until_queued(limiter, waiters, after_release, reverse_wake_up=False):
    """슬롯을 잡고 대기자가 waiters 명이 되면 나간다. 나간 직후 같은 step 에서 after_release 를 부른다.

    그 순간 슬롯은 비었고 맨 앞 대기자는 깨어났지만 아직 돌지 못했다. 슬롯을 잡은 뒤 돌아온다.
    """
    entered = asyncio.Event()

    async def hold():
        async with limiter.slot():
            entered.set()
            await wait_for_queue(limiter, waiters)
            if reverse_wake_up:
                reverse_wake_up_order(limiter)
        await after_release()

    task = asyncio.create_task(hold())
    await entered.wait()
    return task


def reverse_wake_up_order(limiter):
    """늦게 온 요청부터 깨어나게 한다. 시간 제한으로 다시 기다린 요청은 실제로 Condition 의 순서가 바뀐다."""
    limiter._condition._waiters.reverse()


def state(limiter):
    return limiter.running, limiter.waiting


class SteppedClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


# --- 공개 인터페이스 ---


@pytest.mark.parametrize(
    "args",
    [(0, 1), (1, -1), (1, 1, 0), (1, 1, -1.0)],
    ids=["no-running", "negative-waiting", "zero-wait", "negative-wait"],
)
def test_invalid_limits_are_rejected(args):
    with pytest.raises(ValueError):
        AsyncBodyValidationLimiter(*args)


def test_limiter_can_be_built_before_the_event_loop_runs():
    limiter = AsyncBodyValidationLimiter(1, 1, max_wait_seconds=10.0)

    async def scenario():
        async with limiter.slot():
            assert state(limiter) == (1, 0)

    run(scenario)

    assert state(limiter) == (0, 0)
    assert limiter.max_wait_seconds == 10.0


def test_a_limiter_cannot_be_shared_between_event_loops():
    limiter = AsyncBodyValidationLimiter(1, 1)

    async def use():
        async with limiter.slot():
            pass

    asyncio.run(use())

    with pytest.raises(RuntimeError, match="event loop"):
        asyncio.run(use())  # 경합이 없어도 조용히 지나가지 않는다
    assert state(limiter) == (0, 0)


# --- 실행 1 + 대기 1 ---


def test_single_request_runs_immediately():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1)

        async with limiter.slot():
            assert state(limiter) == (1, 0)

        assert state(limiter) == (0, 0)

    run(scenario)


def test_third_request_is_rejected_at_once_while_one_runs_and_one_waits():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1, max_wait_seconds=10.0)
        order = []
        a, b = await run_and_queue(limiter, order, "a", "b")

        started = time.monotonic()
        with pytest.raises(ServerBusyError) as exc_info:
            async with limiter.slot():
                order.append("c-start")
        elapsed = time.monotonic() - started

        assert type(exc_info.value) is ServerBusyError  # 대기시간 초과가 아니라 대기열이 찬 것
        assert (exc_info.value.status_code, exc_info.value.code) == (429, "SERVER_BUSY")
        assert elapsed < 1.0
        assert "c-start" not in order
        assert state(limiter) == (1, 1)  # 거절은 상태를 바꾸지 않는다

        a.release()
        await b.started_within()
        b.release()
        await asyncio.gather(a.task, b.task)

    run(scenario)


def test_waiting_request_runs_after_the_running_one_finishes():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1)
        order = []
        a, b = await run_and_queue(limiter, order, "a", "b")

        assert order == ["a-start"]
        assert state(limiter) == (1, 1)

        a.release()
        await b.started_within()
        assert state(limiter) == (1, 0)
        b.release()
        await asyncio.gather(a.task, b.task)

        assert order == ["a-start", "a-end", "b-start", "b-end"]
        assert state(limiter) == (0, 0)

    run(scenario)


def test_no_queue_means_the_second_request_is_rejected():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 0)
        async with limiter.slot():
            with pytest.raises(ServerBusyError):
                async with limiter.slot():
                    pass

        assert state(limiter) == (0, 0)

    run(scenario)


def test_new_request_is_rejected_rather_than_overtaking_a_woken_waiter():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1)
        order = []

        async def try_to_overtake():
            assert state(limiter) == (0, 1)
            with pytest.raises(ServerBusyError):
                async with limiter.slot():
                    order.append("new-start")

        holder = await hold_until_queued(limiter, 1, try_to_overtake)
        b = Held(limiter, "b", order)
        await holder
        await b.started_within()
        b.release()
        await b.outcome()

        assert order == ["b-start", "b-end"]
        assert state(limiter) == (0, 0)

    run(scenario)


def test_new_request_queues_behind_a_woken_waiter():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 2)
        order = []
        late = []

        async def try_to_overtake():
            assert state(limiter) == (0, 1)
            late.append(Held(limiter, "new", order))

        holder = await hold_until_queued(limiter, 1, try_to_overtake)
        b = Held(limiter, "b", order)
        await holder
        await b.started_within()
        b.release()
        await late[0].started_within()
        late[0].release()
        await asyncio.gather(b.task, late[0].task)

        assert order == ["b-start", "b-end", "new-start", "new-end"]
        assert state(limiter) == (0, 0)

    run(scenario)


# --- 최대 대기시간 ---


def test_waiting_past_10_seconds_is_slot_wait_timeout():
    # 10초에서 0.01초 모자란 시각 → 남은 시간만큼 기다린 뒤 → 10초
    times = iter([100.0, 109.99, 110.0])

    async def scenario():
        limiter = AsyncBodyValidationLimiter(
            1, 1, max_wait_seconds=10.0, clock=lambda: next(times)
        )
        order = []
        a = Held(limiter, "a", order)
        await a.started_within()

        with pytest.raises(SlotWaitTimeout) as exc_info:
            async with limiter.slot():
                order.append("b-start")

        assert isinstance(exc_info.value, ServerBusyError)
        assert (exc_info.value.status_code, exc_info.value.code) == (429, "SERVER_BUSY")
        assert order == ["a-start"]
        assert state(limiter) == (1, 0)  # 대기 자리를 돌려줬다
        a.release()
        await a.outcome()

    run(scenario)


@pytest.mark.parametrize(
    ("limit", "max_wait"),
    [(10.0, 0.02), (0.02, 10.0), (0.02, None), (None, 0.02)],
    ids=["max-wait-shorter", "limit-shorter", "only-limit", "only-max-wait"],
)
def test_the_shorter_of_the_limit_and_max_wait_applies(limit, max_wait):
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1, max_wait_seconds=limit)
        order = []
        a = Held(limiter, "a", order)
        await a.started_within()

        started = time.monotonic()
        with pytest.raises(SlotWaitTimeout):
            async with limiter.slot(max_wait=max_wait):
                pass
        elapsed = time.monotonic() - started

        assert 0.015 <= elapsed < 1.0
        a.release()
        await a.outcome()

    run(scenario)


def test_after_a_queue_timeout_the_next_request_runs():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1, max_wait_seconds=0.01)
        order = []
        a = Held(limiter, "a", order)
        await a.started_within()
        with pytest.raises(SlotWaitTimeout):
            async with limiter.slot():
                pass
        c = Held(limiter, "c", order)  # 대기열 자리가 돌아왔다(429 아님)
        await wait_until(lambda: limiter.waiting == 1)

        a.release()
        await c.started_within()
        c.release()
        await asyncio.gather(a.task, c.task)

        assert state(limiter) == (0, 0)

    run(scenario)


def test_a_slot_that_opened_when_time_ran_out_is_taken():
    # 동기 limiter 와 같다. 시간이 지났어도 깨어났을 때 슬롯이 비어 있으면 실행한다.
    async def scenario():
        clock = SteppedClock()
        limiter = AsyncBodyValidationLimiter(1, 1, max_wait_seconds=10.0, clock=clock)
        order = []
        a, b = await run_and_queue(limiter, order, "a", "b")

        clock.now += 10.0
        a.release()
        await b.started_within()
        b.release()

        assert await b.outcome() is None
        assert state(limiter) == (0, 0)

    run(scenario)


# --- 취소 ---


def test_cancelled_waiter_leaves_the_queue_and_never_runs():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1)
        order = []
        a, b = await run_and_queue(limiter, order, "a", "b")

        b.task.cancel()

        assert isinstance(await b.outcome(), asyncio.CancelledError)
        assert state(limiter) == (1, 0)
        a.release()
        await a.outcome()
        assert order == ["a-start", "a-end"]
        await run_one(limiter, "c", order)  # 자리가 돌아왔다
        assert state(limiter) == (0, 0)

    run(scenario)


def test_cancelled_running_request_returns_its_slot_once():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1)
        order = []
        a, b = await run_and_queue(limiter, order, "a", "b")

        a.task.cancel()

        assert isinstance(await a.outcome(), asyncio.CancelledError)
        await b.started_within()
        assert state(limiter) == (1, 0)  # 두 번 반환했다면 running 은 0 이다
        b.release()
        await b.outcome()
        assert state(limiter) == (0, 0)

    run(scenario)


def test_cancelling_the_head_lets_the_next_waiter_run():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 3)
        order = []
        a, b, c, d = await run_and_queue(limiter, order, "a", "b", "c", "d")

        b.task.cancel()
        await b.outcome()
        assert state(limiter) == (1, 2)

        a.release()
        await c.started_within()
        c.release()
        await d.started_within()
        d.release()
        await asyncio.gather(a.task, c.task, d.task)

        assert order == ["a-start", "a-end", "c-start", "c-end", "d-start", "d-end"]
        assert state(limiter) == (0, 0)

    run(scenario)


def test_waiter_cancelled_right_after_being_woken_leaves_the_slot_free():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1)
        order = []
        waiters = []

        async def cancel_the_waiter():
            assert state(limiter) == (0, 1)
            waiters[0].task.cancel()

        holder = await hold_until_queued(limiter, 1, cancel_the_waiter)
        waiters.append(Held(limiter, "b", order))
        await holder

        assert isinstance(await waiters[0].outcome(), asyncio.CancelledError)
        assert state(limiter) == (0, 0)
        await run_one(limiter, "c", order)  # 슬롯이 주인 없이 사라지지 않았다
        assert state(limiter) == (0, 0)

    run(scenario)


@pytest.mark.parametrize("reverse_wake_up", [False, True], ids=["arrival-order", "reversed"])
def test_cancelling_a_woken_head_starts_the_next_waiter_without_another_release(reverse_wake_up):
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 2)
        order = []
        waiters = []

        async def cancel_the_head():
            assert state(limiter) == (0, 2)
            waiters[0].task.cancel()

        holder = await hold_until_queued(limiter, 2, cancel_the_head, reverse_wake_up)
        waiters.append(Held(limiter, "b", order))
        await wait_until(lambda: limiter.waiting == 1)
        waiters.append(Held(limiter, "c", order))
        await holder

        await waiters[1].started_within()  # 다른 요청이 끝나지 않아도 c 가 시작한다
        waiters[1].release()
        await asyncio.gather(*(waiter.task for waiter in waiters), return_exceptions=True)

        assert order == ["c-start", "c-end"]
        assert state(limiter) == (0, 0)

    run(scenario)


def test_cancel_during_the_release_still_returns_the_slot_and_wakes_the_waiter():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1)
        order = []
        a, b = await run_and_queue(limiter, order, "a", "b")

        # lock 을 잡아 a 가 반환 단계에서 lock 을 기다리게 한 뒤 취소를 두 번 보낸다.
        async with asyncio.timeout(TIMEOUT):
            await limiter._condition.acquire()
        try:
            a.release()
            await wait_until(lambda: "a-end" in order)
            a.task.cancel()
            await asyncio.sleep(0)
            a.task.cancel()
            await asyncio.sleep(0)
            assert not a.task.done()
            assert limiter.running == 1  # 아직 반환 전
        finally:
            limiter._condition.release()

        assert isinstance(await a.outcome(), asyncio.CancelledError)  # 취소는 삼키지 않는다
        await b.started_within()
        assert state(limiter) == (1, 0)
        b.release()
        await b.outcome()
        assert state(limiter) == (0, 0)

    run(scenario)


@pytest.mark.parametrize("offset", [-0.002, -0.001, 0.0, 0.001, 0.002])
def test_cancel_racing_the_queue_timeout_leaves_the_state_consistent(offset):
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1)
        order = []
        a = Held(limiter, "a", order)
        await a.started_within()
        outcomes = set()
        loop = asyncio.get_running_loop()
        for _ in range(30):

            async def wait_in_queue():
                async with limiter.slot(max_wait=0.003):
                    order.append("never")

            waiter = asyncio.create_task(wait_in_queue())
            await wait_until(lambda: limiter.waiting == 1)
            loop.call_later(0.003 + offset, waiter.cancel)
            done = await asyncio.gather(waiter, return_exceptions=True)
            outcomes.add(type(done[0]))
            assert state(limiter) == (1, 0)

        assert outcomes <= {SlotWaitTimeout, asyncio.CancelledError}
        assert "never" not in order
        a.release()
        await a.outcome()
        await run_one(limiter, "next", order)
        assert state(limiter) == (0, 0)

    run(scenario)


def test_cancel_delivered_with_the_timeout_is_not_swallowed(monkeypatch):
    real_on_timeout = asyncio.timeouts.Timeout._on_timeout

    def timeout_then_cancel(timeout):
        real_on_timeout(timeout)  # 대기 제한이 끝난 그 step 에 바깥 취소도 온다
        timeout._task.cancel()

    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1)
        order = []
        a = Held(limiter, "a", order)
        await a.started_within()
        monkeypatch.setattr(asyncio.timeouts.Timeout, "_on_timeout", timeout_then_cancel)

        async def wait_in_queue():
            async with limiter.slot(max_wait=0.01):
                order.append("never")

        done = await asyncio.gather(wait_in_queue(), return_exceptions=True)

        assert isinstance(done[0], asyncio.CancelledError)  # 429 로 바뀌지 않는다
        assert state(limiter) == (1, 0)
        a.release()
        await a.outcome()
        await run_one(limiter, "next", order)
        assert state(limiter) == (0, 0)

    run(scenario)


# --- 예외 ---


def test_exception_in_the_running_request_lets_the_waiter_run():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1)
        order = []
        a = Held(limiter, "a", order, error=ValueError("worker failed"))
        await a.started_within()
        b = Held(limiter, "b", order)
        await wait_until(lambda: limiter.waiting == 1)

        a.release()

        assert isinstance(await a.outcome(), ValueError)
        await b.started_within()
        b.release()
        await b.outcome()
        await run_one(limiter, "c", order)
        assert state(limiter) == (0, 0)

    run(scenario)


def test_rejected_requests_never_return_a_slot_they_did_not_hold():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 0)
        async with limiter.slot():
            for _ in range(5):
                with pytest.raises(ServerBusyError):
                    async with limiter.slot():
                        pass
            assert state(limiter) == (1, 0)  # 거절이 반환으로 이어지면 running 이 줄어든다

        assert state(limiter) == (0, 0)

    run(scenario)


# --- FIFO ---


def test_waiters_run_in_arrival_order():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 5)
        order = []
        names = ["a", "b", "c", "d", "e", "f"]
        requests = await run_and_queue(limiter, order, *names)

        for request in requests:
            await request.started_within()
            request.release()
        await asyncio.gather(*(request.task for request in requests))

        assert order == [f"{name}-{edge}" for name in names for edge in ("start", "end")]

    run(scenario)


def test_order_comes_from_the_queue_not_from_the_notification_order():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 3)
        order = []
        requests = await run_and_queue(limiter, order, "a", "b", "c", "d")
        reverse_wake_up_order(limiter)

        for request in requests:
            await request.started_within()
            request.release()
        await asyncio.gather(*(request.task for request in requests))

        assert [entry for entry in order if entry.endswith("start")] == [
            "a-start", "b-start", "c-start", "d-start",
        ]

    run(scenario)


# --- max_running=2 ---


def test_two_run_at_the_same_time_and_the_third_waits():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(2, 1)
        order = []
        a = Held(limiter, "a", order)
        b = Held(limiter, "b", order)
        await a.started_within()
        await b.started_within()
        c = Held(limiter, "c", order)
        await wait_until(lambda: limiter.waiting == 1)
        assert state(limiter) == (2, 1)

        with pytest.raises(ServerBusyError):
            async with limiter.slot():
                pass

        a.release()
        await c.started_within()
        assert state(limiter) == (2, 0)
        b.release()
        c.release()
        await asyncio.gather(a.task, b.task, c.task)
        assert state(limiter) == (0, 0)

    run(scenario)


@pytest.mark.parametrize("reverse_wake_up", [False, True], ids=["arrival-order", "reversed"])
def test_two_free_slots_start_two_waiters_in_order(reverse_wake_up):
    async def scenario():
        limiter = AsyncBodyValidationLimiter(2, 3)
        order = []
        a = Held(limiter, "a", order)
        b = Held(limiter, "b", order)
        await a.started_within()
        await b.started_within()
        c = Held(limiter, "c", order)
        await wait_until(lambda: limiter.waiting == 1)
        d = Held(limiter, "d", order)
        await wait_until(lambda: limiter.waiting == 2)
        e = Held(limiter, "e", order)
        await wait_until(lambda: limiter.waiting == 3)
        if reverse_wake_up:
            reverse_wake_up_order(limiter)

        a.release()
        b.release()  # 슬롯 둘이 함께 빈다. 앞 대기자가 슬롯을 가져간 뒤 다음 대기자도 깨어나야 한다.
        await c.started_within()
        await d.started_within()
        assert state(limiter) == (2, 1)
        assert not e.started.is_set()

        for request in (c, d):
            request.release()
        await e.started_within()
        e.release()
        await asyncio.gather(a.task, b.task, c.task, d.task, e.task)

        starts = [entry for entry in order if entry.endswith("start")]
        assert starts == ["a-start", "b-start", "c-start", "d-start", "e-start"]
        assert state(limiter) == (0, 0)

    run(scenario)


# --- 상태 정합성 ---


@pytest.mark.parametrize("max_running", [1, 2])
@pytest.mark.parametrize("seed", range(5))
def test_state_stays_consistent_under_a_mix_of_every_outcome(max_running, seed):
    rng = random.Random(seed)
    max_waiting = 2

    async def scenario():
        limiter = AsyncBodyValidationLimiter(max_running, max_waiting, max_wait_seconds=0.02)
        seen = []
        outcomes = []
        running_now = [0]
        most = [0]

        def observe():
            seen.append(state(limiter))

        async def request(delay, work, fail):
            await asyncio.sleep(delay)
            async with limiter.slot():
                running_now[0] += 1
                most[0] = max(most[0], running_now[0])
                observe()
                try:
                    await asyncio.sleep(work)
                    if fail:
                        raise ValueError("failed")
                finally:
                    running_now[0] -= 1

        tasks = [
            asyncio.create_task(
                request(rng.uniform(0, 0.03), rng.uniform(0, 0.005), rng.random() < 0.2)
            )
            for _ in range(60)
        ]
        loop = asyncio.get_running_loop()
        for task in tasks:
            if rng.random() < 0.3:
                loop.call_later(rng.uniform(0, 0.04), task.cancel)

        async def watch():
            while True:
                observe()
                await asyncio.sleep(0)

        watcher = asyncio.create_task(watch())
        async with asyncio.timeout(TIMEOUT):
            outcomes.extend(await asyncio.gather(*tasks, return_exceptions=True))
        watcher.cancel()

        assert all(0 <= r <= max_running and 0 <= w <= max_waiting for r, w in seen)
        assert most[0] <= max_running
        assert state(limiter) == (0, 0)
        kinds = {type(outcome) for outcome in outcomes}
        assert kinds <= {
            type(None), ValueError, ServerBusyError, SlotWaitTimeout, asyncio.CancelledError,
        }
        async with limiter.slot():  # 끝난 뒤에도 그대로 쓸 수 있다
            assert state(limiter) == (1, 0)

    run(scenario)


# --- event loop 와 lock ---


def test_waiting_does_not_block_the_event_loop_or_hold_the_lock():
    async def scenario():
        limiter = AsyncBodyValidationLimiter(1, 1, max_wait_seconds=0.05)
        order = []
        a = Held(limiter, "a", order)
        await a.started_within()
        assert not limiter._condition.locked()  # 실행 중에는 lock 을 잡지 않는다

        async def wait_in_queue():
            async with limiter.slot():
                pass

        waiter = asyncio.create_task(wait_in_queue())
        await wait_until(lambda: limiter.waiting == 1)
        ticks = 0
        while not waiter.done():
            ticks += 1
            assert not limiter._condition.locked()
            await asyncio.sleep(0.001)

        assert ticks > 5  # 대기하는 동안 다른 작업이 계속 돌았다
        assert isinstance(waiter.exception(), SlotWaitTimeout)
        a.release()
        await a.outcome()

    run(scenario)
