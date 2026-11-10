"""warm-up 접수 판단(singleflight·cooldown·재시도 간격·호출 수 상한)과 task 관리. 실제 Modal 은 부르지 않는다."""

import asyncio
import collections
import itertools
import logging
import time

import httpx
import pytest

from app.body_image_validation import warmup_manager
from app.body_image_validation.warmup import ModalWarmupClient, WarmupOutcome, WarmupResult
from app.body_image_validation.warmup_manager import (
    BodyImageWarmupManager,
    WarmupAcceptance,
    WarmupStatus,
    create_warmup_manager,
)
from app.config import body_image_validation as config

COOLDOWN, RETRY, TIMEOUT_RETRY = 20.0, 10.0, 30.0
SECRET = "ws-manager-secret"


def result(outcome=WarmupOutcome.SUCCESS, status_code=200):
    return WarmupResult(outcome, status_code, 0.1)


OK = result()
FAILED = result(WarmupOutcome.CONNECTION_FAILED, None)


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class FakeClient:
    """warm_up() 호출 수를 세고 정해 둔 결과(또는 예외)를 차례로 낸다. block 이면 release() 전까지 붙잡는다."""

    def __init__(self, *results, block=False):
        self.results = list(results) or [OK]
        self.block = block
        self.calls = 0
        self.cancelled = 0
        self.gate = asyncio.Event()

    async def warm_up(self):
        self.calls += 1
        outcome = self.results[min(self.calls, len(self.results)) - 1]
        try:
            if self.block:
                await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def release(self):
        self.gate.set()


def manager_for(client, clock=None, **overrides):
    options = {
        "cooldown": COOLDOWN,
        "retry_interval": RETRY,
        "timeout_retry_interval": TIMEOUT_RETRY,
        "max_starts": 100,
        "start_window": 60.0,
        "clock": clock or Clock(),
    } | overrides
    return BodyImageWarmupManager(client, **options)


async def settle(manager):
    """진행 중인 호출이 끝날 때까지 event loop 를 돌린다."""
    for _ in range(100):
        if not manager.in_progress:
            return
        await asyncio.sleep(0)
    raise AssertionError("warm-up task did not finish")


async def finished(manager):
    """요청 한 번을 접수시키고 호출이 끝날 때까지 기다린다."""
    acceptance = await manager.request()
    await settle(manager)
    return acceptance


def run(scenario):
    """scenario 를 새 event loop 에서 돌리고, loop 에 보고된 미처리 task 예외를 함께 돌려준다."""
    errors = []

    async def main():
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: errors.append(context)
        )
        return await scenario()

    value = asyncio.run(main())
    return value, errors


# --- singleflight ---


def test_ten_concurrent_requests_make_one_call():
    client = FakeClient(block=True)
    manager = manager_for(client)

    async def scenario():
        acceptances = await asyncio.gather(*(manager.request() for _ in range(10)))
        tasks = [t for t in asyncio.all_tasks() if t.get_name() == "body-image-warmup"]
        await asyncio.sleep(0)
        calls_while_running = client.calls
        client.release()
        await settle(manager)
        return acceptances, len(tasks), calls_while_running

    (acceptances, task_count, calls_while_running), errors = run(scenario)

    statuses = [a.status for a in acceptances]
    assert statuses.count(WarmupStatus.STARTED) == 1
    assert statuses.count(WarmupStatus.IN_PROGRESS) == 9
    assert all(a.accepted for a in acceptances)
    assert task_count == 1
    assert calls_while_running == client.calls == 1
    assert errors == []


def test_request_during_a_running_call_reuses_it():
    client = FakeClient(block=True)
    manager = manager_for(client)

    async def scenario():
        first = await manager.request()
        await asyncio.sleep(0)
        second = await manager.request()
        client.release()
        await settle(manager)
        third = await manager.request()
        return first, second, third

    (first, second, third), _ = run(scenario)

    assert first.status is WarmupStatus.STARTED
    assert second == WarmupAcceptance(WarmupStatus.IN_PROGRESS)
    assert third.status is WarmupStatus.COOLDOWN
    assert client.calls == 1


# --- task 관리 ---


def test_request_returns_before_the_call_finishes():
    client = FakeClient(block=True)
    manager = manager_for(client)

    async def scenario():
        acceptance = await manager.request()
        await asyncio.sleep(0)  # task 가 warm_up() 에 들어가 gate 에서 멈춘다
        state = (client.calls, manager.in_progress, manager.last_result)
        client.release()
        await settle(manager)
        return acceptance, state

    (acceptance, (calls, in_progress, last_result)), _ = run(scenario)

    assert acceptance.status is WarmupStatus.STARTED
    assert (calls, in_progress, last_result) == (1, True, None)
    assert manager.in_progress is False
    assert manager.last_result == OK


def test_unexpected_client_error_is_contained_and_rate_limited(caplog):
    clock = Clock()
    manager = manager_for(FakeClient(RuntimeError(f"boom {SECRET}")), clock)

    async def scenario():
        first = await finished(manager)
        clock.now += RETRY - 0.1
        blocked = await manager.request()
        return first, blocked

    with caplog.at_level(logging.INFO, logger=warmup_manager.__name__):
        (first, blocked), errors = run(scenario)

    assert first.status is WarmupStatus.STARTED
    assert blocked.status is WarmupStatus.RATE_LIMITED
    assert manager.last_result is None and manager.in_progress is False
    assert errors == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("error=RuntimeError" in m for m in messages)
    assert all(SECRET not in m for m in messages)
    assert all(r.exc_info is None for r in caplog.records)


def test_aclose_cancels_the_running_call_without_recording_it():
    client = FakeClient(block=True)
    manager = manager_for(client)

    async def scenario():
        await manager.request()
        await asyncio.sleep(0)
        await manager.aclose()
        # aclose 가 돌아온 시점에 취소가 끝나 있어야 한다(asyncio.run 정리에 맡기지 않는다).
        cancelled = client.cancelled
        left = [t for t in asyncio.all_tasks() if t.get_name() == "body-image-warmup"]
        after = await manager.request()
        await manager.aclose()  # 두 번 불러도 된다
        return after, cancelled, left

    (after, cancelled, left), errors = run(scenario)

    assert cancelled == 1
    assert left == []
    assert manager.in_progress is False
    assert manager.last_result is None  # 취소는 성공도 실패도 아니다
    assert after == WarmupAcceptance(WarmupStatus.UNAVAILABLE)
    assert not after.accepted
    assert errors == []


def test_aclose_before_the_task_runs_leaves_nothing_behind():
    client = FakeClient()
    manager = manager_for(client)

    async def scenario():
        await manager.request()
        await manager.aclose()  # task 가 한 번도 실행되기 전
        return await manager.request()

    after, errors = run(scenario)

    assert client.calls == 0
    assert manager.in_progress is False
    assert after.status is WarmupStatus.UNAVAILABLE
    assert errors == []


def test_aclose_without_a_task_stops_new_requests():
    manager = manager_for(FakeClient())

    async def scenario():
        await manager.aclose()
        return await manager.request()

    after, _ = run(scenario)

    assert after.status is WarmupStatus.UNAVAILABLE


def test_a_cancelled_call_does_not_start_the_success_cooldown():
    client = FakeClient(block=True)
    manager = manager_for(client)

    async def scenario():
        await manager.request()
        await asyncio.sleep(0)
        await manager.aclose()

    run(scenario)

    assert manager._succeeded_at is None and manager._retry_not_before is None


# --- cooldown ---


def test_success_starts_a_cooldown_and_then_allows_a_new_call():
    clock = Clock()
    client = FakeClient()
    manager = manager_for(client, clock)

    async def scenario():
        await finished(manager)
        clock.now += COOLDOWN - 0.5
        inside = await manager.request()
        clock.now += 0.5
        after = await finished(manager)
        return inside, after

    (inside, after), _ = run(scenario)

    assert inside.status is WarmupStatus.COOLDOWN and inside.accepted
    assert inside.retry_after == pytest.approx(0.5)
    assert after.status is WarmupStatus.STARTED
    assert client.calls == 2


def test_cooldown_counts_from_when_the_call_finished():
    clock = Clock()
    client = FakeClient(block=True)
    manager = manager_for(client, clock)

    async def scenario():
        await manager.request()
        await asyncio.sleep(0)
        clock.now += 15.0  # cold start 로 15초 걸린 호출
        client.release()
        await settle(manager)
        clock.now += COOLDOWN - 1.0  # 시작부터는 cooldown 이 지났지만 완료부터는 아님
        return await manager.request()

    acceptance, _ = run(scenario)

    assert acceptance.status is WarmupStatus.COOLDOWN


def test_failure_does_not_start_a_cooldown():
    clock = Clock()
    client = FakeClient(FAILED)
    manager = manager_for(client, clock)

    async def scenario():
        await finished(manager)
        clock.now += 1.0
        return await manager.request()

    acceptance, _ = run(scenario)

    assert acceptance.status is WarmupStatus.RATE_LIMITED  # COOLDOWN 이 아니다
    assert not acceptance.accepted


# --- 실패 후 재시도 간격 ---


@pytest.mark.parametrize(
    "outcome",
    [
        WarmupOutcome.CONNECTION_FAILED,
        WarmupOutcome.AUTH_FAILED,
        WarmupOutcome.NOT_FOUND,
        WarmupOutcome.SERVER_ERROR,
        WarmupOutcome.REDIRECT,
        WarmupOutcome.UNEXPECTED_STATUS,
        WarmupOutcome.INVALID_JSON,
        WarmupOutcome.UNEXPECTED_BODY,
    ],
)
def test_failure_blocks_retries_for_the_minimum_interval(outcome):
    clock = Clock()
    client = FakeClient(result(outcome, 503), OK)
    manager = manager_for(client, clock)

    async def scenario():
        await finished(manager)
        clock.now += RETRY - 0.5
        blocked = await manager.request()
        clock.now += 0.5
        retried = await finished(manager)
        return blocked, retried

    (blocked, retried), _ = run(scenario)

    assert blocked.status is WarmupStatus.RATE_LIMITED
    assert blocked.retry_after == pytest.approx(0.5)
    assert retried.status is WarmupStatus.STARTED
    assert client.calls == 2


@pytest.mark.parametrize(
    "outcome", [WarmupOutcome.NETWORK_TIMEOUT, WarmupOutcome.DEADLINE_EXCEEDED]
)
def test_timeouts_use_the_longer_retry_interval(outcome):
    clock = Clock()
    client = FakeClient(result(outcome, None))
    manager = manager_for(client, clock)

    async def scenario():
        await finished(manager)
        clock.now += RETRY  # 일반 실패라면 이미 다시 부를 수 있는 시점
        at_normal = await manager.request()
        clock.now += TIMEOUT_RETRY - RETRY - 0.1
        just_before = await manager.request()
        clock.now += 0.1
        after = await manager.request()
        return at_normal, just_before, after

    (at_normal, just_before, after), _ = run(scenario)

    assert at_normal.status is WarmupStatus.RATE_LIMITED
    assert at_normal.retry_after == pytest.approx(TIMEOUT_RETRY - RETRY)
    assert just_before.status is WarmupStatus.RATE_LIMITED
    assert after.status is WarmupStatus.STARTED


def test_success_after_a_failure_switches_to_the_cooldown():
    clock = Clock()
    client = FakeClient(FAILED, OK)
    manager = manager_for(client, clock)

    async def scenario():
        await finished(manager)
        clock.now += RETRY
        await finished(manager)
        clock.now += 1.0
        return await manager.request()

    acceptance, _ = run(scenario)

    assert acceptance.status is WarmupStatus.COOLDOWN
    assert manager._retry_not_before is None


# --- 호출 수 상한 ---


def test_start_cap_limits_new_calls_within_the_window():
    clock = Clock()
    client = FakeClient()
    manager = manager_for(
        client, clock, cooldown=1.0, retry_interval=1.0, max_starts=2, start_window=60.0
    )

    async def scenario():
        first = await finished(manager)  # t=0
        clock.now += 2.0
        second = await finished(manager)  # t=2
        clock.now += 2.0
        capped = await manager.request()  # t=4: 60초 안에 이미 2번
        clock.now += 56.0
        reopened = await finished(manager)  # t=60: 첫 시작이 창 밖으로 나갔다
        return first, second, capped, reopened

    (first, second, capped, reopened), _ = run(scenario)

    assert [first.status, second.status, reopened.status] == [WarmupStatus.STARTED] * 3
    assert capped.status is WarmupStatus.RATE_LIMITED
    assert capped.retry_after == pytest.approx(56.0)
    assert client.calls == 3


def test_reused_and_skipped_requests_are_not_counted():
    clock = Clock()
    client = FakeClient(block=True)
    manager = manager_for(client, clock, max_starts=1, start_window=60.0)

    async def scenario():
        await manager.request()
        for _ in range(5):
            await manager.request()  # IN_PROGRESS
        client.release()
        await settle(manager)
        for _ in range(5):
            await manager.request()  # COOLDOWN
        return len(manager._starts)

    starts, _ = run(scenario)

    assert starts == 1


def test_start_history_stays_bounded():
    clock = Clock()
    manager = manager_for(FakeClient(), clock, cooldown=1.0, max_starts=3, start_window=10.0)

    async def scenario():
        for _ in range(50):
            await finished(manager)
            clock.now += 5.0

    run(scenario)

    assert len(manager._starts) <= 3


# --- 실제 HTTP client 와 함께 ---


def test_concurrent_requests_send_one_health_call_through_the_real_client():
    calls = []

    async def handle(request):
        calls.append(request)
        await asyncio.sleep(0.01)
        return httpx.Response(200, json={"status": "ok"})

    client = ModalWarmupClient(
        "https://example--lookddak-body-image-staging.modal.run",
        proxy_key="wk-test",
        proxy_secret=SECRET,
        timeout=5.0,
        transport=httpx.MockTransport(handle),
    )
    manager = manager_for(client)

    async def scenario():
        acceptances = await asyncio.gather(*(manager.request() for _ in range(5)))
        await asyncio.wait_for(_wait_idle(manager), timeout=5.0)
        return acceptances, await manager.request()

    (acceptances, after), errors = run(scenario)

    assert len(calls) == 1
    assert [a.status for a in acceptances].count(WarmupStatus.STARTED) == 1
    assert manager.last_result.succeeded
    assert after.status is WarmupStatus.COOLDOWN
    assert errors == []


async def _wait_idle(manager):
    while manager.in_progress:
        await asyncio.sleep(0.005)


# --- 설정 ---


def test_factory_uses_the_code_constants():
    manager = create_warmup_manager(FakeClient())

    assert manager._cooldown == config.WARMUP_COOLDOWN_SECONDS == 20.0
    assert manager._retry_interval == config.WARMUP_MIN_RETRY_INTERVAL_SECONDS == 10.0
    assert manager._timeout_retry_interval == config.WARMUP_TIMEOUT_RETRY_INTERVAL_SECONDS == 30.0
    assert manager._starts.maxlen == config.WARMUP_MAX_STARTS_PER_WINDOW
    assert manager._start_window == config.WARMUP_START_WINDOW_SECONDS


def test_timeout_retry_is_longer_than_the_normal_retry():
    assert config.WARMUP_TIMEOUT_RETRY_INTERVAL_SECONDS > config.WARMUP_MIN_RETRY_INTERVAL_SECONDS


def test_cooldown_is_shorter_than_the_modal_scaledown_window():
    pytest.importorskip("modal")
    from tests import test_body_image_modal_worker as modal_worker_tests

    module = modal_worker_tests.load_production()

    assert config.WARMUP_COOLDOWN_SECONDS < module.WORKER_RESOURCES["scaledown_window"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"cooldown": 0},
        {"retry_interval": -1},
        {"timeout_retry_interval": 0},
        {"start_window": 0},
        {"max_starts": 0},
    ],
)
def test_invalid_limits_are_rejected(overrides):
    with pytest.raises(ValueError):
        manager_for(FakeClient(), **overrides)


# --- 경쟁 조건 (4단계) ---


class TrackingClient(FakeClient):
    """동시에 warm_up() 안에 있는 호출 수의 최댓값을 잰다."""

    def __init__(self, *results, block=False):
        super().__init__(*results, block=block)
        self.active = 0
        self.max_active = 0

    async def warm_up(self):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            return await super().warm_up()
        finally:
            self.active -= 1


def test_request_right_after_registration_reuses_the_task():
    client = FakeClient(block=True)
    manager = manager_for(client)

    async def scenario():
        first = await manager.request()
        second = await manager.request()  # task 가 한 번도 실행되기 전
        client.release()
        await settle(manager)
        return first, second

    (first, second), errors = run(scenario)

    assert (first.status, second.status) == (WarmupStatus.STARTED, WarmupStatus.IN_PROGRESS)
    assert client.calls == 1
    assert errors == []


@pytest.mark.parametrize("ticks", [0, 1, 2, 3, 4])
def test_request_racing_with_completion_never_starts_a_second_call(ticks):
    client = FakeClient(block=True)
    manager = manager_for(client)

    async def scenario():
        await manager.request()
        await asyncio.sleep(0)
        client.release()  # 응답이 막 도착한 상태
        for _ in range(ticks):
            await asyncio.sleep(0)
        racing = await manager.request()
        await settle(manager)
        return racing, await manager.request()

    (racing, after), errors = run(scenario)

    assert racing.status in (WarmupStatus.IN_PROGRESS, WarmupStatus.COOLDOWN)
    assert after.status is WarmupStatus.COOLDOWN
    assert client.calls == 1
    assert errors == []


@pytest.mark.parametrize("close_first", [True, False])
def test_shutdown_racing_with_new_requests_leaves_no_task(close_first):
    client = FakeClient(block=True)
    manager = manager_for(client)

    async def scenario():
        requests = [manager.request() for _ in range(10)]
        calls = [manager.aclose(), *requests] if close_first else [*requests, manager.aclose()]
        outcomes = await asyncio.gather(*calls)
        left = [t for t in asyncio.all_tasks() if t.get_name() == "body-image-warmup"]
        return [o for o in outcomes if o is not None], left, await manager.request()

    (acceptances, left, after), errors = run(scenario)

    statuses = {a.status for a in acceptances}
    assert statuses <= {WarmupStatus.STARTED, WarmupStatus.IN_PROGRESS, WarmupStatus.UNAVAILABLE}
    if close_first:
        assert statuses == {WarmupStatus.UNAVAILABLE} and client.calls == 0
    else:
        assert [a.status for a in acceptances].count(WarmupStatus.STARTED) == 1
    assert client.calls <= 1 and client.cancelled == client.calls
    assert left == [] and not manager.in_progress
    assert manager.last_result is None and manager._succeeded_at is None
    assert after.status is WarmupStatus.UNAVAILABLE
    assert errors == []


@pytest.mark.parametrize("ticks", [0, 1, 2, 3])
def test_cancel_racing_with_the_response_is_recorded_consistently(ticks):
    client = FakeClient(block=True)
    manager = manager_for(client)

    async def scenario():
        await manager.request()
        await asyncio.sleep(0)
        client.release()
        for _ in range(ticks):
            await asyncio.sleep(0)
        await manager.aclose()

    _, errors = run(scenario)

    # 응답이 먼저면 성공으로 남고, 취소가 먼저면 아무것도 남지 않는다. 그 사이 상태는 없다.
    if manager.last_result is None:
        assert client.cancelled == 1 and manager._succeeded_at is None
    else:
        assert manager.last_result == OK and manager._succeeded_at is not None
    assert manager._retry_not_before is None
    assert not manager.in_progress
    assert errors == []


def test_never_more_than_one_call_runs_under_random_interleavings():
    import random

    rng = random.Random(1234)
    clock = Clock()
    client = TrackingClient(
        OK, FAILED, OK, result(WarmupOutcome.DEADLINE_EXCEEDED, None), block=True
    )
    manager = manager_for(client, clock, max_starts=4, start_window=60.0)
    statuses = collections.Counter()

    async def scenario():
        for _ in range(500):
            action = rng.random()
            if action < 0.5:
                statuses[(await manager.request()).status] += 1
            elif action < 0.7:
                client.release()
                client.gate = asyncio.Event()  # 다음 호출은 다시 붙잡는다
            elif action < 0.85:
                clock.now += rng.choice([0.5, 5.0, 10.0, 31.0])
            else:
                await asyncio.sleep(0)
        client.release()
        await manager.aclose()

    _, errors = run(scenario)

    assert client.max_active == 1
    # 마지막 task 는 warm_up() 에 들어가기 전에 종료로 취소될 수 있다.
    assert statuses[WarmupStatus.STARTED] - 1 <= client.calls <= statuses[WarmupStatus.STARTED]
    assert statuses[WarmupStatus.STARTED] > 1  # 실제로 여러 번 시작·완료를 거쳤다
    assert errors == []


def test_aclose_cancels_a_real_http_call_promptly():
    entered = asyncio.Event()

    async def handle(request):
        entered.set()
        await asyncio.sleep(3600)
        return httpx.Response(200, json={"status": "ok"})

    client = ModalWarmupClient(
        "https://example--lookddak-body-image-staging.modal.run",
        proxy_key="wk-test",
        proxy_secret=SECRET,
        timeout=90.0,
        transport=httpx.MockTransport(handle),
    )
    manager = manager_for(client)

    async def scenario():
        await manager.request()
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        started = time.monotonic()
        await asyncio.wait_for(manager.aclose(), timeout=1.0)
        return time.monotonic() - started

    elapsed, errors = run(scenario)

    assert elapsed < 1.0
    assert manager.last_result is None and not manager.in_progress
    assert errors == []


def test_rate_limited_requests_do_not_log_at_info(caplog):
    clock = Clock()
    manager = manager_for(FakeClient(FAILED), clock)

    async def scenario():
        await finished(manager)
        return [await manager.request() for _ in range(50)]

    with caplog.at_level(logging.INFO, logger=warmup_manager.__name__):
        acceptances, _ = run(scenario)

    assert {a.status for a in acceptances} == {WarmupStatus.RATE_LIMITED}
    assert [r for r in caplog.records if r.name == warmup_manager.__name__] == []


def _starts_in_any_window(starts, window):
    return max(sum(1 for s in starts if t <= s < t + window) for t in starts)


@pytest.mark.parametrize(
    "outcome, expected_max",
    [
        (OK, 3),  # 성공만: cooldown 20초 → 60초에 최대 3번, 상한(4)에 닿지 않는다
        (FAILED, 4),  # 빠른 실패만: 재시도 10초로는 6번 → 상한 4번이 걸린다
        (result(WarmupOutcome.AUTH_FAILED, 401), 4),
    ],
)
def test_real_constants_bound_calls_per_window(outcome, expected_max):
    clock = Clock(0.0)
    client = FakeClient(outcome)
    manager = BodyImageWarmupManager(
        client,
        cooldown=config.WARMUP_COOLDOWN_SECONDS,
        retry_interval=config.WARMUP_MIN_RETRY_INTERVAL_SECONDS,
        timeout_retry_interval=config.WARMUP_TIMEOUT_RETRY_INTERVAL_SECONDS,
        max_starts=config.WARMUP_MAX_STARTS_PER_WINDOW,
        start_window=config.WARMUP_START_WINDOW_SECONDS,
        clock=clock,
    )
    starts, statuses = [], collections.Counter()

    async def scenario():
        while clock.now < 300.0:  # 5분 동안 1초마다 요청
            acceptance = await finished(manager)
            statuses[acceptance.status] += 1
            if acceptance.status is WarmupStatus.STARTED:
                starts.append(clock.now)
            clock.now += 1.0

    run(scenario)

    assert _starts_in_any_window(starts, config.WARMUP_START_WINDOW_SECONDS) == expected_max
    assert len(manager._starts) <= config.WARMUP_MAX_STARTS_PER_WINDOW
    if outcome.succeeded:
        assert statuses[WarmupStatus.RATE_LIMITED] == 0


@pytest.mark.parametrize(
    "outcome", [WarmupOutcome.NETWORK_TIMEOUT, WarmupOutcome.DEADLINE_EXCEEDED]
)
def test_after_a_timeout_the_next_call_is_at_least_timeout_plus_interval_apart(outcome):
    clock = Clock(0.0)
    client = FakeClient(result(outcome, None), block=True)
    manager = manager_for(client, clock)
    starts = []

    async def scenario():
        while clock.now < 400.0:
            acceptance = await manager.request()
            if acceptance.status is WarmupStatus.STARTED:
                starts.append(clock.now)
            await asyncio.sleep(0)
            if manager.in_progress and clock.now - starts[-1] >= config.WARMUP_TIMEOUT_SECONDS:
                client.release()  # 90초 만에 시간 초과로 끝난 호출
                await settle(manager)
                client.gate = asyncio.Event()
            clock.now += 1.0
        client.release()
        await manager.aclose()

    run(scenario)

    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert gaps and min(gaps) >= config.WARMUP_TIMEOUT_SECONDS + TIMEOUT_RETRY
