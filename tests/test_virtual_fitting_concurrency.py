"""가상피팅 VTON / LLM 동시 실행 제한.

실제 Runware / S3 는 부르지 않는다. 외부 호출 대역은 worker thread 안에서 threading.Event 를 기다려
"실행 중" 상태로 머물고, 테스트가 Event 를 열어 끝낸다. 큰 sleep 없이 상태 변화는 until 로 확인한다.
"""

import asyncio
import threading
from decimal import Decimal

import pytest
from anyio import to_thread

from app.config import settings
from app.virtual_fitting.concurrency import (
    ConcurrencyLimiter,
    FittingLimiters,
    create_fitting_limiters,
)
from app.virtual_fitting.exceptions import (
    FittingDailyBudgetExceededError,
    FittingModelError,
    FittingPostprocessError,
    FittingQueueTimeoutError,
    FittingServerBusyError,
)
from app.virtual_fitting.providers.base import FittingResult
from app.virtual_fitting.schemas import SyncFittingRequest
from app.virtual_fitting.service import FALLBACK_COMMENT, FALLBACK_TITLE, VirtualFittingService
from tests.virtual_fitting_fixtures import AllowAllBudget, make_bottom, make_top

TIMEOUT = 5.0
COST = Decimal("0.015")


async def until(predicate, timeout=TIMEOUT):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("상태가 기대값이 되지 않았다")
        await asyncio.sleep(0.001)


class Gates:
    """이름별 threading.Event. gated 가 아니면 바로 통과한다."""

    def __init__(self, gated=True):
        self._gated = gated
        self._events = {}
        self._lock = threading.Lock()

    def _event(self, name):
        with self._lock:
            return self._events.setdefault(name, threading.Event())

    def wait(self, name):
        if self._gated:
            assert self._event(name).wait(TIMEOUT), f"{name} 이 열리지 않았다"

    def open(self, name):
        self._event(name).set()

    def open_all(self):
        self._gated = False
        with self._lock:
            for event in self._events.values():
                event.set()


class CallTracker:
    """동시에 실행 중인 호출 수와 그 최대값, 시작·종료 순서를 기록한다(여러 thread 에서 불린다)."""

    def __init__(self):
        self.active = set()
        self.max_active = 0
        self.events = []
        self._lock = threading.Lock()

    def enter(self, name):
        with self._lock:
            self.active.add(name)
            self.max_active = max(self.max_active, len(self.active))
            self.events.append(f"{name}-start")

    def exit(self, name):
        with self._lock:
            self.active.discard(name)
            self.events.append(f"{name}-end")

    def started(self, name):
        return f"{name}-start" in self.events


def _name(url):
    return url.rsplit("/", 1)[-1].removesuffix(".jpg")


class GatedVton:
    def __init__(self, gated=True, errors=None):
        self.gates = Gates(gated)
        self.calls = CallTracker()
        self._errors = errors or {}

    def try_on(self, fitting_input):
        name = _name(fitting_input.person_image_url)
        self.calls.enter(name)
        try:
            self.gates.wait(name)
            if name in self._errors:
                raise self._errors[name]
            return FittingResult(result_image_url=f"https://result/{name}.jpg", cost=COST)
        finally:
            self.calls.exit(name)


class GatedComment:
    """comment / title 이 같은 tracker 를 쓴다. 이름은 'comment-A', 'title-A' 형태다."""

    def __init__(self, gated=True, errors=None):
        self.gates = Gates(gated)
        self.calls = CallTracker()
        self._errors = errors or {}

    def _call(self, name, text):
        self.calls.enter(name)
        try:
            self.gates.wait(name)
            if name in self._errors:
                raise self._errors[name]
            return text
        finally:
            self.calls.exit(name)

    def generate_comment(self, result_image_url, description_summaries):
        name = _name(result_image_url)
        return self._call(f"comment-{name}", f"comment {name}")

    def generate_title(self, comment):
        name = comment.removeprefix("comment ")
        return self._call(f"title-{name}", f"title {name}")


class FakeRepository:
    def __init__(self, *products):
        self._products = {int(p.product_code): p for p in products}

    def find_by_codes(self, product_codes):
        return [self._products[int(c)] for c in product_codes if int(c) in self._products]


class FakeImageStorage:
    def store_remote_image(self, url, key_prefix):
        return f"{key_prefix}/{_name(url)}.jpg"


def _service(vton=None, comment=None, budget=None):
    return VirtualFittingService(
        FakeRepository(make_top("1"), make_bottom("2")),
        vton or GatedVton(gated=False),
        comment or GatedComment(gated=False),
        FakeImageStorage(),
        budget or AllowAllBudget(),
    )


def _request(name):
    return SyncFittingRequest(
        user_image_url=f"https://example.com/{name}.jpg",
        products=[{"product_code": "1"}, {"product_code": "2"}],
    )


def _limiters(vton_running=4, vton_waiting=4, vton_timeout=TIMEOUT, llm_running=4):
    return FittingLimiters(
        vton=ConcurrencyLimiter(vton_running, max_waiting=vton_waiting, wait_timeout=vton_timeout),
        llm=ConcurrencyLimiter(llm_running),
    )


def _start(service, limiters, *names):
    return {
        name: asyncio.create_task(service.fit_with_limiters(_request(name), limiters))
        for name in names
    }


async def _start_in_order(service, limiters, vton, *names):
    """prepare 는 worker thread 에서 끝나는 순서가 정해져 있지 않다. 하나씩 VTON 실행·대기에 들어간
    것을 확인하고 다음을 시작해, 슬롯을 얻는 순서를 이름 순서로 고정한다."""
    tasks = {}
    for name in names:
        waiting = limiters.vton.waiting
        tasks |= _start(service, limiters, name)
        await until(
            lambda name=name, waiting=waiting: (
                vton.calls.started(name) or limiters.vton.waiting > waiting
            )
        )
    return tasks


@pytest.fixture
def configured_limiters(monkeypatch):
    """운영과 같은 create_fitting_limiters 로 만든다. LLM 슬롯 대기 시간 초과의 예외 변환까지 본다."""

    def make(vton_running=4, vton_waiting=4, llm_running=4, llm_wait_timeout=TIMEOUT):
        monkeypatch.setattr(settings, "VIRTUAL_FITTING_VTON_CONCURRENCY", vton_running)
        monkeypatch.setattr(settings, "VIRTUAL_FITTING_VTON_QUEUE_SIZE", vton_waiting)
        monkeypatch.setattr(settings, "VIRTUAL_FITTING_VTON_QUEUE_TIMEOUT_SECONDS", TIMEOUT)
        monkeypatch.setattr(settings, "VIRTUAL_FITTING_LLM_CONCURRENCY", llm_running)
        monkeypatch.setattr(
            settings, "VIRTUAL_FITTING_LLM_SLOT_WAIT_TIMEOUT_SECONDS", llm_wait_timeout
        )
        return create_fitting_limiters()

    return make


class SlotAwareBudget(AllowAllBudget):
    """확인 시점의 VTON running 수를 기록한다. exceed_first 면 첫 확인을 gate 가 열릴 때까지 붙잡았다가 402."""

    def __init__(self, limiters, events=None, exceed_first=False):
        super().__init__()
        self._limiters = limiters
        self.events = events if events is not None else []
        self._exceed_first = exceed_first
        self.gate = threading.Event()

    def ensure_available(self):
        super().ensure_available()
        self.events.append(f"budget(vton running={self._limiters.vton.running})")
        if self._exceed_first and self.checks == 1:
            assert self.gate.wait(TIMEOUT)
            raise FittingDailyBudgetExceededError("오늘 사용액이 상한에 닿았습니다.")


def _borrowed_worker_threads():
    return to_thread.current_default_thread_limiter().borrowed_tokens


# --- ConcurrencyLimiter 단위 ---


def test_invalid_limits_are_rejected():
    for kwargs in (
        {"max_running": 0},
        {"max_running": 1, "max_waiting": -1},
        {"max_running": 1, "wait_timeout": 0},
        {"max_running": 1, "wait_timeout": -1},
    ):
        with pytest.raises(ValueError):
            ConcurrencyLimiter(**kwargs)


def test_limiters_are_built_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_VTON_CONCURRENCY", 2)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_VTON_QUEUE_SIZE", 0)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_VTON_QUEUE_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_LLM_CONCURRENCY", 3)

    limiters = create_fitting_limiters()

    assert limiters.vton is not limiters.llm

    async def run():
        # VTON: 2개 실행, 대기 0 → 3번째는 즉시 거절
        async with limiters.vton.slot(), limiters.vton.slot():
            with pytest.raises(FittingServerBusyError):
                async with limiters.vton.slot():
                    pass
        # LLM: 3개까지 바로 실행된다
        async with limiters.llm.slot(), limiters.llm.slot(), limiters.llm.slot():
            assert limiters.llm.running == 3

    asyncio.run(run())


def test_default_policy_values():
    assert settings.VIRTUAL_FITTING_VTON_CONCURRENCY == 4
    assert settings.VIRTUAL_FITTING_VTON_QUEUE_SIZE == 4
    assert settings.VIRTUAL_FITTING_VTON_QUEUE_TIMEOUT_SECONDS == 30
    assert settings.VIRTUAL_FITTING_LLM_CONCURRENCY == 4
    assert settings.VIRTUAL_FITTING_LLM_SLOT_WAIT_TIMEOUT_SECONDS == 15


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_llm_slot_wait_timeout_is_rejected(monkeypatch, value):
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_LLM_SLOT_WAIT_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError):
        create_fitting_limiters()


def test_llm_slot_wait_timeout_is_a_postprocess_failure_and_vton_timeout_stays_429(monkeypatch):
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_VTON_CONCURRENCY", 1)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_VTON_QUEUE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_LLM_CONCURRENCY", 1)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_LLM_SLOT_WAIT_TIMEOUT_SECONDS", 0.01)
    limiters = create_fitting_limiters()

    async def run():
        async with limiters.llm.slot():
            with pytest.raises(FittingPostprocessError):
                async with limiters.llm.slot():
                    pass
        async with limiters.vton.slot():
            with pytest.raises(FittingQueueTimeoutError):
                async with limiters.vton.slot():
                    pass

    asyncio.run(run())

    assert (limiters.llm.running, limiters.llm.waiting) == (0, 0)
    assert (limiters.vton.running, limiters.vton.waiting) == (0, 0)


def test_cancelled_waiter_leaves_the_queue_and_the_slot_count_unchanged():
    limiter = ConcurrencyLimiter(1, max_waiting=1)

    async def run():
        hold = asyncio.Event()

        async def holder():
            async with limiter.slot():
                await hold.wait()

        holding = asyncio.create_task(holder())
        await until(lambda: limiter.running == 1)
        waiting = asyncio.create_task(limiter.slot().__aenter__())
        await until(lambda: limiter.waiting == 1)

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert (limiter.running, limiter.waiting) == (1, 0)

        hold.set()
        await holding
        assert (limiter.running, limiter.waiting) == (0, 0)

    asyncio.run(run())


def test_slot_handed_over_at_the_moment_of_cancellation_goes_to_the_next_waiter():
    """반환된 슬롯을 넘겨받은 대기자가 깨어나기 전에 취소되면, 그 슬롯은 새지 않고 다음 대기자에게 간다."""
    limiter = ConcurrencyLimiter(1, max_waiting=2)
    order = []

    async def run():
        async def waiter(name):
            async with limiter.slot():
                order.append(name)

        held = limiter.slot()
        await held.__aenter__()
        b = asyncio.create_task(waiter("B"))
        await until(lambda: limiter.waiting == 1)
        c = asyncio.create_task(waiter("C"))
        await until(lambda: limiter.waiting == 2)

        # 반환(B 에게 인계)과 B 취소 사이에 이벤트 루프로 돌아가지 않는다. B 는 아직 깨어나지 않았다.
        await held.__aexit__(None, None, None)
        assert limiter.running == 1 and limiter.waiting == 1
        b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await b
        await c

        assert order == ["C"]
        assert (limiter.running, limiter.waiting) == (0, 0)

    asyncio.run(run())


def test_released_slot_goes_to_the_oldest_waiter_not_a_newcomer():
    limiter = ConcurrencyLimiter(1, max_waiting=2)
    order = []

    async def run():
        hold = asyncio.Event()

        async def use(name, event=None):
            async with limiter.slot():
                order.append(name)
                if event is not None:
                    await event.wait()

        a = asyncio.create_task(use("A", hold))
        await until(lambda: limiter.running == 1)
        b = asyncio.create_task(use("B"))
        await until(lambda: limiter.waiting == 1)

        hold.set()
        late = asyncio.create_task(use("late"))  # A 반환과 같은 차례에 들어온 새 요청
        await asyncio.gather(a, b, late)

        assert order == ["A", "B", "late"]
        assert (limiter.running, limiter.waiting) == (0, 0)

    asyncio.run(run())


# --- VTON: 실행 4 / 대기 4 / 9번째 429 ---


def test_vton_runs_at_most_4_queues_4_and_rejects_the_9th_without_calling_runware():
    vton = GatedVton()
    budget = AllowAllBudget()
    service = _service(vton=vton, budget=budget)
    limiters = _limiters()

    async def run():
        try:
            tasks = await _start_in_order(service, limiters, vton, *"ABCDEFGH")
            await until(lambda: len(vton.calls.active) == 4 and limiters.vton.waiting == 4)

            assert vton.calls.active == {"A", "B", "C", "D"}
            assert limiters.vton.running == 4
            # 사용액 확인은 VTON 슬롯을 얻은 4개만 했다. 대기 중인 E~H 는 아직 하지 않는다.
            assert budget.checks == 4
            # 대기 중인 E~H 는 worker thread 를 잡지 않는다. 지금 빌려 간 thread 는 실행 중인 VTON 4개뿐이다.
            assert _borrowed_worker_threads() == 4

            with pytest.raises(FittingServerBusyError):
                await service.fit_with_limiters(_request("I"), limiters)
            assert not vton.calls.started("I")
            assert limiters.vton.waiting == 4
            assert budget.checks == 4  # 거절된 I 는 사용액 확인도 하지 않았다

            # 하나가 끝나면 가장 먼저 기다린 E 가 바로 실행된다.
            vton.gates.open("A")
            await until(lambda: vton.calls.started("E"))
            assert vton.calls.active == {"B", "C", "D", "E"}
            assert limiters.vton.waiting == 3

            vton.gates.open_all()
            results = await asyncio.gather(*tasks.values())
        finally:
            vton.gates.open_all()

        assert [r.llm_title for r in results] == [f"title {n}" for n in "ABCDEFGH"]
        assert vton.calls.max_active == 4
        assert (limiters.vton.running, limiters.vton.waiting) == (0, 0)
        # 거절된 I 의 비용은 기록되지 않고, 사용액 확인도 슬롯을 얻은 8개만 했다.
        assert budget.recorded == [COST] * 8
        assert budget.checks == 8

    asyncio.run(run())


def test_queue_timeout_is_429_without_calling_runware_or_recording_cost():
    vton = GatedVton()
    budget = AllowAllBudget()
    service = _service(vton=vton, budget=budget)
    limiters = _limiters(vton_running=1, vton_waiting=1, vton_timeout=0.05)

    async def run():
        try:
            a = (await _start_in_order(service, limiters, vton, "A"))["A"]

            with pytest.raises(FittingQueueTimeoutError):
                await service.fit_with_limiters(_request("B"), limiters)

            assert not vton.calls.started("B")
            assert (limiters.vton.running, limiters.vton.waiting) == (1, 0)
            assert budget.recorded == []
            assert budget.checks == 1  # A 만 확인했다. 시간 초과된 B 는 사용액 확인도 하지 않았다

            vton.gates.open("A")
            await a
        finally:
            vton.gates.open_all()

        assert (limiters.vton.running, limiters.vton.waiting) == (0, 0)
        assert budget.recorded == [COST]

    asyncio.run(run())


# --- 하루 사용액 확인은 VTON 슬롯 안에서, try_on 직전에 한다 ---


def test_budget_is_checked_after_the_vton_slot_is_acquired_and_before_try_on():
    events = []
    limiters = _limiters(vton_running=1)
    budget = SlotAwareBudget(limiters, events)

    class OrderedVton(GatedVton):
        def try_on(self, fitting_input):
            events.append(f"try_on(vton running={limiters.vton.running})")
            return super().try_on(fitting_input)

    service = _service(vton=OrderedVton(gated=False), budget=budget)

    asyncio.run(service.fit_with_limiters(_request("A"), limiters))

    assert events == ["budget(vton running=1)", "try_on(vton running=1)"]
    assert budget.recorded == [COST]


def test_prepare_no_longer_checks_the_budget():
    budget = AllowAllBudget()

    _service(budget=budget).prepare(_request("A"))

    assert budget.checks == 0


def test_exceeded_budget_skips_try_on_and_record_and_hands_the_slot_to_the_next_waiter():
    vton = GatedVton(gated=False)
    limiters = _limiters(vton_running=1, vton_waiting=1)
    budget = SlotAwareBudget(limiters, exceed_first=True)
    service = _service(vton=vton, budget=budget)

    async def run():
        try:
            a = _start(service, limiters, "A")["A"]
            await until(lambda: budget.checks == 1)  # A 가 슬롯을 얻고 사용액을 확인하는 중
            b = _start(service, limiters, "B")["B"]
            await until(lambda: limiters.vton.waiting == 1)

            budget.gate.set()
            with pytest.raises(FittingDailyBudgetExceededError):
                await a
            result = await b
        finally:
            budget.gate.set()
        return result

    result = asyncio.run(run())

    # A: try_on X, record X. A 가 반환한 슬롯을 B 가 받아 정상 처리했다.
    assert not vton.calls.started("A")
    assert vton.calls.started("B")
    assert result.llm_title == "title B"
    assert budget.checks == 2
    assert budget.recorded == [COST]
    assert (limiters.vton.running, limiters.vton.waiting) == (0, 0)


def test_vton_error_still_returns_the_slot_to_the_waiting_request():
    vton = GatedVton(errors={"A": FittingModelError("boom")})
    budget = AllowAllBudget()
    service = _service(vton=vton, budget=budget)
    limiters = _limiters(vton_running=1, vton_waiting=1)

    async def run():
        try:
            tasks = await _start_in_order(service, limiters, vton, "A", "B")
            vton.gates.open("A")
            with pytest.raises(FittingModelError):
                await tasks["A"]
            vton.gates.open("B")
            result = await tasks["B"]
        finally:
            vton.gates.open_all()

        assert result.llm_title == "title B"
        assert (limiters.vton.running, limiters.vton.waiting) == (0, 0)
        assert budget.recorded == [COST]  # 실패한 A 는 기록하지 않는다 (기존과 같음)

    asyncio.run(run())


# --- VTON 슬롯은 VTON 호출만 감싼다 ---


def test_vton_slot_is_released_before_llm_so_the_next_vton_starts():
    vton = GatedVton()
    comment = GatedComment()
    service = _service(vton=vton, comment=comment)
    limiters = _limiters(vton_running=1, vton_waiting=1)

    async def run():
        try:
            tasks = await _start_in_order(service, limiters, vton, "A", "E")

            vton.gates.open("A")
            # A 는 LLM comment 에서 멈춰 있다. 그래도 E 의 VTON 은 바로 시작한다.
            await until(lambda: comment.calls.started("comment-A"))
            await until(lambda: vton.calls.started("E"))
            assert comment.calls.active == {"comment-A"}
            assert limiters.vton.running == 1 and limiters.vton.waiting == 0

            vton.gates.open_all()
            comment.gates.open_all()
            await asyncio.gather(*tasks.values())
        finally:
            vton.gates.open_all()
            comment.gates.open_all()

    asyncio.run(run())


# --- LLM: 실제 요청 단위로 슬롯을 얻고 반환한다 ---


def test_llm_runs_at_most_4_requests_at_once():
    comment = GatedComment()
    service = _service(comment=comment)
    limiters = _limiters(vton_running=6, vton_waiting=0, llm_running=4)

    async def run():
        try:
            tasks = _start(service, limiters, *"ABCDEF")
            await until(lambda: len(comment.calls.active) == 4 and limiters.llm.waiting == 2)
            # LLM 슬롯을 기다리는 2개도 worker thread 를 잡지 않는다.
            assert _borrowed_worker_threads() == 4

            comment.gates.open_all()
            await asyncio.gather(*tasks.values())
        finally:
            comment.gates.open_all()

        assert comment.calls.max_active == 4
        assert (limiters.llm.running, limiters.llm.waiting) == (0, 0)

    asyncio.run(run())


def test_comment_returns_its_llm_slot_and_title_acquires_a_new_one():
    """LLM 1개일 때 A comment → B comment → A title 순서면, comment 와 title 은 한 슬롯이 아니다."""
    comment = GatedComment()
    service = _service(comment=comment)
    limiters = _limiters(llm_running=1)

    async def run():
        try:
            tasks = _start(service, limiters, "A")
            await until(lambda: comment.calls.started("comment-A"))
            tasks |= _start(service, limiters, "B")
            await until(lambda: limiters.llm.waiting == 1)  # B comment 대기

            comment.gates.open("comment-A")
            # A 의 comment 슬롯은 반환되어 먼저 기다린 B 의 comment 로 갔고, A 의 title 은 새로 기다린다.
            await until(lambda: comment.calls.started("comment-B"))
            assert not comment.calls.started("title-A")
            assert limiters.llm.running == 1 and limiters.llm.waiting == 1

            comment.gates.open("comment-B")
            await until(lambda: comment.calls.started("title-A"))
            comment.gates.open_all()
            results = await asyncio.gather(*tasks.values())
        finally:
            comment.gates.open_all()

        assert comment.calls.events[:6] == [
            "comment-A-start",
            "comment-A-end",
            "comment-B-start",
            "comment-B-end",
            "title-A-start",
            "title-A-end",
        ]
        assert comment.calls.max_active == 1
        assert [r.llm_title for r in results] == ["title A", "title B"]

    asyncio.run(run())


@pytest.mark.parametrize("failing", ["comment-A", "title-A"])
def test_llm_error_returns_the_slot_and_keeps_the_fallback_pair(failing):
    comment = GatedComment(
        gated=False, errors={failing: FittingPostprocessError("Runware LLM HTTP 오류: 500")}
    )
    service = _service(comment=comment)
    limiters = _limiters(llm_running=1)

    async def run():
        failed = await service.fit_with_limiters(_request("A"), limiters)
        assert (limiters.llm.running, limiters.llm.waiting) == (0, 0)
        succeeded = await service.fit_with_limiters(_request("B"), limiters)
        return failed, succeeded

    failed, succeeded = asyncio.run(run())

    assert (failed.llm_comment, failed.llm_title) == (FALLBACK_COMMENT, FALLBACK_TITLE)
    assert failed.result_image_key == "virtual-fitting/results/A.jpg"
    assert (succeeded.llm_comment, succeeded.llm_title) == ("comment B", "title B")
    if failing == "comment-A":
        assert not comment.calls.started("title-A")


def test_vton_and_llm_slots_are_independent():
    """LLM 슬롯이 모두 차 있어도 VTON 은 진행되고, VTON 슬롯이 차 있어도 LLM 은 진행된다."""
    vton = GatedVton()
    comment = GatedComment()
    service = _service(vton=vton, comment=comment)
    limiters = _limiters(vton_running=1, vton_waiting=1, llm_running=1)

    async def run():
        try:
            tasks = _start(service, limiters, "A")
            vton.gates.open("A")
            await until(lambda: comment.calls.started("comment-A"))  # LLM 1/1 사용 중

            tasks |= _start(service, limiters, "B")
            await until(lambda: vton.calls.started("B"))  # VTON 1/1 사용 중
            assert comment.calls.active == {"comment-A"} and vton.calls.active == {"B"}

            comment.gates.open_all()
            vton.gates.open_all()
            await asyncio.gather(*tasks.values())
        finally:
            comment.gates.open_all()
            vton.gates.open_all()

    asyncio.run(run())


# --- LLM 슬롯 대기 시간 초과 → LLM 을 부르지 않고 기존 fallback ---


def test_llm_wait_within_the_timeout_runs_comment_and_title_normally(configured_limiters):
    comment = GatedComment()
    service = _service(comment=comment)
    limiters = configured_limiters(llm_running=1, llm_wait_timeout=TIMEOUT)

    async def run():
        try:
            tasks = _start(service, limiters, "A")
            await until(lambda: comment.calls.started("comment-A"))
            tasks |= _start(service, limiters, "B")
            await until(lambda: limiters.llm.waiting == 1)

            comment.gates.open_all()
            return await asyncio.gather(*tasks.values())
        finally:
            comment.gates.open_all()

    results = asyncio.run(run())

    assert [(r.llm_comment, r.llm_title) for r in results] == [
        ("comment A", "title A"),
        ("comment B", "title B"),
    ]
    assert (limiters.llm.running, limiters.llm.waiting) == (0, 0)


def test_comment_slot_wait_timeout_falls_back_without_calling_the_llm(configured_limiters):
    comment = GatedComment()
    service = _service(comment=comment)
    limiters = configured_limiters(llm_running=1, llm_wait_timeout=0.05)

    async def run():
        try:
            a = _start(service, limiters, "A")["A"]
            await until(lambda: comment.calls.started("comment-A"))  # A 가 LLM 슬롯 1/1 사용 중

            b = await service.fit_with_limiters(_request("B"), limiters)
            # B 의 comment 슬롯 대기가 끝난 뒤에도 A 는 슬롯을 그대로 쥐고 있다(누수·중복 없음).
            assert (limiters.llm.running, limiters.llm.waiting) == (1, 0)

            comment.gates.open_all()
            return await a, b
        finally:
            comment.gates.open_all()

    a, b = asyncio.run(run())

    assert (b.llm_comment, b.llm_title) == (FALLBACK_COMMENT, FALLBACK_TITLE)
    assert b.result_image_key == "virtual-fitting/results/B.jpg"  # S3 저장까지 갔다
    assert not comment.calls.started("comment-B")
    assert not comment.calls.started("title-B")
    assert (a.llm_comment, a.llm_title) == ("comment A", "title A")
    assert (limiters.llm.running, limiters.llm.waiting) == (0, 0)


def test_title_slot_wait_timeout_discards_the_generated_comment(configured_limiters):
    """B 의 comment 는 만들어졌지만 title 슬롯을 못 얻으면, 기존 정책대로 둘 다 fallback 이다."""
    comment = GatedComment()
    service = _service(comment=comment)
    # A 의 comment 대기는 B 의 comment 가 끝나는 즉시 풀린다(수 ms). B 의 title 대기만 시간을 넘긴다.
    limiters = configured_limiters(llm_running=1, llm_wait_timeout=0.3)

    async def run():
        try:
            b = _start(service, limiters, "B")["B"]
            await until(lambda: comment.calls.started("comment-B"))
            a = _start(service, limiters, "A")["A"]
            await until(lambda: limiters.llm.waiting == 1)  # A comment 대기

            comment.gates.open("comment-B")
            # B 의 comment 슬롯은 A 에게 넘어갔다. A 의 comment 는 열지 않으므로 B 의 title 은 시간을 넘긴다.
            b_result = await b

            comment.gates.open_all()
            return await a, b_result
        finally:
            comment.gates.open_all()

    a, b = asyncio.run(run())

    assert comment.calls.started("comment-B")  # comment 는 실제로 만들어졌지만
    assert not comment.calls.started("title-B")  # title 은 부르지 않았고
    assert (b.llm_comment, b.llm_title) == (FALLBACK_COMMENT, FALLBACK_TITLE)  # 둘 다 버렸다
    assert (a.llm_comment, a.llm_title) == ("comment A", "title A")
    assert (limiters.llm.running, limiters.llm.waiting) == (0, 0)


def test_llm_slot_wait_does_not_hold_worker_threads(configured_limiters):
    comment = GatedComment()
    service = _service(comment=comment)
    limiters = configured_limiters(vton_running=5, vton_waiting=0, llm_running=4)

    async def run():
        try:
            tasks = _start(service, limiters, *"ABCDE")
            await until(lambda: len(comment.calls.active) == 4 and limiters.llm.waiting == 1)
            # 5번째는 LLM 슬롯을 async 로 기다린다. 빌려 간 thread 는 실행 중인 LLM 4개뿐이다.
            assert _borrowed_worker_threads() == 4

            comment.gates.open_all()
            await asyncio.gather(*tasks.values())
        finally:
            comment.gates.open_all()

    asyncio.run(run())

    assert comment.calls.max_active == 4
    assert (limiters.llm.running, limiters.llm.waiting) == (0, 0)


def test_sync_fit_and_fit_with_limiters_give_the_same_result():
    service = _service()

    limited = asyncio.run(service.fit_with_limiters(_request("A"), _limiters()))

    assert limited == service.fit(_request("A"))
