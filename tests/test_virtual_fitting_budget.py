"""가상피팅 하루 사용액 상한. 실제 PostgreSQL 과 Runware 없이 fake 로 검증한다."""

import io
import json
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from urllib.error import HTTPError

import psycopg
import pytest

from app.clients.s3 import ImageStorageError
from app.virtual_fitting.budget import FittingBudget
from app.virtual_fitting.exceptions import (
    FittingBalanceExhaustedError,
    FittingDailyBudgetExceededError,
    FittingDatabaseError,
    FittingImageStorageError,
    FittingModelError,
)
from app.virtual_fitting.providers.base import FittingInput, FittingResult
from app.virtual_fitting.providers.comment import MockCommentProvider
from app.virtual_fitting.providers.runware import (
    RUNWARE_ENDPOINT,
    RUNWARE_MODEL,
    RunwarePrunaProvider,
    build_payload,
)
from app.virtual_fitting.providers.runware_usage import (
    RunwareUsageClient,
    build_usage_payload,
    parse_total_spend,
)
from app.virtual_fitting.repositories.usage_repository import (
    ADD_SQL,
    RECONCILE_SQL,
    DailyUsage,
    UsageRepository,
)
from app.virtual_fitting.schemas import SyncFittingRequest
from app.virtual_fitting.service import VirtualFittingService
from tests.virtual_fitting_fixtures import AllowAllBudget, make_top

NOW = datetime(2026, 9, 28, 3, 0, tzinfo=UTC)
TODAY = date(2026, 9, 28)
LIMIT = Decimal(5)
INTERVAL = timedelta(minutes=10)


class FakeUsageRepository:
    def __init__(self, amount="0", reconciled_at=NOW):
        self.usage = DailyUsage(Decimal(amount), reconciled_at)
        self.added = []
        self.reconciled = []

    def get(self, day):
        return self.usage

    def add(self, day, cost):
        self.added.append((day, cost))

    def reconcile(self, day, runware_amount):
        self.reconciled.append((day, runware_amount))
        return max(self.usage.amount, runware_amount)


class FakeUsageSource:
    def __init__(self, spend="0"):
        self.spend = Decimal(spend)
        self.days = []

    def spend_on(self, day):
        self.days.append(day)
        return self.spend


def _budget(repository, source=None, now=NOW):
    return FittingBudget(
        repository,
        source or FakeUsageSource(),
        daily_limit=LIMIT,
        reconcile_interval=INTERVAL,
        now=lambda: now,
    )


# --- FittingBudget ---


def test_under_the_limit_passes_without_asking_runware_when_recently_reconciled():
    repository = FakeUsageRepository("4.99", reconciled_at=NOW - timedelta(minutes=9))
    source = FakeUsageSource()

    _budget(repository, source).ensure_available()

    assert source.days == []
    assert repository.reconciled == []


def test_reaching_the_limit_is_rejected_with_402():
    with pytest.raises(FittingDailyBudgetExceededError) as caught:
        _budget(FakeUsageRepository("5")).ensure_available()

    assert caught.value.status_code == 402
    assert caught.value.code == "FITTING_DAILY_BUDGET_EXCEEDED"
    assert caught.value.message == (
        "오늘 사용할 수 있는 가상 피팅 한도를 초과했습니다. 내일 다시 시도해주세요."
    )


@pytest.mark.parametrize(
    "reconciled_at",
    [None, NOW - timedelta(minutes=10), NOW - timedelta(hours=3)],
    ids=["never", "exactly-interval", "long-ago"],
)
def test_stale_usage_is_reconciled_with_runware_for_today(reconciled_at):
    repository = FakeUsageRepository("1", reconciled_at=reconciled_at)
    source = FakeUsageSource("2.5")

    _budget(repository, source).ensure_available()

    assert source.days == [TODAY]
    assert repository.reconciled == [(TODAY, Decimal("2.5"))]


def test_decision_uses_the_reconciled_amount():
    # 우리 합계는 1달러지만 Runware 는 응답 못 받은 호출까지 합쳐 5달러라고 한다.
    repository = FakeUsageRepository("1", reconciled_at=None)

    with pytest.raises(FittingDailyBudgetExceededError):
        _budget(repository, FakeUsageSource("5")).ensure_available()


def test_record_adds_the_cost_to_today():
    repository = FakeUsageRepository()

    _budget(repository).record(Decimal("0.03"))

    assert repository.added == [(TODAY, Decimal("0.03"))]


def test_today_is_the_utc_date():
    # KST 2026-09-29 08:59 는 아직 UTC 9월 28일이다.
    kst_morning = datetime(2026, 9, 28, 23, 59, tzinfo=UTC)
    repository = FakeUsageRepository()

    _budget(repository, now=kst_morning).record(Decimal(1))

    assert repository.added == [(TODAY, Decimal(1))]


# --- UsageRepository ---


class FakeCursor:
    def __init__(self, row):
        self._row = row
        self.executed = []

    def execute(self, sql, params):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeConnection:
    def __init__(self, row=None, error=None):
        self.cursor_instance = FakeCursor(row)
        self._error = error
        self.autocommit = False

    def cursor(self):
        if self._error is not None:
            raise self._error
        return self.cursor_instance


class FakePool:
    def __init__(self, connection):
        self.connection_instance = connection

    @contextmanager
    def connection(self):
        yield self.connection_instance


def test_missing_row_means_nothing_spent_and_never_reconciled():
    repository = UsageRepository(FakePool(FakeConnection(row=None)))

    assert repository.get(TODAY) == DailyUsage(Decimal(0), None)


def test_add_is_a_single_upsert_that_adds_in_the_database():
    connection = FakeConnection()

    UsageRepository(FakePool(connection)).add(TODAY, Decimal("0.03"))

    assert connection.cursor_instance.executed == [(ADD_SQL, (TODAY, Decimal("0.03")))]
    assert "usage_amount.amount + EXCLUDED.amount" in ADD_SQL


def test_reconcile_keeps_the_larger_amount_and_returns_it():
    connection = FakeConnection(row=(Decimal("2.5"),))

    amount = UsageRepository(FakePool(connection)).reconcile(TODAY, Decimal("2.5"))

    assert amount == Decimal("2.5")
    assert connection.cursor_instance.executed == [(RECONCILE_SQL, (TODAY, Decimal("2.5")))]
    assert "GREATEST(usage_amount.amount, EXCLUDED.amount)" in RECONCILE_SQL


def test_database_errors_become_domain_errors():
    connection = FakeConnection(error=psycopg.OperationalError("down"))

    with pytest.raises(FittingDatabaseError):
        UsageRepository(FakePool(connection)).get(TODAY)


# --- Runware: includeCost / 사용량 조회 / 402 ---


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self._body


class FakeOpener:
    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return FakeResponse(self._payload)


def _usage_response(total_spend):
    return {"data": [{"usage": {"timeseries": {"data": [], "meta": {"totalSpend": total_spend}}}}]}


def test_fitting_request_asks_runware_for_the_cost():
    [task] = build_payload(FittingInput("https://u", ["https://g"], "p"), "t")

    assert task["includeCost"] is True


def test_fitting_cost_is_parsed_as_decimal():
    opener = FakeOpener({"data": [{"imageURL": "https://im.runware.ai/r.jpg", "cost": 0.0132}]})

    result = RunwarePrunaProvider("key", opener=opener).try_on(
        FittingInput("https://u", ["https://g"], "p")
    )

    assert result.cost == Decimal("0.0132")


def test_runware_402_is_balance_exhausted_not_a_model_failure():
    error = HTTPError(RUNWARE_ENDPOINT, 402, "Payment Required", {}, io.BytesIO(b"{}"))

    with pytest.raises(FittingBalanceExhaustedError) as caught:
        RunwarePrunaProvider("key", opener=FakeOpener(error=error)).try_on(
            FittingInput("https://u", ["https://g"], "p")
        )

    assert caught.value.status_code == 402
    assert caught.value.code == "FITTING_BALANCE_EXHAUSTED"
    assert caught.value.message == (
        "현재 가상 피팅 서비스를 이용할 수 없습니다. 잠시 후 다시 시도해주세요."
    )


def test_usage_payload_is_one_utc_day_of_the_fitting_model_only():
    [task] = build_usage_payload(TODAY, "t")

    assert task["taskType"] == "accountManagement"
    assert task["operation"] == "getUsageActivity"
    assert (task["startDate"], task["endDate"]) == ("2026-09-28", "2026-09-28")
    assert task["models"] == [RUNWARE_MODEL]
    assert task["timezone"] == "UTC"


def test_usage_client_returns_total_spend_exactly():
    opener = FakeOpener(_usage_response(1.2345))

    assert RunwareUsageClient("key", opener=opener).spend_on(TODAY) == Decimal("1.2345")
    [request] = opener.requests
    assert request.get_header("Authorization") == "Bearer key"


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"data": []}'])
def test_unreadable_usage_response_is_a_model_error(body):
    with pytest.raises(FittingModelError):
        parse_total_spend(body)


# --- Service 흐름 ---


class RepositoryOf:
    def __init__(self, *products):
        self._products = {p.product_code: p for p in products}

    def find_by_codes(self, codes):
        return [self._products[c] for c in codes if c in self._products]


class CostlyProvider:
    def __init__(self, cost):
        self._cost = cost
        self.calls = 0

    def try_on(self, fitting_input):
        self.calls += 1
        return FittingResult("https://im.runware.ai/r.jpg", cost=self._cost)


class BrokenStorage:
    def store_remote_image(self, url, key_prefix):
        raise ImageStorageError("s3 down")


class ExhaustedBudget(AllowAllBudget):
    def ensure_available(self):
        raise FittingDailyBudgetExceededError("over")


class RaisingCommentProvider:
    """fallback 으로 흡수되지 않는 예외를 던지는 LLM 후처리."""

    def generate_comment(self, image_url, summaries):
        raise FittingBalanceExhaustedError("Runware LLM 잔액이 부족합니다.")

    def generate_title(self, comment):
        raise AssertionError("comment 가 실패하면 title 은 불리지 않는다")


REQUEST = SyncFittingRequest(
    user_image_url="https://example.com/u.png", products=[{"product_code": "1"}]
)


def test_cost_is_recorded_even_when_llm_postprocess_escapes():
    budget = AllowAllBudget()
    service = VirtualFittingService(
        RepositoryOf(make_top("1")),
        CostlyProvider(Decimal("0.02")),
        RaisingCommentProvider(),
        BrokenStorage(),
        budget,
    )

    with pytest.raises(FittingBalanceExhaustedError):
        service.fit(REQUEST)

    assert budget.recorded == [Decimal("0.02")]


def test_cost_is_recorded_even_when_s3_storage_fails():
    budget = AllowAllBudget()
    service = VirtualFittingService(
        RepositoryOf(make_top("1")),
        CostlyProvider(Decimal("0.02")),
        MockCommentProvider(),
        BrokenStorage(),
        budget,
    )

    with pytest.raises(FittingImageStorageError):
        service.fit(REQUEST)

    assert budget.recorded == [Decimal("0.02")]


def test_exhausted_budget_never_calls_runware():
    provider = CostlyProvider(Decimal("0.02"))
    service = VirtualFittingService(
        RepositoryOf(make_top("1")),
        provider,
        MockCommentProvider(),
        BrokenStorage(),
        ExhaustedBudget(),
    )

    with pytest.raises(FittingDailyBudgetExceededError):
        service.fit(REQUEST)

    assert provider.calls == 0
