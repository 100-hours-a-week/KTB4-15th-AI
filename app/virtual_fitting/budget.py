"""가상피팅 하루 사용액 상한 (2026-09-28 결정).

  ① try_on 전 : 오늘 합계가 상한에 닿았으면 402 FITTING_DAILY_BUDGET_EXCEEDED 로 막는다.
                마지막 보정이 FITTING_USAGE_RECONCILE_SECONDS 보다 오래됐으면 먼저 보정한다.
  ③ try_on 후 : Runware 가 알려준 이 호출의 금액(includeCost)을 오늘 합계에 더한다.

판단은 지연 없는 우리 합계로 한다. Runware 집계(getUsageActivity)는 늦게 반영되므로
판단에 쓰지 않고, 우리가 응답을 못 받은 호출(타임아웃 등)의 금액을 메우는 보정에만 쓴다.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Protocol

from app.virtual_fitting.exceptions import FittingDailyBudgetExceededError
from app.virtual_fitting.repositories.usage_repository import UsageRepository


class UsageSource(Protocol):
    def spend_on(self, day: date) -> Decimal: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


class FittingBudget:
    def __init__(
        self,
        repository: UsageRepository,
        usage_source: UsageSource,
        *,
        daily_limit: Decimal,
        reconcile_interval: timedelta,
        now: Callable[[], datetime] = _utc_now,
    ):
        self._repository = repository
        self._usage_source = usage_source
        self._daily_limit = daily_limit
        self._reconcile_interval = reconcile_interval
        self._now = now

    #       동시성 — 합계가 $4.99 일 때 요청 5개가 거의 동시에 이 확인을 지나면? 각 호출의 금액은
    #       record() 에서야 더해진다. 하루 실제 지출은 상한을 얼마까지 넘을 수 있고, 그 폭은 무엇에 비례하나?
    #       현재 스레드풀의 최대 개수는 40개이고 요청당 0.015달러라 총 0.6달러로 감당 할 만 하다.
    # sabu: 보정 실패 — Runware 사용량 조회가 실패(타임아웃·5xx)하면 이 피팅 요청은 어떻게 되나?
    #       보정은 사용자의 피팅을 막을 만큼 중요한 단계인가?
    def ensure_available(self) -> None:
        now = self._now()
        today = now.date()
        usage = self._repository.get(today)
        amount = usage.amount
        if self._is_stale(usage.reconciled_at, now):
            amount = self._repository.reconcile(today, self._usage_source.spend_on(today))
        if amount >= self._daily_limit:
            raise FittingDailyBudgetExceededError(
                f"오늘 가상피팅 사용액 {amount} USD 가 상한 {self._daily_limit} USD 에 닿았습니다."
            )

    def record(self, cost: Decimal) -> None:
        self._repository.add(self._now().date(), cost)

    def _is_stale(self, reconciled_at: datetime | None, now: datetime) -> bool:
        return reconciled_at is None or now - reconciled_at >= self._reconcile_interval
