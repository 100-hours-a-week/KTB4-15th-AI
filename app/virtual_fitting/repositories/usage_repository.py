"""가상피팅 하루 사용액 (scripts/db/schema.sql 의 usage_amount + scripts/db/usage_amount_split_totals.sql).

하루 한 줄에 두 합계를 따로 둔다.
  recorded_amount   : 개별 비용 기록(V1 budget.record · V2 TaskCostRepository)으로 더한 합계. add() 만 바꾼다.
  reconciled_amount : Runware 사용량 조회로 확인한 하루 사용액(가장 큰 값, 줄지 않음). reconcile() 만 바꾼다.
  legacy_amount     : 변경 SQL 적용 시점의 amount(출처 미구분). 바꾸지 않는다. 새 날짜는 0.
판단 금액 = GREATEST(legacy_amount + recorded_amount, reconciled_amount) — 개별 비용을 보정값에 더하지 않는다.
amount 는 저장된 판단 금액이다. 쓸 때마다 GREATEST(amount, 위 두 값)로만 다시 계산한다(줄지 않음).

한계: 판단 금액은 실시간 계정 사용액이 아니다. 아직 받지 못한 비용은 빠져 있고, 보정값에만 있는 비용과 그 뒤의
새 개별 비용이 섞이면 다음 보정까지 실제보다 작게 볼 수 있다. 하루 상한을 넘지 않는다고 보장하지 않는다.
"""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg

from app.virtual_fitting.exceptions import FittingDatabaseError

# 판단 금액. 읽을 때도 같은 식으로 계산한다(amount 를 따로 믿지 않는다).
EFFECTIVE_SQL = "GREATEST(amount, legacy_amount + recorded_amount, reconciled_amount)"

SELECT_SQL = f"SELECT {EFFECTIVE_SQL}, reconciled_at FROM usage_amount WHERE usage_date = %(day)s"

# 개별 비용은 recorded_amount 에만 더한다. 더하기는 DB 가 한 문장으로 한다(동시 요청끼리 덮어쓰지 않게).
# ON CONFLICT 의 usage_amount.* 는 잠근 뒤의 기존 값이다.
ADD_SQL = """
INSERT INTO usage_amount (usage_date, recorded_amount, amount) VALUES (%(day)s, %(cost)s, %(cost)s)
ON CONFLICT (usage_date) DO UPDATE SET
    recorded_amount = usage_amount.recorded_amount + EXCLUDED.recorded_amount,
    amount = GREATEST(
        usage_amount.amount,
        usage_amount.legacy_amount + usage_amount.recorded_amount + EXCLUDED.recorded_amount,
        usage_amount.reconciled_amount
    )
"""

# Runware 합계는 reconciled_amount 에만 반영한다. 늦게 도착한 작은(오래된) 합계로 줄이지 않는다.
RECONCILE_SQL = """
INSERT INTO usage_amount (usage_date, reconciled_amount, amount, reconciled_at)
VALUES (%(day)s, %(total)s, %(total)s, now())
ON CONFLICT (usage_date) DO UPDATE SET
    reconciled_amount = GREATEST(usage_amount.reconciled_amount, EXCLUDED.reconciled_amount),
    amount = GREATEST(
        usage_amount.amount,
        usage_amount.legacy_amount + usage_amount.recorded_amount,
        usage_amount.reconciled_amount,
        EXCLUDED.reconciled_amount
    ),
    reconciled_at = now()
RETURNING amount
"""


def add_params(day: date, cost: Decimal) -> dict[str, Any]:
    return {"day": day, "cost": cost}


@dataclass(frozen=True)
class DailyUsage:
    amount: Decimal  # 판단 금액
    reconciled_at: datetime | None


class UsageRepository:
    """쓸 때만 pool 에서 connection 을 빌리고 즉시 반환한다."""

    def __init__(self, connection_pool: Any):
        self._connection_pool = connection_pool

    def get(self, day: date) -> DailyUsage:
        """그날 줄이 아직 없으면 0 원, 보정 기록 없음으로 본다."""
        row = self._execute(SELECT_SQL, {"day": day}, fetch=True)
        if row is None:
            return DailyUsage(amount=Decimal(0), reconciled_at=None)
        amount, reconciled_at = row
        return DailyUsage(amount=Decimal(amount), reconciled_at=reconciled_at)

    def add(self, day: date, cost: Decimal) -> None:
        """개별 비용 기록. recorded_amount 에만 더한다."""
        self._execute(ADD_SQL, add_params(day, cost))

    def reconcile(self, day: date, runware_amount: Decimal) -> Decimal:
        """Runware 합계를 reconciled_amount 에 반영하고 보정 뒤의 판단 금액을 돌려준다."""
        (amount,) = self._execute(RECONCILE_SQL, {"day": day, "total": runware_amount}, fetch=True)
        return Decimal(amount)

    def _execute(self, sql: str, params: dict, *, fetch: bool = False) -> Any:
        try:
            with self._connection_pool.connection() as connection:
                connection.autocommit = True
                with connection.cursor() as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchone() if fetch else None
        except psycopg.Error as error:
            raise FittingDatabaseError("AI PostgreSQL 사용액 조회/기록에 실패했습니다.") from error
