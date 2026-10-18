"""가상피팅 하루 사용액 (scripts/db/schema.sql 의 usage_amount)."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg

from app.virtual_fitting.exceptions import FittingDatabaseError

SELECT_SQL = "SELECT amount, reconciled_at FROM usage_amount WHERE usage_date = %s"

# 더하기는 DB 가 한 문장으로 한다. 파이썬에서 읽고 더해서 쓰면 동시 요청끼리 서로의 금액을 덮어쓴다.
ADD_SQL = """
INSERT INTO usage_amount (usage_date, amount) VALUES (%s, %s)
ON CONFLICT (usage_date) DO UPDATE SET amount = usage_amount.amount + EXCLUDED.amount
"""

# 우리 합계와 Runware 합계는 둘 다 실제보다 작게만 틀린다 — 큰 쪽이 실제에 더 가깝다.
RECONCILE_SQL = """
INSERT INTO usage_amount (usage_date, amount, reconciled_at) VALUES (%s, %s, now())
ON CONFLICT (usage_date) DO UPDATE
    SET amount = GREATEST(usage_amount.amount, EXCLUDED.amount), reconciled_at = now()
RETURNING amount
"""


@dataclass(frozen=True)
class DailyUsage:
    amount: Decimal
    reconciled_at: datetime | None


class UsageRepository:
    """쓸 때만 pool 에서 connection 을 빌리고 즉시 반환한다."""

    def __init__(self, connection_pool: Any):
        self._connection_pool = connection_pool

    def get(self, day: date) -> DailyUsage:
        """그날 줄이 아직 없으면 0 원, 보정 기록 없음으로 본다."""
        row = self._execute(SELECT_SQL, (day,), fetch=True)
        if row is None:
            return DailyUsage(amount=Decimal(0), reconciled_at=None)
        amount, reconciled_at = row
        return DailyUsage(amount=Decimal(amount), reconciled_at=reconciled_at)

    def add(self, day: date, cost: Decimal) -> None:
        self._execute(ADD_SQL, (day, cost))

    def reconcile(self, day: date, runware_amount: Decimal) -> Decimal:
        """보정 뒤의 합계를 돌려준다."""
        (amount,) = self._execute(RECONCILE_SQL, (day, runware_amount), fetch=True)
        return Decimal(amount)

    def _execute(self, sql: str, params: tuple, *, fetch: bool = False) -> Any:
        try:
            with self._connection_pool.connection() as connection:
                connection.autocommit = True
                with connection.cursor() as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchone() if fetch else None
        except psycopg.Error as error:
            raise FittingDatabaseError("AI PostgreSQL 사용액 조회/기록에 실패했습니다.") from error
