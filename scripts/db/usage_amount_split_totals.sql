-- 가상피팅 하루 사용액(usage_amount)을 "개별 비용 기록 합계"와 "Runware 보정값"으로 나누는 변경 SQL (AI PostgreSQL).
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/usage_amount_split_totals.sql
--
-- 왜: 이전에는 amount 하나에 개별 비용을 더하고(ADD) 보정 때 GREATEST(amount, Runware 합계)를 남겼다. 그래서
--     "작업 A 가 이미 포함된 Runware 합계로 보정 → 늦게 작업 A 비용 기록"이면 같은 비용이 두 번 들어갔다(2.00 → 2.02 → 2.04).
--
-- 적용 전제·순서
--   1. scripts/db/schema.sql 로 usage_amount 가 이미 있어야 한다(이 파일은 테이블을 만들지 않는다).
--      schema.sql 의 CREATE TABLE IF NOT EXISTS 를 다시 실행해도 기존 테이블 구조는 바뀌지 않으므로 이 파일이 따로 필요하다.
--   2. 이 SQL 을 먼저 적용하고, 그다음 새 코드(UsageRepository 가 새 컬럼을 쓰는 버전)를 배포한다.
--      새 코드는 새 컬럼이 없으면 DB 오류(FittingDatabaseError)를 낸다.
--   3. 적용 뒤에도 이전 코드는 amount 에만 쓰므로 계속 동작하지만, 이전 코드가 "새 날짜의 첫 줄"을 만들면
--      그 금액은 legacy_amount 가 아니라 amount 에만 남는다. 이전·새 코드를 섞어 오래 운영하지 않는다(V1 포함 AI 서버를 함께 교체).
--   Dev·운영 DB 적용은 배포 절차에서 따로 결정한다.
--
-- 재실행: 한 트랜잭션이고, 이미 적용됐으면 아무것도 바꾸지 않는다(ADD COLUMN IF NOT EXISTS, legacy_amount IS NULL 인 줄만 채움).
--
-- 컬럼 의미 (판단 금액 = GREATEST(amount, legacy_amount + recorded_amount, reconciled_amount))
--   recorded_amount   : 이 변경 이후 개별 비용 기록(V1 budget.record · V2 TaskCostRepository)으로 더한 하루 합계.
--   reconciled_amount : 이 변경 이후 Runware getUsageActivity 로 확인한 하루 사용액 중 가장 큰 값(줄지 않음).
--   legacy_amount     : 전환 시점의 amount 그대로. 개별 기록과 보정이 섞여 있어 출처를 나눌 수 없다. 이후 바꾸지 않는다.
--                       같은 날짜의 새 개별 기록은 이 값 위에 이어서 센다(legacy + recorded). 새 날짜는 0 이다.
--   amount            : 저장된 판단 금액. 새 코드는 쓸 때마다 위 GREATEST 로만 다시 계산한다(직접 더하거나 보정값으로 덮지 않음, 줄지 않음).
--
-- 기존 데이터 전환 기준과 남는 한계
--   - 기존 행의 amount 를 legacy_amount 로 옮길 뿐 다시 더하거나 0 으로 만들지 않는다. 판단 금액은 전환 전과 같다.
--   - 기존 amount 안의 과거 이중 합산은 근거가 없으므로 자동으로 되돌리지 않는다.
--   - 전환 당일: 전환 전 보정에 이미 포함된 작업의 비용이 전환 뒤 늦게 기록되면 legacy_amount + recorded_amount 에서
--     다시 셀 수 있다(그날 하루, 보수적으로 크게). 다음 날짜부터는 legacy_amount = 0 이라 이 경우가 없다.
BEGIN;

ALTER TABLE usage_amount
    ADD COLUMN IF NOT EXISTS legacy_amount     numeric,
    ADD COLUMN IF NOT EXISTS recorded_amount   numeric NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS reconciled_amount numeric NOT NULL DEFAULT 0;

-- DEFAULT 없이 추가한 뒤 기존 행만 채운다(기존 행을 0 으로 채우지 않기 위해). ALTER 가 테이블을 잠그므로 그사이 새 행은 없다.
UPDATE usage_amount SET legacy_amount = amount WHERE legacy_amount IS NULL;

ALTER TABLE usage_amount
    ALTER COLUMN legacy_amount SET DEFAULT 0,
    ALTER COLUMN legacy_amount SET NOT NULL;

COMMIT;
