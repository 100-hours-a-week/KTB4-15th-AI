-- 가상피팅 V2: 모든 AI 서버·프로세스가 공유하는 Pruna 생성 요청 제한과 작업별 비용 기록 (AI PostgreSQL).
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/virtual_fitting_pruna_limits.sql
-- 신규 테이블 생성용이다. CREATE TABLE IF NOT EXISTS 는 같은 이름이 있으면 건너뛸 뿐 기존 구조를 바꾸지 않는다.
-- 기존 테이블(virtual_fitting_async_jobs, usage_amount)은 바꾸지 않는다. usage_amount 변경은 usage_amount_split_totals.sql.
--
-- 제한 값(진행 중 3건·HTTP 동시 2건·최소 간격 10초)은 코드 설정(settings)에서 정하고, 여기에는 상태만 둔다.
-- 자리를 잡고 놓는 SQL 은 app/virtual_fitting/repositories/pruna_limiter.py 가 짧은 트랜잭션으로 실행한다.

-- 한 줄짜리 공유 상태. 모든 제출이 이 행을 잠가 차례로 자리를 잡는다(짧게, 외부 호출 중에는 잡지 않음).
--   next_send_at: 다음 생성 요청을 보낼 수 있는 가장 이른 시각(최소 간격·Runware 대기 안내를 줄이지 않고 반영)
CREATE TABLE IF NOT EXISTS virtual_fitting_pruna_limiter (
    id            smallint     PRIMARY KEY CHECK (id = 1),
    next_send_at  timestamptz  NOT NULL DEFAULT 'epoch'  -- 1970-01-01 UTC: 처음에는 바로 보낼 수 있다
);
INSERT INTO virtual_fitting_pruna_limiter (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- 공유 보류(8단계 보완). 위 CREATE 는 이미 있는 테이블을 바꾸지 않으므로 컬럼은 ALTER 로 추가한다(재실행해도 그대로).
--   send_hold_reason: NULL 이 아니면 어느 Worker 도 새 생성 요청 권한을 받지 못한다. Runware 대기 안내가 유효하지만
--                     시각으로 표현할 수 없게 클 때 기록한다(retry_after_unrepresentable / retry_after_out_of_range).
--   send_hold_at    : 처음 보류한 시각. 보류 중 다시 기록해도 처음 원인·시각을 유지한다.
-- 자동으로 풀리지 않는다. 원인 확인 뒤 운영자가 직접 해제한다:
--   UPDATE virtual_fitting_pruna_limiter SET send_hold_reason = NULL, send_hold_at = NULL WHERE id = 1;
ALTER TABLE virtual_fitting_pruna_limiter
    ADD COLUMN IF NOT EXISTS send_hold_reason text,
    ADD COLUMN IF NOT EXISTS send_hold_at     timestamptz;

-- 진행 중인 생성 작업 자리(작업당 하나). 생성 요청 전에 잡고, 생성이 끝났다는 근거(완료·확정 종료·명확한 미접수)가
-- 있을 때만 지운다. 접수 응답·UNKNOWN·Worker 종료·lease 만료·지연만으로는 지우지 않는다.
CREATE TABLE IF NOT EXISTS virtual_fitting_pruna_in_flight (
    fitting_job_id  bigint       PRIMARY KEY,
    task_uuid       uuid         NOT NULL UNIQUE,
    acquired_at     timestamptz  NOT NULL
);

-- 생성 요청 HTTP 전송 중 자리. 전송이 끝나면 지운다. 프로세스가 죽어 지우지 못해도 held_until 이 지나면 빈 자리로 본다.
CREATE TABLE IF NOT EXISTS virtual_fitting_pruna_sends (
    fitting_job_id  bigint       PRIMARY KEY,
    held_until      timestamptz  NOT NULL
);

-- 생성 작업(task_uuid)별 비용 기록. 같은 task 의 비용을 두 번 더하지 않게 한다(usage_amount.recorded_amount 에
-- 더하는 것과 같은 트랜잭션). Runware 보정값은 usage_amount.reconciled_amount 에 따로 두고, 판단은 둘 중 큰 값으로 한다
-- (scripts/db/usage_amount_split_totals.sql). 그래서 보정 뒤 늦게 기록한 비용도 보정값에 다시 더해지지 않는다.
CREATE TABLE IF NOT EXISTS virtual_fitting_task_costs (
    task_uuid       uuid         PRIMARY KEY,
    fitting_job_id  bigint       NOT NULL,
    cost            numeric      NOT NULL CHECK (cost >= 0),
    usage_date      date         NOT NULL,
    recorded_at     timestamptz  NOT NULL DEFAULT now()
);
