-- 가상피팅 V2 Watchdog(15단계) 점검·알림 기록 (AI PostgreSQL). 새 테이블만 만든다(기존 테이블은 바꾸지 않음).
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/virtual_fitting_watchdog.sql
-- 적용 순서: virtual_fitting_async_jobs.sql → virtual_fitting_fallback.sql → (이 파일) → 새 코드 배포. 재실행해도 그대로다.
-- 기존 변경 SQL(usage_amount_split_totals.sql 등)의 배포 조건(구·신버전 동시 쓰기 금지)은 그대로 지킨다.

-- 작업별 점검 기록. virtual_fitting_async_jobs.updated_at 을 건드리지 않아 실제 지연 시간을 초기화하지 않는다.
--   next_check_at          : 이 시각 전에는 어느 Watchdog 인스턴스도 이 작업을 다시 점검하지 않는다(DB 선점, 오래된 작업만
--                            반복 점검하는 굶주림 방지 — 점검한 작업은 뒤로 간다).
--   follow_up_published_at : 마지막으로 POSTPROCESS·RUN_FALLBACK 을 다시 발행한 시각(점검마다 무조건 발행하지 않음).
--   cleanup_settled_at     : 최종 결과 뒤 비용 누락 점검을 끝낸 시각(생성 종료를 확인했지만 비용이 보고되지 않음 등).
--                            이 값이 있으면 그 작업의 비용 누락 점검 대상에서 빠진다(비용을 0 으로 기록한 것이 아님).
CREATE TABLE IF NOT EXISTS virtual_fitting_watchdog_checks (
    fitting_job_id          bigint       PRIMARY KEY,
    checked_at              timestamptz  NOT NULL,
    next_check_at           timestamptz  NOT NULL,
    follow_up_published_at  timestamptz,
    cleanup_settled_at      timestamptz
);
-- 이 파일의 이전 초안으로 만든 테이블에도 같은 컬럼을 둔다(재실행해도 그대로).
ALTER TABLE virtual_fitting_watchdog_checks ADD COLUMN IF NOT EXISTS cleanup_settled_at timestamptz;

-- 운영 이상 알림 기록. 같은 대상(scope)·원인(reason)은 한 줄이고, 재알림 간격 안에서는 다시 보내지 않는다(인스턴스·재시작 무관).
--   scope: 'job:<fitting_job_id>' 또는 'global:<이름>'. reason: 안전한 원인 코드(외부 오류 원문·URL·토큰 없음).
CREATE TABLE IF NOT EXISTS virtual_fitting_watchdog_alerts (
    scope             text         NOT NULL,
    reason            text         NOT NULL,
    stage             text         NOT NULL,
    fitting_job_id    bigint,
    first_seen_at     timestamptz  NOT NULL,
    last_seen_at      timestamptz  NOT NULL,
    last_notified_at  timestamptz,
    seen_count        integer      NOT NULL DEFAULT 1,
    PRIMARY KEY (scope, reason)
);
