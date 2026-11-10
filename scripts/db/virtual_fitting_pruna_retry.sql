-- 가상피팅 V2 Pruna 확실한 미접수 재요청(보완 2-2). 기존 작업 테이블에 컬럼 2개를 더하고 작은 시도 기록 테이블을 만든다.
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/virtual_fitting_pruna_retry.sql
--
-- 적용 전제·순서: virtual_fitting_async_jobs.sql → virtual_fitting_fallback.sql → virtual_fitting_auth_hold.sql →
-- virtual_fitting_runware_image_ref.sql → virtual_fitting_ai_result_storage.sql → (이 파일) → 새 코드.
-- 새 코드는 이 컬럼·테이블을 읽고 쓰므로 없으면 DB 오류다. 재실행해도 바뀌는 것이 없다(IF NOT EXISTS, 제약은 없을 때만).
-- 기존 행은 두 컬럼 모두 NULL(미접수 기록 없음)이 되고, 시도 기록 테이블에는 아무것도 채우지 않는다(아래 "기존 작업" 참고).
--
-- 확실한 미접수: 현재 전송 시도(task)에 연결된 serviceOverloaded·inferenceRateLimitExceeded(공통 분류
-- app/virtual_fitting/providers/runware_errors.py). 같은 fitting_job_id 전체에서 최초 1회 + 재요청 3회 = 최대 4회 전송.
-- 접수 불명확(timeout·429/5xx·retryAfter 만·다른 task 응답 등)은 이 기록을 쓰지 않는다(기존 조회 복구).
--
--   pruna_not_accepted_at : 현재 task_uuid 의 시도를 확실한 미접수로 기록한 시각. 다음 시도를 시작할 때(새 task_uuid 와
--                           같은 트랜잭션) NULL 로 돌아간다. 이 값이 있으면 "보냈을 수 있음(접수 불명확)"이 아니다.
--   pruna_retry_at        : 다음 재요청을 보낼 수 있는 가장 이른 시각(DB 시각). 미접수를 기록할 때 한 번 정하고, 중복 알림·
--                           재수신으로 다시 계산하지 않는다. NULL 이고 pruna_not_accepted_at 이 있으면 자동 재요청 보류(운영 확인):
--                           대기 안내를 시각으로 표현할 수 없거나, 미접수로 기록한 시도에 모순되는 근거가 왔다.
ALTER TABLE virtual_fitting_async_jobs
    ADD COLUMN IF NOT EXISTS pruna_not_accepted_at timestamptz,
    ADD COLUMN IF NOT EXISTS pruna_retry_at        timestamptz;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'virtual_fitting_async_jobs_pruna_retry_check'
          AND conrelid = 'virtual_fitting_async_jobs'::regclass
    ) THEN
        ALTER TABLE virtual_fitting_async_jobs
            ADD CONSTRAINT virtual_fitting_async_jobs_pruna_retry_check
            CHECK (pruna_retry_at IS NULL OR pruna_not_accepted_at IS NOT NULL);
    END IF;
END
$$;

-- 비동기 Pruna 전송 시도(작업당 최대 4줄). 요청 본문·이미지·URL 은 복제하지 않는다(작업 테이블의 원래 입력을 그대로 쓴다).
--   attempt_no          : 1(최초)~4. 작업 안에서 겹치지 않는다. 다음 번호는 기존 최대값 + 1.
--   task_uuid           : 그 시도의 Runware taskUUID. 현재 시도는 virtual_fitting_async_jobs.task_uuid 와 같고, 이전 시도의 늦은
--                         알림은 이 표로 찾는다(현재 task 로 적용하지 않음).
--   started_at          : 전송 시작 기록(= 그 시도의 pruna_submit_started_at). 자리 확보와 같은 트랜잭션에서 외부 전송 직전에 확정.
--   not_accepted_*      : 확실한 미접수 근거(안전한 오류 코드·확인 경로·시각). 처음 기록만 남는다.
--   next_attempt_at     : 미접수 기록 때 정한 다음 전송 가능 시각(작업의 pruna_retry_at 과 같은 값). 소진·보류면 NULL.
--   conflict_*          : 미접수로 기록한 시도에 접수·생성 결과 근거가 뒤늦게 왔다(운영 확인). 처음 기록만 남는다.
-- 기존 작업: 이 표가 생기기 전에 시작한 작업은 줄이 없다. 코드는 호출 시작 기록(pruna_submit_started_at)이 있으면 1회 전송한
-- 것으로 본다(0 으로 보지 않음). 그 작업이 확실한 미접수를 받으면 그때 현재 task 를 1번 시도로 기록한다.
CREATE TABLE IF NOT EXISTS virtual_fitting_pruna_attempts (
    fitting_job_id       bigint       NOT NULL CHECK (fitting_job_id > 0),
    attempt_no           smallint     NOT NULL CHECK (attempt_no BETWEEN 1 AND 4),
    task_uuid            uuid         NOT NULL UNIQUE,
    started_at           timestamptz  NOT NULL,
    not_accepted_code    text         CHECK (not_accepted_code ~ '^[A-Za-z0-9_.-]{1,64}$'),
    not_accepted_source  text         CHECK (
        not_accepted_source IN ('submit_response', 'webhook', 'status_query')
    ),
    not_accepted_at      timestamptz,
    next_attempt_at      timestamptz,
    conflict_reason      text         CHECK (conflict_reason ~ '^[A-Za-z0-9_.-]{1,64}$'),
    conflict_at          timestamptz,
    PRIMARY KEY (fitting_job_id, attempt_no),
    CHECK (
        (not_accepted_at IS NULL AND not_accepted_code IS NULL AND not_accepted_source IS NULL
            AND next_attempt_at IS NULL)
        OR (not_accepted_at IS NOT NULL AND not_accepted_code IS NOT NULL
            AND not_accepted_source IS NOT NULL)
    ),
    CHECK ((conflict_at IS NULL) = (conflict_reason IS NULL))
);
