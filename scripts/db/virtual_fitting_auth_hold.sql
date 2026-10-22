-- 가상피팅 V2 Runware 인증 오류 보류 기록(15단계 보완). virtual_fitting_async_jobs 를 바꾸는 변경 SQL (AI PostgreSQL).
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/virtual_fitting_auth_hold.sql
--
-- 적용 전제·순서: virtual_fitting_async_jobs.sql → virtual_fitting_fallback.sql → (이 파일) → 새 코드.
-- 새 코드는 이 컬럼을 SELECT 하므로 없으면 DB 오류다. 재실행해도 바뀌는 것이 없다(ADD COLUMN IF NOT EXISTS,
-- 제약은 없을 때만 추가). 기존 행은 모두 NULL(보류 기록 없음)이 된다. 값을 채우거나 추측하지 않는다.
--
-- Runware 가 우리 API 키를 거절했다는 확인된 사실(정상 Webhook 토큰으로 받은 invalidApiKey, 또는 생성 요청에 대한
-- HTTP 401·403 응답)을 그 비동기 생성 시도(task_uuid)에 묶어 남긴다. Webhook·요청 Worker 재수신·Watchdog 이 같은 기록으로
-- "운영 확인(operator_review)"을 판단한다(메모리·로그가 아니라 DB). Webhook 토큰 누락·불일치(401)는 기록하지 않는다.
--   auth_hold_task_uuid : 거절된 생성 시도의 taskUUID(= 그때의 task_uuid). fallback task 와 구분한다.
--   auth_hold_reason    : 안전한 원인 코드만(invalidApiKey, http_401, http_403). 외부 오류 원문·API Key·토큰은 저장하지 않는다.
--   auth_hold_at        : 처음 기록한 시각. 중복 수신·늦은 접수 응답으로 바뀌거나 지워지지 않는다.
-- 자동으로 풀리지 않는다(설정 변경·Watchdog 실행만으로 해제하지 않음). 해제는 운영 판단(설계 문서 "인증 오류 보류").
ALTER TABLE virtual_fitting_async_jobs
    ADD COLUMN IF NOT EXISTS auth_hold_task_uuid uuid,
    ADD COLUMN IF NOT EXISTS auth_hold_reason    text,
    ADD COLUMN IF NOT EXISTS auth_hold_at        timestamptz;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'virtual_fitting_async_jobs_auth_hold_check'
          AND conrelid = 'virtual_fitting_async_jobs'::regclass
    ) THEN
        ALTER TABLE virtual_fitting_async_jobs
            ADD CONSTRAINT virtual_fitting_async_jobs_auth_hold_check CHECK (
                (auth_hold_task_uuid IS NULL AND auth_hold_reason IS NULL AND auth_hold_at IS NULL)
                OR (auth_hold_task_uuid IS NOT NULL AND auth_hold_at IS NOT NULL
                    AND auth_hold_reason ~ '^[A-Za-z0-9_.-]{1,64}$')
            );
    END IF;
END
$$;
