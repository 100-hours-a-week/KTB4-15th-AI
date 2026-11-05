-- 가상피팅 V2 Runware 결과 참조 보존. virtual_fitting_async_jobs 를 바꾸는 변경 SQL (AI PostgreSQL).
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/virtual_fitting_runware_image_ref.sql
--
-- 적용 전제·순서: virtual_fitting_async_jobs.sql → virtual_fitting_fallback.sql → virtual_fitting_auth_hold.sql → (이 파일)
-- → 새 코드. 새 코드는 이 컬럼을 SELECT 하므로 없으면 DB 오류다. 재실행해도 바뀌는 것이 없다(ADD COLUMN IF NOT EXISTS,
-- 제약은 없을 때만 추가). 기존 행은 모두 NULL(받은 결과 참조 없음)이 된다. 값을 채우거나 추측하지 않는다.
--
-- 비동기 생성 시도(task_uuid)의 성공 알림(Webhook 또는 Watchdog 의 getResponse)이 준 결과 참조를 남긴다. 추후 결과 이미지를
-- 내려받아 복구하기 위한 내부 기록이며, BE 메시지·후처리 큐 메시지·결과 S3 Key 에는 쓰지 않는다. 결과 이미지 확인은 지금처럼
-- async_image_key(Runware 가 uploadEndpoint 로 직접 올린 S3 객체)로 한다.
--   runware_image_url  : Runware 응답의 imageURL 원본 그대로(쿼리의 token·서명·만료 값 포함). 허용 호스트(im.runware.ai)의
--                        https URL 만 저장한다(코드에서 확인). 우리 업로드용 Presigned PUT(S3 버킷 호스트)은 저장하지 않는다.
--                        다운로드 자격이 담길 수 있으므로 로그·BE·큐 메시지에 쓰지 않는다.
--   runware_image_uuid : Runware 응답의 imageUUID. Runware 가 만든 결과 이미지 식별자로, 우리가 만든 요청 식별자 task_uuid 와 다르다.
-- 성공 전환(RECEIVED·WAITING_PRUNA → POSTPROCESSING)과 같은 UPDATE 로 처음 기록한다. 그 뒤에는 후처리 중(POSTPROCESSING /
-- POSTPROCESS, 결과 없음)이고 같은 결과로 확인된(다른 쪽 참조가 일치) 늦은 알림이 비어 있는 컬럼만 보완한다. 이미 있는 값은
-- 바꾸거나 지우지 않는다. fallback(동기) 생성 결과는 여기에 두지 않는다(fallback_image_key).
ALTER TABLE virtual_fitting_async_jobs
    ADD COLUMN IF NOT EXISTS runware_image_url  text,
    ADD COLUMN IF NOT EXISTS runware_image_uuid uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'virtual_fitting_async_jobs_runware_image_url_check'
          AND conrelid = 'virtual_fitting_async_jobs'::regclass
    ) THEN
        ALTER TABLE virtual_fitting_async_jobs
            ADD CONSTRAINT virtual_fitting_async_jobs_runware_image_url_check CHECK (
                runware_image_url IS NULL
                OR (left(runware_image_url, 8) = 'https://' AND length(runware_image_url) <= 2048)
            );
    END IF;
END
$$;
