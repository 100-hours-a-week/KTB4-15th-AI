-- 가상피팅 V2 결과 저장 방식 구분. virtual_fitting_async_jobs 를 바꾸는 변경 SQL (AI PostgreSQL).
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/virtual_fitting_ai_result_storage.sql
--
-- 적용 전제·순서: virtual_fitting_async_jobs.sql → virtual_fitting_fallback.sql → virtual_fitting_auth_hold.sql →
-- virtual_fitting_runware_image_ref.sql → (이 파일) → 새 코드. 새 코드는 이 컬럼을 SELECT 하므로 없으면 DB 오류다.
-- 재실행해도 바뀌는 것이 없다(ADD COLUMN IF NOT EXISTS). NOT NULL DEFAULT false 추가는 기존 행을 다시 쓰지 않는다(PG 11+).
--
--   ai_stores_result : true 면 이 작업의 Pruna 결과는 AI 가 Runware 결과 URL 에서 내려받아 async_image_key 에 저장한다
--                      (uploadEndpoint 없이 요청한 작업). false(기존 행·현재 요청 전부)는 Runware 가 uploadEndpoint 로
--                      async_image_key 에 직접 올리는 기존 방식이다.
-- 기존 기록으로는 두 방식을 구분할 수 없다: async_image_key 는 같은 .webp Key 이고, 결과 URL 은 기존 방식 작업에도 저장되며,
-- 객체 유무·시각은 진행 상황일 뿐 요청 방식의 근거가 아니다. 그래서 요청을 준비할 때(task_uuid 저장과 같은 UPDATE) 한 번만
-- 정하고 그 뒤에는 바꾸지 않는다. 기존 행에 true 를 채우지 않는다.
ALTER TABLE virtual_fitting_async_jobs
    ADD COLUMN IF NOT EXISTS ai_stores_result boolean NOT NULL DEFAULT false;
