-- 가상피팅 V2 fallback(13단계) 진행 기록. virtual_fitting_async_jobs 를 바꾸는 변경 SQL (AI PostgreSQL).
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/virtual_fitting_fallback.sql
--
-- 적용 전제·순서: virtual_fitting_async_jobs.sql 로 테이블이 있어야 한다. 그 CREATE TABLE IF NOT EXISTS 를 다시 실행해도
-- 기존 테이블은 바뀌지 않으므로 이 파일이 따로 필요하다. 이 SQL 을 먼저 적용한 뒤 새 코드를 배포한다(새 코드는 이 컬럼을
-- SELECT 하므로 없으면 DB 오류). 재실행해도 바뀌는 것이 없다(ADD COLUMN IF NOT EXISTS · CREATE INDEX IF NOT EXISTS).
-- 기존 행은 모두 NULL 이 된다(fallback 을 시작한 적 없음). 값을 채우거나 추측하지 않는다.
--
--   fallback_task_uuid  : V1 동기 생성(fallback) 요청의 taskUUID. 비동기 task_uuid 와 별개다(늦은 Webhook·작업 조회는
--                         계속 task_uuid 로 한다). 외부 호출 전에 저장하고 재수신 때 재사용한다. 비용·공유 제한 자리를
--                         이 시도와 연결한다.
--   fallback_started_at : 생성 요청을 보내기 직전 기록(fallback_attempt_count 증가·공유 제한 자리와 같은 트랜잭션).
--                         "보냈을 수 있음"이며 결과 근거가 아니다. 값이 있으면 다시 생성하지 않는다.
--   fallback_image_key  : 생성 결과를 S3(virtual-fitting/results/{id}/pruna-sync.jpg)에 저장했다는 기록. 이 값이 있으면
--                         생성·저장은 끝났고 제목·코멘트·결과 저장부터 이어 간다.
-- Presigned URL·Runware 결과 URL 은 저장하지 않는다.
ALTER TABLE virtual_fitting_async_jobs
    ADD COLUMN IF NOT EXISTS fallback_task_uuid  uuid,
    ADD COLUMN IF NOT EXISTS fallback_started_at timestamptz,
    ADD COLUMN IF NOT EXISTS fallback_image_key  text;

CREATE UNIQUE INDEX IF NOT EXISTS virtual_fitting_async_jobs_fallback_task_uuid_key
    ON virtual_fitting_async_jobs (fallback_task_uuid);
