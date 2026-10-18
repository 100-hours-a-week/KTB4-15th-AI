-- 가상피팅 V2 비동기 작업 (AI PostgreSQL). schema.sql 과 같이 여러 번 실행해도 된다.
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/virtual_fitting_async_jobs.sql
--
-- 한 줄 = BE 의 fitting_job_id 하나. AI 는 이 번호를 새로 만들지 않고, BE DB 와 따로라서 외래키도 없다.
-- 같은 fitting_job_id 가 다시 오면 새 줄을 만들거나 덮어쓰지 않는다(PRIMARY KEY + ON CONFLICT DO NOTHING).
-- Presigned URL·인증 토큰·이미지 파일은 저장하지 않는다. S3 Key 만 둔다.
-- 시각은 모두 timestamptz(UTC 기준으로 다룬다). requested_at 은 BE 요청 시각, created_at 은 AI 가 저장한 시각.
-- 상태 전환·실행 권한·횟수 한도 판단은 이 파일이 아니라 코드(5단계 이후)가 한다. 여기서는 값의 범위만 막는다.
CREATE TABLE IF NOT EXISTS virtual_fitting_async_jobs (
    fitting_job_id                bigint       PRIMARY KEY CHECK (fitting_job_id > 0),
    -- 같은 작업 번호로 다른 입력이 왔는지 비교할 값. 만드는 방법은 이후 단계에서 정한다.
    request_fingerprint           text         NOT NULL CHECK (request_fingerprint <> ''),
    -- 원본 입력. 완료 후 정리할 수 있게 NULL 을 허용한다.
    user_image_key                text,
    products                      text[]       CHECK (
        products IS NULL
        OR (cardinality(products) >= 1 AND array_position(products, NULL) IS NULL)
    ),
    -- Runware taskUUID. 호출 준비 전에는 NULL. 값이 있으면 작업끼리 겹치지 않는다.
    task_uuid                     uuid         UNIQUE,
    async_image_key               text,
    result_image_key              text,
    status                        text         NOT NULL DEFAULT 'RECEIVED' CHECK (
        status IN (
            'RECEIVED', 'WAITING_PRUNA', 'POSTPROCESSING', 'FALLBACK_RUNNING', 'COMPLETED', 'FAILED'
        )
    ),
    requested_at                  timestamptz  NOT NULL,
    created_at                    timestamptz  NOT NULL DEFAULT now(),
    updated_at                    timestamptz  NOT NULL DEFAULT now(),
    pruna_submit_started_at       timestamptz,
    runware_accepted_at           timestamptz,
    next_action                   text         DEFAULT 'SUBMIT_PRUNA' CHECK (
        next_action IN ('SUBMIT_PRUNA', 'POSTPROCESS', 'RUN_FALLBACK', 'PUBLISH_RESULT')
    ),
    verified_image_key            text,
    image_verified_at             timestamptz,
    -- be-fitting-response 로 보낼 최종 메시지(code/data/message).
    result_payload                jsonb        CHECK (
        result_payload IS NULL OR jsonb_typeof(result_payload) = 'object'
    ),
    result_message_status         text         NOT NULL DEFAULT 'PENDING' CHECK (
        result_message_status IN ('PENDING', 'PUBLISHED', 'RETRY_EXHAUSTED')
    ),
    result_message_attempt_count  integer      NOT NULL DEFAULT 0 CHECK (result_message_attempt_count >= 0),
    llm_generation_attempt_count  integer      NOT NULL DEFAULT 0 CHECK (llm_generation_attempt_count >= 0),
    fallback_attempt_count        integer      NOT NULL DEFAULT 0 CHECK (fallback_attempt_count >= 0),
    s3_check_attempt_count        integer      NOT NULL DEFAULT 0 CHECK (s3_check_attempt_count >= 0),
    worker_lease_token            text,
    worker_lease_until            timestamptz
);
