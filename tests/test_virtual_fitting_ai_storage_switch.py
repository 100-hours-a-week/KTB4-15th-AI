import hashlib
import json
import uuid
from datetime import datetime

import pytest

from app.virtual_fitting.providers.base import FittingInput
from app.virtual_fitting.providers.runware_async import build_async_payload
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    NewAsyncFittingJob,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository
from tests.test_virtual_fitting_v2_scenarios import (  # noqa: F401 (pytest fixture)
    JOB,
    age,
    client,
    env,
    expire_lease,
    finish,
    row,
    submitted,
)
from tests.v2_integration_harness import (
    USER_IMAGE_KEY,
    WEBHOOK_URL,
    Stack,
    post_webhook,
    prepared_by_previous_version,
)

KEY = f"virtual-fitting/results/{JOB}/pruna-async.webp"
# 배경 제거 결과(최종 결과·LLM 입력). 원본 KEY 는 그대로 남는다.
TRANSPARENT_KEY = f"virtual-fitting/results/{JOB}/transparent.webp"
SECRETS = ("fake-cdn-token", "FAKESIGNATURE", WEBHOOK_URL.split("token=")[1])


def payload_of(env, task):  # noqa: F811
    return env.runware.submissions[task].payload


def presigned_puts(env):  # noqa: F811
    return [key for method, key in env.s3.presigned if method == "put_object"]


def assert_no_secret(text):
    for secret in SECRETS:
        assert secret not in text, secret


def prepare_existing(env, job_id, *, ai, key=None):  # noqa: F811
    """이전 실행이 준비만 하고 보내기 전에 멈춘 작업(같은 BE 요청, task·Key·저장 방식이 DB 에 있음)."""
    from app.virtual_fitting.repositories.async_job_repository import AsyncJobRepository
    from app.virtual_fitting.request_worker import request_fingerprint

    message = env.request(job_id)
    codes = tuple(item["product_code"] for item in message["products"])
    pool = env.pools[-1]
    jobs, state = AsyncJobRepository(pool), AsyncJobStateRepository(pool)
    created = jobs.create(
        NewAsyncFittingJob(
            job_id,
            request_fingerprint(USER_IMAGE_KEY, codes),
            USER_IMAGE_KEY,
            codes,
            datetime.fromisoformat(message["requested_at"]),
        )
    )
    assert created.created
    grant = state.acquire_lease(job_id, NextAction.SUBMIT_PRUNA, 60)
    task = str(uuid.uuid4())
    prepared = state.prepare_pruna_submission(
        job_id,
        grant.token,
        task,
        key or f"virtual-fitting/results/{job_id}/pruna-async.webp",
        ai_stores_result=ai,
    )
    assert prepared.applied and prepared.job.ai_stores_result is ai
    state.release_lease(job_id, grant.token)
    return task


# === 1~3. 신규 작업: true·task·Key 함께, PUT 발급 0회, uploadEndpoint 없음 ===


def test_new_job_is_prepared_as_ai_storage_and_sent_without_upload_endpoint(env):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    current = row(env)
    assert (current.ai_stores_result, current.task_uuid, current.async_image_key) == (
        True,
        task,
        KEY,
    )
    assert presigned_puts(env) == []  # 결과 PUT URL 발급 0회
    assert env.s3.presigned == [("get_object", USER_IMAGE_KEY)]  # 사용자 이미지 GET 은 그대로
    task_payload = payload_of(env, task)
    assert "uploadEndpoint" not in task_payload  # null·빈 문자열도 아님
    assert task_payload["webhookURL"] == WEBHOOK_URL and task_payload["taskUUID"] == task
    assert (task_payload["deliveryMethod"], task_payload["outputType"]) == ("async", "URL")
    assert (task_payload["outputFormat"], task_payload["outputQuality"]) == ("WEBP", 90)
    assert task_payload["numberResults"] == 1
    person = task_payload["inputs"]["referenceImages"][0]
    assert person["role"] == "person" and USER_IMAGE_KEY in person["image"]
    assert [r["role"] for r in task_payload["inputs"]["referenceImages"][1:]] == [
        "garment",
        "garment",
    ]
    assert (current.status, current.next_action) == (AsyncJobStatus.WAITING_PRUNA, None)


# === 4~6. 이미 준비된 작업은 DB 기록대로 ===


def test_prepared_direct_upload_job_keeps_the_previous_contract_on_redelivery(env):  # noqa: F811
    task = prepare_existing(env, JOB, ai=False)
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    stack.request_once()
    current = row(env)
    assert (current.ai_stores_result, current.task_uuid) == (
        False,
        task,
    )  # 새 코드라고 true 로 바꾸지 않음
    task_payload = payload_of(env, task)
    assert presigned_puts(env) == [KEY] and task_payload["uploadEndpoint"].split("?")[0].endswith(
        KEY
    )
    assert (task_payload["outputFormat"], task_payload["outputQuality"]) == ("WEBP", 90)


def test_prepared_ai_job_keeps_ai_storage_on_redelivery(env):  # noqa: F811
    task = prepare_existing(env, JOB, ai=True)
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    stack.request_once()
    assert row(env).ai_stores_result is True and row(env).task_uuid == task
    assert presigned_puts(env) == [] and "uploadEndpoint" not in payload_of(env, task)


def test_prepared_legacy_jpg_job_keeps_jpg_without_webp_quality(env):  # noqa: F811
    jpg = f"virtual-fitting/results/{JOB}/pruna-async.jpg"
    task = prepare_existing(env, JOB, ai=False, key=jpg)
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    stack.request_once()
    task_payload = payload_of(env, task)
    assert row(env).async_image_key == jpg  # 이미 준비한 Key 를 .webp 로 바꾸지 않음
    assert task_payload["outputFormat"] == "JPG" and "outputQuality" not in task_payload
    assert presigned_puts(env) == [jpg] and "uploadEndpoint" in task_payload


# === 7. 준비 DB 저장 실패·commit 응답 유실·경합 ===


def flaky_prepare(env, mode):  # noqa: F811
    """첫 prepare_pruna_submission 을 실패시킨다. mode=before: 저장 전 실패, after: 저장 뒤 응답 유실."""
    from app.virtual_fitting.exceptions import FittingDatabaseError

    original = AsyncJobStateRepository.prepare_pruna_submission
    calls = []

    def prepare(self, *args, **kwargs):
        calls.append(kwargs.get("ai_stores_result"))
        if len(calls) == 1:
            if mode == "after":
                original(self, *args, **kwargs)
            raise FittingDatabaseError("injected")
        return original(self, *args, **kwargs)

    env.monkeypatch.setattr(AsyncJobStateRepository, "prepare_pruna_submission", prepare)
    return calls


@pytest.mark.parametrize("mode", ["before", "after"])
def test_prepare_failure_sends_nothing_and_redelivery_uses_the_db_record(env, mode):  # noqa: F811
    calls = flaky_prepare(env, mode)
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    (first,) = stack.request_once()
    assert (
        first.delete is False and env.runware.count("async_submit") == 0
    )  # 확인 전에는 보내지 않음
    env.sqs.expire_visibility("be-fitting-request")
    stack.request_once()
    current = row(env)
    assert calls == [True, True] and current.ai_stores_result is True
    assert env.runware.count("async_submit") == 1
    assert "uploadEndpoint" not in payload_of(env, current.task_uuid)


def test_job_prepared_concurrently_by_another_version_follows_the_final_db_record(env):  # noqa: F811
    # 다른 Worker(이전 버전)가 먼저 준비했다: 이 Worker 의 true 는 적용되지 않고 DB 의 false 로 조립한다.
    prepared_by_previous_version(env.monkeypatch, JOB)
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    assert row(env).ai_stores_result is False and "uploadEndpoint" in payload_of(env, task)


# === 8. 호출 시작·접수 불명확은 재전송하지 않음 ===


def test_started_but_unknown_is_never_resent_and_keeps_its_mode(env):  # noqa: F811
    env.runware.async_modes.append("timeout")
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    env.sqs.expire_visibility("be-fitting-request")
    expire_lease(env)
    stack.request_once()
    assert env.runware.count("async_submit") == 1
    current = row(env)
    assert current.task_uuid == task and current.ai_stores_result is True
    assert current.pruna_submit_started_at is not None and current.runware_accepted_at is None


# === 9. 저장 방식과 Provider 인자 불일치는 전송 전 거절 ===


INPUT = FittingInput("https://bucket/u.png?X-Amz-Signature=s", ["https://img/g.jpg"], "p")
TASK = "a770f077-f413-47de-9dac-be0b26a35da6"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ai_stores_result": True, "upload_endpoint": "https://bucket/k.webp?X-Amz-Signature=s"},
        {"ai_stores_result": True, "upload_endpoint": None, "output_format": "JPG"},
        {"ai_stores_result": False, "upload_endpoint": None},
        {"ai_stores_result": "yes", "upload_endpoint": None},
    ],
)
def test_mode_and_arguments_must_agree(kwargs):
    kwargs.setdefault("output_format", "WEBP")
    with pytest.raises(ValueError):
        build_async_payload(INPUT, task_uuid=TASK, webhook_url=WEBHOOK_URL, **kwargs)


def test_default_contract_still_requires_the_endpoint_and_includes_it():
    (task,) = build_async_payload(
        INPUT,
        task_uuid=TASK,
        webhook_url=WEBHOOK_URL,
        upload_endpoint="https://bucket/k.webp?X-Amz-Signature=s",
        output_format="WEBP",
    )
    assert task["uploadEndpoint"].startswith("https://bucket/k.webp")
    (ai,) = build_async_payload(
        INPUT, task_uuid=TASK, webhook_url=WEBHOOK_URL, upload_endpoint=None, ai_stores_result=True
    )
    assert set(task) - set(ai) == {"uploadEndpoint"}  # 차이는 uploadEndpoint 하나


def test_ai_job_with_a_non_webp_key_is_rejected_before_sending(env):  # noqa: F811
    prepare_existing(env, JOB, ai=True, key=f"virtual-fitting/results/{JOB}/pruna-async.jpg")
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    (outcome,) = stack.request_once()
    assert outcome.delete is False and env.runware.count("async_submit") == 0
    assert row(env).pruna_submit_started_at is None  # 보내지 않은 것이 확실해 시작 기록을 되돌렸다


# === 10. 신규 요청 전체 흐름(결과 URL → 다운로드·S3 → LLM → BE 결과) ===


def test_new_request_flows_to_the_be_result_with_the_original_webp_bytes(env):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    body = env.runware.complete(task)
    assert post_webhook(client(), body).status_code == 200
    finish(env, stack)
    final = row(env)
    assert final.status is AsyncJobStatus.COMPLETED
    assert final.result_message_status is ResultMessageStatus.PUBLISHED
    url = env.runware.submissions[task].result_url
    original = env.runware.cdn[url]
    stored_body, content_type, _ = env.s3.objects[KEY]
    assert stored_body == original and content_type == "image/webp"  # 원본 bytes 그대로
    assert env.s3.metadata[KEY] == {
        "fitting-job-id": str(JOB),
        "task-uuid": task,
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    # 저장된 URL 그대로(token 포함) 원본을 받고, 그다음 배경 제거 결과를 받는다.
    assert env.runware.downloads[0] == url and len(env.runware.downloads) == 2
    assert env.runware.rembg_inputs == [KEY] and env.runware.count("rembg") == 1
    transparent, transparent_type, _ = env.s3.objects[TRANSPARENT_KEY]
    assert transparent_type == "image/webp" and transparent != original
    assert (
        KEY not in env.s3.heads
    )  # 결과는 Runware 업로드 확인(HEAD) 경로를 쓰지 않음(사용자 이미지 HEAD 는 기존대로)
    (be,) = env.sqs.bodies("be-fitting-response")
    assert set(be) == {"fitting_job_id", "status", "result_image_key", "llm_title", "llm_comment"}
    assert be["result_image_key"] == TRANSPARENT_KEY and be["status"] == "COMPLETED"
    assert final.verified_image_key == final.async_image_key == KEY  # 원본 확인 기록은 그대로
    # 내부 큐에는 작업 참조만
    for message in env.sqs.bodies("ai-fitting-response"):
        assert set(message) == {"fitting_job_id", "task_uuid", "action"}
    assert env.runware.count("llm_comment") == 1 and env.runware.count("llm_title") == 1
    assert ("get_object", KEY) in env.s3.presigned  # RemBG 에는 원본의 Presigned GET
    assert (
        "get_object",
        TRANSPARENT_KEY,
    ) in env.s3.presigned  # LLM 에는 투명 이미지의 Presigned GET
    assert_no_secret(json.dumps(env.sqs.bodies("be-fitting-response")))


# === 11. 새 작업과 기존 직접 업로드 작업 혼합 ===


def test_new_and_existing_jobs_run_side_by_side(env):  # noqa: F811
    legacy, new = JOB, JOB + 1
    prepared_by_previous_version(env.monkeypatch, legacy)
    stack = Stack(env).mark_running()
    legacy_task = submitted(env, stack, legacy)
    new_task = submitted(env, stack, new)
    client_ = client()
    for task in (legacy_task, new_task):
        assert post_webhook(client_, env.runware.complete(task)).status_code == 200
    for _ in range(4):
        stack.postprocess_once()
    expire_lease(env, legacy)
    expire_lease(env, new)
    stack.publish_once()
    assert row(env, legacy).ai_stores_result is False and row(env, new).ai_stores_result is True
    assert row(env, legacy).status is row(env, new).status is AsyncJobStatus.COMPLETED
    legacy_key = f"virtual-fitting/results/{legacy}/pruna-async.webp"
    new_key = f"virtual-fitting/results/{new}/pruna-async.webp"
    assert legacy_key in env.s3.heads and new_key not in env.s3.heads  # 기존: HEAD 확인
    legacy_clear = f"virtual-fitting/results/{legacy}/transparent.webp"
    new_clear = f"virtual-fitting/results/{new}/transparent.webp"
    # 원본은 새 작업만 AI 가 저장(기존 객체는 덮어쓰지 않음). 투명 결과·RemBG 호출 표시는 두 작업 모두 별도 Key 에 저장.
    markers = [f"virtual-fitting/results/{job}/rembg-call.json" for job in (legacy, new)]
    assert sorted(p["Key"] for p in env.s3.puts) == sorted(
        [new_key, legacy_clear, new_clear, *markers]
    )
    assert presigned_puts(env) == [legacy_key]
    assert sorted(env.runware.rembg_inputs) == sorted([legacy_key, new_key])
    assert sorted(b["result_image_key"] for b in env.sqs.bodies("be-fitting-response")) == sorted(
        [legacy_clear, new_clear]
    )


# === 12. 접수 응답보다 빠른 Webhook(접수 시각 NULL) + URL 없는 성공 ===


def test_fast_uuid_only_webhook_before_the_accept_record_is_recovered(env):  # noqa: F811
    stack = Stack(env).mark_running()
    hook = client()
    seen = {}

    def webhook_first(submission):
        # 접수 응답을 돌려주기 전에(DB 에 접수 기록이 없을 때) UUID 만 담긴 성공 알림이 먼저 온다.
        before = row(env)
        seen["accepted_before"] = before.runware_accepted_at
        body = json.dumps(
            {
                "taskType": "imageInference",
                "taskUUID": submission.task_uuid,
                "imageUUID": submission.image_uuid,
                "cost": 0.02,
            }
        ).encode()
        seen["status"] = post_webhook(hook, body).status_code
        return "accept"

    env.runware.async_modes.append(webhook_first)
    task = submitted(env, stack)
    current = row(env)
    assert seen == {"accepted_before": None, "status": 200}
    # 늦은 접수 응답이 상태를 되돌리지 않았고, 성공 전환이 접수 시각을 채웠다(복구 시점 기준).
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert current.runware_accepted_at is not None and current.runware_image_url is None
    assert env.sqs.bodies("ai-fitting-response") == []  # URL 없이 POSTPROCESS 를 보내지 않음
    # 3분 뒤 Watchdog 이 같은 task 를 조회해 URL 을 저장하고 POSTPROCESS 를 보낸다.
    age(env, "runware_accepted_at", 200)
    env.runware.success_status(task)
    stack.watchdog_once()
    assert row(env).runware_image_url == env.runware.submissions[task].result_url
    assert [b["action"] for b in env.sqs.bodies("ai-fitting-response")] == ["POSTPROCESS"]
    finish(env, stack)
    assert row(env).status is AsyncJobStatus.COMPLETED and KEY in env.s3.metadata


def test_recovery_time_is_not_pushed_back_by_lease_updates(env):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    body = json.dumps(
        {
            "taskType": "imageInference",
            "taskUUID": task,
            "imageUUID": env.runware.submissions[task].image_uuid,
        }
    ).encode()
    assert post_webhook(client(), body).status_code == 200
    age(env, "runware_accepted_at", 200)
    env.sql(  # Worker 재수신·lease 갱신으로 updated_at 이 방금 바뀌었다
        "UPDATE virtual_fitting_async_jobs SET updated_at = statement_timestamp() WHERE fitting_job_id = %s",
        (JOB,),
    )
    env.runware.success_status(task)
    stack.watchdog_once()
    assert row(env).runware_image_url is not None


# === 13. URL 저장 실패 503 → 재전송, Webhook 누락 → 조회 복구 ===


def test_url_not_saved_is_503_and_redelivery_completes(env):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    env.sql(
        """
        CREATE FUNCTION fail_url() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.runware_image_url IS NOT NULL THEN RAISE EXCEPTION 'injected'; END IF;
            RETURN NEW;
        END $$;
        CREATE TRIGGER fail_url BEFORE UPDATE ON virtual_fitting_async_jobs
            FOR EACH ROW EXECUTE FUNCTION fail_url();
        """
    )
    body = env.runware.complete(task)
    response = post_webhook(client(), body)
    assert response.status_code == 503 and response.json()["code"] == "WEBHOOK_RESULT_REF_NOT_SAVED"
    assert row(env).status is AsyncJobStatus.WAITING_PRUNA
    env.sql("DROP TRIGGER fail_url ON virtual_fitting_async_jobs")
    assert post_webhook(client(), body).status_code == 200
    finish(env, stack)
    assert row(env).status is AsyncJobStatus.COMPLETED


def test_missing_webhook_is_recovered_by_the_status_query(env):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    age(env, "runware_accepted_at", 200)
    env.runware.complete(task)  # 결과는 생겼지만 Webhook 이 오지 않았다
    env.runware.success_status(task)
    stack.watchdog_once()
    current = row(env)
    assert current.status is AsyncJobStatus.POSTPROCESSING
    assert current.runware_image_url == env.runware.submissions[task].result_url
    finish(env, stack)
    assert row(env).status is AsyncJobStatus.COMPLETED


# === 14. S3 성공 뒤 DB 실패·Worker 중단·중복 큐 → 기존 객체 재사용 ===


def test_s3_saved_then_db_failure_resumes_without_a_second_download(env):  # noqa: F811
    from app.virtual_fitting.exceptions import FittingDatabaseError

    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    assert post_webhook(client(), env.runware.complete(task)).status_code == 200
    original = AsyncJobStateRepository.record_image_verified
    failed = []

    def fail_once(self, *args, **kwargs):
        if not failed:
            failed.append(1)
            raise FittingDatabaseError("injected")
        return original(self, *args, **kwargs)

    env.monkeypatch.setattr(AsyncJobStateRepository, "record_image_verified", fail_once)
    (first,) = stack.postprocess_once()
    assert first.delete is False and KEY in env.s3.objects and row(env).verified_image_key is None
    # 같은 메시지 재수신 + 중복 메시지(예: 재발행)
    env.sqs.put(
        "ai-fitting-response", {"fitting_job_id": JOB, "task_uuid": task, "action": "POSTPROCESS"}
    )
    env.sqs.expire_visibility("ai-fitting-response")
    expire_lease(env)
    for _ in range(3):
        stack.postprocess_once()
    expire_lease(env)
    stack.publish_once()
    assert row(env).status is AsyncJobStatus.COMPLETED
    # 원본 1번 + 배경 제거 결과 1번(+ RemBG 호출 표시 1번). 다시 받지도 올리지도 않음
    assert len(env.runware.downloads) == 2 and len(env.s3.puts) == 3
    assert env.runware.count("rembg") == 1  # 배경 제거도 한 번(유료 API 재호출 없음)
    assert env.runware.count("llm_comment") == 1  # LLM 도 한 번


# === 16. 저장 오류는 재생성·fallback·FAILED 가 아니다 ===


def test_expired_or_missing_result_url_never_regenerates_or_falls_back(env):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    assert post_webhook(client(), env.runware.complete(task)).status_code == 200
    env.runware.cdn.clear()  # URL 만료·삭제(404)
    for _ in range(3):
        stack.postprocess_once()
        env.sqs.expire_visibility("ai-fitting-response")
        expire_lease(env)
    current = row(env)
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert current.result_payload is None and current.fallback_attempt_count == 0
    assert env.runware.count("sync_generate") == 0 and env.runware.count("async_submit") == 1
    assert len(env.sqs.live("ai-fitting-response")) == 1  # 메시지는 남는다(기존 재수신·DLQ 정책)


# === 17. 최종 결과 저장 뒤 발행 실패 → LLM 재실행 없이 발행 ===


def test_publish_failure_after_the_result_is_recovered_without_llm(env):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    assert post_webhook(client(), env.runware.complete(task)).status_code == 200
    for _ in range(2):
        stack.postprocess_once()
    assert row(env).next_action is NextAction.PUBLISH_RESULT
    env.sqs.fault("be-fitting-response", "send", "throttled")
    expire_lease(env)
    stack.publish_once()
    assert env.sqs.bodies("be-fitting-response") == []
    calls = (env.runware.count("llm_comment"), env.runware.count("llm_title"))
    expire_lease(env)
    stack.publish_once()
    assert row(env).result_message_status is ResultMessageStatus.PUBLISHED
    assert (env.runware.count("llm_comment"), env.runware.count("llm_title")) == calls
    assert len(env.runware.downloads) == 2 and env.runware.count("rembg") == 1


# === 18. 내부 큐 삭제 시점·재발행 제한 ===


def test_message_is_kept_until_the_result_is_saved_and_republish_is_bounded(env):  # noqa: F811
    stack = Stack(env).mark_running()
    task = submitted(env, stack)
    assert post_webhook(client(), env.runware.complete(task)).status_code == 200
    env.runware.cdn.clear()
    (outcome,) = stack.postprocess_once()
    assert outcome.delete is False  # 저장이 안 됐으면 지우지 않는다
    # 오래 진행이 없으면 Watchdog 은 한 번만 다시 발행하고, 그 뒤에는 알림만 남긴다(DLQ 우회 없음).
    for _ in range(3):
        expire_lease(env)
        age(env, "runware_accepted_at", 7200)  # AI 후처리 단계 시작(성공 전환) 기준 장기 미진행
        env.sql("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
        env.sql(
            "UPDATE virtual_fitting_watchdog_checks SET follow_up_published_at = "
            "LEAST(follow_up_published_at, now() - interval '1 hour')"
        )
        stack.watchdog_once()
    actions = [b["action"] for b in env.sqs.bodies("ai-fitting-response")]
    assert actions == ["POSTPROCESS", "POSTPROCESS"]  # Webhook 1번 + Watchdog 재발행 1번
    assert env.network_attempts == []
    _ = task
