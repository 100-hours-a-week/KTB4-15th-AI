"""요청 메시지 삭제(인계 판단)와 Watchdog 재발행 기준을 실제 조립 경로(요청 Worker·Webhook·후처리 Worker·Watchdog)로 검증한다.

- 요청 메시지는 DB 기록과 그 단계를 실제로 이어 갈 주체(후처리 Worker·Watchdog·결과 전송 루프)의 준비 상태로만 지운다.
- AI 저장 작업의 후처리 재발행은 같은 단계에서 한 번만이고, lease·재수신(updated_at 변경)으로 다시 열리지 않는다.
외부 경계는 v2_integration_harness 의 가짜(SQS·S3·Runware·결과 다운로드)다. 실제 네트워크·DNS 는 쓰지 않는다.
"""

import json

from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from tests.test_virtual_fitting_v2_scenarios import (  # noqa: F401 (pytest fixture)
    JOB,
    age,
    client,
    env,
    expire_lease,
    finish,
    restart,
    row,
)
from tests.v2_integration_harness import Stack, post_webhook

REQUEST = "be-fitting-request"
FOLLOW_UP = "ai-fitting-response"


def fast_webhook(env, *, uuid_only=False):  # noqa: F811
    """접수 응답을 기록하기 전에 성공 Webhook 이 먼저 온다(그 뒤 접수 응답)."""
    seen = {}

    def deliver_first(submission):
        if uuid_only:
            body = json.dumps(
                {
                    "taskType": "imageInference",
                    "taskUUID": submission.task_uuid,
                    "imageUUID": submission.image_uuid,
                    "cost": 0.02,
                }
            ).encode()
        else:
            body = env.runware.complete(submission.task_uuid)
        seen["status"] = row(env).status
        seen["webhook"] = post_webhook(client(), body).status_code
        return "accept"

    env.runware.async_modes.append(deliver_first)
    return seen


def request_messages(env):  # noqa: F811
    return len(env.sqs.live(REQUEST))


def redeliver_request(env, stack):  # noqa: F811
    env.sqs.expire_visibility(REQUEST)
    expire_lease(env)
    return stack.request_once()


# === 요청 메시지: 빠른 Webhook 뒤 후처리 인계 ===


def test_fast_webhook_then_postprocess_handoff_deletes_the_request_message(env):  # noqa: F811
    stack = Stack(env).mark_running()
    seen = fast_webhook(env)
    env.send_request(JOB)
    (outcome,) = stack.request_once()
    current = row(env)
    assert seen == {"status": AsyncJobStatus.RECEIVED, "webhook": 200}
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert current.runware_image_url is not None  # 필요한 결과 참조가 저장됐다
    assert [b["action"] for b in env.sqs.bodies(FOLLOW_UP)] == ["POSTPROCESS"]
    assert outcome.delete is True and request_messages(env) == 0
    finish(env, stack)
    assert row(env).result_message_status is ResultMessageStatus.PUBLISHED


def test_fast_webhook_without_a_running_postprocess_worker_keeps_the_message(env):  # noqa: F811
    stack = Stack(env).mark_running(postprocess=False)
    fast_webhook(env)
    env.send_request(JOB)
    (outcome,) = stack.request_once()
    assert row(env).next_action is NextAction.POSTPROCESS
    assert outcome.delete is False and request_messages(env) == 1


def test_fast_webhook_without_a_running_watchdog_keeps_the_message(env):  # noqa: F811
    # 후처리 메시지가 유실돼도 DB 기록으로 다시 발행할 주체(Watchdog)가 없으면 인계됐다고 보지 않는다.
    stack = Stack(env).mark_running(watchdog=False)
    fast_webhook(env)
    env.send_request(JOB)
    (outcome,) = stack.request_once()
    assert outcome.delete is False and request_messages(env) == 1


def test_redelivery_after_a_late_ready_path_deletes_without_running_anything_again(env):  # noqa: F811
    stack = Stack(env).mark_running(postprocess=False)
    fast_webhook(env)
    env.send_request(JOB)
    stack.request_once()
    assert request_messages(env) == 1
    stack.mark_running()  # 후처리 Worker 가 이제 돈다
    (outcome,) = redeliver_request(env, stack)
    assert outcome.delete is True and request_messages(env) == 0
    # 재수신은 생성·LLM·비용을 다시 하지 않는다.
    assert env.runware.count("async_submit") == 1
    assert env.runware.count("llm_comment") == env.runware.count("llm_title") == 0
    assert len(env.sql("SELECT 1 FROM virtual_fitting_task_costs")) == 1


# === 요청 메시지: UUID 만 받은 참조 복구 대기 ===


def test_uuid_only_fast_webhook_is_handed_to_the_watchdog(env):  # noqa: F811
    stack = Stack(env).mark_running(postprocess=False)  # 참조 복구 단계에는 Watchdog 이 필요하다
    fast_webhook(env, uuid_only=True)
    env.send_request(JOB)
    (outcome,) = stack.request_once()
    current = row(env)
    assert current.runware_image_url is None and current.runware_image_uuid is not None
    assert env.sqs.bodies(FOLLOW_UP) == []
    assert outcome.delete is True and request_messages(env) == 0


def test_uuid_only_fast_webhook_without_a_watchdog_keeps_the_message(env):  # noqa: F811
    stack = Stack(env).mark_running(watchdog=False)
    seen = fast_webhook(env, uuid_only=True)
    env.send_request(JOB)
    (outcome,) = stack.request_once()
    assert seen["webhook"] == 503  # 기존: 이어 갈 경로가 없으면 Webhook 도 완료로 답하지 않는다
    assert outcome.delete is False and request_messages(env) == 1


# === 요청 메시지: 사전 검증 실패 결과 저장 → 결과 전송 루프 인계 ===


def missing_user_image(env):  # noqa: F811
    env.s3.objects.clear()  # 원본 사진 없음 → INVALID_REQUEST 실패 결과(기존 분류 그대로)


def test_saved_failure_result_is_handed_to_the_publisher_on_first_processing(env):  # noqa: F811
    stack = Stack(env).mark_running(postprocess=False, watchdog=False)  # 결과 전송 루프만 필요
    missing_user_image(env)
    env.send_request(JOB)
    (outcome,) = stack.request_once()
    current = row(env)
    assert current.next_action is NextAction.PUBLISH_RESULT
    assert current.result_payload["error_code"] == "INVALID_REQUEST"
    assert outcome.delete is True and request_messages(env) == 0
    assert env.runware.count("async_submit") == 0
    stack.publish_once()
    assert env.sqs.bodies("be-fitting-response")[0]["error_code"] == "INVALID_REQUEST"


def test_saved_failure_result_without_a_publisher_keeps_the_message(env):  # noqa: F811
    stack = Stack(env).mark_running(publisher=False)
    missing_user_image(env)
    env.send_request(JOB)
    (outcome,) = stack.request_once()
    assert row(env).next_action is NextAction.PUBLISH_RESULT
    assert outcome.delete is False and request_messages(env) == 1


def test_published_duplicate_request_is_deleted_and_conflicts_are_kept(env):  # noqa: F811
    stack = Stack(env).mark_running()
    fast_webhook(env)
    env.send_request(JOB)
    stack.request_once()
    finish(env, stack)
    assert row(env).result_message_status is ResultMessageStatus.PUBLISHED
    env.send_request(JOB)  # 같은 요청의 중복
    (duplicate,) = stack.request_once()
    assert duplicate.action == "already_published_deleted" and duplicate.delete is True
    env.send_request(JOB, codes=("123456",))  # 같은 작업 번호, 다른 입력
    (conflict,) = stack.request_once()
    assert conflict.action == "input_conflict_kept" and conflict.delete is False
    env.sqs.put(REQUEST, "not json")
    (invalid,) = stack.request_once()
    assert invalid.delete is False
    assert env.runware.count("async_submit") == 1


# === Watchdog: AI 저장 작업의 실제 진행과 재발행 1회 ===


def ai_store_failing(env, stack):  # noqa: F811
    """결과 URL 은 저장됐지만 내려받을 수 없다(만료 404): 후처리가 진행하지 못하는 AI 저장 작업."""
    env.send_request(JOB)
    stack.request_once()
    task = row(env).task_uuid
    assert post_webhook(client(), env.runware.complete(task)).status_code == 200
    env.runware.cdn.clear()
    return task


def watchdog_check(env, stack):  # noqa: F811
    env.sql("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
    return [a for r in stack.watchdog_once() for a in r.actions]


def consume_and_fail(env, stack):  # noqa: F811
    """후처리 Worker 가 메시지를 받아 실패한다(lease 선점·반납으로 updated_at 이 바뀐다)."""
    env.sqs.expire_visibility(FOLLOW_UP)
    expire_lease(env)
    stack.postprocess_once()


def postprocess_messages(env):  # noqa: F811
    return [b["action"] for b in env.sqs.bodies(FOLLOW_UP)]


def test_lease_and_redelivery_do_not_postpone_the_stall_judgement(env):  # noqa: F811
    stack = Stack(env).mark_running()
    ai_store_failing(env, stack)
    consume_and_fail(env, stack)
    # 성공 전환(이 단계 시작) 뒤 10분이 지났다. 재수신·lease 갱신은 방금 있었다(updated_at 최신).
    age(env, "runware_accepted_at", 700)
    env.sql(
        "UPDATE virtual_fitting_async_jobs SET updated_at = statement_timestamp() WHERE fitting_job_id = %s",
        (JOB,),
    )
    actions = watchdog_check(env, stack)
    assert "republish_postprocess_ok" in actions
    assert postprocess_messages(env) == ["POSTPROCESS", "POSTPROCESS"]  # Webhook 1 + 재발행 1


def test_consumption_after_a_republish_does_not_reopen_the_limit(env):  # noqa: F811
    stack = Stack(env).mark_running()
    ai_store_failing(env, stack)
    age(env, "runware_accepted_at", 7200)  # 이 단계(성공 전환)는 2시간 전에 시작했다
    assert "republish_postprocess_ok" in watchdog_check(env, stack)
    for _ in range(3):
        # 재발행한 메시지도 소비에 계속 실패한다(재수신마다 lease 가 updated_at 을 바꾼다) → DLQ 로 가는 중.
        consume_and_fail(env, stack)
        env.sql(  # 재발행 기록은 그 소비보다 앞선다(재발행 → 소비 순서). 단계 시작보다는 뒤다.
            "UPDATE virtual_fitting_watchdog_checks SET follow_up_published_at = "
            "follow_up_published_at - interval '5 minutes'"
        )
        actions = watchdog_check(env, stack)
        assert "republish_postprocess_held" in actions
    assert postprocess_messages(env) == ["POSTPROCESS", "POSTPROCESS"]
    assert env.sql(
        "SELECT seen_count >= 1 FROM virtual_fitting_watchdog_alerts "
        "WHERE scope = %s AND reason = 'postprocess_stalled'",
        (f"job:{JOB}",),
    ) == [(True,)]  # 그 뒤로는 운영 알림
    current = row(env)
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert current.fallback_attempt_count == 0 and env.runware.count("sync_generate") == 0


def test_the_limit_survives_a_watchdog_restart(env):  # noqa: F811
    stack = Stack(env).mark_running()
    ai_store_failing(env, stack)
    age(env, "runware_accepted_at", 700)
    assert "republish_postprocess_ok" in watchdog_check(env, stack)
    consume_and_fail(env, stack)
    stack = restart(env)  # 새 pool·새 Watchdog(메모리 상태 없음)
    assert "republish_postprocess_held" in watchdog_check(env, stack)
    assert postprocess_messages(env) == ["POSTPROCESS", "POSTPROCESS"]


def test_a_real_checkpoint_opens_the_next_stage_once(env):  # noqa: F811
    stack = Stack(env).mark_running()
    ai_store_failing(env, stack)
    age(env, "runware_accepted_at", 700)
    assert "republish_postprocess_ok" in watchdog_check(env, stack)
    # 실제 진척: 재발행 뒤에 결과 이미지 저장이 확인됐다(체크포인트). 그 뒤 LLM 단계에서 10분 넘게 멈췄다.
    env.sql(
        "UPDATE virtual_fitting_watchdog_checks SET follow_up_published_at = "
        "statement_timestamp() - interval '800 seconds'"
    )
    key = row(env).async_image_key
    env.sql(
        "UPDATE virtual_fitting_async_jobs SET verified_image_key = %s, "
        "image_verified_at = statement_timestamp() - interval '700 seconds' WHERE fitting_job_id = %s",
        (key, JOB),
    )
    assert "republish_postprocess_ok" in watchdog_check(env, stack)  # 다음 단계의 복구 1회
    # 그 단계에서도 진척 없이 시간이 흐르면(lease·재수신 갱신만 있어도) 다시 발행하지 않는다.
    env.sql(
        "UPDATE virtual_fitting_async_jobs SET updated_at = statement_timestamp() WHERE fitting_job_id = %s",
        (JOB,),
    )
    assert "republish_postprocess_held" in watchdog_check(env, stack)
    assert postprocess_messages(env) == ["POSTPROCESS"] * 3


def test_failed_republish_is_not_counted_and_is_retried(env):  # noqa: F811
    stack = Stack(env).mark_running()
    ai_store_failing(env, stack)
    age(env, "runware_accepted_at", 700)
    env.sqs.fault(FOLLOW_UP, "send", "throttled")
    assert "republish_postprocess_failed" in watchdog_check(env, stack)
    assert "republish_postprocess_ok" in watchdog_check(env, stack)
    assert postprocess_messages(env) == ["POSTPROCESS", "POSTPROCESS"]
