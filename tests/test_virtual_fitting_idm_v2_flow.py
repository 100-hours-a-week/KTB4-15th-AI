"""V2 전체 흐름에서 IDM-VTON fallback 과 배경 제거(RemBG)의 공존: 실제 요청 Worker·Webhook·후처리 Worker·결과 전송 루프·
Watchdog(테스트 harness) + PostprocessHandler(RemBG 켬) + IDM 핸들러.

S3·SQS·Runware HTTP(Pruna·RemBG·LLM)는 harness 의 가짜, Modal 과 상품 이미지 분류만 이 파일의 가짜다. 가짜 Modal 은 받은
Presigned PUT 헤더(Content-Type·x-amz-meta-*)로 harness S3 에 WebP 를 올린다. 실제 유료 API 는 부르지 않는다.
- Pruna 비동기 성공: pruna-async.webp → RemBG 1회 → transparent.webp 가 결과. IDM 은 부르지 않는다.
- Pruna 확정 생성 실패(timeoutProvider·providerError): IDM-VTON → idm-vton.webp 가 결과. RemBG·Pruna 동기 생성은 없다.
- RemBG 실패: FITTING_BACKGROUND_REMOVAL_FAILED. IDM 으로 넘어가지 않는다.
"""

from urllib.parse import urlparse

import pytest

from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.idm_fallback_handler import IdmVtonFallbackHandler, idm_result_key
from app.virtual_fitting.idm_image_validator import IdmImageClass
from app.virtual_fitting.idm_modal_client import IdmCallKind, IdmCallResult, validate_output
from app.virtual_fitting.postprocess_handler import PostprocessHandler
from app.virtual_fitting.postprocess_worker import PostprocessHandlers
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository
from tests.test_virtual_fitting_idm_fallback_handler import FakeClassifier, webp_bytes
from tests.test_virtual_fitting_v2_scenarios import (  # noqa: F401 (pytest fixture env)
    ASYNC_KEY,
    JOB,
    REMBG_CALL_KEY,
    SYNC_KEY,
    TRANSPARENT_KEY,
    age,
    all_costs,
    client,
    env,
    expire_lease,
    finish,
    post_webhook,
    pruna_sync_handlers,
    restart,
    row,
)
from tests.v2_integration_harness import Stack

KEY = idm_result_key(JOB)
SUCCESS_FIELDS = {"fitting_job_id", "status", "result_image_key", "llm_title", "llm_comment"}
FAILURE_FIELDS = {"fitting_job_id", "status", "error_code", "error_message"}


class HarnessModal:
    """spawn 1회마다 harness S3 에 결과를 올리고 성공 요약을 돌려준다.

    outcome: "success" | "failed"(올리지 않고 끝난 실패) | "unknown"(올렸지만 spawn 응답을 잃음 → 결과 불명확)
    """

    def __init__(self, s3, outcome="success"):
        self.s3, self.requests, self.outcome = s3, [], outcome

    def resolve(self):
        return self

    def spawn(self, request):
        self.requests.append(request)
        if self.outcome == "failed":
            return request, None
        key = urlparse(request["upload_url"]).path.lstrip("/")
        headers = {k.lower(): v for k, v in request["upload_headers"].items()}
        body = webp_bytes()
        self.s3.store(key, body, headers["content-type"])
        self.s3.metadata[key] = {
            k[len("x-amz-meta-") :]: v for k, v in headers.items() if k.startswith("x-amz-meta-")
        }
        self.body = body
        if self.outcome == "unknown":
            return None, IdmCallResult(IdmCallKind.UNKNOWN, "ConnectionError")
        return request, None

    def wait(self, call, timeout_seconds, fitting_job_id, attempt_id):
        import hashlib

        if self.outcome == "failed":
            return validate_output(
                {"status": "failed", "error_kind": "inference_failed"}, fitting_job_id, attempt_id
            )
        return validate_output(
            {
                "status": "succeeded",
                "fitting_job_id": fitting_job_id,
                "attempt_id": attempt_id,
                "content_type": "image/webp",
                "bytes": len(self.body),
                "sha256": hashlib.sha256(self.body).hexdigest(),
                "width": 384,
                "height": 512,
            },
            fitting_job_id,
            attempt_id,
        )


def idm_handlers(modal, classifier):
    """실제 연결과 같은 조립(POSTPROCESS=PostprocessHandler·RemBG 켬, RUN_FALLBACK=IDM-VTON). Modal·분류만 가짜."""
    return PostprocessHandlers(
        postprocess=PostprocessHandler(),
        run_fallback=IdmVtonFallbackHandler(modal=modal, classifier=classifier, poll_seconds=0.01),
    )


def idm_stack(env, classification=IdmImageClass.GARMENT_ONLY, outcome="success"):  # noqa: F811
    modal = HarnessModal(env.s3, outcome)
    classifier = FakeClassifier(classification)
    return Stack(env, idm_handlers(modal, classifier)).mark_running(), modal, classifier


def submit(env, stack, codes):  # noqa: F811
    env.send_request(JOB, codes=codes)
    stack.request_once()
    return row(env).task_uuid


def llm_image_keys(env):  # noqa: F811
    """코멘트 LLM 에 넘긴 이미지 URL 의 S3 Key(Presigned GET, 서명 쿼리 제외)."""
    keys = []
    for request in env.runware.llm_requests:
        content = request["messages"][1]["content"]
        if isinstance(content, list):
            keys.append(urlparse(content[1]["image_url"]["url"]).path.lstrip("/"))
    return keys


def assert_no_background_removal(env):  # noqa: F811
    """RemBG 호출·투명 결과·호출 표시·RemBG 비용 기록이 모두 없다."""
    assert env.runware.count("rembg") == 0 and env.runware.rembg_inputs == []
    assert TRANSPARENT_KEY not in env.s3.objects and REMBG_CALL_KEY not in env.s3.objects
    assert not [p for p in env.s3.puts if p["Key"] in (TRANSPARENT_KEY, REMBG_CALL_KEY)]
    current = row(env)
    tasks = {task for task, _ in all_costs(env)}
    # Pruna 생성 task 비용만(있다면)
    assert tasks <= {current.task_uuid, current.fallback_task_uuid}


def redeliver(env, task, action="RUN_FALLBACK"):  # noqa: F811
    env.sqs.put("ai-fitting-response", {"fitting_job_id": JOB, "task_uuid": task, "action": action})


def test_pruna_success_uses_rembg_and_does_not_call_idm(env):  # noqa: F811
    stack, modal, classifier = idm_stack(env)
    task = submit(env, stack, ("123456",))
    assert post_webhook(client(), env.runware.complete(task)).status_code == 200
    finish(env, stack)
    final = row(env)
    # 정상 생성 경로는 그대로: 원본 저장 → RemBG 1회 → transparent.webp 가 결과·LLM 입력
    assert final.result_image_key == TRANSPARENT_KEY and final.status is AsyncJobStatus.COMPLETED
    assert final.async_image_key == ASYNC_KEY and ASYNC_KEY in env.s3.objects
    assert env.runware.count("rembg") == 1 and env.runware.rembg_inputs == [ASYNC_KEY]
    assert llm_image_keys(env) == [TRANSPARENT_KEY]
    (payload,) = env.sqs.bodies("be-fitting-response")
    assert set(payload) == SUCCESS_FIELDS and payload["result_image_key"] == TRANSPARENT_KEY
    assert modal.requests == [] and classifier.calls == []
    assert env.runware.count("sync_generate") == 0 and final.fallback_attempt_count == 0


@pytest.mark.parametrize("code", ["timeoutProvider", "providerError"])
def test_confirmed_failure_calls_idm_once_without_background_removal(env, code):  # noqa: F811
    stack, modal, _ = idm_stack(env)
    task = submit(env, stack, ("123456",))
    assert post_webhook(client(), env.runware.failure(task, code)).status_code == 200
    finish(env, stack)
    final = row(env)
    assert len(modal.requests) == 1 and env.runware.count("sync_generate") == 0
    assert final.status is AsyncJobStatus.COMPLETED
    assert final.result_message_status is ResultMessageStatus.PUBLISHED
    assert final.fallback_image_key == KEY and final.fallback_attempt_count == 1
    assert final.result_image_key == KEY and KEY in env.s3.objects
    # IDM 원본 idm-vton.webp 를 그대로 LLM 입력·결과로 쓴다. 배경 제거는 하지 않는다.
    assert llm_image_keys(env) == [KEY]
    assert_no_background_removal(env)
    (payload,) = env.sqs.bodies("be-fitting-response")
    # 기존 BE 결과 메시지 형식(성공 필드 5개) 그대로, 이미지 Key 만 IDM 결과
    assert set(payload) == SUCCESS_FIELDS
    assert payload["status"] == "COMPLETED" and payload["result_image_key"] == KEY
    # 내부 큐에는 작업 참조만
    for message in env.sqs.bodies("ai-fitting-response"):
        assert set(message) == {"fitting_job_id", "task_uuid", "action"}
    # 중복 수신·재처리: 추가 생성·배경 제거·재발행 없음
    redeliver(env, task)
    stack.postprocess_once()
    assert len(modal.requests) == 1 and env.runware.count("rembg") == 0
    assert len(env.sqs.bodies("be-fitting-response")) == 1


def test_provider_error_in_the_submit_response_runs_idm(env):  # noqa: F811
    env.runware.async_modes.append(
        lambda submission: (
            "errors",
            [{"code": "providerError", "message": "x", "taskUUID": submission.task_uuid}],
        )
    )
    stack, modal, _ = idm_stack(env)
    submit(env, stack, ("123456",))
    assert [b["action"] for b in env.sqs.bodies("ai-fitting-response")] == ["RUN_FALLBACK"]
    finish(env, stack)
    assert row(env).result_image_key == KEY and len(modal.requests) == 1
    assert_no_background_removal(env)
    assert env.runware.count("sync_generate") == 0


def test_provider_error_from_get_response_runs_idm(env):  # noqa: F811
    stack, modal, _ = idm_stack(env)
    task = submit(env, stack, ("123456",))
    env.runware.statuses[task] = {
        "errors": [{"taskUUID": task, "code": "providerError", "status": "error"}]
    }
    age(env, "runware_accepted_at", 400)  # Webhook 없이 3분이 지나 Watchdog 이 getResponse 로 확인
    stack.watchdog_once()
    assert ("get_response", task) in env.runware.calls
    assert [b["action"] for b in env.sqs.bodies("ai-fitting-response")] == ["RUN_FALLBACK"]
    finish(env, stack)
    assert row(env).result_image_key == KEY and len(modal.requests) == 1
    assert_no_background_removal(env)


def test_http_500_without_a_task_error_does_not_start_idm(env):  # noqa: F811
    env.runware.async_modes.append(("http", 500))  # 본문에 이 task 의 오류 코드가 없다(결과 불명확)
    stack, modal, classifier = idm_stack(env)
    env.send_request(JOB, codes=("123456",))
    stack.request_once()
    current = row(env)
    assert current.next_action is not NextAction.RUN_FALLBACK
    assert current.fallback_attempt_count == 0 and current.fallback_started_at is None
    assert "RUN_FALLBACK" not in [b["action"] for b in env.sqs.bodies("ai-fitting-response")]
    stack.postprocess_once()
    assert modal.requests == [] and classifier.calls == [] and current.result_payload is None


def test_background_removal_failure_is_final_and_does_not_start_idm(env):  # noqa: F811
    env.runware.rembg_modes.append(("http", 500))
    stack, modal, classifier = idm_stack(env)
    task = submit(env, stack, ("123456",))
    assert post_webhook(client(), env.runware.complete(task)).status_code == 200
    finish(env, stack)
    final = row(env)
    assert final.status is AsyncJobStatus.FAILED and env.runware.count("rembg") == 1
    (payload,) = env.sqs.bodies("be-fitting-response")
    assert set(payload) == FAILURE_FIELDS
    assert payload["error_code"] == "FITTING_BACKGROUND_REMOVAL_FAILED"
    # RemBG 실패로 RUN_FALLBACK 을 만들지 않는다(IDM·Pruna 동기 생성 없음)
    assert "RUN_FALLBACK" not in [b["action"] for b in env.sqs.bodies("ai-fitting-response")]
    assert final.fallback_attempt_count == 0 and final.fallback_started_at is None
    assert modal.requests == [] and classifier.calls == []
    assert env.runware.count("sync_generate") == 0
    # 늦은 RUN_FALLBACK·POSTPROCESS 재수신에도 아무것도 다시 부르지 않는다
    redeliver(env, task)
    redeliver(env, task, "POSTPROCESS")
    stack.postprocess_once()
    stack.postprocess_once()
    assert modal.requests == [] and env.runware.count("rembg") == 1


def test_idm_failure_does_not_call_pruna_sync_or_rembg(env):  # noqa: F811
    stack, modal, _ = idm_stack(env, outcome="failed")
    task = submit(env, stack, ("123456",))
    assert post_webhook(client(), env.runware.failure(task, "providerError")).status_code == 200
    finish(env, stack)
    (payload,) = env.sqs.bodies("be-fitting-response")
    assert set(payload) == FAILURE_FIELDS and payload["error_code"] == "FITTING_MODEL_FAILED"
    assert len(modal.requests) == 1 and row(env).status is AsyncJobStatus.FAILED
    assert env.runware.count("sync_generate") == 0
    assert_no_background_removal(env)
    redeliver(env, task)
    stack.postprocess_once()
    assert len(modal.requests) == 1


def test_unconfirmed_idm_call_is_held_and_recovered_from_s3_without_a_new_call(env):  # noqa: F811
    stack, modal, classifier = idm_stack(env, outcome="unknown")
    task = submit(env, stack, ("123456",))
    assert post_webhook(client(), env.runware.failure(task, "timeoutProvider")).status_code == 200
    stack.postprocess_once()
    held = row(env)
    # Modal 은 결과를 올렸지만 spawn 응답을 잃었다: 실패로 확정하지 않고 보류(이미지 기록 없음)
    assert len(modal.requests) == 1 and KEY in env.s3.objects
    assert held.result_payload is None and held.fallback_image_key is None
    assert held.fallback_started_at is not None and held.fallback_attempt_count == 1
    # Worker 재시작 + 같은 메시지 재수신: 다시 부르지 않고 고정 Key 로 이어 간다
    stack = restart(env, idm_handlers(modal, classifier))
    env.sqs.expire_visibility("ai-fitting-response")
    expire_lease(env)
    finish(env, stack)
    final = row(env)
    assert len(modal.requests) == 1 and final.result_image_key == KEY
    assert final.result_message_status is ResultMessageStatus.PUBLISHED
    assert_no_background_removal(env)


def test_idm_result_saved_before_the_db_record_is_recovered_by_the_watchdog(env, monkeypatch):  # noqa: F811
    stack, modal, classifier = idm_stack(env)
    task = submit(env, stack, ("123456",))
    assert post_webhook(client(), env.runware.failure(task, "timeoutProvider")).status_code == 200
    real = AsyncJobStateRepository.record_fallback_image
    calls = []

    def fail_once(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise FittingDatabaseError("injected after S3 put")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(AsyncJobStateRepository, "record_fallback_image", fail_once)
    stack.postprocess_once()
    assert KEY in env.s3.objects and row(env).fallback_image_key is None
    stack = restart(env, idm_handlers(modal, classifier))
    age(env, "updated_at", 700)
    expire_lease(env)
    stack.watchdog_once()  # 고정 Key 의 작업·시도 메타데이터 확인 → 기록 → RUN_FALLBACK 재발행(Modal 미호출)
    assert row(env).fallback_image_key == KEY
    env.sqs.expire_visibility("ai-fitting-response")
    finish(env, stack)
    final = row(env)
    assert final.result_image_key == KEY and len(modal.requests) == 1
    assert final.result_message_status is ResultMessageStatus.PUBLISHED
    assert_no_background_removal(env)


def test_late_messages_after_the_idm_result_change_nothing(env):  # noqa: F811
    stack, modal, _ = idm_stack(env)
    task = submit(env, stack, ("123456",))
    assert post_webhook(client(), env.runware.failure(task, "timeoutProvider")).status_code == 200
    finish(env, stack)
    before = row(env)
    assert before.status is AsyncJobStatus.COMPLETED
    # 늦은 성공·실패 Webhook, 중복 RUN_FALLBACK·POSTPROCESS
    for body in (env.runware.complete(task), env.runware.failure(task, "providerError")):
        assert post_webhook(client(), body).status_code == 200
    redeliver(env, task)
    redeliver(env, task, "POSTPROCESS")
    for _ in range(3):
        stack.postprocess_once()
    expire_lease(env)
    stack.publish_once()
    after = row(env)
    assert (after.status, after.result_payload, after.result_image_key) == (
        before.status,
        before.result_payload,
        before.result_image_key,
    )
    assert len(modal.requests) == 1 and len(env.sqs.bodies("be-fitting-response")) == 1
    assert_no_background_removal(env)


def test_pruna_sync_result_left_by_the_previous_version_is_not_used_by_idm(
    env,  # noqa: F811
    monkeypatch,
):
    """전환기: 이전 연결(Pruna 동기 fallback)이 pruna-sync.webp 를 저장한 뒤 DB 기록 전에 멈췄다.

    IDM 핸들러는 그 결과를 조회·기록하지 않는다. idm-vton.webp 만 확인해 업로드 기한 전에는 보류, 기한 뒤에는 기존 정책대로
    FITTING_MODEL_FAILED. Modal·Pruna 동기 재생성·RemBG 는 부르지 않는다.
    """
    from app.config import settings
    from app.virtual_fitting.idm_fallback_handler import RECOVERY_MARGIN_SECONDS

    env.runware.async_modes.append(
        lambda submission: (
            "errors",
            [{"code": "timeoutProvider", "message": "x", "taskUUID": submission.task_uuid}],
        )
    )
    stack = Stack(env, pruna_sync_handlers()).mark_running()
    env.send_request(JOB, codes=("123456",))
    stack.request_once()
    real = AsyncJobStateRepository.record_fallback_image

    def fail(self, *args, **kwargs):
        raise FittingDatabaseError("stopped after S3 put")

    monkeypatch.setattr(AsyncJobStateRepository, "record_fallback_image", fail)
    stack.postprocess_once()
    assert SYNC_KEY in env.s3.objects and row(env).fallback_image_key is None
    monkeypatch.setattr(AsyncJobStateRepository, "record_fallback_image", real)
    modal, classifier = HarnessModal(env.s3), FakeClassifier()
    heads, gets = len(env.s3.heads), len(env.s3.gets)
    stack = restart(env, idm_handlers(modal, classifier))  # 새 배포: RUN_FALLBACK = IDM-VTON
    env.sqs.expire_visibility("ai-fitting-response")
    expire_lease(env)
    stack.postprocess_once()
    held = row(env)
    assert held.fallback_image_key is None and held.result_payload is None  # 업로드 기한 전: 보류
    age(
        env,
        "fallback_started_at",
        settings.IDM_VTON_UPLOAD_URL_EXPIRES_SECONDS + RECOVERY_MARGIN_SECONDS + 60,
    )
    env.sqs.expire_visibility("ai-fitting-response")
    expire_lease(env)
    finish(env, stack)
    final = row(env)
    assert final.status is AsyncJobStatus.FAILED and final.fallback_image_key is None
    (payload,) = env.sqs.bodies("be-fitting-response")
    assert payload["error_code"] == "FITTING_MODEL_FAILED"
    assert SYNC_KEY not in env.s3.heads[heads:] and SYNC_KEY not in env.s3.gets[gets:]
    assert modal.requests == [] and classifier.calls == []
    assert env.runware.count("sync_generate") == 1  # 이전 연결이 보낸 1회뿐
    assert_no_background_removal(env)
    assert KEY not in env.s3.objects


def test_pruna_failure_with_top_and_bottom_fails_without_idm(env):  # noqa: F811
    stack, modal, classifier = idm_stack(env)
    task = submit(env, stack, ("123456", "789012"))
    assert post_webhook(client(), env.runware.failure(task, "timeoutProvider")).status_code == 200
    finish(env, stack)
    (payload,) = env.sqs.bodies("be-fitting-response")
    assert payload["status"] == "FAILED" and payload["error_code"] == "FITTING_MODEL_FAILED"
    assert modal.requests == [] and classifier.calls == []
    assert env.runware.count("sync_generate") == 0 and row(env).status is AsyncJobStatus.FAILED


def test_pruna_failure_with_a_model_worn_image_fails_without_idm(env):  # noqa: F811
    stack, modal, classifier = idm_stack(env, IdmImageClass.MODEL_WORN)
    task = submit(env, stack, ("123456",))
    assert post_webhook(client(), env.runware.failure(task, "timeoutProvider")).status_code == 200
    finish(env, stack)
    (payload,) = env.sqs.bodies("be-fitting-response")
    assert payload["status"] == "FAILED" and modal.requests == [] and len(classifier.calls) == 1
