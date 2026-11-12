"""V2 전체 흐름에서 IDM-VTON fallback: 실제 요청 Worker·Webhook·후처리 Worker·결과 전송 루프(테스트 harness) + IDM 핸들러.

S3·SQS·Runware HTTP 는 harness 의 가짜, Modal 과 상품 이미지 분류만 이 파일의 가짜다. 가짜 Modal 은 받은 Presigned PUT
헤더(Content-Type·x-amz-meta-*)로 harness S3 에 WebP 를 올린다. Pruna 동기 생성(sync_generate)은 한 번도 불리지 않아야 한다.
"""

from urllib.parse import urlparse

from app.virtual_fitting.idm_fallback_handler import IdmVtonFallbackHandler, idm_result_key
from app.virtual_fitting.idm_image_validator import IdmImageClass
from app.virtual_fitting.idm_modal_client import validate_output
from app.virtual_fitting.postprocess_handler import PostprocessHandler
from app.virtual_fitting.postprocess_worker import PostprocessHandlers
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    ResultMessageStatus,
)
from tests.test_virtual_fitting_idm_fallback_handler import FakeClassifier, webp_bytes
from tests.test_virtual_fitting_v2_scenarios import (  # noqa: F401 (pytest fixture env)
    ASYNC_KEY,
    JOB,
    client,
    env,
    finish,
    post_webhook,
    row,
)
from tests.v2_integration_harness import Stack

KEY = idm_result_key(JOB)


class HarnessModal:
    """spawn 1회마다 harness S3 에 결과를 올리고 성공 요약을 돌려준다."""

    def __init__(self, s3):
        self.s3, self.requests = s3, []

    def resolve(self):
        return self

    def spawn(self, request):
        self.requests.append(request)
        key = urlparse(request["upload_url"]).path.lstrip("/")
        headers = {k.lower(): v for k, v in request["upload_headers"].items()}
        body = webp_bytes()
        self.s3.store(key, body, headers["content-type"])
        self.s3.metadata[key] = {
            k[len("x-amz-meta-") :]: v for k, v in headers.items() if k.startswith("x-amz-meta-")
        }
        self.body = body
        return request, None

    def wait(self, call, timeout_seconds, fitting_job_id, attempt_id):
        import hashlib

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


def idm_stack(env, classification=IdmImageClass.GARMENT_ONLY):  # noqa: F811
    modal = HarnessModal(env.s3)
    classifier = FakeClassifier(classification)
    handlers = PostprocessHandlers(
        postprocess=PostprocessHandler(),
        run_fallback=IdmVtonFallbackHandler(modal=modal, classifier=classifier, poll_seconds=0.01),
    )
    return Stack(env, handlers).mark_running(), modal, classifier


def submit(env, stack, codes):  # noqa: F811
    env.send_request(JOB, codes=codes)
    stack.request_once()
    return row(env).task_uuid


def test_pruna_success_does_not_call_idm(env):  # noqa: F811
    stack, modal, classifier = idm_stack(env)
    task = submit(env, stack, ("123456",))
    assert post_webhook(client(), env.runware.complete(task)).status_code == 200
    finish(env, stack)
    final = row(env)
    assert final.result_image_key == ASYNC_KEY and final.status is AsyncJobStatus.COMPLETED
    assert modal.requests == [] and classifier.calls == []
    assert env.runware.count("sync_generate") == 0


def test_pruna_failure_with_one_top_calls_idm_once_and_publishes_the_same_format(env):  # noqa: F811
    stack, modal, _ = idm_stack(env)
    task = submit(env, stack, ("123456",))
    assert post_webhook(client(), env.runware.failure(task, "timeoutProvider")).status_code == 200
    finish(env, stack)
    final = row(env)
    assert len(modal.requests) == 1 and env.runware.count("sync_generate") == 0
    assert final.status is AsyncJobStatus.COMPLETED
    assert final.result_message_status is ResultMessageStatus.PUBLISHED
    assert final.fallback_image_key == KEY and final.fallback_attempt_count == 1
    (payload,) = env.sqs.bodies("be-fitting-response")
    # 기존 BE 결과 메시지 형식(성공 필드 5개) 그대로, 이미지 Key 만 IDM 결과
    assert set(payload) == {
        "fitting_job_id",
        "status",
        "result_image_key",
        "llm_title",
        "llm_comment",
    }
    assert payload["status"] == "COMPLETED" and payload["result_image_key"] == KEY
    # 중복 수신·재처리: 추가 생성 없음
    env.sqs.put(
        "ai-fitting-response", {"fitting_job_id": JOB, "task_uuid": task, "action": "RUN_FALLBACK"}
    )
    stack.postprocess_once()
    assert len(modal.requests) == 1


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
