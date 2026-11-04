"""기존 GetObject/PutObject 범위: generic MIME 후처리·실제 Key·중복 방지 회귀. 외부 호출 없음."""

import copy
from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace

import pytest

from app.clients.s3 import S3ObjectCheck, S3ObjectState
from app.virtual_fitting.postprocess_handler import PostprocessHandler
from app.virtual_fitting.postprocess_worker import HandlerStatus
from app.virtual_fitting.repositories.async_job_repository import AsyncJobStatus, NextAction

KEY = "virtual-fitting/results/123/pruna-async.webp"
NOW = datetime(2026, 11, 3, tzinfo=UTC)


class MemoryState:
    def __init__(self, job):
        self.job = job

    def claim_s3_check_attempt(self, *_):
        self.job.s3_check_attempt_count += 1
        return SimpleNamespace(applied=True, job=self.job)

    def record_image_verified(self, _id, _token, key):
        self.job.verified_image_key = key
        self.job.image_verified_at = NOW
        return SimpleNamespace(applied=True)

    def holds_lease(self, *_):
        return True

    def claim_llm_generation_attempt(self, *_):
        self.job.llm_generation_attempt_count += 1
        return SimpleNamespace(applied=True)

    def save_final_result(self, _id, _token, payload, key):
        self.job.result_payload, self.job.result_image_key = payload, key
        return SimpleNamespace(applied=True)


class ExistingStorage:
    """메타데이터 변경 API·버전 조회 없이 기존 HEAD/Presigned GET 만 제공한다."""
    def __init__(self, content_type):
        self.heads, self.urls = [], []
        self.object_properties = {"content_type": content_type, "modified": NOW,
            "tags": {"existing": "untouched"}, "acl": "existing", "retention": "existing"}

    def head_object(self, key):
        self.heads.append(key)
        return S3ObjectCheck(S3ObjectState.EXISTS, key,
            content_type=self.object_properties["content_type"], last_modified=NOW)

    def presigned_get_url(self, key, *, expires_in):
        self.urls.append(key)
        return f"https://local.invalid/{key}"

    def __getattr__(self, name):
        raise AssertionError(f"unexpected additional storage operation: {name}")


@pytest.mark.parametrize("extension", ["webp", "jpg", "png"])
@pytest.mark.parametrize("content_type", ["image/webp", "binary/octet-stream", "application/octet-stream", None])
def test_original_async_key_reaches_llm_and_result_without_metadata_changes(extension, content_type):
    key = KEY.replace(".webp", "." + extension)
    job = SimpleNamespace(fitting_job_id=123, result_payload=None, verified_image_key=None,
        async_image_key=key, status=AsyncJobStatus.POSTPROCESSING, next_action=NextAction.POSTPROCESS,
        s3_check_attempt_count=0, llm_generation_attempt_count=0, products=["123"])
    state = MemoryState(job)
    storage = ExistingStorage(content_type)
    original = copy.deepcopy(storage.object_properties)
    comments_seen = []

    def comment(url, _summaries):
        comments_seen.append(url)
        return "기존 코멘트"

    handler = PostprocessHandler(
        jobs=SimpleNamespace(find_by_fitting_job_id=lambda _: job), state=state,
        products=SimpleNamespace(find_by_codes=lambda _: [SimpleNamespace(product_code="123", description_summary="셔츠")]),
        storage=storage, comments=SimpleNamespace(generate_comment=comment, generate_title=lambda _: "기존 제목"),
        offsets=(0,),
    )
    context = SimpleNamespace(may_continue=lambda: True)
    assert handler(job, "lease", context).status is HandlerStatus.DONE
    assert job.result_payload == {"fitting_job_id": 123, "status": "COMPLETED", "result_image_key": key,
                                 "llm_title": "기존 제목", "llm_comment": "기존 코멘트"}
    assert job.verified_image_key == key and job.image_verified_at == NOW
    assert job.s3_check_attempt_count == 1 and job.llm_generation_attempt_count == 2
    assert comments_seen == [f"https://local.invalid/{key}"]
    assert storage.heads == [key] and storage.urls == [key]
    assert storage.object_properties == original
    # 재수신에는 HEAD/LLM/서명도 반복하지 않는다.
    assert handler(job, "lease", context).status is HandlerStatus.DONE
    assert storage.heads == [key] and storage.urls == [key] and len(comments_seen) == 1
    assert job.s3_check_attempt_count == 1 and job.llm_generation_attempt_count == 2


@pytest.mark.parametrize(("extension", "fmt"), [("webp", "WEBP"), ("jpg", "JPG")])
def test_submitter_uses_persisted_output_contract_and_actual_person_url(extension, fmt):
    import json

    from app.virtual_fitting.providers.base import FittingInput
    from app.virtual_fitting.providers.runware_async import RunwarePrunaAsyncProvider
    from app.virtual_fitting.pruna_submitter import PrunaSubmitter

    sent = []
    task = "a770f077-f413-47de-9dac-be0b26a35da6"

    def opener(request, timeout):
        sent.append(json.loads(request.data)[0])
        response = BytesIO(json.dumps({"data": [{"taskUUID": task, "status": "processing"}]}).encode())
        response.status = 200
        return response

    submitter = object.__new__(PrunaSubmitter)
    submitter._provider = RunwarePrunaAsyncProvider("test-only", opener=opener)
    submitter._state = SimpleNamespace(record_submission_accepted=lambda _: SimpleNamespace(applied=True))
    submitter._limiter = SimpleNamespace(finish_send=lambda _: None)
    submitter._keepalive = 60
    submitter._webhook_url = "https://local.invalid/webhook"
    submitter._decide = lambda _: True
    person_url = "https://local.invalid/actual-user-key.webp?signature=mock"
    job = SimpleNamespace(fitting_job_id=123, task_uuid=task, async_image_key=KEY.replace(".webp", "." + extension))
    assert submitter._send(job, "lease", FittingInput(person_url, ["https://local.invalid/shirt.jpg"], ""),
                            "https://local.invalid/" + job.async_image_key, SimpleNamespace(beat=lambda: True))
    assert len(sent) == 1 and sent[0]["outputFormat"] == fmt
    assert (sent[0].get("outputQuality") == 90) if fmt == "WEBP" else "outputQuality" not in sent[0]
    assert sent[0]["inputs"]["referenceImages"] == [
        {"image": person_url, "role": "person"},
        {"image": "https://local.invalid/shirt.jpg", "role": "garment"},
    ]

