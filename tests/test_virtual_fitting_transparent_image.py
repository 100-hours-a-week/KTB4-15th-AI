"""배경 제거(Runware RemBG v1.4 → 투명 WebP) 검증. V2 비동기·V1 fallback 공용.

1. Provider: 요청 형식(성공한 단독 테스트 설정)·taskUUID 지정·응답 해석·오류 분류(retryable·결과 불명확)
2. Alpha 검증: RGBA·Alpha 0/255·투명 픽셀 존재
3. 저장(가짜 S3): 원본과 다른 Key·Metadata·재사용, 작업당 1회 호출(S3 호출 표시), 어떤 실패에도 재호출 없음,
   호출 여부 불명확이면 보류, 받은 결과의 저장 일시 오류는 같은 URL 로만 다시, 비용 기록(작업별로 정해진 taskUUID)
4. 실제 조립(PostgreSQL + 가짜 SQS/S3/Runware): LLM 입력·결과 Key·비용·예산, 최초 호출 실패 → 최종 실패, 중복 수신·
   재시작·Watchdog 에도 재호출 없음, V1 fallback(정상·실패·이전 .jpg 원본), 재시도 표 없음, 설정 off 회귀, 처리 시간 로그
"""

import hashlib
import io
import json
from decimal import Decimal
from urllib.error import HTTPError

import pytest
from PIL import Image

from app.clients.s3 import ConditionalPutOutcome, ConditionalPutResult, S3ImageStorage
from app.virtual_fitting import transparent_image
from app.virtual_fitting.providers.runware_rembg import (
    REMBG_MODEL,
    RembgError,
    RembgResult,
    RunwareRembgProvider,
    build_payload,
    parse_result,
)
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.result_image import ResultImageError
from app.virtual_fitting.transparent_image import (
    BackgroundRemovalFailed,
    BackgroundRemovalPending,
    rembg_call_key,
    rembg_task_uuid,
    store_transparent_image,
    transparent_result_key,
    validate_transparent_webp,
)
from tests.test_virtual_fitting_v2_scenarios import (  # noqa: F401 (pytest fixture)
    JOB,
    SYNC_KEY,
    age,
    client,
    costs,
    env,
    expire_lease,
    finish,
    rembg_costs,
    restart,
    row,
    submitted,
)
from tests.v2_integration_harness import (
    BUCKET,
    COMMENT,
    TITLE,
    FakeS3Backend,
    Stack,
    post_webhook,
    transparent_webp_bytes,
    webp_bytes,
)

TASK = "11111111-2222-4333-8444-555555555555"
REMBG_TASK = "99999999-2222-4333-8444-555555555555"
SOURCE = f"virtual-fitting/results/{JOB}/pruna-async.webp"
CLEAR = f"virtual-fitting/results/{JOB}/transparent.webp"
RESULT_URL = "https://im.runware.ai/image/ws/0.5/ii/rembg.webp?token=x"
MARKER = f"virtual-fitting/results/{JOB}/rembg-call.json"
FAILED_CODE = "FITTING_BACKGROUND_REMOVAL_FAILED"


def rgba_webp(alpha_values) -> bytes:
    """세로 줄마다 Alpha 를 바꾼 RGBA WebP(lossless 라 Alpha 가 그대로 보존된다)."""
    image = Image.new("RGBA", (len(alpha_values), 4))
    for x, alpha in enumerate(alpha_values):
        for y in range(4):
            image.putpixel((x, y), (255, 255, 255, alpha))
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", lossless=True)
    return buffer.getvalue()


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


# === 1. Provider ===


class _Body(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Opener:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, request, timeout=None):
        self.requests.append((request, timeout))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _Body(json.dumps(item).encode())


def success(task=REMBG_TASK, cost=0.0006, url=RESULT_URL):
    item = {
        "taskType": "imageBackgroundRemoval",
        "taskUUID": task,
        "imageUUID": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        "imageURL": url,
    }
    if cost is not None:
        item["cost"] = cost
    return {"data": [item]}


def provider(*responses):
    opener = Opener(*responses)
    return RunwareRembgProvider("key-1", opener=opener, new_task_uuid=lambda: REMBG_TASK), opener


def http_error(code):
    return HTTPError("https://api.runware.ai/v1", code, "x", {}, io.BytesIO(b'{"errors":[]}'))


def test_payload_uses_the_tested_settings_and_sync_webp_output():
    (task,) = build_payload("https://bucket/source.webp?X-Amz-Signature=s", REMBG_TASK)
    assert task == {
        "taskType": "removeBackground",
        "taskUUID": REMBG_TASK,
        "model": "runware:109@1",
        "inputs": {"image": "https://bucket/source.webp?X-Amz-Signature=s"},
        "settings": {
            "alphaMatting": True,
            "postProcessMask": True,
            "returnOnlyMask": False,
            "rgba": [255, 255, 255, 0],
        },
        "outputType": "URL",
        "outputFormat": "WEBP",
        "deliveryMethod": "sync",
        "includeCost": True,
    }
    assert REMBG_MODEL == "runware:109@1"


def test_provider_sends_one_request_with_the_vton_key_and_parses_cost():
    rembg, opener = provider(success())
    result = rembg.remove_background("https://bucket/source.webp")
    ((request, timeout),) = opener.requests
    assert request.full_url == "https://api.runware.ai/v1" and timeout == 30.0
    assert request.get_header("Authorization") == "Bearer key-1"
    assert json.loads(request.data)[0]["inputs"]["image"] == "https://bucket/source.webp"
    assert result.image_url == RESULT_URL and result.cost == Decimal("0.0006")
    assert result.task_uuid == REMBG_TASK
    assert RESULT_URL not in repr(result)  # 결과 URL(token) 은 repr 에 남기지 않는다


def test_missing_or_bad_cost_keeps_the_result():
    assert parse_result(json.dumps(success(cost=None)).encode(), REMBG_TASK).cost is None
    assert parse_result(json.dumps(success(cost="x")).encode(), REMBG_TASK).cost is None


@pytest.mark.parametrize(
    ("response", "reason", "code"),
    [
        (TimeoutError("t"), "timeout", "timeout"),
        (http_error(504), "timeout", "timeout"),
        (http_error(500), "http_error", "http_500"),
        (OSError("reset"), "connection_failed", "connection_failed"),
        ({"data": []}, "invalid_response", "invalid_response"),
        (
            success(task="00000000-0000-4000-8000-000000000000"),
            "invalid_response",
            "invalid_response",
        ),
        (http_error(429), "http_error", "http_429"),
        (http_error(400), "http_error", "http_400"),
        (http_error(401), "http_error", "http_401"),
        (http_error(402), "balance_exhausted", "http_402"),
        ({"errors": [{"code": "serviceOverloaded"}]}, "api_error", "serviceOverloaded"),
        ({"errors": [{"code": "invalidApiKey"}]}, "api_error", "invalidApiKey"),
        (success(url="https://evil.example/x.webp"), "invalid_result_ref", "invalid_result_ref"),
        (success(url="http://im.runware.ai/x.webp"), "invalid_result_ref", "invalid_result_ref"),
    ],
)
def test_failures_are_classified_without_resending(response, reason, code):
    rembg, opener = provider(response)
    with pytest.raises(RembgError) as caught:
        rembg.remove_background("https://bucket/source.webp")
    assert (caught.value.reason, caught.value.error_code) == (reason, code)
    assert len(opener.requests) == 1  # Provider 는 다시 보내지 않는다


def test_given_task_uuid_is_sent_and_matched():
    given = "12345678-2222-4333-8444-555555555555"
    rembg, opener = provider(success(task=given))
    result = rembg.remove_background("https://bucket/source.webp", task_uuid=given)
    ((request, _),) = opener.requests
    assert json.loads(request.data)[0]["taskUUID"] == given and result.task_uuid == given


def test_actual_response_task_type_is_matched_by_task_uuid():
    """실측 응답의 taskType 은 imageBackgroundRemoval 이다(요청한 removeBackground 와 다름)."""
    body = success()
    body["data"][0]["taskType"] = "imageBackgroundRemoval"
    assert parse_result(json.dumps(body).encode(), REMBG_TASK).image_url == RESULT_URL


def test_non_https_source_is_rejected_before_calling():
    rembg, opener = provider()
    with pytest.raises(ValueError):
        rembg.remove_background("http://bucket/source.webp")
    assert opener.requests == []


# === 2. Alpha 검증 ===


def test_transparent_webp_reports_rgba_alpha_range_and_ratio():
    stats = validate_transparent_webp(rgba_webp([0, 0, 128, 255]))
    assert (stats.mode, stats.alpha_min, stats.alpha_max) == ("RGBA", 0, 255)
    assert (stats.translucent_ratio, stats.clear_ratio) == (0.75, 0.5)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (webp_bytes(), "no_alpha"),  # RGB(Alpha 없음, 검은 배경 그대로)
        (rgba_webp([128, 255, 255]), "no_transparency"),  # 완전 투명 픽셀 없음
        (rgba_webp([0, 0, 200]), "no_foreground"),  # 인물까지 지워짐
        (png_bytes(), "not_webp"),  # 투명 PNG 여도 WebP 가 아니면 거절
        (b"", "empty"),
    ],
)
def test_images_without_real_transparency_are_rejected(body, reason):
    with pytest.raises(ResultImageError) as caught:
        validate_transparent_webp(body)
    assert caught.value.reason == reason and caught.value.retryable is False


# === 3. 저장(가짜 S3) ===


class Rembg:
    def __init__(self, *modes):
        self.modes, self.calls, self.tasks = list(modes), [], []

    def remove_background(self, image_url, *, task_uuid=None):
        self.calls.append(image_url)
        self.tasks.append(task_uuid)
        mode = self.modes.pop(0) if self.modes else "success"
        if isinstance(mode, RembgError):
            raise mode
        return RembgResult(
            task_uuid, "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", RESULT_URL, Decimal("0.0006")
        )


class Costs:
    """TaskCostRepository 대역: 같은 taskUUID 는 한 번만 더한다."""

    def __init__(self):
        self.records = {}

    def record(self, task_uuid, job_id, cost, usage_date):
        if task_uuid in self.records:
            return False
        self.records[task_uuid] = (job_id, cost)
        return True


class Case:
    """저장 한 묶음(같은 S3 로 여러 번 실행 = 재수신·재시작)."""

    def __init__(self, *modes, bodies=None):
        backend = FakeS3Backend()
        backend.store(SOURCE, webp_bytes(), "image/webp")
        client_ = backend.client()
        self.backend = backend
        self.storage = S3ImageStorage(client_, bucket=BUCKET, presign_client=client_)
        self.rembg, self.costs = Rembg(*modes), Costs()
        self.bodies = list(bodies or [])  # 차례로 돌려줄 다운로드 결과(bytes 또는 ResultImageError)
        self.downloads = []

    def download(self, url):
        self.downloads.append(url)
        body = self.bodies.pop(0) if self.bodies else transparent_webp_bytes()
        if isinstance(body, Exception):
            raise body
        return body

    def marker(self):
        found = self.backend.objects.get(MARKER)
        return json.loads(found[0]) if found else None

    def run(self, task=TASK, **kwargs):
        return store_transparent_image(
            self.storage,
            fitting_job_id=JOB,
            task_uuid=task,
            source_key=SOURCE,
            rembg=self.rembg,
            costs=self.costs,
            download=self.download,
            source_url_expires_seconds=600,
            sleep=lambda seconds: None,
            **kwargs,
        )


def test_original_and_transparent_results_are_stored_under_separate_keys():
    case = Case()
    original = case.backend.objects[SOURCE]
    image = case.run()
    assert transparent_result_key(JOB) == CLEAR == image.key and not image.reused
    assert rembg_call_key(JOB) == MARKER
    assert case.backend.objects[SOURCE] == original  # 원본 그대로
    body, content_type, _ = case.backend.objects[CLEAR]
    assert content_type == "image/webp" and body == transparent_webp_bytes()
    assert case.backend.metadata[CLEAR] == {
        "fitting-job-id": str(JOB),
        "task-uuid": TASK,  # 원본을 만든 생성 task
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    # RemBG 는 원본의 Presigned GET 을 읽고, 결과는 im.runware.ai 에서 내려받는다.
    assert len(case.rembg.calls) == 1 and f"/{SOURCE}?" in case.rembg.calls[0]
    assert ("get_object", SOURCE) in case.backend.presigned and case.downloads == [RESULT_URL]
    assert (image.alpha.mode, image.alpha.alpha_min, image.alpha.alpha_max) == ("RGBA", 0, 255)
    # taskUUID 는 작업·원본 task 로 정해진다. 호출 직전에 표시 객체를 만들고, 비용은 그 taskUUID 로 한 번.
    expected = rembg_task_uuid(JOB, TASK)
    assert case.rembg.tasks == [expected] and image.rembg_task_uuid == expected
    marker = case.marker()
    assert (marker["state"], marker["rembg_task_uuid"], marker["fitting_job_id"]) == (
        "called",
        expected,
        JOB,
    )
    assert RESULT_URL not in json.dumps(marker) and "token" not in json.dumps(marker)
    assert case.costs.records == {expected: (JOB, Decimal("0.0006"))}
    assert image.store_tries == 1 and image.rembg_ms is not None and image.store_ms is not None


def test_rembg_task_uuid_is_stable_per_job_and_source():
    other = "22222222-2222-4333-8444-555555555555"
    assert rembg_task_uuid(JOB, TASK) == rembg_task_uuid(JOB, TASK)
    assert (
        len({rembg_task_uuid(JOB, TASK), rembg_task_uuid(JOB, other), rembg_task_uuid(1, TASK)})
        == 3
    )


def test_saved_result_is_reused_without_calling_the_paid_api_again():
    case = Case()
    first = case.run()
    second = case.run()
    assert second.reused and second.key == first.key and second.sha256 == first.sha256
    assert len(case.rembg.calls) == 1 and len(case.downloads) == 1
    assert [p["Key"] for p in case.backend.puts] == [MARKER, CLEAR] and len(case.costs.records) == 1


def test_object_from_another_task_is_a_conflict_without_calling_the_api():
    case = Case()
    case.run()
    with pytest.raises(ResultImageError) as caught:
        case.run(task="22222222-2222-4333-8444-555555555555")
    assert caught.value.reason == "existing_object_conflict" and len(case.rembg.calls) == 1


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (RembgError("timeout", retryable=True), "timeout"),
        (RembgError("http_error", retryable=True, http_status=429), "http_429"),
        (RembgError("http_error", retryable=True, http_status=500), "http_500"),
        (RembgError("http_error", retryable=False, http_status=400), "http_400"),
        (RembgError("balance_exhausted", retryable=False, http_status=402), "http_402"),
        (RembgError("api_error", retryable=True, code="serviceOverloaded"), "serviceOverloaded"),
        (RembgError("connection_failed", retryable=True), "connection_failed"),
        (RembgError("invalid_response", retryable=False), "invalid_response"),
    ],
)
def test_any_call_failure_is_final_without_calling_again(error, reason):
    case = Case(error)
    with pytest.raises(BackgroundRemovalFailed) as caught:
        case.run()
    assert caught.value.reason == reason and len(case.rembg.calls) == 1
    assert (case.marker()["state"], case.marker()["reason"]) == ("failed", reason)
    assert CLEAR not in case.backend.objects and case.costs.records == {}
    # 재수신·재시작: 표시가 "failed" 라 부르지 않고 같은 최종 실패로 이어 간다.
    with pytest.raises(BackgroundRemovalFailed) as again:
        case.run()
    assert again.value.reason == reason and len(case.rembg.calls) == 1


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (webp_bytes(), "result_no_alpha"),  # 검은 배경 그대로(RGB)
        (rgba_webp([128, 255]), "result_no_transparency"),
        (rgba_webp([0, 0, 200]), "result_no_foreground"),
        (ResultImageError("http_error", http_status=403), "result_http_403"),  # 결과 URL 거절·만료
    ],
)
def test_unusable_result_is_final_without_calling_again(body, reason):
    case = Case(bodies=[body])
    with pytest.raises(BackgroundRemovalFailed) as caught:
        case.run()
    assert caught.value.reason == reason and len(case.rembg.calls) == 1
    assert case.marker()["state"] == "failed" and CLEAR not in case.backend.objects
    assert len(case.costs.records) == 1  # 쓸 수 없는 결과도 비용은 나갔다(한 번 기록)


def test_store_hiccup_retries_the_same_result_url_only():
    case = Case(bodies=[ResultImageError("timeout"), transparent_webp_bytes()])
    image = case.run()
    assert image.key == CLEAR and image.store_tries == 2
    assert len(case.rembg.calls) == 1 and case.downloads == [RESULT_URL, RESULT_URL]


def test_called_without_a_stored_result_is_held_without_calling_again():
    """응답을 받은 뒤 결과 저장이 계속 실패(또는 호출 중 중단)하면 다음 실행은 다시 부르지 않고 보류한다."""
    case = Case(bodies=[ResultImageError("timeout")] * 3)
    with pytest.raises(ResultImageError) as caught:
        case.run()
    assert caught.value.retryable and len(case.downloads) == 3  # 같은 URL 로 최대 3번
    assert case.marker()["state"] == "called" and CLEAR not in case.backend.objects
    for _ in range(2):  # 재수신·재시작·Watchdog 재발행
        with pytest.raises(BackgroundRemovalPending) as held:
            case.run()
        assert held.value.reason == "rembg_call_unconfirmed"
    assert len(case.rembg.calls) == 1 and len(case.costs.records) == 1


class MarkerPut:
    """호출 표시 PUT 결과를 바꾸는 저장소 감싸기."""

    def __init__(self, storage, outcome, *, store=False, other=False):
        self._storage, self.outcome, self.store, self.other = storage, outcome, store, other

    def put_object_if_absent(self, body, key, *, content_type, metadata):
        if key != MARKER:
            return self._storage.put_object_if_absent(
                body, key, content_type=content_type, metadata=metadata
            )
        if self.store:  # 저장은 됐는데 응답을 잃었다
            self._storage.put_object_if_absent(
                body, key, content_type=content_type, metadata=metadata
            )
        if self.other:  # 다른 실행이 먼저 표시를 만들었다
            marker = json.loads(body)
            marker["claim"] = "other-run"
            self._storage.upload_bytes(json.dumps(marker).encode(), key, "application/json")
        return ConditionalPutResult(self.outcome, key)

    def __getattr__(self, name):
        return getattr(self._storage, name)


@pytest.mark.parametrize(
    ("outcome", "store", "other", "calls", "held"),
    [
        (ConditionalPutOutcome.UNKNOWN, True, False, 1, None),  # 우리 표시로 확인 → 부른다
        (ConditionalPutOutcome.UNKNOWN, False, False, 0, "rembg_call_unconfirmed"),
        (ConditionalPutOutcome.EXISTS, False, True, 0, "rembg_call_unconfirmed"),
        (ConditionalPutOutcome.CONFLICT, False, True, 0, "rembg_call_unconfirmed"),
        (ConditionalPutOutcome.FAILED, False, False, 0, "rembg_call_marker_put_failed"),
    ],
)
def test_only_the_run_that_created_the_marker_calls(outcome, store, other, calls, held):
    case = Case()
    case.storage = MarkerPut(case.storage, outcome, store=store, other=other)
    if held is None:
        assert case.run().key == CLEAR
    else:
        with pytest.raises(BackgroundRemovalPending) as caught:
            case.run()
        assert caught.value.reason == held
    assert len(case.rembg.calls) == calls


def test_source_presign_failure_happens_before_the_marker():
    from app.clients.s3 import ImageStorageError

    case = Case()

    def fail(*args, **kwargs):
        raise ImageStorageError("signing failed")

    case.storage.presigned_get_url = fail
    with pytest.raises(ImageStorageError):
        case.run()
    assert case.marker() is None and case.rembg.calls == []  # 호출 기회를 쓰지 않았다


def test_concurrent_save_by_another_run_is_confirmed_and_reused():
    """결과 조건부 PUT 이 412(같은 결과가 이미 저장)면 그 객체를 확인해 쓴다(덮어쓰지 않음)."""
    case = Case()
    other_body = rgba_webp([0, 255])

    def download(url):
        case.backend.store(CLEAR, other_body, "image/webp")
        case.backend.metadata[CLEAR] = {
            "fitting-job-id": str(JOB),
            "task-uuid": TASK,
            "sha256": hashlib.sha256(other_body).hexdigest(),
        }
        return transparent_webp_bytes()

    case.download = download
    image = case.run()
    assert image.reused is False and image.sha256 == hashlib.sha256(other_body).hexdigest()
    assert case.backend.objects[CLEAR][0] == other_body


def test_invalid_inputs_are_rejected_before_any_call():
    case = Case()
    for task, source in (("x", SOURCE), (TASK, CLEAR), (TASK, MARKER)):
        with pytest.raises(ValueError):
            store_transparent_image(
                case.storage,
                fitting_job_id=JOB,
                task_uuid=task,
                source_key=source,
                rembg=case.rembg,
            )
    assert case.rembg.calls == [] and case.marker() is None


# === 4. 실제 조립(통합) ===


def completed_by_webhook(env, stack):  # noqa: F811
    task = submitted(env, stack)
    assert post_webhook(client(), env.runware.complete(task)).status_code == 200
    return task


def redeliver(env):  # noqa: F811
    env.sqs.expire_visibility("ai-fitting-response")
    expire_lease(env)


def env_marker(env):  # noqa: F811
    found = env.s3.objects.get(MARKER)
    return json.loads(found[0]) if found else None


def test_llm_gets_the_transparent_image_and_be_gets_its_key(env):  # noqa: F811
    stack = Stack(env).mark_running()
    task = completed_by_webhook(env, stack)
    finish(env, stack)
    final = row(env)
    assert final.status is AsyncJobStatus.COMPLETED
    assert final.result_message_status is ResultMessageStatus.PUBLISHED
    assert (final.async_image_key, final.verified_image_key) == (SOURCE, SOURCE)
    assert final.result_image_key == CLEAR
    comment_request, title_request = env.runware.llm_requests
    url = comment_request["messages"][1]["content"][1]["image_url"]["url"]
    assert f"/{CLEAR}?" in url and SOURCE not in url  # LLM 에는 배경 제거된 최종 이미지
    assert title_request["messages"][1]["content"] == f"코디 설명: {COMMENT}"
    (be,) = env.sqs.bodies("be-fitting-response")
    assert be == {
        "fitting_job_id": JOB,
        "status": "COMPLETED",
        "result_image_key": CLEAR,
        "llm_title": TITLE,
        "llm_comment": COMMENT,
    }
    assert env.runware.rembg_inputs == [SOURCE] and env.runware.count("rembg") == 1
    stats = validate_transparent_webp(env.s3.objects[CLEAR][0])
    assert (stats.alpha_min, stats.alpha_max) == (0, 255)
    assert env.s3.metadata[CLEAR]["task-uuid"] == task
    # 비용은 Pruna task·RemBG 호출 taskUUID 별로 한 번씩. 재시도 기록 표 없이 동작한다.
    assert rembg_costs(env) == [(rembg_task_uuid(JOB, task), Decimal("0.0006"))]
    assert costs(env) == [(task, Decimal("0.023"))]
    assert env_marker(env)["state"] == "called"
    assert env.sql("SELECT to_regclass('virtual_fitting_rembg_attempts')") == [(None,)]


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("timeout", "timeout"),
        (("http", 429), "http_429"),
        (("http", 500), "http_500"),
        (("errors", [{"code": "serviceOverloaded", "message": "x"}]), "serviceOverloaded"),
        ("opaque", "result_no_alpha"),  # 결과 이미지 검증 실패
    ],
)
def test_first_call_failure_fails_the_job_without_retry_regeneration_or_fallback(env, mode, reason):  # noqa: F811
    env.runware.rembg_modes.append(mode)
    stack = Stack(env).mark_running()
    task = completed_by_webhook(env, stack)
    finish(env, stack)
    final = row(env)
    assert final.status is AsyncJobStatus.FAILED
    assert final.result_message_status is ResultMessageStatus.PUBLISHED
    (be,) = env.sqs.bodies("be-fitting-response")
    assert be == {
        "fitting_job_id": JOB,
        "status": "FAILED",
        "error_code": FAILED_CODE,
        "error_message": "가상 피팅 결과 이미지의 배경 제거에 실패했습니다.",
    }
    assert env.runware.count("rembg") == 1  # 재호출 없음
    assert (env_marker(env)["state"], env_marker(env)["reason"]) == ("failed", reason)
    # RemBG 실패는 Pruna 실패가 아니다: 다시 생성하지 않고 fallback 하지 않으며 원본은 복구용으로 남는다.
    assert env.runware.count("async_submit") == 1 and env.runware.count("sync_generate") == 0
    assert final.task_uuid == task and final.fallback_attempt_count == 0
    assert final.verified_image_key == SOURCE and SOURCE in env.s3.objects
    assert final.result_image_key is None and CLEAR not in env.s3.objects
    assert env.runware.count("llm_comment") == 0
    # 늦은 중복 메시지가 와도 부르지 않는다
    env.sqs.put(
        "ai-fitting-response", {"fitting_job_id": JOB, "task_uuid": task, "action": "POSTPROCESS"}
    )
    finish(env, stack)
    assert env.runware.count("rembg") == 1 and len(env.sqs.bodies("be-fitting-response")) == 1


def test_unconfirmed_call_is_held_across_restart_and_watchdog_without_calling_again(env):  # noqa: F811
    """응답은 받았지만 결과 저장이 끝나지 않은 채 멈췄다(호출 여부·결과 불명확). 다시 부르지 않고 운영 확인으로 보류한다."""
    from app.virtual_fitting import result_image
    from app.virtual_fitting.result_image import ResultImageError as Error

    real = result_image.download_runware_image

    def download(url, *args, **kwargs):
        if "/rembg-" in url:
            raise Error("timeout")
        return real(url, *args, **kwargs)

    env.monkeypatch.setattr(result_image, "download_runware_image", download)
    env.monkeypatch.setattr(transparent_image, "STORE_RETRY_SECONDS", 0.0)
    stack = Stack(env).mark_running()
    task = completed_by_webhook(env, stack)
    (outcome,) = stack.postprocess_once()
    assert outcome.delete is False and env_marker(env)["state"] == "called"
    # Worker 재시작 + 재수신 + 중복 메시지
    stack = restart(env)
    env.sqs.put(
        "ai-fitting-response", {"fitting_job_id": JOB, "task_uuid": task, "action": "POSTPROCESS"}
    )
    redeliver(env)
    stack.postprocess_once()
    # 메시지가 DLQ 로 빠진 뒤 Watchdog 재발행 1회
    for message in env.sqs.live("ai-fitting-response"):
        message.deleted = True
    expire_lease(env)
    age(env, "image_verified_at", 7200)
    age(env, "runware_accepted_at", 7200)
    env.sql("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
    stack.watchdog_once()
    stack.postprocess_once()
    current = row(env)
    assert env.runware.count("rembg") == 1  # 어느 경로에서도 다시 부르지 않았다
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert current.result_payload is None and current.fallback_attempt_count == 0
    assert env.runware.count("sync_generate") == 0 and env.runware.count("llm_comment") == 0
    assert len(rembg_costs(env)) == 1  # 받은 응답의 비용은 한 번 기록됐다


def test_db_failure_after_the_transparent_save_reuses_it_on_redelivery(env):  # noqa: F811
    from app.virtual_fitting.exceptions import FittingDatabaseError
    from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository

    original = AsyncJobStateRepository.claim_llm_generation_attempt
    failed = []

    def fail_once(self, *args, **kwargs):
        if not failed:
            failed.append(1)
            raise FittingDatabaseError("injected")
        return original(self, *args, **kwargs)

    env.monkeypatch.setattr(AsyncJobStateRepository, "claim_llm_generation_attempt", fail_once)
    stack = Stack(env).mark_running()
    completed_by_webhook(env, stack)
    (first,) = stack.postprocess_once()
    assert first.delete is False and CLEAR in env.s3.objects and row(env).result_payload is None
    # Worker 재시작 + 같은 메시지 재수신 + 중복 메시지(lease 로 한 번에 하나만 실행)
    task = row(env).task_uuid
    stack = restart(env)
    env.sqs.put(
        "ai-fitting-response", {"fitting_job_id": JOB, "task_uuid": task, "action": "POSTPROCESS"}
    )
    redeliver(env)
    finish(env, stack)
    assert row(env).status is AsyncJobStatus.COMPLETED and row(env).result_image_key == CLEAR
    assert env.runware.count("rembg") == 1  # 유료 배경 제거 재호출 없음
    assert env.runware.count("llm_comment") == 1 and env.runware.count("llm_title") == 1
    assert len(rembg_costs(env)) == 1  # 비용 중복 기록 없음


def test_watchdog_republish_after_the_transparent_save_completes_without_calling_again(env):  # noqa: F811
    from app.virtual_fitting.exceptions import FittingDatabaseError
    from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository

    original = AsyncJobStateRepository.claim_llm_generation_attempt
    failed = []

    def fail_once(self, *args, **kwargs):
        if not failed:
            failed.append(1)
            raise FittingDatabaseError("injected")
        return original(self, *args, **kwargs)

    env.monkeypatch.setattr(AsyncJobStateRepository, "claim_llm_generation_attempt", fail_once)
    stack = Stack(env).mark_running()
    completed_by_webhook(env, stack)
    stack.postprocess_once()
    # 메시지가 DLQ 로 빠진 것처럼 내부 큐에서 사라진 뒤, 확인 시각 기준으로 오래 진행이 없다.
    for message in env.sqs.live("ai-fitting-response"):
        message.deleted = True
    expire_lease(env)
    age(env, "image_verified_at", 7200)
    age(env, "runware_accepted_at", 7200)
    env.sql("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
    stack.watchdog_once()
    actions = [b["action"] for b in env.sqs.bodies("ai-fitting-response")]
    assert actions == ["POSTPROCESS", "POSTPROCESS"]  # Webhook 1번 + Watchdog 재발행 1번
    finish(env, stack)
    final = row(env)
    assert final.status is AsyncJobStatus.COMPLETED and final.result_image_key == CLEAR
    assert env.runware.count("rembg") == 1 and env.runware.count("sync_generate") == 0
    assert len(rembg_costs(env)) == 1


def fallback_flow(env):  # noqa: F811
    env.runware.async_modes.append(
        lambda submission: (
            "errors",
            [{"code": "timeoutProvider", "message": "x", "taskUUID": submission.task_uuid}],
        )
    )
    stack = Stack(env).mark_running()
    env.send_request(JOB)
    stack.request_once()
    return stack


def test_v1_fallback_result_goes_through_the_same_background_removal(env):  # noqa: F811
    stack = fallback_flow(env)
    finish(env, stack)
    final = row(env)
    assert final.status is AsyncJobStatus.COMPLETED and final.result_image_key == CLEAR
    assert final.fallback_image_key == SYNC_KEY  # 원본(pruna-sync.webp)은 복구용으로 보존
    assert env.s3.objects[SYNC_KEY][1] == "image/webp"
    assert env.runware.rembg_inputs == [SYNC_KEY]
    assert env.s3.metadata[CLEAR]["task-uuid"] == final.fallback_task_uuid
    url = env.runware.llm_requests[0]["messages"][1]["content"][1]["image_url"]["url"]
    assert f"/{CLEAR}?" in url
    assert env.sqs.bodies("be-fitting-response")[0]["result_image_key"] == CLEAR
    assert rembg_costs(env) == [(rembg_task_uuid(JOB, final.fallback_task_uuid), Decimal("0.0006"))]


def test_v1_fallback_background_removal_failure_does_not_generate_again(env):  # noqa: F811
    env.runware.rembg_modes.append(("http", 500))
    stack = fallback_flow(env)
    finish(env, stack)
    final = row(env)
    assert env.runware.count("sync_generate") == 1 and final.fallback_attempt_count == 1
    assert env.runware.count("rembg") == 1 and final.status is AsyncJobStatus.FAILED
    assert final.fallback_image_key == SYNC_KEY and SYNC_KEY in env.s3.objects
    assert env.sqs.bodies("be-fitting-response")[0]["error_code"] == FAILED_CODE


def test_fallback_original_saved_by_the_previous_version_as_jpg_is_used(env):  # noqa: F811
    """이전 코드가 저장한 pruna-sync.jpg 원본(기록된 Key)도 그대로 배경 제거에 쓴다."""
    from app.clients.s3 import ImageStorageError

    legacy = f"virtual-fitting/results/{JOB}/pruna-sync.jpg"
    real = transparent_image.result_image_get_url
    calls = []

    def fail_once(*args, **kwargs):  # 원본 기록 직후, RemBG 호출 전에 멈추게 한다
        calls.append(1)
        if len(calls) == 1:
            raise ImageStorageError("signing failed")
        return real(*args, **kwargs)

    env.monkeypatch.setattr(transparent_image, "result_image_get_url", fail_once)
    stack = fallback_flow(env)
    stack.postprocess_once()
    assert row(env).fallback_image_key == SYNC_KEY and row(env).result_payload is None
    assert env.runware.count("rembg") == 0 and env_marker(env) is None
    env.s3.store(legacy, content_type="image/jpeg")
    env.sql(
        "UPDATE virtual_fitting_async_jobs SET fallback_image_key = %s WHERE fitting_job_id = %s",
        (legacy, JOB),
    )
    redeliver(env)
    finish(env, stack)
    final = row(env)
    assert final.status is AsyncJobStatus.COMPLETED and final.result_image_key == CLEAR
    assert final.fallback_image_key == legacy and env.runware.rembg_inputs == [legacy]


def test_rembg_cost_counts_toward_the_daily_budget(env):  # noqa: F811
    """하루 개별 기록 합계 = Pruna + RemBG. 4.9764 + 0.023 + 0.0006 = 5.0000 → 다음 요청은 예산 초과."""
    env.sql(
        "INSERT INTO usage_amount (usage_date, amount, legacy_amount, recorded_amount, "
        "reconciled_amount, reconciled_at) VALUES ((now() AT TIME ZONE 'UTC')::date, 4.9764, 0, "
        "4.9764, 0, now())"
    )
    stack = Stack(env).mark_running()
    completed_by_webhook(env, stack)
    finish(env, stack)
    assert row(env).status is AsyncJobStatus.COMPLETED
    assert env.sql("SELECT recorded_amount, amount FROM usage_amount") == [
        (Decimal("5.0000"), Decimal("5.0000"))
    ]
    second = JOB + 1
    env.send_request(second)
    stack.request_once()
    assert row(env, second).result_payload["error_code"] == "FITTING_DAILY_BUDGET_EXCEEDED"
    assert env.runware.count("async_submit") == 1  # 두 번째 작업은 생성 요청을 보내지 않았다


def test_disabled_setting_keeps_the_original_result_contract(env):  # noqa: F811
    from app.config import settings

    env.monkeypatch.setattr(settings, "VIRTUAL_FITTING_REMOVE_BACKGROUND_ENABLED", False)
    stack = Stack(env).mark_running()
    completed_by_webhook(env, stack)
    finish(env, stack)
    final = row(env)
    assert final.status is AsyncJobStatus.COMPLETED
    assert final.result_image_key == final.verified_image_key == SOURCE
    assert env.runware.count("rembg") == 0 and CLEAR not in env.s3.objects
    assert env_marker(env) is None
    url = env.runware.llm_requests[0]["messages"][1]["content"][1]["image_url"]["url"]
    assert f"/{SOURCE}?" in url


def test_timing_events_are_traceable_by_fitting_job_id(env, caplog):  # noqa: F811
    import logging

    caplog.set_level(logging.INFO, logger="app.virtual_fitting.diagnostics")
    stack = Stack(env).mark_running()
    completed_by_webhook(env, stack)
    finish(env, stack)
    lines = [r.getMessage() for r in caplog.records if f"fitting_job_id={JOB} " in r.getMessage()]
    steps = [
        line.split("step=")[1].split()[0] for line in lines if "event=postprocess_timing " in line
    ]
    assert steps == ["pruna_image_store", "rembg_api", "rembg_store", "llm_comment", "llm_title"]
    (summary,) = [line for line in lines if "event=postprocess_timing_summary" in line]
    for field in (
        "pruna_image_ms=",
        "rembg_api_ms=",
        "rembg_store_ms=",
        "rembg_store_tries=1",
        "llm_comment_ms=",
        "llm_title_ms=",
        "handler_ms=",
        "since_pruna_submit_ms=",
        "rembg_cost=0.0006",
    ):
        assert field in summary, field
    (sent,) = [line for line in lines if "event=result_sent " in line]
    assert (
        "send_ms=" in sent and "since_job_created_ms=" in sent and "since_pruna_submit_ms=" in sent
    )
