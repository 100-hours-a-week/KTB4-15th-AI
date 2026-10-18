"""12단계: V2 fallback 이 재사용할 V1 동기 처리 경계.

실제 Runware·S3 는 부르지 않는다. 실제 RunwarePrunaProvider·S3ImageStorage 를 쓰고 HTTP·boto3 계층만 가짜로 바꿔
유료 생성 요청 횟수를 센다. V1 Service 는 단계 메서드(prepare_fitting·run_vton·store_result)만 따로 불러
예산·비용 기록·LLM 이 함께 돌지 않는지 확인한다(13단계 RUN_FALLBACK 핸들러는 아직 없다).
"""

import io
import json
import uuid
from decimal import Decimal
from urllib.error import HTTPError, URLError

import pytest

from app.clients.s3 import ImageStorageError, S3ImageStorage
from app.virtual_fitting.exceptions import (
    FittingImageStorageError,
    FittingModelError,
    FittingTimeoutError,
    ProductNotFoundError,
)
from app.virtual_fitting.providers.base import FittingInput, FittingResult
from app.virtual_fitting.providers.runware import RUNWARE_ENDPOINT, RunwarePrunaProvider
from app.virtual_fitting.service import VirtualFittingService
from tests.test_virtual_fitting_service import (
    BOTTOM,
    RESULT_KEY,
    TOP,
    FakeFittingProvider,
    FakeImageStorage,
    FakeRepository,
)

PRESIGNED = (
    "https://bucket.s3.ap-northeast-2.amazonaws.com/users/15/body.png?X-Amz-Signature=SECRETSIG"
)
CDN_URL = "https://im.runware.ai/image/os/a14d18/result.jpg"
FALLBACK_KEY = "virtual-fitting/results/123/pruna-sync.jpg"
TASK = "a770f077-f413-47de-9dac-be0b26a35da6"


def _png() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png()


class Response:
    def __init__(self, body):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return self.body


class CountingOpener:
    """유료 생성 요청을 센다. error 면 던지고, 아니면 body 를 돌려준다."""

    def __init__(self, body=None, error=None):
        self.body, self.error, self.requests = body, error, []

    def __call__(self, request, timeout):
        self.requests.append(json.loads(request.data))
        if self.error is not None:
            raise self.error
        return Response(self.body)


def fitting_input():
    return FittingInput(PRESIGNED, ["https://img/top.jpg", "https://img/bottom.jpg"], "prompt")


def success(cost=0.015):
    item = {"taskType": "imageInference", "imageURL": CDN_URL}
    if cost is not None:
        item["cost"] = cost
    return {"data": [item]}


# --- 동기 생성 Provider: 호출 한 번 = 유료 요청 한 번 ---


def test_one_try_on_is_one_paid_request_and_returns_image_and_cost():
    opener = CountingOpener(success())
    result = RunwarePrunaProvider("k", opener=opener).try_on(fitting_input())
    assert result == FittingResult(CDN_URL, Decimal("0.015"))
    (payload,) = opener.requests
    assert len(payload) == 1 and payload[0]["deliveryMethod"] == "sync"
    assert payload[0]["includeCost"] is True
    assert payload[0]["inputs"]["referenceImages"][0] == {"image": PRESIGNED, "role": "person"}


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (URLError(TimeoutError("timed out")), FittingTimeoutError),
        (TimeoutError(), FittingTimeoutError),
        (HTTPError(RUNWARE_ENDPOINT, 504, "gw", {}, io.BytesIO(b"")), FittingTimeoutError),
        (HTTPError(RUNWARE_ENDPOINT, 500, "err", {}, io.BytesIO(b"{}")), FittingModelError),
        (HTTPError(RUNWARE_ENDPOINT, 429, "busy", {}, io.BytesIO(b"{}")), FittingModelError),
        (ConnectionResetError(), FittingModelError),
        (URLError("connection refused"), FittingModelError),
    ],
)
def test_timeouts_and_unclear_responses_are_not_resent(error, expected):
    opener = CountingOpener(error=error)
    with pytest.raises(expected):
        RunwarePrunaProvider("k", opener=opener).try_on(fitting_input())
    assert len(opener.requests) == 1  # 자동으로 다시 보내지 않는다


@pytest.mark.parametrize("body", [b"not json", {"data": []}, {"errors": [{"message": "x"}]}])
def test_bad_responses_are_not_resent(body):
    opener = CountingOpener(body)
    with pytest.raises(FittingModelError):
        RunwarePrunaProvider("k", opener=opener).try_on(fitting_input())
    assert len(opener.requests) == 1


def test_caller_task_uuid_is_sent_and_default_stays_random():
    opener = CountingOpener(success())
    provider = RunwarePrunaProvider("k", opener=opener)
    provider.try_on(fitting_input(), task_uuid=TASK.upper())
    provider.try_on(fitting_input())
    provider.try_on(fitting_input())
    sent = [request[0]["taskUUID"] for request in opener.requests]
    assert sent[0] == TASK  # V2: 호출부가 정한 값(정규화)
    assert sent[1] != sent[2] and uuid.UUID(sent[1]).version == 4  # V1: 기존처럼 매번 새로


def test_invalid_task_uuid_is_rejected_before_sending():
    opener = CountingOpener(success())
    with pytest.raises(ValueError):
        RunwarePrunaProvider("k", opener=opener).try_on(fitting_input(), task_uuid="nope")
    assert opener.requests == []


def test_missing_cost_stays_none_not_zero():
    result = RunwarePrunaProvider("k", opener=CountingOpener(success(None))).try_on(fitting_input())
    assert result.cost is None


# --- V1 Service 단계 메서드만 쓰는 경로 ---


class Forbidden:
    """부르면 안 되는 의존성(예산·사용액 기록·LLM)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"fallback 재사용 경로는 {name} 를 부르지 않는다")

        return call


class KeyedStorage(FakeImageStorage):
    def __init__(self):
        super().__init__()
        self.keyed = []

    def store_remote_image_at(self, url, key):
        self.keyed.append((url, key))
        return key


def reuse_service(provider=None, storage=None):
    budget, comments = Forbidden(), Forbidden()
    service = VirtualFittingService(
        FakeRepository(TOP, BOTTOM),
        provider or FakeFittingProvider(result_image_url=CDN_URL),
        comments,
        storage or KeyedStorage(),
        budget,
    )
    return service, budget, comments


def test_reuse_steps_run_only_input_generation_and_keyed_storage():
    provider, storage = FakeFittingProvider(result_image_url=CDN_URL), KeyedStorage()
    service, budget, comments = reuse_service(provider, storage)

    prepared = service.prepare_fitting(["2", "1"], PRESIGNED)  # V2: 호출 직전에 발급한 원본 URL
    result = service.run_vton(prepared)
    key = service.store_result(result, key=FALLBACK_KEY)

    assert prepared.fitting_input.person_image_url == PRESIGNED
    assert prepared.fitting_input.garment_image_urls == [
        "https://img/top.jpg",
        "https://img/bottom.jpg",
    ]
    assert [p.product_code for p in prepared.products] == ["1", "2"]  # 상의 → 하의
    assert len(provider.inputs) == 1  # 생성 한 번
    assert key == FALLBACK_KEY and storage.keyed == [(CDN_URL, FALLBACK_KEY)]
    assert storage.urls == []  # V1 의 새 이름 저장은 쓰지 않았다
    # 예산 확인·사용액 기록·LLM 은 이 경로에서 돌지 않는다(13단계 호출부의 책임).
    assert budget.calls == [] and comments.calls == []


def test_prepare_fitting_keeps_v1_validation():
    service, budget, _ = reuse_service()
    with pytest.raises(ProductNotFoundError):
        service.prepare_fitting(["999"], PRESIGNED)
    assert budget.calls == []


def test_v1_store_result_default_is_unchanged():
    storage = KeyedStorage()
    service, _, _ = reuse_service(storage=storage)
    assert service.store_result(FittingResult(CDN_URL)) == RESULT_KEY
    assert storage.urls == [(CDN_URL, "virtual-fitting/results")] and storage.keyed == []


def test_keyed_storage_failure_keeps_the_v1_error_code():
    class Broken(KeyedStorage):
        def store_remote_image_at(self, url, key):
            raise ImageStorageError("download failed")

    service, _, _ = reuse_service(storage=Broken())
    with pytest.raises(FittingImageStorageError):
        service.store_result(FittingResult(CDN_URL), key=FALLBACK_KEY)


def test_run_vton_returns_cost_without_recording_it():
    provider = RunwarePrunaProvider("k", opener=CountingOpener(success(0.02)))
    service, budget, comments = reuse_service(provider=provider)
    result = service.run_vton(service.prepare_fitting(["1"], PRESIGNED))
    # 비용은 돌려주기만 한다. 기록은 호출부 한 곳(V1: fit 의 record_cost, V2: 13단계 TaskCostRepository)만 한다.
    assert result.cost == Decimal("0.02") and budget.calls == [] and comments.calls == []


# --- S3: 정해진 Key 에 저장 ---


class FakeS3Client:
    def __init__(self):
        self.puts = []

    def put_object(self, **params):
        self.puts.append(params)


def test_store_remote_image_at_downloads_and_puts_at_the_exact_key():
    client = FakeS3Client()
    storage = S3ImageStorage(client, bucket="b", opener=lambda request, timeout: Response(PNG))
    assert storage.store_remote_image_at(CDN_URL, FALLBACK_KEY) == FALLBACK_KEY
    ((put),) = client.puts
    assert put["Key"] == FALLBACK_KEY and put["ContentType"] == "image/png"


def test_store_remote_image_at_rejects_bad_keys_before_downloading():
    downloads = []
    storage = S3ImageStorage(
        FakeS3Client(), bucket="b", opener=lambda r, t: downloads.append(1) or Response(PNG)
    )
    for bad in ("", "/abs/key.jpg", "s3://b/k.jpg", "a/../b.jpg"):
        with pytest.raises(ValueError):
            storage.store_remote_image_at(CDN_URL, bad)
    assert downloads == []


def test_store_remote_image_at_failure_does_not_leak_the_url():
    def opener(request, timeout):
        raise OSError("boom")

    storage = S3ImageStorage(FakeS3Client(), bucket="b", opener=opener)
    with pytest.raises(ImageStorageError) as error:
        storage.store_remote_image_at(PRESIGNED, FALLBACK_KEY)
    assert "SECRETSIG" not in str(error.value) and "http" not in str(error.value)


def test_v1_store_remote_image_still_uses_a_new_name():
    client = FakeS3Client()
    storage = S3ImageStorage(client, bucket="b", opener=lambda request, timeout: Response(PNG))
    key = storage.store_remote_image(CDN_URL, "virtual-fitting/results")
    assert key.startswith("virtual-fitting/results/") and key.endswith(".png")
    assert key != FALLBACK_KEY


# --- Worker 연결 상태 유지 ---


def test_worker_stays_disabled_and_follow_up_paths_not_ready():
    # 13단계에서 RUN_FALLBACK 이 연결됐지만 활성화·결과 발행·복구 경로는 그대로다.
    from app.config import settings
    from app.virtual_fitting import postprocess_worker as pw
    from app.virtual_fitting.pruna_submitter import NextStep, NoFollowUpPaths

    assert pw.connected_postprocess_handlers().missing() == []
    assert not any(NoFollowUpPaths().ready_for(None, step) for step in NextStep)
    assert settings.VIRTUAL_FITTING_POSTPROCESS_WORKER_ENABLED is False
    assert settings.VIRTUAL_FITTING_REQUEST_WORKER_ENABLED is False
