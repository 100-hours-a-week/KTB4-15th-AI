"""IDM-VTON RUN_FALLBACK 핸들러: 입력 검증 → 호출 기회 1회 선점 → Modal 1회 → S3 WebP 확인 → 제목·코멘트 → 결과.

실제 10단계 FollowUpProcessor(lease·heartbeat) + IdmVtonFallbackHandler + 실제 AsyncJobStateRepository(격리 PostgreSQL)로
돌린다. S3·Modal·상품 이미지 분류·LLM·상품 조회는 가짜다. 가짜 Modal 은 spawn 횟수를 세고, 성공이면 받은 업로드 헤더
(Content-Type·x-amz-meta-*)로 가짜 S3 에 WebP 를 올린다. Pruna 동기 Provider·PrunaLimiter 는 주입하지 않는다.
"""

import hashlib
import io
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from PIL import Image

from app.clients.s3 import ImageStorageError, S3ObjectCheck, S3ObjectState
from app.virtual_fitting.exceptions import FittingDatabaseError, FittingPostprocessError
from app.virtual_fitting.idm_fallback_handler import (
    IDM_CONTENT_TYPE,
    META_ATTEMPT,
    META_JOB,
    RECOVERY_MARGIN_SECONDS,
    IdmVtonFallbackHandler,
    ResultState,
    idm_result_key,
    inspect_fallback_result,
)
from app.virtual_fitting.idm_image_validator import (
    GarmentImageCheck,
    IdmImageClass,
    IdmImageUnavailable,
)
from app.virtual_fitting.idm_modal_client import (
    IdmCallKind,
    IdmCallResult,
    IdmModalUnavailable,
    validate_output,
)
from app.virtual_fitting.postprocess_worker import PostprocessHandlers
from app.virtual_fitting.repositories.async_job_repository import AsyncJobStatus, NextAction
from app.virtual_fitting.repositories.async_job_state import ChangeReason
from app.virtual_fitting.service import FALLBACK_COMMENT, FALLBACK_TITLE
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_fallback_handler import FakeProducts, fallback_job, in_flight
from tests.test_virtual_fitting_postprocess_handler import FakeComments
from tests.test_virtual_fitting_postprocess_worker import Handler, counts, item, processor
from tests.test_virtual_fitting_pruna_submit import JOB, FakeClock, db, job  # noqa: F401
from tests.virtual_fitting_fixtures import make_bottom, make_top

KEY = idm_result_key(JOB)
TOP = make_top("123456", sub_category="후디", image_url="https://img.example/top.jpg")
BOTTOM = make_bottom("789012", image_url="https://img.example/bottom.jpg")
UPLOAD_EXPIRES = 1800


def webp_bytes(size=(384, 512), color=(40, 60, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="WEBP")
    return buffer.getvalue()


class FakeStorage:
    """사용자 원본 HEAD·Presigned GET/PUT·고정 Key 객체(본문·Content-Type·메타데이터·시각)를 흉내 낸다."""

    def __init__(self, user_state=S3ObjectState.EXISTS):
        self.user_state = user_state
        self.objects = {}  # key -> (body, content_type, metadata, last_modified)
        self.presigned, self.heads, self.reads = [], [], []
        self.head_faults, self.read_faults = [], []
        self.presign_error = False
        self.lock = threading.Lock()

    def put(self, key, body, content_type=IDM_CONTENT_TYPE, metadata=None, modified=None):
        with self.lock:
            self.objects[key] = (
                body,
                content_type,
                dict(metadata or {}),
                modified or datetime.now(UTC),
            )

    def presigned_get_url(self, key, *, expires_in):
        if self.presign_error:
            raise ImageStorageError("signing failed")
        self.presigned.append(("get", key, expires_in, None, None))
        return f"https://bucket.s3.ap-northeast-2.amazonaws.com/{key}?X-Amz-Signature=GET"

    def presigned_put_url(self, key, *, expires_in, content_type=None, metadata=None):
        if self.presign_error:
            raise ImageStorageError("signing failed")
        self.presigned.append(("put", key, expires_in, content_type, dict(metadata or {})))
        return f"https://bucket.s3.ap-northeast-2.amazonaws.com/{key}?X-Amz-Signature=PUT"

    def head_object(self, key):
        self.heads.append(key)
        if self.head_faults:
            return S3ObjectCheck(self.head_faults.pop(0), key)
        if key.startswith("users/"):
            return S3ObjectCheck(self.user_state, key)
        with self.lock:
            found = self.objects.get(key)
        if found is None:
            return S3ObjectCheck(S3ObjectState.NOT_FOUND, key)
        body, content_type, metadata, modified = found
        return S3ObjectCheck(
            S3ObjectState.EXISTS,
            key,
            http_status=200,
            content_length=len(body),
            content_type=content_type,
            last_modified=modified,
            metadata=metadata,
        )

    def read_object(self, key, *, max_bytes):
        self.reads.append(key)
        if self.read_faults:
            raise self.read_faults.pop(0)
        with self.lock:
            found = self.objects.get(key)
        if found is None:
            raise ImageStorageError("missing")
        return found[0]


class FakeCall:
    def __init__(self, request):
        self.request = request


class FakeModal:
    """spawn 횟수를 센다. outcome: success / failed:<kind> / unknown / upload_then_fail / running.

    wait 는 waits 만큼 None(실행 중)을 돌려준 뒤 결과를 낸다. success 는 받은 헤더로 storage 에 WebP 를 올린다.
    """

    def __init__(self, storage, outcome="success", waits=0, body=None, during_wait=None):
        self.storage, self.outcome, self.waits = storage, outcome, waits
        self.body = body or webp_bytes()
        self.requests, self.wait_calls = [], 0
        self.unavailable = False
        self.during_wait = during_wait
        self.gate = None

    def resolve(self):
        if self.unavailable:
            raise IdmModalUnavailable("not deployed")
        return self

    def _upload(self, request):
        headers = {k.lower(): v for k, v in request["upload_headers"].items()}
        metadata = {
            k[len("x-amz-meta-") :]: v for k, v in headers.items() if k.startswith("x-amz-meta-")
        }
        self.storage.put(KEY, self.body, headers["content-type"], metadata)

    def spawn(self, request):
        self.requests.append(request)
        if self.outcome == "unknown":
            return None, IdmCallResult(IdmCallKind.UNKNOWN, "spawn_ConnectionError")
        if self.outcome in ("success", "upload_then_fail"):
            self._upload(request)
        return FakeCall(request), None

    def wait(self, call, timeout_seconds, fitting_job_id, attempt_id):
        self.wait_calls += 1
        if self.gate is not None:
            self.gate.wait(10)
        if self.during_wait is not None:
            self.during_wait()
        if self.outcome == "running" or self.wait_calls <= self.waits:
            return None
        if self.outcome == "success":
            output = {
                "status": "succeeded",
                "fitting_job_id": fitting_job_id,
                "attempt_id": attempt_id,
                "content_type": IDM_CONTENT_TYPE,
                "bytes": len(self.body),
                "sha256": hashlib.sha256(self.body).hexdigest(),
                "width": 384,
                "height": 512,
            }
            return validate_output(output, fitting_job_id, attempt_id)
        if self.outcome == "upload_then_fail":
            return validate_output(
                {"status": "failed", "error_kind": "internal_error"}, fitting_job_id, attempt_id
            )
        if self.outcome.startswith("failed:"):
            kind = self.outcome.split(":", 1)[1]
            return validate_output(
                {"status": "failed", "error_kind": kind}, fitting_job_id, attempt_id
            )
        return IdmCallResult(IdmCallKind.UNKNOWN, "wait_ConnectionError")


class FakeClassifier:
    def __init__(self, classification=IdmImageClass.GARMENT_ONLY, error=None):
        self.classification, self.error, self.calls = classification, error, []

    def classify_url(self, url):
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        return GarmentImageCheck(self.classification, "test")


class FakeColors:
    def __init__(self, color="블랙"):
        self.color = color

    def find_color(self, product_code):
        return self.color


def env(db, **kwargs):  # noqa: F811
    clock = FakeClock()
    storage = kwargs.pop("storage", None) or FakeStorage()
    e = SimpleNamespace(
        clock=clock,
        storage=storage,
        modal=kwargs.pop("modal", None) or FakeModal(storage, **kwargs.pop("modal_kwargs", {})),
        classifier=kwargs.pop("classifier", None) or FakeClassifier(),
        comments=kwargs.pop("comments", None) or FakeComments(),
        products=kwargs.pop("products", None) or FakeProducts(TOP),
    )
    e.handler = IdmVtonFallbackHandler(
        jobs=db["jobs"],
        state=db["state"],
        products=e.products,
        storage=e.storage,
        comments=e.comments,
        modal=e.modal,
        classifier=e.classifier,
        colors=FakeColors(),
        clock=lambda: clock.sleep(1) or clock(),  # 대기 한 번에 1초씩 흐른다
        poll_seconds=1,
        max_wait_seconds=kwargs.pop("max_wait_seconds", 30),
        upload_expires_seconds=UPLOAD_EXPIRES,
        result_url_expires_seconds=600,
    )
    assert not kwargs, kwargs
    return e


def idm_job(db, products=("123456",)):  # noqa: F811
    """확인된 Pruna 실패로 FALLBACK_RUNNING / RUN_FALLBACK 이 된 작업. 기본은 상의 1개."""
    task = fallback_job(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET products = %s WHERE fitting_job_id = %s",
        (list(products), JOB),
    )
    return task


def run(db, e, **kwargs):  # noqa: F811
    return processor(
        db, PostprocessHandlers(postprocess=Handler(), run_fallback=e.handler), **kwargs
    ).process(item(JOB, job(db).task_uuid, NextAction.RUN_FALLBACK))


def expire_lease(db):  # noqa: F811
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second' "
        "WHERE fitting_job_id = %s AND worker_lease_token IS NOT NULL",
        (JOB,),
    )


def age_start(db, seconds):  # noqa: F811
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET fallback_started_at = now() - make_interval(secs => %s) "
        "WHERE fitting_job_id = %s",
        (seconds, JOB),
    )


def model_failed():
    return {
        "fitting_job_id": JOB,
        "status": "FAILED",
        "error_code": "FITTING_MODEL_FAILED",
        "error_message": "가상 피팅 이미지 생성에 실패했습니다.",
    }


def costs_and_slots(db):  # noqa: F811
    return (
        db["sql"]("SELECT count(*) FROM virtual_fitting_task_costs")[0][0],
        in_flight(db),
    )


# --- 정상 ---


def test_success_calls_modal_once_and_saves_the_completed_result(db):  # noqa: F811
    idm_job(db)
    e = env(db)
    outcome = run(db, e)
    current = job(db)

    (request,) = e.modal.requests  # 생성 함수 정확히 1번
    attempt = current.fallback_task_uuid
    assert request["fitting_job_id"] == JOB and request["attempt_id"] == attempt
    assert request["garment_image_url"] == TOP.image_url  # 상품 DB 의 외부 URL 그대로
    assert request["garment_description"] == "black hoodie"
    assert request["person_image_url"].startswith(
        "https://bucket.s3.ap-northeast-2.amazonaws.com/users/"
    )
    assert request["upload_headers"] == {
        "Content-Type": "image/webp",
        f"x-amz-meta-{META_JOB}": str(JOB),
        f"x-amz-meta-{META_ATTEMPT}": attempt,
    }
    put = [p for p in e.storage.presigned if p[0] == "put"]
    assert put == [
        ("put", KEY, UPLOAD_EXPIRES, "image/webp", {META_JOB: str(JOB), META_ATTEMPT: attempt})
    ]
    expected_deadline = current.fallback_started_at.timestamp() + UPLOAD_EXPIRES - 60
    assert request["not_after_epoch"] == pytest.approx(expected_deadline)
    assert current.fallback_attempt_count == 1 and current.fallback_started_at is not None
    assert current.fallback_image_key == KEY and current.result_image_key == KEY
    assert current.result_payload == {
        "fitting_job_id": JOB,
        "status": "COMPLETED",
        "result_image_key": KEY,
        "llm_title": "차분한 데일리룩",
        "llm_comment": "좋은 코디입니다.\n잘 어울립니다.",
    }
    assert current.next_action is NextAction.PUBLISH_RESULT
    assert current.status is AsyncJobStatus.FALLBACK_RUNNING  # 최종 상태는 결과 전송 루프가 확정
    ((llm_url, _),) = e.comments.comment_calls
    assert f"/{KEY}?" in llm_url  # LLM 은 IDM 결과 이미지로
    assert KEY in e.storage.reads  # 본문을 읽어 sha256·WebP 확인
    assert costs_and_slots(db) == (0, [])  # Pruna 비용·공유 자리를 쓰지 않음
    assert outcome.action == "handler_done"


def test_repeated_messages_and_restarts_do_not_call_again(db):  # noqa: F811
    idm_job(db)
    e = env(db)
    run(db, e)
    saved = job(db)
    for _ in range(3):
        expire_lease(db)
        run(db, env(db, modal=e.modal))
    assert len(e.modal.requests) == 1 and job(db).result_payload == saved.result_payload


# --- 입력 검증: Modal 미호출 ---


@pytest.mark.parametrize(
    "products, codes",
    [((BOTTOM,), ["789012"]), ((TOP, BOTTOM), ["123456", "789012"])],
    ids=["bottom", "top_and_bottom"],
)
def test_unsupported_combinations_do_not_call_modal(db, products, codes):  # noqa: F811
    idm_job(db, codes)
    e = env(db, products=FakeProducts(*products))
    run(db, e)
    current = job(db)
    assert e.modal.requests == [] and e.classifier.calls == []
    assert current.fallback_attempt_count == 0 and current.fallback_started_at is None
    assert current.result_payload == model_failed()
    assert current.next_action is NextAction.PUBLISH_RESULT


def test_non_https_product_image_is_failed_before_classification_and_claim(db):  # noqa: F811
    idm_job(db)
    http_top = make_top("123456", sub_category="후디", image_url="http://img.example/top.jpg")
    e = env(db, products=FakeProducts(http_top))
    run(db, e)
    current = job(db)
    assert e.classifier.calls == [] and e.modal.requests == []
    assert current.fallback_attempt_count == 0 and current.fallback_started_at is None
    assert current.result_payload == model_failed()


@pytest.mark.parametrize(
    "classification",
    [IdmImageClass.MODEL_WORN, IdmImageClass.AMBIGUOUS, IdmImageClass.INVALID],
)
def test_non_garment_only_images_do_not_call_modal(db, classification):  # noqa: F811
    idm_job(db)
    e = env(db, classifier=FakeClassifier(classification))
    run(db, e)
    current = job(db)
    assert e.classifier.calls == [TOP.image_url] and e.modal.requests == []
    assert current.fallback_attempt_count == 0 and current.result_payload == model_failed()


@pytest.mark.parametrize(
    "setup, reason",
    [
        (
            lambda e: setattr(e.classifier, "error", IdmImageUnavailable("timeout")),
            "garment_image_unavailable",
        ),
        (lambda e: setattr(e.modal, "unavailable", True), "idm_function_unavailable"),
        (lambda e: setattr(e.storage, "presign_error", True), "presign_failed"),
        (
            lambda e: setattr(e.storage, "user_state", S3ObjectState.TRANSIENT_ERROR),
            "user_image_transient_error",
        ),
    ],
)
def test_temporary_problems_before_the_call_keep_the_attempt(db, setup, reason):  # noqa: F811
    idm_job(db)
    e = env(db)
    setup(e)
    outcome = run(db, e)
    current = job(db)
    assert outcome.action == "handler_deferred_kept" and not outcome.delete
    assert e.modal.requests == [] and current.fallback_attempt_count == 0
    assert current.fallback_started_at is None and current.result_payload is None


def test_missing_user_image_is_a_failure_without_a_call(db):  # noqa: F811
    idm_job(db)
    e = env(db, storage=FakeStorage(S3ObjectState.NOT_FOUND))
    run(db, e)
    assert (
        e.modal.requests == [] and job(db).result_payload["error_code"] == "INVALID_REQUEST"
    )  # 기존 원본 없음 코드


# --- 생성 실패·불명확: 재호출 없음 ---


def test_finished_failure_without_a_result_is_failed_once(db):  # noqa: F811
    idm_job(db)
    e = env(db, modal_kwargs={"outcome": "failed:preprocess_rejected"})
    run(db, e)
    assert job(db).result_payload == model_failed() and job(db).fallback_attempt_count == 1
    expire_lease(db)
    run(db, e)
    assert len(e.modal.requests) == 1


def test_failure_after_upload_uses_the_uploaded_result(db):  # noqa: F811
    idm_job(db)
    e = env(db, modal_kwargs={"outcome": "upload_then_fail"})
    run(db, e)
    assert job(db).result_payload["status"] == "COMPLETED" and len(e.modal.requests) == 1


def test_unknown_spawn_is_held_until_the_upload_window_closes(db):  # noqa: F811
    idm_job(db)
    e = env(db, modal_kwargs={"outcome": "unknown"})
    first = run(db, e)
    assert first.action == "handler_deferred_kept" and job(db).fallback_attempt_count == 1
    # 재수신(Worker 재시작): 업로드 URL 이 아직 유효 → 실행 중일 수 있어 보류, 다시 부르지 않음
    expire_lease(db)
    assert run(db, env(db, modal=e.modal, storage=e.storage)).action == "handler_deferred_kept"
    assert job(db).result_payload is None
    # 만료 + 여유가 지나도 결과가 없으면 최종 실패
    age_start(db, UPLOAD_EXPIRES + RECOVERY_MARGIN_SECONDS + 5)
    expire_lease(db)
    run(db, env(db, modal=e.modal, storage=e.storage))
    assert job(db).result_payload == model_failed()
    assert len(e.modal.requests) == 1


def test_wait_timeout_then_late_upload_is_recovered_without_a_new_call(db):  # noqa: F811
    idm_job(db)
    e = env(db, modal_kwargs={"outcome": "running"}, max_wait_seconds=5)
    outcome = run(db, e)
    assert outcome.action == "handler_deferred_kept" and job(db).result_payload is None
    attempt = job(db).fallback_task_uuid
    # Modal 이 나중에 끝나 올린 결과
    e.storage.put(KEY, webp_bytes(), metadata={META_JOB: str(JOB), META_ATTEMPT: attempt})
    expire_lease(db)
    run(db, env(db, modal=e.modal, storage=e.storage))
    current = job(db)
    assert current.result_payload["result_image_key"] == KEY and len(e.modal.requests) == 1


def test_stop_during_wait_detaches_without_a_new_call(db):  # noqa: F811
    idm_job(db)
    stop = threading.Event()
    storage = FakeStorage()
    modal = FakeModal(storage, outcome="running", during_wait=stop.set)
    e = env(db, storage=storage, modal=modal)
    outcome = run(db, e, should_stop=stop.is_set)
    assert outcome.action == "handler_deferred_kept" and modal.wait_calls == 1
    assert len(modal.requests) == 1 and job(db).fallback_started_at is not None


def test_lease_and_visibility_are_renewed_while_waiting(db):  # noqa: F811
    from tests.test_virtual_fitting_request_worker import FakeSqs

    idm_job(db)
    e = env(db, modal_kwargs={"waits": 4})
    sqs = FakeSqs()
    run(db, e, sqs=sqs)
    # heartbeat 간격 0: 대기 사이마다 lease·Visibility 를 연장했다
    assert len(sqs.visibility_calls) >= 4 and job(db).result_payload["status"] == "COMPLETED"


def test_keepalive_failure_during_wait_is_not_completed(db):  # noqa: F811
    from tests.test_virtual_fitting_request_worker import FakeSqs

    idm_job(db)
    e = env(db, modal_kwargs={"outcome": "running"})
    outcome = run(db, e, sqs=FakeSqs(visibility_ok=False))
    assert not outcome.delete and len(e.modal.requests) <= 1
    assert job(db).result_payload is None


# --- S3 결과 확인·복구 ---


def test_s3_saved_but_db_record_failed_is_recovered_without_regeneration(db, monkeypatch):  # noqa: F811
    from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository

    idm_job(db)
    real = AsyncJobStateRepository.record_fallback_image
    calls = []

    def fail_once(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise FittingDatabaseError("injected after S3 put")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(AsyncJobStateRepository, "record_fallback_image", fail_once)
    e = env(db)
    assert run(db, e).action == "handler_error_kept"
    assert KEY in e.storage.objects and job(db).fallback_image_key is None
    expire_lease(db)
    run(db, env(db, modal=e.modal, storage=e.storage))
    assert job(db).result_payload["result_image_key"] == KEY and len(e.modal.requests) == 1


@pytest.mark.parametrize(
    "tamper, reason",
    [
        (
            lambda s, a: s.put(KEY, webp_bytes(), metadata={META_JOB: "999", META_ATTEMPT: a}),
            "metadata_mismatch",
        ),
        (
            lambda s, a: s.put(
                KEY, webp_bytes(), metadata={META_JOB: str(JOB), META_ATTEMPT: "other"}
            ),
            "metadata_mismatch",
        ),
        (
            lambda s, a: s.put(
                KEY, b"not an image", metadata={META_JOB: str(JOB), META_ATTEMPT: a}
            ),
            "decode_failed",
        ),
        (
            lambda s, a: s.put(
                KEY, webp_bytes(), "image/jpeg", {META_JOB: str(JOB), META_ATTEMPT: a}
            ),
            "content_type",
        ),
    ],
)
def test_recovery_rejects_objects_that_do_not_match_the_attempt(db, tamper, reason):  # noqa: F811
    idm_job(db)
    e = env(db, modal_kwargs={"outcome": "unknown"})
    run(db, e)
    current = job(db)
    tamper(e.storage, current.fallback_task_uuid)
    check = inspect_fallback_result(e.storage, current, upload_expires_seconds=UPLOAD_EXPIRES)
    assert (check.state, check.reason) == (ResultState.INVALID, reason)
    expire_lease(db)
    run(db, env(db, modal=e.modal, storage=e.storage))
    assert job(db).result_payload == model_failed() and len(e.modal.requests) == 1


def test_reported_success_with_a_different_body_is_failed(db):  # noqa: F811
    idm_job(db)
    storage = FakeStorage()
    modal = FakeModal(storage)
    modal.body = webp_bytes()
    original_upload = modal._upload

    def corrupt(request):
        original_upload(request)
        _, ct, meta, modified = storage.objects[KEY]
        storage.objects[KEY] = (webp_bytes(color=(1, 2, 3)), ct, meta, modified)

    modal._upload = corrupt
    run(db, env(db, storage=storage, modal=modal))
    assert job(db).result_payload == model_failed()


def test_transient_s3_errors_during_recovery_are_held(db):  # noqa: F811
    idm_job(db)
    e = env(db, modal_kwargs={"outcome": "unknown"})
    run(db, e)
    age_start(db, UPLOAD_EXPIRES + RECOVERY_MARGIN_SECONDS + 5)
    e.storage.head_faults.append(S3ObjectState.TRANSIENT_ERROR)
    expire_lease(db)
    assert run(db, env(db, modal=e.modal, storage=e.storage)).action == "handler_deferred_kept"
    assert job(db).result_payload is None


PRUNA_SYNC_KEYS = (
    f"virtual-fitting/results/{JOB}/pruna-sync.jpg",
    f"virtual-fitting/results/{JOB}/pruna-sync.webp",
)


@pytest.mark.parametrize("legacy", PRUNA_SYNC_KEYS)
def test_pruna_sync_results_are_not_looked_up_or_recorded(db, legacy):  # noqa: F811
    """IDM 복구는 idm-vton.webp 만 본다. 시작 뒤 저장된 Pruna 동기 결과만 있어도 쓰지 않는다(HEAD·GET 도 없음)."""
    idm_job(db)
    e = env(db, modal_kwargs={"outcome": "unknown"})
    run(db, e)
    e.storage.put(legacy, b"legacy", "image/jpeg" if legacy.endswith(".jpg") else "image/webp")
    expire_lease(db)
    assert run(db, env(db, modal=e.modal, storage=e.storage)).action == "handler_deferred_kept"
    current = job(db)
    assert current.fallback_image_key is None and current.result_payload is None
    # 업로드 기한이 지나면 기존 정책대로 실패 확정. Pruna 결과로 완료하지 않는다.
    age_start(db, UPLOAD_EXPIRES + RECOVERY_MARGIN_SECONDS + 5)
    expire_lease(db)
    run(db, env(db, modal=e.modal, storage=e.storage))
    current = job(db)
    assert current.fallback_image_key is None and current.result_payload == model_failed()
    assert legacy not in e.storage.heads and legacy not in e.storage.reads
    assert len(e.modal.requests) == 1  # 재호출 없음


def recovery_job(started_ago):
    return SimpleNamespace(
        fitting_job_id=JOB,
        fallback_task_uuid="attempt-1",
        fallback_started_at=datetime.now(UTC) - timedelta(seconds=started_ago),
    )


CLOSED = UPLOAD_EXPIRES + RECOVERY_MARGIN_SECONDS + 5


@pytest.mark.parametrize(
    ("setup", "started_ago", "state", "reason"),
    [
        ("valid", 60, ResultState.FOUND, "verified"),
        ("none", 60, ResultState.PENDING, "upload_window_open"),
        ("none", CLOSED, ResultState.ABSENT, "upload_window_closed"),
        (S3ObjectState.ACCESS_DENIED, CLOSED, ResultState.UNKNOWN, "head_access_denied"),
        (S3ObjectState.TRANSIENT_ERROR, CLOSED, ResultState.UNKNOWN, "head_transient_error"),
        ("other_attempt", CLOSED, ResultState.INVALID, "metadata_mismatch"),
        ("pruna_sync_only", 60, ResultState.PENDING, "upload_window_open"),
        ("pruna_sync_only", CLOSED, ResultState.ABSENT, "upload_window_closed"),
    ],
)
def test_inspect_fallback_result_checks_only_the_idm_key(setup, started_ago, state, reason):
    storage = FakeStorage()
    metadata = {META_JOB: str(JOB), META_ATTEMPT: "attempt-1"}
    if setup == "valid":
        storage.put(KEY, webp_bytes(), metadata=metadata)
    elif setup == "other_attempt":
        storage.put(KEY, webp_bytes(), metadata={**metadata, META_ATTEMPT: "attempt-0"})
    elif setup == "pruna_sync_only":
        for legacy in PRUNA_SYNC_KEYS:
            storage.put(legacy, webp_bytes(), "image/webp", metadata)
    elif isinstance(setup, S3ObjectState):
        storage.head_faults.append(setup)  # 403·5xx·timeout 은 없다는 근거가 아니다
    check = inspect_fallback_result(
        storage, recovery_job(started_ago), upload_expires_seconds=UPLOAD_EXPIRES
    )
    assert (check.state, check.reason, check.key) == (state, reason, KEY)
    assert storage.heads == [KEY]  # Pruna 동기 결과 Key 는 HEAD 하지 않는다
    assert set(storage.reads) <= {KEY}


# --- DB 1회 선점 ---


def test_claim_is_once_and_requires_the_lease(db):  # noqa: F811
    idm_job(db)
    state = db["state"]
    grant = state.acquire_lease(JOB, NextAction.RUN_FALLBACK, 60)
    first = state.claim_idm_vton_attempt(JOB, grant.token)
    second = state.claim_idm_vton_attempt(JOB, grant.token)
    assert first.applied and first.job.fallback_attempt_count == 1
    assert first.job.fallback_started_at is not None
    assert (second.applied, second.reason) == (False, ChangeReason.LIMIT_REACHED)
    state.release_lease(JOB, grant.token)
    other = state.claim_idm_vton_attempt(JOB, "not-the-token")
    assert other.reason is ChangeReason.LEASE_INVALID


def test_concurrent_workers_call_modal_once(db):  # noqa: F811
    idm_job(db)
    storage = FakeStorage()
    modal = FakeModal(storage)
    modal.gate = threading.Event()
    e1, e2 = env(db, storage=storage, modal=modal), env(db, storage=storage, modal=modal)
    results = []
    first = threading.Thread(target=lambda: results.append(run(db, e1)))
    first.start()
    for _ in range(100):
        if modal.requests:
            break
        threading.Event().wait(0.05)
    second = run(db, e2)  # 첫 Worker 가 lease 를 쥐고 Modal 결과를 기다리는 중
    modal.gate.set()
    first.join(10)
    assert second.action == "lease_lease_held" and not second.delete
    assert len(modal.requests) == 1 and job(db).result_payload["status"] == "COMPLETED"


# --- 후처리 정책 재사용 ---


def test_llm_failures_use_the_existing_default_text(db):  # noqa: F811
    idm_job(db)
    failing = FakeComments(
        comments=(FittingPostprocessError("x"), FittingPostprocessError("y")), titles=()
    )
    run(db, env(db, comments=failing))
    current = job(db)
    assert current.result_payload["llm_title"] == FALLBACK_TITLE
    assert current.result_payload["llm_comment"] == FALLBACK_COMMENT
    assert current.result_payload["result_image_key"] == KEY
    assert counts(db)[1] == 2  # 코멘트 2회 후 기본 문구(제목은 부르지 않음)


def test_upload_window_uses_the_recorded_start():
    from app.virtual_fitting.idm_fallback_handler import upload_window_closed

    start = datetime(2026, 11, 11, tzinfo=UTC)
    limit = timedelta(seconds=UPLOAD_EXPIRES + RECOVERY_MARGIN_SECONDS)
    assert (
        upload_window_closed(start, UPLOAD_EXPIRES, start + limit - timedelta(seconds=1)) is False
    )
    assert upload_window_closed(start, UPLOAD_EXPIRES, start + limit) is True
    assert upload_window_closed(None, UPLOAD_EXPIRES) is None
