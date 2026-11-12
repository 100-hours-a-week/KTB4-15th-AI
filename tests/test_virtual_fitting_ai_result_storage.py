import functools
import hashlib
import json
import logging
import threading
import time
from decimal import Decimal

import pytest

from app.clients.s3 import S3ImageStorage, S3ObjectState
from app.clients.sqs import SqsOutcome
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.postprocess_handler import PostprocessHandler
from app.virtual_fitting.postprocess_worker import PostprocessHandlers
from app.virtual_fitting.providers.runware_status import TaskStatus, TaskStatusKind, parse_status
from app.virtual_fitting.pruna_submitter import NextStep
from app.virtual_fitting.repositories.async_job_repository import AsyncJobStatus, NextAction
from app.virtual_fitting.repositories.pruna_limiter import TaskCostRepository
from app.virtual_fitting.result_image import ResultImageError, store_runware_result_image
from app.virtual_fitting.runware_webhook import (
    EventOutcome,
    RunwareWebhookService,
    WebhookFollowUpError,
    WebhookFollowUpNotReadyError,
    WebhookResultRefNotSavedError,
    parse_webhook,
)
from app.virtual_fitting.service import FALLBACK_COMMENT, FALLBACK_TITLE
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_postprocess_handler import FakeComments, FakeProducts
from tests.test_virtual_fitting_postprocess_handler import FakeStorage as HeadStorage
from tests.test_virtual_fitting_postprocess_worker import (
    ReadyPaths,
    item,
    postprocessing,
    processor,
)
from tests.test_virtual_fitting_pruna_submit import JOB, FakeClock, db, job, limiter  # noqa: F401
from tests.test_virtual_fitting_pruna_submit import _new_job as _submit_job
from tests.test_virtual_fitting_request_worker import FakeSqs
from tests.test_virtual_fitting_result_image import RGB, FakeS3
from tests.test_virtual_fitting_watchdog import (  # noqa: F401 (pytest fixture)
    FakeStatus,
    age,
    clear_lease,
    reasons,
    watchdog,
    wdb,
)
from tests.test_virtual_fitting_webhook import (  # noqa: F401 (pytest fixture)
    FakeQueue,
    api,
    post,
    queue_settings,
    task_costs,
)

URL = (
    "https://im.runware.ai/image/ws/2/ii/AI-RESULT.webp"
    "?token=tok-SECRET-789&X-Amz-Signature=sig-SECRET-012&Expires=1700000000"
)
OTHER_URL = "https://im.runware.ai/image/ws/2/ii/OTHER.webp"
IMAGE_UUID = "550e8400-e29b-41d4-a716-446655440000"
OTHER_UUID = "7d444840-9dc0-11d1-b245-5ffdce74fad2"
SECRETS = ("tok-SECRET-789", "sig-SECRET-012", "AI-RESULT.webp", "im.runware.ai/")


@pytest.fixture(autouse=True)
def _follow_up_queue(monkeypatch):
    """내부 큐 설정(가짜 URL, 실제 SQS 호출 없음). Webhook 처리는 설정이 없으면 기록 전에 멈춘다."""
    from app.config import settings

    monkeypatch.setattr(settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(
        settings,
        "AI_FITTING_RESPONSE_QUEUE_URL",
        "https://sqs.ap-northeast-2.amazonaws.com/123456789012/ai-fitting-response",
    )


def key_of(job_id=JOB):
    return f"virtual-fitting/results/{job_id}/pruna-async.webp"


# --- 준비 ---


def ai_submitted(db, job_id=JOB, accepted=True):  # noqa: F811
    """AI 저장 방식으로 준비·전송 시작한 작업(테스트 전용 선택). 실제 요청 코드는 기본값(직접 업로드)을 쓴다."""
    db["jobs"].create(_submit_job(job_id))
    token = db["state"].acquire_lease(job_id, NextAction.SUBMIT_PRUNA, 60).token
    task = f"a770f077-f413-47de-9dac-{job_id:012d}"
    prepared = db["state"].prepare_pruna_submission(
        job_id, token, task, key_of(job_id), ai_stores_result=True
    )
    assert prepared.applied and prepared.job.ai_stores_result
    assert limiter(db, interval=0).begin_submission(job_id, token, task).granted
    if accepted:
        assert db["state"].record_submission_accepted(task).applied
    return task


def hook(db, queue=None, ready=(NextStep.RECOVER_RESULT_REF,)):  # noqa: F811
    """9단계 처리(Webhook 과 같은 서비스). 기본은 Watchdog 참조 복구 경로가 준비된 배포."""
    return RunwareWebhookService(
        jobs=db["jobs"],
        state=db["state"],
        limiter=limiter(db, interval=0),
        costs=TaskCostRepository(db["pool"]),
        queue=queue if queue is not None else FakeQueue(),
        follow_up=ReadyPaths(*ready),
    )


def success(task, **refs):
    body = {"taskType": "imageInference", "taskUUID": task, "cost": 0.02, **refs}
    return json.dumps(body).encode()


def ai_postprocessing(db, job_id=JOB, **refs):  # noqa: F811
    """AI 작업이 Webhook 성공(받은 참조 그대로)까지 끝나 POSTPROCESSING / POSTPROCESS 인 상태. 실행 권한은 비운다."""
    task = ai_submitted(db, job_id)
    refs = refs or {"imageURL": URL, "imageUUID": IMAGE_UUID}
    hook(db).handle(parse_webhook(success(task, **refs)))
    clear_lease(db, job_id)
    return task


def refs(db, job_id=JOB):  # noqa: F811
    current = job(db, job_id)
    return current.runware_image_url, current.runware_image_uuid


class FakePresign:
    def generate_presigned_url(self, method, Params, ExpiresIn):
        return f"https://bucket.s3.ap-northeast-2.amazonaws.com/{Params['Key']}?X-Amz-Signature=GET"


class Downloads:
    """결과 다운로드 자리. 부른 URL 을 기록한다. body·error·during(실행 중 부수효과)을 정할 수 있다."""

    def __init__(self, body=RGB, error=None, during=None):
        self.body, self.error, self.during, self.calls = body, error, during, []

    def __call__(self, url):
        self.calls.append(url)
        if self.during is not None:
            self.during()
        if self.error is not None:
            raise self.error
        return self.body


def handler_env(db, downloads=None, comments=None, s3=None):  # noqa: F811
    s3 = s3 or FakeS3()
    storage = S3ImageStorage(s3, bucket="test-bucket", presign_client=FakePresign())
    downloads = downloads or Downloads()
    clock = FakeClock()
    handler = PostprocessHandler(
        jobs=db["jobs"],
        state=db["state"],
        products=FakeProducts(),
        storage=storage,
        comments=comments or FakeComments(),
        clock=clock,
        sleep=clock.sleep,
        result_url_expires_seconds=600,
        store_result=functools.partial(store_runware_result_image, download=downloads),
        remove_background_enabled=False,  # 배경 제거는 test_virtual_fitting_transparent_image.py
    )
    return handler, s3, downloads


def run(db, handler, *, sqs=None, follow_up=None, should_stop=lambda: False, **kwargs):  # noqa: F811
    task = job(db).task_uuid
    return processor(
        db,
        PostprocessHandlers(postprocess=handler),
        sqs=sqs,
        follow_up=follow_up,
        should_stop=should_stop,
        **kwargs,
    ).process(item(JOB, task))


def counts(db):  # noqa: F811
    current = job(db)
    return (current.s3_check_attempt_count, current.fallback_attempt_count)


# === 1·2. 저장 방식 구분: 기존 작업은 기존 경로, AI 작업은 새 함수 ===


def test_existing_and_default_jobs_keep_direct_upload_meaning(db):  # noqa: F811
    task = postprocessing(db)  # 기존 테스트 도우미 = 실제 요청과 같은 기본 준비
    assert job(db).ai_stores_result is False and task
    # 이미 준비된 작업은 저장 방식이 바뀌지 않는다(같은 UPDATE 로 한 번만).
    token = db["state"].acquire_lease(JOB + 1, NextAction.SUBMIT_PRUNA, 60)
    assert token.job is None  # 없는 작업
    db["jobs"].create(_submit_job(JOB + 1))
    lease = db["state"].acquire_lease(JOB + 1, NextAction.SUBMIT_PRUNA, 60).token
    first = db["state"].prepare_pruna_submission(JOB + 1, lease, OTHER_UUID, key_of(JOB + 1))
    again = db["state"].prepare_pruna_submission(
        JOB + 1, lease, OTHER_UUID, key_of(JOB + 1), ai_stores_result=True
    )
    assert first.applied and not again.applied and again.job.ai_stores_result is False


def test_direct_upload_job_uses_the_existing_s3_check_not_the_download(db):  # noqa: F811
    postprocessing(db)
    db["sql"](  # 기존 작업에도 1단계 이후 URL 이 저장될 수 있다. 그래도 경로를 바꾸지 않는다.
        "UPDATE virtual_fitting_async_jobs SET runware_image_url = %s WHERE fitting_job_id = %s",
        (URL, JOB),
    )
    clock = FakeClock()
    heads = HeadStorage(clock, states=[S3ObjectState.EXISTS])
    downloads = Downloads()
    handler = PostprocessHandler(
        jobs=db["jobs"],
        state=db["state"],
        products=FakeProducts(),
        storage=heads,
        comments=FakeComments(),
        clock=clock,
        sleep=clock.sleep,
        result_url_expires_seconds=600,
        store_result=functools.partial(store_runware_result_image, download=downloads),
        remove_background_enabled=False,  # 배경 제거는 test_virtual_fitting_transparent_image.py
    )
    outcome = run(db, handler)
    assert outcome.action == "handler_done" and downloads.calls == []
    assert len(heads.heads) == 1 and job(db).next_action is NextAction.PUBLISH_RESULT


def test_ai_job_downloads_stores_verifies_and_saves_the_result(db, caplog):  # noqa: F811
    ai_postprocessing(db)
    handler, s3, downloads = handler_env(db)
    with caplog.at_level(logging.DEBUG):
        outcome = run(db, handler)
    current = job(db)
    assert outcome.action == "handler_done" and outcome.delete is False  # 발행 경로 미준비면 남긴다
    assert downloads.calls == [URL]  # 저장된 원본 URL 그대로(token 포함)
    stored = s3.objects[key_of()]
    assert stored["Body"] == RGB and stored["ContentType"] == "image/webp"
    assert stored["Metadata"]["sha256"] == hashlib.sha256(RGB).hexdigest()
    assert current.verified_image_key == key_of() and current.image_verified_at is not None
    assert current.result_payload == {
        "fitting_job_id": JOB,
        "status": "COMPLETED",
        "result_image_key": key_of(),
        "llm_title": "차분한 데일리룩",
        "llm_comment": "좋은 코디입니다.\n잘 어울립니다.",
    }
    assert current.next_action is NextAction.PUBLISH_RESULT and counts(db) == (0, 0)
    assert_no_secret("\n".join(r.getMessage() for r in caplog.records))


def test_ai_result_deleted_only_when_the_publish_path_can_continue(db):  # noqa: F811
    ai_postprocessing(db)
    handler, _, _ = handler_env(db)
    outcome = run(db, handler, follow_up=ReadyPaths(NextStep.PUBLISH_RESULT))
    assert (outcome.action, outcome.delete) == ("handler_done", True)


def test_ai_llm_failures_keep_default_text_and_be_payload(db):  # noqa: F811
    ai_postprocessing(db)
    from app.virtual_fitting.exceptions import FittingPostprocessError

    failing = FakeComments(comments=(FittingPostprocessError("x"), FittingPostprocessError("y")))
    handler, _, _ = handler_env(db, comments=failing)
    run(db, handler)
    payload = job(db).result_payload
    assert (payload["llm_title"], payload["llm_comment"]) == (FALLBACK_TITLE, FALLBACK_COMMENT)
    assert set(payload) == {
        "fitting_job_id",
        "status",
        "result_image_key",
        "llm_title",
        "llm_comment",
    }
    assert failing.title_calls == []  # 코멘트 → 제목 순서·기본 문구 정책 그대로


# === 3. 유효 URL 저장 실패 → 503, 재수신 뒤 저장 ===


FAIL_URL_TRIGGER = """
CREATE FUNCTION fail_url() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.runware_image_url IS NOT NULL AND OLD.runware_image_url IS NULL THEN
        RAISE EXCEPTION 'injected failure';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER fail_url BEFORE UPDATE ON virtual_fitting_async_jobs
    FOR EACH ROW EXECUTE FUNCTION fail_url();
"""


def test_success_with_url_not_saved_is_503_and_redelivery_saves(db, queue_settings):  # noqa: F811
    task = ai_submitted(db)
    db["sql"](FAIL_URL_TRIGGER)
    queue = FakeQueue()
    with pytest.raises(WebhookResultRefNotSavedError) as error:
        hook(db, queue).handle(parse_webhook(success(task, imageURL=URL, imageUUID=IMAGE_UUID)))
    assert error.value.status_code == 503
    assert job(db).status is AsyncJobStatus.WAITING_PRUNA and refs(db) == (None, None)
    assert queue.sent == [] and task_costs(db) == []
    db["sql"]("DROP TRIGGER fail_url ON virtual_fitting_async_jobs")
    (result,) = hook(db, queue).handle(
        parse_webhook(success(task, imageURL=URL, imageUUID=IMAGE_UUID))
    )
    assert result.outcome is EventOutcome.APPLIED and refs(db) == (URL, IMAGE_UUID)
    assert [p["action"] for _, p in queue.sent] == ["POSTPROCESS"]


def test_late_url_not_saved_is_503_only_while_the_ai_job_still_needs_it(db, queue_settings):  # noqa: F811
    ai_task = ai_postprocessing(db, imageUUID=IMAGE_UUID)
    direct_task = postprocessing(db, JOB + 1)
    db["sql"](FAIL_URL_TRIGGER)
    with pytest.raises(WebhookResultRefNotSavedError):
        hook(db).handle(parse_webhook(success(ai_task, imageURL=URL, imageUUID=IMAGE_UUID)))
    assert refs(db) == (None, IMAGE_UUID)
    # 기존 방식 작업은 기존 정책(fill_failed 경고, 처리 계속)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET runware_image_uuid = %s WHERE fitting_job_id = %s",
        (IMAGE_UUID, JOB + 1),
    )
    (result,) = hook(db).handle(
        parse_webhook(success(direct_task, imageURL=URL, imageUUID=IMAGE_UUID))
    )
    assert result.outcome is EventOutcome.DUPLICATE
    # S3 저장이 이미 확인된 AI 작업은 URL 을 다시 요구하지 않는다(503 아님).
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET verified_image_key = %s, image_verified_at = now() "
        "WHERE fitting_job_id = %s",
        (key_of(), JOB),
    )
    (result,) = hook(db).handle(parse_webhook(success(ai_task, imageURL=URL, imageUUID=IMAGE_UUID)))
    assert result.outcome is EventOutcome.DUPLICATE


def test_conflict_or_unconfirmed_url_is_not_turned_into_503(db, queue_settings):  # noqa: F811
    task = ai_postprocessing(db, imageUUID=IMAGE_UUID)
    db["sql"](FAIL_URL_TRIGGER)  # 저장을 시도했다면 실패했을 것이다
    hook(db).handle(parse_webhook(success(task, imageURL=OTHER_URL, imageUUID=OTHER_UUID)))
    hook(db).handle(parse_webhook(success(task, imageURL=OTHER_URL)))  # 근거 부족(조회로 확인)
    hook(db).handle(parse_webhook(success(task, imageURL="https://evil.example/x.webp")))
    assert refs(db) == (None, IMAGE_UUID)


def test_router_answers_503_then_200_after_redelivery(api):  # noqa: F811
    task = ai_submitted(api["db"])
    api["db"]["sql"](FAIL_URL_TRIGGER)
    body = success(task, imageURL=URL, imageUUID=IMAGE_UUID)
    response = post(api, body)
    assert response.status_code == 503
    assert response.json()["code"] == "WEBHOOK_RESULT_REF_NOT_SAVED"
    assert_no_secret(response.text)
    api["db"]["sql"]("DROP TRIGGER fail_url ON virtual_fitting_async_jobs")
    assert post(api, body).status_code == 200
    assert refs(api["db"]) == (URL, IMAGE_UUID)


def test_uuid_only_success_is_2xx_only_with_a_live_recovery_path(api, monkeypatch):  # noqa: F811
    from types import SimpleNamespace

    from app.virtual_fitting import watchdog as watchdog_module

    task = ai_submitted(api["db"])
    body = success(task, imageUUID=IMAGE_UUID)
    # Watchdog 이 돌지 않는 배포: 기록은 남기되 완료로 답하지 않는다(기존 NOT_READY 503, Runware 재전송).
    monkeypatch.setattr(watchdog_module, "_registered", None)
    response = post(api, body)
    assert response.status_code == 503 and response.json()["code"] == "WEBHOOK_FOLLOW_UP_NOT_READY"
    monkeypatch.setattr(watchdog_module, "_registered", SimpleNamespace(alive=True))
    assert post(api, body).status_code == 200
    assert (
        api["boto"].bodies == []
    )  # URL 이 없으면 POSTPROCESS 를 보내지 않는다(Worker 가 할 일 없음)
    current = job(api["db"])
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert refs(api["db"]) == (None, IMAGE_UUID) and current.ai_stores_result


# === 4·5. Watchdog 의 같은 task 조회로 참조 확보 ===


def ai_waiting_ref(db, **refs_):  # noqa: F811
    task = ai_postprocessing(db, **refs_)
    age(db, "runware_accepted_at", 200)  # 접수 확인 뒤 조회 시작(3분) 이후
    return task


def status(task, url=URL, image_uuid=IMAGE_UUID):
    return TaskStatus(
        TaskStatusKind.SUCCEEDED, task, Decimal("0.02"), image_url=url, image_uuid=image_uuid
    )


def test_uuid_only_job_gets_its_url_from_the_watchdog_query_then_is_stored(wdb, queue_settings):  # noqa: F811
    task = ai_waiting_ref(wdb, imageUUID=IMAGE_UUID)
    handler, s3, downloads = handler_env(wdb)
    first = run(wdb, handler)
    assert (
        first.action == "handler_deferred_kept" and downloads.calls == []
    )  # URL 없다고 다운로드 가정 안 함
    w = watchdog(wdb, FakeStatus(status(task)))
    reports = w.run_once()
    assert refs(wdb) == (URL, IMAGE_UUID)
    actions = [a for r in reports for a in r.actions]
    assert "result_ref_recovered" in actions
    assert [p for _, p in w.queue.sent] == [
        {"fitting_job_id": JOB, "task_uuid": task, "action": "POSTPROCESS"}
    ]  # 큐에는 작업 참조만(URL·bytes 없음)
    assert task_costs(wdb) == [(task, Decimal("0.02"), job(wdb).pruna_submit_started_at.date())]
    second = run(wdb, handler)
    assert second.action == "handler_done" and downloads.calls == [URL]
    assert s3.objects[key_of()]["Body"] == RGB


def test_query_url_without_uuid_fills_after_confirmation_for_ai_jobs_only(wdb):  # noqa: F811
    task = ai_waiting_ref(wdb, imageUUID=IMAGE_UUID)
    watchdog(wdb, FakeStatus(status(task, image_uuid=None))).run_once()
    assert refs(wdb) == (URL, IMAGE_UUID)  # 같은 task 직접 조회 = 확인 근거


def test_both_refs_null_get_a_first_reference_from_a_confirmed_query(wdb):  # noqa: F811
    # 성공 알림의 URL 이 허용되지 않아 참조가 하나도 없는 AI 작업
    task = ai_waiting_ref(wdb, imageURL="http://im.runware.ai/x.webp")
    assert refs(wdb) == (None, None)
    watchdog(wdb, FakeStatus(status(task))).run_once()
    assert refs(wdb) == (URL, IMAGE_UUID)


def test_direct_job_never_uses_the_query_confirmation_rule(wdb, queue_settings):  # noqa: F811
    task = postprocessing(wdb)
    change = wdb["state"].fill_pruna_image_ref(
        task, image_url=URL, image_uuid=IMAGE_UUID, confirmed_by_query=True
    )
    assert not change.applied and refs(wdb) == (None, None)
    # Watchdog 은 기존 작업을 참조 복구 대상으로 보지 않는다(기존 점검 경로 그대로).
    age(wdb, "runware_accepted_at", 200)
    status_client = FakeStatus(status(task))
    watchdog(wdb, status_client).run_once()
    assert status_client.calls == [] and refs(wdb) == (None, None)


def test_external_body_cannot_claim_query_confirmation(wdb, queue_settings):  # noqa: F811
    task = ai_waiting_ref(wdb, imageUUID=IMAGE_UUID)
    body = success(task, imageURL=URL, from_status_query=True, fromStatusQuery=True)
    (event,) = parse_webhook(body)
    assert event.from_status_query is False
    hook(wdb).handle([event])
    assert refs(wdb) == (None, IMAGE_UUID)  # URL 만으로는 연결하지 않음(근거 부족)


def test_ambiguous_query_results_are_not_picked(wdb):  # noqa: F811
    task = ai_waiting_ref(wdb, imageUUID=IMAGE_UUID)
    body = json.dumps(
        {
            "data": [
                {"taskUUID": task, "status": "success", "imageURL": URL, "imageUUID": IMAGE_UUID},
                {
                    "taskUUID": task,
                    "status": "success",
                    "imageURL": OTHER_URL,
                    "imageUUID": OTHER_UUID,
                },
            ]
        }
    ).encode()
    parsed = parse_status(body, task)
    assert parsed.kind is TaskStatusKind.SUCCEEDED and parsed.image_url is None
    assert parsed.image_ref_invalid
    w = watchdog(wdb, FakeStatus(parsed))
    reports = w.run_once()
    assert refs(wdb) == (None, IMAGE_UUID)
    assert "result_ref_unrecovered" in [a for r in reports for a in r.actions]
    assert ("result_ref_missing", "postprocess", JOB) in reasons(w.sink)
    assert job(wdb).status is AsyncJobStatus.POSTPROCESSING  # 생성 실패로 바꾸지 않는다


@pytest.mark.parametrize(
    "kind", [TaskStatusKind.NOT_FOUND, TaskStatusKind.UNKNOWN, TaskStatusKind.PROCESSING]
)
def test_query_failure_or_not_found_is_not_a_generation_failure(wdb, kind):  # noqa: F811
    ai_waiting_ref(wdb, imageUUID=IMAGE_UUID)
    w = watchdog(wdb, FakeStatus(kind))
    w.run_once()
    current = job(wdb)
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert current.result_payload is None and counts(wdb) == (0, 0)
    assert ("result_ref_missing", "postprocess", JOB) in reasons(w.sink)
    assert w.queue.sent == []


# === 6. 다른 task·UUID 충돌·늦은 완료·fallback 경합 ===


def test_query_result_with_another_uuid_is_not_linked(wdb):  # noqa: F811
    task = ai_waiting_ref(wdb, imageUUID=IMAGE_UUID)
    watchdog(wdb, FakeStatus(status(task, url=OTHER_URL, image_uuid=OTHER_UUID))).run_once()
    assert refs(wdb) == (None, IMAGE_UUID)


def test_other_task_query_never_touches_this_job(wdb):  # noqa: F811
    ai_waiting_ref(wdb, imageUUID=IMAGE_UUID)
    other = ai_postprocessing(wdb, JOB + 1, imageUUID=OTHER_UUID)
    wdb["state"].fill_pruna_image_ref(
        other, image_url=URL, image_uuid=OTHER_UUID, confirmed_by_query=True
    )
    assert refs(wdb) == (None, IMAGE_UUID) and refs(wdb, JOB + 1) == (URL, OTHER_UUID)


@pytest.mark.parametrize(
    "arrange",
    [
        "status = 'COMPLETED', next_action = NULL",
        "status = 'FALLBACK_RUNNING', next_action = 'RUN_FALLBACK'",
        (
            "next_action = 'PUBLISH_RESULT', result_payload = jsonb_build_object("
            "'fitting_job_id', fitting_job_id, 'status', 'FAILED', 'error_code', 'X', "
            "'error_message', 'x')"
        ),
    ],
)
def test_confirmed_fill_respects_the_state_at_update_time(wdb, arrange):  # noqa: F811
    task = ai_postprocessing(wdb, imageUUID=IMAGE_UUID)
    wdb["sql"](f"UPDATE virtual_fitting_async_jobs SET {arrange} WHERE fitting_job_id = %s", (JOB,))
    change = wdb["state"].fill_pruna_image_ref(
        task, image_url=URL, image_uuid=IMAGE_UUID, confirmed_by_query=True
    )
    assert not change.applied and refs(wdb) == (None, IMAGE_UUID)


# === 7. DB 저장 뒤 내부 큐 발행 실패·누락 복구, DLQ 우회 방지 ===


def test_publish_failure_after_db_record_is_recovered_once_then_held(wdb, queue_settings):  # noqa: F811
    task = ai_submitted(wdb)
    queue = FakeQueue(SqsOutcome.FAILED)
    with pytest.raises(WebhookFollowUpError):
        hook(wdb, queue).handle(parse_webhook(success(task, imageURL=URL, imageUUID=IMAGE_UUID)))
    assert refs(wdb) == (URL, IMAGE_UUID) and job(wdb).status is AsyncJobStatus.POSTPROCESSING
    clear_lease(wdb)
    # 장기 미진행: AI 후처리 단계 시작(성공 전환)이 2시간 전. 판단 기준은 updated_at 이 아니라 단계 시작 기록이다.
    age(wdb, "runware_accepted_at", 7200)
    w = watchdog(wdb)
    w.run_once()
    assert [p["action"] for _, p in w.queue.sent] == ["POSTPROCESS"]  # DB 기록으로 한 번 다시 발행
    # 그 뒤에도 진행이 없으면(소비가 계속 실패해 DLQ 로 가는 중일 수 있음) 새 메시지를 또 만들지 않는다.
    for _ in range(3):
        wdb["sql"]("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
        wdb["sql"](
            "UPDATE virtual_fitting_watchdog_checks SET follow_up_published_at = "
            "now() - interval '1 hour'"
        )  # 재발행 간격(600초)이 지나도
        age(wdb, "updated_at", 0)  # lease·재수신으로 updated_at 이 바뀌어도(진척 아님)
        w.run_once()
    assert len(w.queue.sent) == 1
    assert ("postprocess_stalled", "postprocess", JOB) in reasons(w.sink)
    assert counts(wdb) == (0, 0) and job(wdb).next_action is NextAction.POSTPROCESS


def test_failed_republish_is_retried_because_it_was_not_consumed(wdb):  # noqa: F811
    ai_postprocessing(wdb)
    age(wdb, "runware_accepted_at", 601)  # AI 후처리 단계 시작 기준 장기 미진행
    w = watchdog(wdb)
    w.queue.outcomes = [SqsOutcome.FAILED]
    w.run_once()
    wdb["sql"]("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
    w.run_once()
    assert [p["action"] for _, p in w.queue.sent] == ["POSTPROCESS", "POSTPROCESS"]


# === 8·9. S3 성공 뒤 DB 실패·Worker 종료 → 기존 객체 재사용, 단계 생략 ===


class FailingVerify:
    """record_image_verified 를 처음 한 번 실패시킨다(S3 PUT 뒤 DB 기록 실패)."""

    def __init__(self, real):
        self.real, self.failed = real, False

    def __getattr__(self, name):
        attr = getattr(self.real, name)
        if name != "record_image_verified" or self.failed:
            return attr

        def fail(*args, **kwargs):
            self.failed = True
            raise FittingDatabaseError("injected")

        return fail


def test_s3_saved_but_db_record_failed_is_recovered_without_download(db):  # noqa: F811
    ai_postprocessing(db)
    s3 = FakeS3()
    first_handler, _, downloads = handler_env(db, s3=s3)
    first_handler._given = (db["jobs"], FailingVerify(db["state"]), *first_handler._given[2:])
    first = run(db, first_handler)
    assert first.action == "handler_error_kept" and job(db).verified_image_key is None
    assert key_of() in s3.objects and len(downloads.calls) == 1
    later, _, again = handler_env(db, s3=s3, downloads=Downloads(error=AssertionError("no")))
    second = run(db, later)
    assert second.action == "handler_done" and again.calls == []  # 만료됐을 URL 을 다시 받지 않음
    assert job(db).verified_image_key == key_of() and len(s3.puts()) == 1


def test_verified_or_saved_result_skips_download_and_upload(db):  # noqa: F811
    ai_postprocessing(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET verified_image_key = %s, image_verified_at = now() "
        "WHERE fitting_job_id = %s",
        (key_of(), JOB),
    )
    handler, s3, downloads = handler_env(db)
    assert run(db, handler).action == "handler_done"
    assert downloads.calls == [] and s3.calls == []  # LLM 은 확인된 Key 의 Presigned GET 만
    llm = FakeComments()
    handler2, s3_2, downloads2 = handler_env(db, comments=llm)
    run(db, handler2)  # 최종 결과가 있으면 LLM·저장을 다시 하지 않는다
    assert llm.comment_calls == [] and downloads2.calls == [] and s3_2.calls == []


# === 10. 중복 메시지·동시 Worker·Watchdog 경합 ===


def test_concurrent_workers_store_only_once(db):  # noqa: F811
    task = ai_postprocessing(db)
    gate, entered = threading.Event(), threading.Event()
    s3 = FakeS3()

    def slow():
        entered.set()
        gate.wait(5)

    handler, _, downloads = handler_env(db, s3=s3, downloads=Downloads(during=slow))
    results = {}
    first = threading.Thread(target=lambda: results.setdefault("a", run(db, handler)))
    first.start()
    assert entered.wait(5)
    second = processor(db, PostprocessHandlers(postprocess=handler)).process(
        item(JOB, task, message_id="m-2")
    )
    assert second.action == "lease_lease_held" and not second.delete
    # 실행 중(lease 보유)인 작업은 Watchdog 도 참조 복구·재발행하지 않는다.
    gate.set()
    first.join(5)
    assert results["a"].action == "handler_done" and len(downloads.calls) == 1
    assert len(s3.puts()) == 1


def test_watchdog_skips_a_leased_job(wdb):  # noqa: F811
    task = ai_waiting_ref(wdb, imageUUID=IMAGE_UUID)
    wdb["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60)
    status_client = FakeStatus(status(task))
    watchdog(wdb, status_client).run_once()
    assert status_client.calls == [] and refs(wdb) == (None, IMAGE_UUID)


# === 11·12. blocking 다운로드 중 연장, 실행 권한 상실·종료 뒤 늦은 완료 ===


def test_lease_and_visibility_keep_extending_during_a_blocking_download(db):  # noqa: F811
    ai_postprocessing(db)
    leases = []

    def blocking():
        for _ in range(4):
            time.sleep(0.06)
            leases.append(job(db).worker_lease_until)

    handler, _, _ = handler_env(db, downloads=Downloads(during=blocking))
    sqs = FakeSqs()
    outcome = run(db, handler, sqs=sqs, keepalive_seconds=0.03)
    assert outcome.action == "handler_done"
    assert len(sqs.visibility_calls) >= 3 and leases[0] < leases[-1]


def test_lost_lease_during_download_blocks_the_late_db_write(db):  # noqa: F811
    ai_postprocessing(db)
    s3 = FakeS3()

    def lose_lease():
        db["sql"](
            "UPDATE virtual_fitting_async_jobs SET worker_lease_token = 'someone-else', "
            "worker_lease_until = now() + interval '1 minute' WHERE fitting_job_id = %s",
            (JOB,),
        )

    handler, _, _ = handler_env(db, s3=s3, downloads=Downloads(during=lose_lease))
    outcome = run(db, handler)
    assert outcome.action in ("handler_deferred_kept", "keepalive_failed_kept")
    current = job(db)
    assert current.verified_image_key is None and current.result_payload is None
    assert key_of() in s3.objects  # 객체는 남고 다음 실행이 확인해 재사용한다
    clear_lease(db)
    later, _, again = handler_env(db, s3=s3, downloads=Downloads(error=AssertionError("no")))
    assert run(db, later).action == "handler_done" and again.calls == []


def test_stop_requested_during_download_records_nothing(db):  # noqa: F811
    ai_postprocessing(db)
    stop = threading.Event()
    handler, s3, downloads = handler_env(db, downloads=Downloads(during=stop.set))
    outcome = run(db, handler, should_stop=stop.is_set)
    assert outcome.action == "handler_deferred_kept" and len(downloads.calls) == 1
    assert job(db).verified_image_key is None and key_of() in s3.objects


# === 13. 다운로드·S3·DB 오류는 재생성·fallback 아님 ===


@pytest.mark.parametrize(
    "error",
    [
        ResultImageError("timeout"),
        ResultImageError("http_error", http_status=403),  # URL 만료·접근 거부
        ResultImageError("not_webp"),
        ResultImageError("corrupt"),
        ResultImageError("s3_put_failed", http_status=403),
        ResultImageError("existing_object_conflict"),
    ],
    ids=lambda e: e.reason,
)
def test_storage_errors_never_fall_back_or_fail_the_job(db, error, caplog):  # noqa: F811
    ai_postprocessing(db)
    handler, s3, _ = handler_env(db, downloads=Downloads(error=error))
    with caplog.at_level(logging.DEBUG):
        for _ in range(3):  # 반복돼도(기존 후처리 재시도 소진 흉내)
            outcome = run(db, handler)
            assert outcome.action == "handler_deferred_kept" and not outcome.delete
            clear_lease(db)
    current = job(db)
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert current.result_payload is None and counts(db) == (0, 0) and s3.puts() == []
    assert_no_secret("\n".join(r.getMessage() for r in caplog.records))


def test_missing_url_in_the_worker_is_deferred_not_failed(db):  # noqa: F811
    ai_postprocessing(db, imageUUID=IMAGE_UUID)
    handler, s3, downloads = handler_env(db)
    outcome = run(db, handler, follow_up=ReadyPaths(*NextStep))
    assert outcome.action == "handler_deferred_kept" and not outcome.delete
    assert downloads.calls == [] and s3.calls == [] and counts(db) == (0, 0)


def test_watchdog_never_uses_s3_absence_for_ai_jobs(wdb):  # noqa: F811
    ai_postprocessing(wdb)
    wdb["sql"](  # 기존 작업이면 복구용 HEAD → fallback 판단까지 가는 기록
        "UPDATE virtual_fitting_async_jobs SET s3_check_attempt_count = 2 WHERE fitting_job_id = %s",
        (JOB,),
    )
    age(wdb, "runware_accepted_at", 601)  # AI 후처리 단계 시작 기준 장기 미진행
    w = watchdog(wdb)
    w.run_once()
    assert w._storage.heads == []  # 객체 유무로 판단하지 않는다
    assert job(wdb).next_action is NextAction.POSTPROCESS and counts(wdb) == (2, 0)
    assert [p["action"] for _, p in w.queue.sent] == ["POSTPROCESS"]


# === 16. URL·token·서명값 비노출 ===


def assert_no_secret(text):
    for secret in SECRETS:
        assert secret not in text, secret


def test_queue_messages_and_logs_never_carry_the_url(wdb, queue_settings, caplog):  # noqa: F811
    with caplog.at_level(logging.DEBUG):
        task = ai_waiting_ref(wdb, imageUUID=IMAGE_UUID)
        w = watchdog(wdb, FakeStatus(status(task)))
        w.run_once()
        handler, _, _ = handler_env(wdb)
        run(wdb, handler)
    sent = json.dumps([payload for _, payload in w.queue.sent])
    assert_no_secret(sent)
    assert_no_secret("\n".join(r.getMessage() for r in caplog.records))
    assert_no_secret(repr(job(wdb)))
    assert job(wdb).verified_image_key == key_of()


def test_uuid_only_success_without_a_recovery_path_is_not_completed(db):  # noqa: F811
    task = ai_submitted(db)
    queue = FakeQueue()
    with pytest.raises(WebhookFollowUpNotReadyError) as error:
        hook(db, queue, ready=()).handle(parse_webhook(success(task, imageUUID=IMAGE_UUID)))
    assert error.value.pending == [(task, "recover_result_ref")]
    assert refs(db) == (None, IMAGE_UUID) and job(db).status is AsyncJobStatus.POSTPROCESSING
    assert queue.sent == []  # 기록은 남고(재전송은 멱등), POSTPROCESS 는 URL 이 생긴 뒤에


def test_uuid_only_success_publishes_nothing_until_the_url_is_recovered(wdb):  # noqa: F811
    task = ai_submitted(wdb)
    queue = FakeQueue()
    (result,) = hook(wdb, queue).handle(parse_webhook(success(task, imageUUID=IMAGE_UUID)))
    assert result.outcome is EventOutcome.APPLIED and result.published is None and queue.sent == []
    assert len(task_costs(wdb)) == 1  # 비용·자리 반환은 기존대로


# === 실제 요청 경로(4단계 전환): 처음 준비하는 작업은 AI 저장 방식 ===


def test_real_submission_prepares_new_jobs_for_ai_storage(db):  # noqa: F811
    from tests.test_virtual_fitting_pruna_submit import FakeProvider, build, valid_message

    provider = FakeProvider()
    processor_, storage = build(db, provider)
    assert processor_.process(valid_message()).action == "submitted"
    (call,) = provider.calls
    assert call["upload"] is None and call["ai_stores_result"] is True  # DB 에 확정된 방식을 명시
    assert [kind for kind, _ in storage.calls] == ["GET"]  # 결과 PUT URL 발급 없음
    assert job(db).ai_stores_result is True
