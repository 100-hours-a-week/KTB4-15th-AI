"""9단계: Runware Webhook 수신 (Router·Service·Repository·SQS 연결).

실제 Runware·AWS 는 부르지 않는다. 본문은 공식 문서 예시를 바탕으로 만든 Mock 이다(실제 계정 본문 미확인).
DB 는 격리 PostgreSQL(TEST_DATABASE_URL), SQS 는 가짜 boto3 client(실제 SqsQueueClient 경유) 또는 가짜 큐 객체다.
"""

import json
import logging
import threading
import time
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.clients.sqs import FittingQueue, SqsOperation, SqsOutcome, SqsResult
from app.config import settings
from app.config.logging_config import RedactQuerySecretsFilter, redact_query_secrets
from app.virtual_fitting import webhook_router
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.repositories.async_job_repository import AsyncJobStatus, NextAction
from app.virtual_fitting.repositories.pruna_limiter import TaskCostRepository
from app.virtual_fitting.repositories.usage_repository import UsageRepository
from app.virtual_fitting.runware_webhook import (
    EventKind,
    EventOutcome,
    RunwareWebhookService,
    WebhookFollowUpError,
    WebhookFollowUpNotReadyError,
    WebhookPayloadError,
    WebhookUnknownTaskError,
    parse_webhook,
    webhook_url_matches_token,
)
from tests.test_virtual_fitting_async_job_state import (  # noqa: F401 (pytest fixture)
    lock_waiters,
    pg,
)
from tests.test_virtual_fitting_pruna_submit import (  # noqa: F401 (pytest fixture)
    JOB,
    db,
    job,
    limiter,
    ready_job,
)

TOKEN = "test-webhook-token-123"
REGION = "ap-northeast-2"
QUEUE_URL = f"https://sqs.{REGION}.amazonaws.com/123456789012/ai-fitting-response"
OTHER_TASK = "a770f077-f413-47de-9dac-999999999999"
IMAGE_URL = "https://im.runware.ai/image/os/a14d18/ws/2/ii/SECRET-IMAGE.jpg"


def success_body(task, cost=0.023, **extra):
    body = {
        "taskType": "imageInference",
        "taskUUID": task,
        "imageUUID": "550e8400-e29b-41d4-a716-446655440000",
        "imageURL": IMAGE_URL,
        **extra,
    }
    if cost is not None:
        body["cost"] = cost
    return json.dumps(body).encode()


def error_body(task, code="timeoutProvider", **extra):
    item = {
        "taskUUID": task,
        "code": code,
        "status": "error",
        "message": "The external provider did not respond within the timeout window.",
        **extra,
    }
    return json.dumps({"errors": [item]}).encode()


# --- 준비 ---


def submitted(db, job_id=JOB, accepted=True):  # noqa: F811
    """task_uuid 저장·공유 자리 확보·호출 시작 기록까지 마친 작업(accepted 면 WAITING_PRUNA)."""
    token, task = ready_job(db, job_id)
    assert limiter(db, interval=0).begin_submission(job_id, token, task).granted
    if accepted:
        assert db["state"].record_submission_accepted(task).applied
    return task


class FakeQueue:
    """SqsQueueClient.send_json 자리. outcomes 를 차례로 돌려준다(없으면 SUCCEEDED)."""

    def __init__(self, *outcomes):
        self.outcomes, self.sent = list(outcomes), []

    def send_json(self, queue, payload):
        self.sent.append((queue, dict(payload)))
        outcome = self.outcomes.pop(0) if self.outcomes else SqsOutcome.SUCCEEDED
        return SqsResult(
            SqsOperation.SEND, queue, outcome, "ok" if outcome is SqsOutcome.SUCCEEDED else "x"
        )


class Flaky:
    """감싼 객체의 method 를 처음 n 번 실패시킨다(부분 실패 흉내)."""

    def __init__(self, real, method, failures=1):
        self.real, self.method, self.failures = real, method, failures

    def __getattr__(self, name):
        attr = getattr(self.real, name)
        if name != self.method:
            return attr

        def call(*args, **kwargs):
            if self.failures > 0:
                self.failures -= 1
                raise FittingDatabaseError("injected")
            return attr(*args, **kwargs)

        return call


@pytest.fixture
def queue_settings(monkeypatch):
    monkeypatch.setattr(settings, "AWS_REGION", REGION)
    monkeypatch.setattr(settings, "AI_FITTING_RESPONSE_QUEUE_URL", QUEUE_URL)


def service(db, queue=None, *, state=None, costs=None, limiter_=None):  # noqa: F811
    return RunwareWebhookService(
        jobs=db["jobs"],
        state=state or db["state"],
        limiter=limiter_ or limiter(db, interval=0),
        costs=costs or TaskCostRepository(db["pool"]),
        queue=queue if queue is not None else FakeQueue(),
    )


def in_flight(db):  # noqa: F811
    return db["sql"]("SELECT fitting_job_id, task_uuid::text FROM virtual_fitting_pruna_in_flight")


def task_costs(db):  # noqa: F811
    return db["sql"]("SELECT task_uuid::text, cost, usage_date FROM virtual_fitting_task_costs")


def recorded_total(db, day):  # noqa: F811
    rows = db["sql"]("SELECT recorded_amount FROM usage_amount WHERE usage_date = %s", (day,))
    return rows[0][0] if rows else None


START_DAY = datetime.now(UTC).date()


# --- 본문 해석 ---


def test_official_single_success_object_is_supported():
    (event,) = parse_webhook(success_body("A770F077-F413-47DE-9DAC-BE0B26A35DA6"))
    assert event.kind is EventKind.SUCCEEDED
    assert event.task_uuid == "a770f077-f413-47de-9dac-be0b26a35da6"
    assert event.cost == Decimal("0.023")  # float 를 거치지 않는다


def test_errors_array_and_data_array_are_supported():
    (failed,) = parse_webhook(error_body(OTHER_TASK))
    assert (failed.kind, failed.error_code, failed.recoverable) == (
        EventKind.FAILED,
        "timeoutProvider",
        True,
    )
    body = json.dumps(
        {
            "data": [
                json.loads(success_body(OTHER_TASK)),
                {"taskType": "imageInference", "taskUUID": OTHER_TASK, "status": "processing"},
            ],
            "errors": [{"taskUUID": OTHER_TASK, "code": "invalidModel"}],
        }
    ).encode()
    kinds = [event.kind for event in parse_webhook(body)]
    assert kinds == [EventKind.SUCCEEDED, EventKind.PROCESSING, EventKind.FAILED]


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (b"not json", "invalid_json"),
        (b"[]", "not_object"),
        (b'{"data": {}}', "unsupported_shape"),
        (b'{"errors": []}', "empty"),
        (b'{"errors": ["x"]}', "invalid_item"),
        (success_body(OTHER_TASK, taskType="videoInference"), "unsupported_task_type"),
        (
            json.dumps({"taskUUID": OTHER_TASK, "imageURL": IMAGE_URL}).encode(),
            "unsupported_task_type",
        ),
        (success_body("not-a-uuid"), "invalid_task_uuid"),
        (
            json.dumps(
                {"taskType": "imageInference", "taskUUID": 123, "imageURL": IMAGE_URL}
            ).encode(),
            "invalid_task_uuid",
        ),
        (
            json.dumps({"taskType": "imageInference", "taskUUID": OTHER_TASK}).encode(),
            "unclear_completion",
        ),
        (success_body(OTHER_TASK, status="error"), "unsupported_status"),
        (success_body(OTHER_TASK, status="queued"), "unsupported_status"),
        (json.dumps({"errors": [{"taskUUID": OTHER_TASK}]}).encode(), "invalid_error_code"),
        (json.dumps({"errors": [{"code": "timeoutProvider"}]}).encode(), "invalid_task_uuid"),
        (error_body(OTHER_TASK, taskType="videoInference"), "unsupported_task_type"),
        (b"{" + b" " * (300 * 1024) + b"}", "body_too_large"),
    ],
)
def test_unsupported_bodies_are_rejected(body, reason):
    with pytest.raises(WebhookPayloadError) as error:
        parse_webhook(body)
    assert error.value.reason == reason


@pytest.mark.parametrize("bad", ['"0.02"', "-0.01", "NaN", "Infinity", "true", "{}"])
def test_invalid_cost_is_not_a_cost(bad):
    body = success_body(OTHER_TASK, cost=None)[:-1] + f', "cost": {bad}}}'.encode()
    (event,) = parse_webhook(body)
    assert event.cost is None and event.cost_invalid


def test_missing_cost_is_not_zero():
    (event,) = parse_webhook(success_body(OTHER_TASK, cost=None))
    assert event.cost is None and not event.cost_invalid
    (event,) = parse_webhook(success_body(OTHER_TASK, cost=0))
    assert event.cost == Decimal(0)  # 명시적인 0 은 비용이다


def test_only_confirmed_codes_fall_back():
    for code in ("providerRateLimitExceeded", "serviceOverloaded", "invalidApiKey", "somethingNew"):
        (event,) = parse_webhook(error_body(OTHER_TASK, code=code, retryAfter=5))
        assert not event.recoverable


# --- 성공·실패 처리 (Service + 격리 DB) ---


def test_success_moves_to_postprocess_records_cost_releases_slot_and_publishes(db, queue_settings):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    (result,) = service(db, queue).handle(parse_webhook(success_body(task)))
    current = job(db)
    assert result.outcome is EventOutcome.APPLIED and result.published is NextAction.POSTPROCESS
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    # imageURL 은 저장하지 않는다. 이미지 확인은 async_image_key 로 후처리 Worker 가 한다.
    assert current.async_image_key == f"virtual-fitting/results/{JOB}/pruna-async.jpg"
    assert current.verified_image_key is None and current.result_image_key is None
    assert queue.sent == [
        (
            FittingQueue.AI_FITTING_RESPONSE,
            {"fitting_job_id": JOB, "task_uuid": task, "action": "POSTPROCESS"},
        )
    ]
    assert task_costs(db) == [(task, Decimal("0.023"), START_DAY)]
    assert recorded_total(db, START_DAY) == Decimal("0.023")
    assert in_flight(db) == []


def test_confirmed_recoverable_failure_queues_fallback(db, queue_settings):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    (result,) = service(db, queue).handle(parse_webhook(error_body(task)))
    current = job(db)
    assert (current.status, current.next_action) == (
        AsyncJobStatus.FALLBACK_RUNNING,
        NextAction.RUN_FALLBACK,
    )
    assert current.fallback_attempt_count == 0  # 실제 fallback 은 하지 않았다(13단계)
    assert (
        queue.sent[0][1]["action"] == "RUN_FALLBACK" and result.published is NextAction.RUN_FALLBACK
    )
    assert in_flight(db) == []


@pytest.mark.parametrize(
    "code", ["invalidReferenceImage", "providerRateLimitExceeded", "unknownNewCode"]
)
def test_other_failures_save_a_v2_failure_result_without_fallback(db, queue_settings, code):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    # 결과는 저장하지만 결과 발행 경로(14단계)가 없어 완료로 응답하지 않는다(503).
    with pytest.raises(WebhookFollowUpNotReadyError) as not_ready:
        service(db, queue).handle(parse_webhook(error_body(task, code=code)))
    (result,) = not_ready.value.results
    assert not_ready.value.pending == [(task, "publish_result")]
    current = job(db)
    assert current.next_action is NextAction.PUBLISH_RESULT
    assert current.status is AsyncJobStatus.WAITING_PRUNA  # FAILED 확정은 결과 발행 뒤(14단계)
    assert current.result_payload == {
        "fitting_job_id": JOB,
        "status": "FAILED",
        "error_code": "FITTING_MODEL_FAILED",
        "error_message": "가상 피팅 이미지 생성에 실패했습니다.",
    }
    assert queue.sent == [] and result.published is None  # PUBLISH_RESULT 는 이 큐로 보내지 않는다
    assert in_flight(db) == []


def test_processing_changes_nothing(db, queue_settings):  # noqa: F811
    task = submitted(db)
    body = json.dumps(
        {"data": [{"taskType": "imageInference", "taskUUID": task, "status": "processing"}]}
    ).encode()
    queue = FakeQueue()
    (result,) = service(db, queue).handle(parse_webhook(body))
    assert result.outcome is EventOutcome.PROCESSING
    assert job(db).status is AsyncJobStatus.WAITING_PRUNA
    assert in_flight(db) == [(JOB, task)] and queue.sent == [] and task_costs(db) == []


def test_unregistered_task_or_no_start_record_has_no_side_effects(db, queue_settings):  # noqa: F811
    queue = FakeQueue()
    with pytest.raises(WebhookUnknownTaskError):
        service(db, queue).handle(parse_webhook(success_body(OTHER_TASK)))
    _, task = ready_job(db, 7)  # task 는 저장했지만 호출 시작 기록이 없다
    with pytest.raises(WebhookUnknownTaskError):
        service(db, queue).handle(parse_webhook(success_body(task)))
    assert (
        job(db, 7).status is AsyncJobStatus.RECEIVED
        and job(db, 7).next_action is NextAction.SUBMIT_PRUNA
    )
    assert queue.sent == [] and task_costs(db) == []


def test_one_unknown_task_in_a_batch_rejects_the_whole_batch_before_side_effects(
    db,  # noqa: F811
    queue_settings,
):
    task = submitted(db)
    body = json.dumps(
        {"data": [json.loads(success_body(task)), json.loads(success_body(OTHER_TASK))]}
    ).encode()
    with pytest.raises(WebhookUnknownTaskError):
        service(db).handle(parse_webhook(body))
    assert job(db).status is AsyncJobStatus.WAITING_PRUNA and task_costs(db) == []


# --- 선도착·순서 ---


def test_webhook_before_the_accept_response_and_the_late_accept_does_not_revert(db, queue_settings):  # noqa: F811
    task = submitted(db, accepted=False)
    assert job(db).status is AsyncJobStatus.RECEIVED
    service(db).handle(parse_webhook(success_body(task)))
    assert job(db).status is AsyncJobStatus.POSTPROCESSING
    late = db["state"].record_submission_accepted(task)  # 8단계 접수 응답이 뒤늦게 기록된다
    assert (late.job.status, late.job.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert late.job.runware_accepted_at is not None


def test_failure_before_the_accept_response_keeps_the_failure_result(db, queue_settings):  # noqa: F811
    task = submitted(db, accepted=False)
    with pytest.raises(WebhookFollowUpNotReadyError):
        service(db).handle(parse_webhook(error_body(task, code="invalidReferenceImage")))
    late = db["state"].record_submission_accepted(task)
    # 수정 전에는 RECEIVED 라는 이유로 WAITING_PRUNA / next_action=NULL 이 되어 PUBLISH_RESULT 가 지워졌다.
    assert (late.job.status, late.job.next_action) == (
        AsyncJobStatus.RECEIVED,
        NextAction.PUBLISH_RESULT,
    )
    assert late.job.result_payload["status"] == "FAILED"


def test_recoverable_failure_before_accept_is_not_reverted(db, queue_settings):  # noqa: F811
    task = submitted(db, accepted=False)
    service(db).handle(parse_webhook(error_body(task)))
    late = db["state"].record_submission_accepted(task)
    assert late.job.status is AsyncJobStatus.FALLBACK_RUNNING


def test_success_then_failure_keeps_postprocess(db, queue_settings):  # noqa: F811
    task = submitted(db)
    service(db).handle(parse_webhook(success_body(task, cost=None)))
    queue = FakeQueue()
    (result,) = service(db, queue).handle(parse_webhook(error_body(task, cost=0.023)))
    assert result.outcome is EventOutcome.IGNORED and queue.sent == []
    assert job(db).status is AsyncJobStatus.POSTPROCESSING
    assert task_costs(db) == [(task, Decimal("0.023"), START_DAY)]  # 늦은 알림의 비용은 반영


def test_failure_then_success_keeps_fallback(db, queue_settings):  # noqa: F811
    task = submitted(db)
    service(db).handle(parse_webhook(error_body(task)))
    queue = FakeQueue()
    (result,) = service(db, queue).handle(parse_webhook(success_body(task)))
    assert result.outcome is EventOutcome.IGNORED and queue.sent == []
    assert job(db).status is AsyncJobStatus.FALLBACK_RUNNING


def test_late_webhook_after_the_final_result_preserves_it(db, queue_settings):  # noqa: F811
    task = submitted(db)
    final = {
        "fitting_job_id": JOB,
        "status": "COMPLETED",
        "result_image_key": "virtual-fitting/results/123/final.jpg",
        "llm_title": "t",
        "llm_comment": "c",
    }
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET status = 'COMPLETED', next_action = NULL, "
        "result_message_status = 'PUBLISHED', result_payload = %s::jsonb, "
        "result_image_key = 'virtual-fitting/results/123/final.jpg' WHERE fitting_job_id = %s",
        (json.dumps(final), JOB),
    )
    queue = FakeQueue()
    for body in (success_body(task), error_body(task)):
        (result,) = service(db, queue).handle(parse_webhook(body))
        assert result.outcome is EventOutcome.IGNORED
    current = job(db)
    assert current.status is AsyncJobStatus.COMPLETED and current.result_payload == final
    assert current.result_image_key == "virtual-fitting/results/123/final.jpg"
    assert queue.sent == []
    # 종료된 작업이어도 실제 비용과 진행 중 자리는 정리한다.
    assert len(task_costs(db)) == 1 and in_flight(db) == []


# --- 중복·동시 수신 ---


def test_same_webhook_three_times_changes_state_cost_and_slot_once(db, queue_settings):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    outcomes = [
        service(db, queue).handle(parse_webhook(success_body(task)))[0].outcome for _ in range(3)
    ]
    assert outcomes == [EventOutcome.APPLIED, EventOutcome.DUPLICATE, EventOutcome.DUPLICATE]
    assert len(task_costs(db)) == 1 and recorded_total(db, START_DAY) == Decimal("0.023")
    assert in_flight(db) == []
    # 후속 메시지는 상태가 그대로면 다시 보낸다(중복 발행 허용, 후처리 Worker 가 DB 상태·lease 로 막는다).
    assert [payload["action"] for _, payload in queue.sent] == ["POSTPROCESS"] * 3


def test_concurrent_identical_webhooks(db, queue_settings):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    results, errors = [], []
    barrier = threading.Barrier(5)

    def run():
        try:
            barrier.wait()
            results.extend(service(db, queue).handle(parse_webhook(success_body(task))))
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert not errors, errors
    assert sum(r.outcome is EventOutcome.APPLIED for r in results) == 1
    assert len(task_costs(db)) == 1 and recorded_total(db, START_DAY) == Decimal("0.023")
    assert in_flight(db) == [] and job(db).status is AsyncJobStatus.POSTPROCESSING


def test_concurrent_success_and_failure_pick_one_path(db, queue_settings):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    bodies = [success_body(task), error_body(task)] * 3
    barrier = threading.Barrier(len(bodies))
    errors = []

    def run(body):
        try:
            barrier.wait()
            service(db, queue).handle(parse_webhook(body))
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run, args=(b,)) for b in bodies]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert not errors, errors
    status = job(db).status
    assert status in (AsyncJobStatus.POSTPROCESSING, AsyncJobStatus.FALLBACK_RUNNING)
    expected = "POSTPROCESS" if status is AsyncJobStatus.POSTPROCESSING else "RUN_FALLBACK"
    assert {payload["action"] for _, payload in queue.sent} == {expected}


# --- 비용 ---


def test_cost_missing_first_and_sent_later_is_recorded_without_reverting(db, queue_settings):  # noqa: F811
    task = submitted(db)
    service(db).handle(parse_webhook(success_body(task, cost=None)))
    assert task_costs(db) == []
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET verified_image_key = 'virtual-fitting/results/123/pruna-async.jpg' WHERE fitting_job_id = %s",
        (JOB,),
    )
    service(db).handle(parse_webhook(success_body(task, cost=0.031)))
    assert task_costs(db) == [(task, Decimal("0.031"), START_DAY)]
    assert (
        job(db).status is AsyncJobStatus.POSTPROCESSING and job(db).verified_image_key is not None
    )


def test_invalid_cost_is_not_recorded_but_the_event_is_processed(db, queue_settings):  # noqa: F811
    task = submitted(db)
    body = success_body(task, cost=None)[:-1] + b', "cost": -1}'
    service(db).handle(parse_webhook(body))
    assert task_costs(db) == [] and recorded_total(db, START_DAY) is None
    assert job(db).status is AsyncJobStatus.POSTPROCESSING


def test_cost_date_is_the_submit_start_date_and_does_not_move_with_late_delivery(
    db,  # noqa: F811
    queue_settings,
):
    task = submitted(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET pruna_submit_started_at = '2026-10-16 23:59:58+00' WHERE fitting_job_id = %s",
        (JOB,),
    )
    service(db).handle(parse_webhook(success_body(task)))  # 실제 도착은 다음 날(오늘)
    service(db).handle(parse_webhook(success_body(task, cost=0.5)))  # 더 늦은 중복·다른 금액
    assert task_costs(db) == [(task, Decimal("0.023"), date(2026, 10, 16))]
    assert recorded_total(db, date(2026, 10, 16)) == Decimal("0.023")
    assert UsageRepository(db["pool"]).get(START_DAY).amount == 0


def test_cost_keeps_the_recorded_and_reconciled_split(db, queue_settings):  # noqa: F811
    task = submitted(db)
    UsageRepository(db["pool"]).reconcile(START_DAY, Decimal("0.023"))  # 보정이 먼저 이 비용을 포함
    service(db).handle(parse_webhook(success_body(task)))
    assert UsageRepository(db["pool"]).get(START_DAY).amount == Decimal("0.023")  # 다시 더하지 않음


# --- 자리 ---


def test_slot_of_another_attempt_is_not_released(db, queue_settings):  # noqa: F811
    task = submitted(db)
    # 같은 작업 번호의 자리를 다른 시도(task)가 쥐고 있다(예: 이후 시도·fallback 예약).
    db["sql"](
        "UPDATE virtual_fitting_pruna_in_flight SET task_uuid = %s WHERE fitting_job_id = %s",
        (OTHER_TASK, JOB),
    )
    service(db).handle(parse_webhook(success_body(task)))
    assert in_flight(db) == [(JOB, OTHER_TASK)]


def test_slot_is_released_once(db, queue_settings):  # noqa: F811
    task = submitted(db)
    shared = limiter(db, interval=0)
    service(db, limiter_=shared).handle(parse_webhook(success_body(task)))
    assert shared.release_in_flight(JOB, task_uuid=task) is False


# --- 부분 실패와 재수신 ---


@pytest.mark.parametrize("outcome", [SqsOutcome.FAILED, SqsOutcome.UNKNOWN])
def test_sqs_failure_after_db_is_retried_on_redelivery(db, queue_settings, outcome):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue(outcome)
    with pytest.raises(WebhookFollowUpError):
        service(db, queue).handle(parse_webhook(success_body(task)))
    assert job(db).status is AsyncJobStatus.POSTPROCESSING  # 기록은 남았다
    assert len(task_costs(db)) == 1 and in_flight(db) == []
    (result,) = service(db, queue).handle(parse_webhook(success_body(task)))  # 재수신
    assert result.outcome is EventOutcome.DUPLICATE and result.published is NextAction.POSTPROCESS
    assert len(queue.sent) == 2 and len(task_costs(db)) == 1


def test_cost_failure_after_state_change_is_completed_on_redelivery(db, queue_settings):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    costs = Flaky(TaskCostRepository(db["pool"]), "record")
    with pytest.raises(FittingDatabaseError):
        service(db, queue, costs=costs).handle(parse_webhook(success_body(task)))
    assert job(db).status is AsyncJobStatus.POSTPROCESSING
    assert task_costs(db) == [] and in_flight(db) == [(JOB, task)] and queue.sent == []
    service(db, queue, costs=costs).handle(parse_webhook(success_body(task)))
    assert len(task_costs(db)) == 1 and in_flight(db) == [] and len(queue.sent) == 1


def test_slot_release_failure_is_completed_on_redelivery(db, queue_settings):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    shared = Flaky(limiter(db, interval=0), "release_in_flight")
    with pytest.raises(FittingDatabaseError):
        service(db, queue, limiter_=shared).handle(parse_webhook(error_body(task)))
    assert job(db).status is AsyncJobStatus.FALLBACK_RUNNING and in_flight(db) == [(JOB, task)]
    service(db, queue, limiter_=shared).handle(parse_webhook(error_body(task)))
    assert in_flight(db) == [] and [p["action"] for _, p in queue.sent] == ["RUN_FALLBACK"]


def test_partly_processed_batch_is_safe_to_redeliver(db, queue_settings):  # noqa: F811
    first, second = submitted(db, 1), submitted(db, 2)
    body = json.dumps(
        {"data": [json.loads(success_body(first)), json.loads(success_body(second))]}
    ).encode()
    queue = FakeQueue(SqsOutcome.SUCCEEDED, SqsOutcome.UNKNOWN)
    with pytest.raises(WebhookFollowUpError):
        service(db, queue).handle(parse_webhook(body))
    service(db, queue).handle(parse_webhook(body))
    assert len(task_costs(db)) == 2 and in_flight(db) == []
    assert job(db, 1).status is job(db, 2).status is AsyncJobStatus.POSTPROCESSING
    assert recorded_total(db, START_DAY) == Decimal("0.046")


def test_follow_up_message_contains_only_ids(db, queue_settings):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    service(db, queue).handle(parse_webhook(success_body(task)))
    ((queue_name, payload),) = queue.sent
    assert queue_name is FittingQueue.AI_FITTING_RESPONSE
    assert set(payload) == {"fitting_job_id", "task_uuid", "action"}
    assert IMAGE_URL not in json.dumps(payload)


# --- Router (실제 app·의존성 조립, 가짜 DB pool·boto3) ---


class FakeBotoSqs:
    def __init__(self, error=None):
        self.bodies, self.urls, self.error = [], [], error

    def send_message(self, QueueUrl, MessageBody):
        if self.error is not None:
            raise self.error
        self.urls.append(QueueUrl)
        self.bodies.append(json.loads(MessageBody))
        return {"MessageId": f"m-{len(self.bodies)}"}


@pytest.fixture
def api(db, monkeypatch, queue_settings):  # noqa: F811
    """app.main.app 그대로. DB pool 과 boto3 SQS client 만 격리 DB·가짜로 바꾼다(lifespan 은 돌리지 않음)."""
    from app.main import app

    boto = FakeBotoSqs()
    monkeypatch.setattr(settings, "WEBHOOK_TOKEN", TOKEN)
    monkeypatch.setattr(webhook_router, "get_connection_pool", lambda: db["pool"])
    monkeypatch.setattr(webhook_router, "get_webhook_sqs_client", lambda: boto)
    # Webhook 이 부르면 안 되는 것들
    forbidden = []

    def no_call(name):
        def call(*args, **kwargs):
            forbidden.append(name)
            raise AssertionError(f"webhook must not call {name}")

        return call

    import app.virtual_fitting.providers.runware as v1
    import app.virtual_fitting.providers.runware_comment as comment
    from app.clients import s3

    monkeypatch.setattr(s3.S3ImageStorage, "__init__", no_call("S3ImageStorage"))
    monkeypatch.setattr(v1.RunwarePrunaProvider, "__init__", no_call("V1 RunwarePrunaProvider"))
    monkeypatch.setattr(
        comment.RunwareCommentProvider, "__init__", no_call("RunwareCommentProvider")
    )
    client = TestClient(app, raise_server_exceptions=False)
    return {"client": client, "boto": boto, "db": db, "forbidden": forbidden}


def post(api, body, token=TOKEN, params=None):
    query = params if params is not None else ({"token": token} if token is not None else {})
    return api["client"].post("/api/v2/webhooks/runware", content=body, params=query)


def test_router_success_end_to_end(api):
    db_ = api["db"]
    task = submitted(db_)
    response = post(api, success_body(task))
    assert response.status_code == 200
    assert response.json() == {
        "code": "WEBHOOK_PROCESSED",
        "data": {
            "results": [{"task_uuid": task, "outcome": "applied", "published": "POSTPROCESS"}]
        },
        "message": "알림을 처리했습니다.",
    }
    assert api["boto"].urls == [QUEUE_URL]
    assert api["boto"].bodies == [
        {"fitting_job_id": JOB, "task_uuid": task, "action": "POSTPROCESS"}
    ]
    assert job(db_).status is AsyncJobStatus.POSTPROCESSING and in_flight(db_) == []
    assert api["forbidden"] == []
    # 같은 알림 재전송도 200
    again = post(api, success_body(task))
    assert again.status_code == 200 and again.json()["data"]["results"][0]["outcome"] == "duplicate"


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"token": "wrong"},
        {"token": ""},
        {"token": TOKEN.upper()},
        [("token", TOKEN), ("token", TOKEN)],
    ],
)
def test_router_auth_failure_has_no_side_effects(api, params, monkeypatch):
    task = submitted(api["db"])
    called = []
    monkeypatch.setattr(webhook_router, "open_webhook_service", lambda: called.append(1))
    response = post(api, success_body(task), params=params)
    assert response.status_code == 401 and response.json()["code"] == "UNAUTHORIZED"
    assert called == [] and api["boto"].bodies == []
    assert job(api["db"]).status is AsyncJobStatus.WAITING_PRUNA and task_costs(api["db"]) == []


def test_router_without_configured_token_rejects_everything(api, monkeypatch):
    monkeypatch.setattr(settings, "WEBHOOK_TOKEN", "")
    task = submitted(api["db"])
    for params in ({"token": ""}, {}, {"token": "anything"}):
        response = post(api, success_body(task), params=params)
        assert response.status_code == 500 and response.json()["code"] == "WEBHOOK_NOT_CONFIGURED"
    assert job(api["db"]).status is AsyncJobStatus.WAITING_PRUNA


def test_router_bad_bodies_and_unknown_tasks_are_400(api):
    submitted(api["db"])
    for body in (b"{not json", success_body("bad"), b'{"foo": 1}'):
        response = post(api, body)
        assert response.status_code == 400 and response.json()["code"] == "INVALID_REQUEST"
    response = post(api, success_body(OTHER_TASK))
    assert response.status_code == 400 and response.json()["code"] == "WEBHOOK_UNKNOWN_TASK"
    assert api["boto"].bodies == [] and job(api["db"]).status is AsyncJobStatus.WAITING_PRUNA


def test_router_sqs_failure_is_5xx_and_redelivery_recovers(api):
    from botocore.exceptions import EndpointConnectionError, ReadTimeoutError

    task = submitted(api["db"])
    api["boto"].error = ReadTimeoutError(endpoint_url=QUEUE_URL)  # 응답 없음 → UNKNOWN
    first = post(api, success_body(task))
    assert first.status_code == 503 and first.json()["code"] == "WEBHOOK_FOLLOW_UP_FAILED"
    api["boto"].error = EndpointConnectionError(endpoint_url=QUEUE_URL)  # 연결 실패 → FAILED
    assert post(api, success_body(task)).status_code == 503
    assert (
        job(api["db"]).status is AsyncJobStatus.POSTPROCESSING and len(task_costs(api["db"])) == 1
    )
    api["boto"].error = None
    recovered = post(api, success_body(task))
    assert recovered.status_code == 200
    assert api["boto"].bodies == [
        {"fitting_job_id": JOB, "task_uuid": task, "action": "POSTPROCESS"}
    ]
    assert len(task_costs(api["db"])) == 1


def test_router_missing_queue_config_stops_before_recording(api, monkeypatch):
    monkeypatch.setattr(settings, "AI_FITTING_RESPONSE_QUEUE_URL", "")
    task = submitted(api["db"])
    response = post(api, success_body(task))
    assert response.status_code == 500 and response.json()["code"] == "SQS_CONFIG_ERROR"
    assert job(api["db"]).status is AsyncJobStatus.WAITING_PRUNA and task_costs(api["db"]) == []


def test_router_database_error_is_5xx(api, monkeypatch):
    def broken():
        raise FittingDatabaseError("down")

    monkeypatch.setattr(webhook_router, "get_connection_pool", broken)
    response = post(api, success_body(OTHER_TASK))
    assert response.status_code == 500 and response.json()["code"] == "DATABASE_ERROR"


def test_router_times_out_with_503_instead_of_waiting(api, monkeypatch):
    monkeypatch.setattr(webhook_router, "WEBHOOK_DEADLINE_SECONDS", 0.2)
    release = threading.Event()

    def slow(events):
        release.wait(5)
        return []

    monkeypatch.setattr(webhook_router, "_process", slow)
    started = time.monotonic()
    response = post(api, success_body(OTHER_TASK))
    release.set()
    assert response.status_code == 503 and response.json()["code"] == "WEBHOOK_TIMEOUT"
    assert time.monotonic() - started < 2


def test_router_does_not_need_the_internal_api_key(api, monkeypatch):
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "internal")
    task = submitted(api["db"])
    assert post(api, success_body(task)).status_code == 200


def test_router_logs_no_token_url_or_body(api, caplog):
    task = submitted(api["db"])
    with caplog.at_level(logging.DEBUG):
        post(api, error_body(task, code="invalidReferenceImage"))
        post(api, success_body(task), token="wrong-secret-value")
        post(api, b"{broken " + IMAGE_URL.encode())
    # 서버 코드(app.*)의 로그만 본다. httpx(테스트 client)는 자기 요청 URL 을 로그에 남긴다.
    server = [r for r in caplog.records if r.name.startswith("app")]
    assert server  # 처리 로그는 남는다
    text = "\n".join(r.getMessage() for r in server)
    for secret in (TOKEN, "wrong-secret-value", "SECRET-IMAGE", "did not respond", "token="):
        assert secret not in text


def test_access_log_filter_hides_the_token():
    assert (
        redact_query_secrets("/api/v2/webhooks/runware?token=abc&x=1")
        == "/api/v2/webhooks/runware?token=<redacted>&x=1"
    )
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:5", "POST", f"/api/v2/webhooks/runware?token={TOKEN}", "1.1", 200),
        None,
    )
    assert RedactQuerySecretsFilter().filter(record)
    assert TOKEN not in record.getMessage() and "token=<redacted>" in record.getMessage()


# --- 토큰과 8단계 URL ---


def test_webhook_url_must_carry_the_same_token():
    url = f"https://ai.example.com/api/v2/webhooks/runware?token={TOKEN}"
    assert webhook_url_matches_token(url, TOKEN)
    assert not webhook_url_matches_token(url, "other")
    assert not webhook_url_matches_token(url, "")
    assert not webhook_url_matches_token(url.replace("https", "http"), TOKEN)
    assert not webhook_url_matches_token(url.replace("/runware", "/other"), TOKEN)
    assert not webhook_url_matches_token(url + f"&token={TOKEN}", TOKEN)
    assert not webhook_url_matches_token("", TOKEN)


def test_request_worker_does_not_start_with_a_mismatched_webhook_token(monkeypatch):
    from app.virtual_fitting import request_worker

    monkeypatch.setattr(settings, "WEBHOOK_TOKEN", TOKEN)
    monkeypatch.setattr(
        settings,
        "VIRTUAL_FITTING_RUNWARE_WEBHOOK_URL",
        "https://ai.example.com/api/v2/webhooks/runware?token=other",
    )
    from app.config import database

    monkeypatch.setattr(database, "get_connection_pool", lambda: pytest.fail("pool must not open"))
    with pytest.raises(ValueError) as error:
        request_worker.create_request_worker()
    assert TOKEN not in str(error.value) and "other" not in str(error.value)


def test_follow_up_paths_are_still_not_ready():
    from app.virtual_fitting.pruna_submitter import NextStep, NoFollowUpPaths

    paths = NoFollowUpPaths()
    assert not any(paths.ready_for(None, step) for step in NextStep)
    assert settings.VIRTUAL_FITTING_REQUEST_WORKER_ENABLED is False
