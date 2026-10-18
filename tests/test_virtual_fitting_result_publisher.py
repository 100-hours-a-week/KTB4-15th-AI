"""14단계: 최종 결과 발행(be-fitting-response)·재시도·PUBLISH_RESULT 경로.

실제 ResultPublishProcessor·AsyncJobRepository·AsyncJobStateRepository(격리 PostgreSQL)를 쓰고 SQS 는 가짜 객체
(또는 실제 SqsQueueClient + 가짜 boto3 client)로 바꾼다. 결과는 실제 저장 경로(Webhook·POSTPROCESS·RUN_FALLBACK·요청 Worker)로
만든다. 실제 AWS·Runware 는 부르지 않는다.
"""

import asyncio
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.clients.sqs import FittingQueue, SqsOperation, SqsOutcome, SqsResult
from app.config import settings
from app.virtual_fitting import result_publisher as rp
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.pruna_submitter import NextStep, deletion_allowed
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository
from app.virtual_fitting.result_publisher import (
    PublishOutcome,
    ResultPublisherWorker,
    ResultPublishPaths,
    ResultPublishProcessor,
    register_result_publisher,
)
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_pruna_submit import JOB, db, job  # noqa: F401 (pytest fixture)

QUEUE_URL = "https://sqs.ap-northeast-2.amazonaws.com/123456789012/be-fitting-response"
ASYNC_KEY = f"virtual-fitting/results/{JOB}/pruna-async.jpg"
SYNC_KEY = f"virtual-fitting/results/{JOB}/pruna-sync.jpg"


@pytest.fixture(autouse=True)
def be_queue(monkeypatch):
    monkeypatch.setattr(settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", QUEUE_URL)
    register_result_publisher(None)
    yield
    register_result_publisher(None)


class FakeSqs:
    """send_json 만. outcomes(SqsOutcome·reason 또는 Exception)를 차례로. gate 가 있으면 보내는 중 멈춘다."""

    def __init__(self, *outcomes, gate=None):
        self.outcomes, self.gate, self.sent = list(outcomes), gate, []

    def send_json(self, queue, payload):
        self.sent.append((queue, json.loads(json.dumps(payload))))
        if self.gate is not None:
            self.gate.wait(10)
        item = self.outcomes.pop(0) if self.outcomes else (SqsOutcome.SUCCEEDED, "ok")
        if isinstance(item, BaseException):
            raise item
        outcome, reason = item
        return SqsResult(SqsOperation.SEND, queue, outcome, reason, message_id="m-1")


def publisher(db, sqs, state=None):  # noqa: F811
    return ResultPublishProcessor(
        db["jobs"], state or db["state"], sqs, lease_seconds=60, retry_backoff_seconds=(5.0, 30.0)
    )


def success_payload(key=ASYNC_KEY, title="차분한 데일리룩"):
    return {
        "fitting_job_id": JOB,
        "status": "COMPLETED",
        "result_image_key": key,
        "llm_title": title,
        "llm_comment": "선택하신 상품으로 완성한 가상 피팅 결과입니다.",
    }


FAILED_PAYLOAD = {
    "fitting_job_id": JOB,
    "status": "FAILED",
    "error_code": "FITTING_MODEL_FAILED",
    "error_message": "가상 피팅 이미지 생성에 실패했습니다.",
}


# --- 결과를 실제 저장 경로로 만든다 ---


def saved_by_postprocess(db):  # noqa: F811
    from app.clients.s3 import S3ObjectState
    from tests.test_virtual_fitting_postprocess_handler import FakeComments
    from tests.test_virtual_fitting_postprocess_handler import env as postprocess_env
    from tests.test_virtual_fitting_postprocess_handler import run as run_postprocess
    from tests.test_virtual_fitting_postprocess_worker import postprocessing

    postprocessing(db)
    comments = FakeComments(comments=["선택하신 상품으로 완성한 가상 피팅 결과입니다."])
    run_postprocess(db, postprocess_env(db, [S3ObjectState.EXISTS], comments=comments))
    assert job(db).result_payload == success_payload()


def saved_by_fallback(db):  # noqa: F811
    from tests.test_virtual_fitting_fallback_handler import env as fallback_env
    from tests.test_virtual_fitting_fallback_handler import fallback_job
    from tests.test_virtual_fitting_fallback_handler import run as run_fallback

    fallback_job(db)
    run_fallback(db, fallback_env(db))
    assert job(db).result_payload["result_image_key"] == SYNC_KEY


def saved_by_webhook(db, code="someNewCode"):  # noqa: F811
    from app.virtual_fitting.repositories.pruna_limiter import TaskCostRepository
    from app.virtual_fitting.runware_webhook import (
        RunwareWebhookService,
        WebhookFollowUpNotReadyError,
        parse_webhook,
    )
    from tests.test_virtual_fitting_pruna_submit import limiter
    from tests.test_virtual_fitting_webhook import FakeQueue, error_body, submitted

    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        settings,
        "AI_FITTING_RESPONSE_QUEUE_URL",
        "https://sqs.ap-northeast-2.amazonaws.com/123456789012/ai-fitting-response",
    )
    try:
        task = submitted(db)
        service = RunwareWebhookService(
            jobs=db["jobs"],
            state=db["state"],
            limiter=limiter(db),
            costs=TaskCostRepository(db["pool"]),
            queue=FakeQueue(),
        )
        with pytest.raises(WebhookFollowUpNotReadyError):
            service.handle(parse_webhook(error_body(task, code=code)))
    finally:
        monkey.undo()
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_token = NULL, worker_lease_until = NULL "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    assert job(db).result_payload == FAILED_PAYLOAD


def saved_by_request_worker(db):  # noqa: F811
    from app.virtual_fitting.request_worker import PrecheckOutcome, PrecheckStatus, RequestProcessor
    from tests.test_virtual_fitting_pruna_submit import FixedCheck
    from tests.test_virtual_fitting_request_worker import FakeSqs as RequestSqs
    from tests.test_virtual_fitting_request_worker import valid_message

    check = FixedCheck(PrecheckOutcome(PrecheckStatus.FAILED, "PRODUCT_NOT_FOUND", "상품 없음"))
    RequestProcessor(
        db["jobs"], db["state"], check, RequestSqs(), lease_seconds=60, visibility_seconds=120
    ).process(valid_message())
    assert job(db).result_payload["error_code"] == "PRODUCT_NOT_FOUND"


def expire_lease(db):  # noqa: F811
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second' "
        "WHERE fitting_job_id = %s AND worker_lease_token IS NOT NULL",
        (JOB,),
    )


# --- 정확한 JSON 과 최종 상태 ---


@pytest.mark.parametrize(
    ("save", "expected", "final"),
    [
        (saved_by_postprocess, lambda: success_payload(), AsyncJobStatus.COMPLETED),
        (saved_by_fallback, None, AsyncJobStatus.COMPLETED),
        (saved_by_webhook, lambda: FAILED_PAYLOAD, AsyncJobStatus.FAILED),
    ],
)
def test_saved_result_is_sent_exactly_and_finalized(db, save, expected, final):  # noqa: F811
    save(db)
    stored = job(db).result_payload
    sqs = FakeSqs()
    assert publisher(db, sqs).publish(JOB) is PublishOutcome.PUBLISHED
    ((queue, body),) = sqs.sent
    assert queue is FittingQueue.BE_FITTING_RESPONSE
    assert body == stored and (
        expected is None or body == expected()
    )  # code/data/message 로 감싸지 않음
    assert "code" not in body and "data" not in body
    current = job(db)
    assert (current.status, current.next_action, current.result_message_status) == (
        final,
        None,
        ResultMessageStatus.PUBLISHED,
    )
    assert current.result_message_attempt_count == 1 and current.worker_lease_token is None
    assert current.result_payload == stored  # 결과는 그대로
    assert current.user_image_key is not None and current.products  # 원본 입력 보존


def test_fallback_result_uses_the_saved_sync_key(db):  # noqa: F811
    saved_by_fallback(db)
    sqs = FakeSqs()
    publisher(db, sqs).publish(JOB)
    assert sqs.sent[0][1]["result_image_key"] == SYNC_KEY


@pytest.mark.parametrize(
    "save", [saved_by_request_worker, saved_by_webhook, saved_by_postprocess, saved_by_fallback]
)
def test_every_entry_point_is_found_by_the_due_query(db, save):  # noqa: F811
    save(db)
    assert db["jobs"].find_result_publish_due(20) == [JOB]
    assert publisher(db, FakeSqs()).publish(JOB) is PublishOutcome.PUBLISHED
    assert db["jobs"].find_result_publish_due(20) == []


def test_saved_result_survives_a_restart(db):  # noqa: F811
    saved_by_webhook(db)
    # "재시작": 새 pool 사용자·새 Repository·새 Processor
    fresh_state = AsyncJobStateRepository(db["pool"])
    sqs = FakeSqs()
    assert (
        ResultPublishProcessor(db["jobs"], fresh_state, sqs).publish(JOB)
        is PublishOutcome.PUBLISHED
    )
    assert len(sqs.sent) == 1


# --- 중복 실행 방지 ---


def test_concurrent_publishers_send_once(db):  # noqa: F811
    saved_by_webhook(db)
    gate = threading.Event()
    sqs = FakeSqs(gate=gate)
    results, barrier = [], threading.Barrier(5)

    def go():
        barrier.wait()
        results.append(publisher(db, sqs).publish(JOB))

    threads = [threading.Thread(target=go) for _ in range(5)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 10
    while len(results) < 4:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    gate.set()
    for thread in threads:
        thread.join(10)
    assert len(sqs.sent) == 1
    assert sorted(r.value for r in results) == ["published"] + ["skipped"] * 4


# --- 전송 결과별 ---


@pytest.mark.parametrize(
    ("outcome", "reason", "backoff"),
    [
        (SqsOutcome.UNKNOWN, "timeout", 5),
        (SqsOutcome.FAILED, "throttled", 5),
        (SqsOutcome.FAILED, "access_denied", 300),
    ],
)
def test_failed_or_unknown_retries_later_with_the_same_payload(db, outcome, reason, backoff):  # noqa: F811
    saved_by_webhook(db)
    sqs = FakeSqs((outcome, reason))
    assert publisher(db, sqs).publish(JOB) is PublishOutcome.RETRY_LATER
    current = job(db)
    assert (
        current.result_message_attempt_count == 1
        and current.result_message_status is ResultMessageStatus.PENDING
    )
    assert (
        current.result_payload == FAILED_PAYLOAD and current.status is AsyncJobStatus.WAITING_PRUNA
    )
    # 실행 권한을 backoff 동안 쥔 채 둔다 → 그동안 대상에서 빠진다.
    remaining = db["sql"](
        "SELECT EXTRACT(EPOCH FROM worker_lease_until - now()) FROM virtual_fitting_async_jobs"
    )[0][0]
    assert backoff - 2 < remaining <= backoff
    assert db["jobs"].find_result_publish_due(20) == []
    assert publisher(db, sqs).publish(JOB) is PublishOutcome.SKIPPED and len(sqs.sent) == 1
    expire_lease(db)
    assert publisher(db, sqs).publish(JOB) is PublishOutcome.PUBLISHED
    assert sqs.sent[0][1] == sqs.sent[1][1] == FAILED_PAYLOAD


def test_unexpected_exception_while_sending_is_unknown(db):  # noqa: F811
    saved_by_webhook(db)
    sqs = FakeSqs(RuntimeError("boom"))
    assert publisher(db, sqs).publish(JOB) is PublishOutcome.RETRY_LATER
    assert job(db).result_message_attempt_count == 1


def test_at_most_three_sends_then_exhausted_and_never_again(db):  # noqa: F811
    saved_by_postprocess(db)
    sqs = FakeSqs(*[(SqsOutcome.UNKNOWN, "timeout")] * 5)
    outcomes = []
    for _ in range(5):
        outcomes.append(publisher(db, sqs).publish(JOB))  # 매번 새 Processor(재시작·Worker 변경)
        expire_lease(db)
    assert outcomes[:3] == [
        PublishOutcome.RETRY_LATER,
        PublishOutcome.RETRY_LATER,
        PublishOutcome.EXHAUSTED,
    ]
    assert outcomes[3:] == [PublishOutcome.SKIPPED] * 2
    assert len(sqs.sent) == 3  # 4번째 자동 전송 없음
    current = job(db)
    assert current.result_message_status is ResultMessageStatus.RETRY_EXHAUSTED
    assert current.result_message_attempt_count == 3
    assert current.result_payload == success_payload() and current.result_image_key == ASYNC_KEY
    assert current.status is AsyncJobStatus.POSTPROCESSING  # 생성 실패·발행 성공으로 처리하지 않음
    assert current.next_action is NextAction.PUBLISH_RESULT
    assert db["jobs"].find_result_publish_due(20) == []


def test_claimed_then_stopped_before_sending_keeps_the_count(db):  # noqa: F811
    saved_by_webhook(db)
    token = db["state"].acquire_lease(JOB, NextAction.PUBLISH_RESULT, 60).token
    assert db["state"].claim_result_message_attempt(JOB, token).applied  # 여기서 프로세스가 멈췄다
    expire_lease(db)
    sqs = FakeSqs()
    assert publisher(db, sqs).publish(JOB) is PublishOutcome.PUBLISHED
    assert job(db).result_message_attempt_count == 2  # 되돌리지 않았다


def test_attempts_already_used_without_success_are_exhausted_without_sending(db):  # noqa: F811
    saved_by_webhook(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET result_message_attempt_count = 3 WHERE fitting_job_id = %s",
        (JOB,),
    )
    sqs = FakeSqs()
    assert publisher(db, sqs).publish(JOB) is PublishOutcome.EXHAUSTED
    assert sqs.sent == [] and job(db).result_message_status is ResultMessageStatus.RETRY_EXHAUSTED


class NotRecording:
    """mark_result_published 만 실패시킨다(SQS 성공 뒤 AI DB 기록 실패)."""

    def __init__(self, real, error=False):
        self.real, self.error = real, error

    def __getattr__(self, name):
        return getattr(self.real, name)

    def mark_result_published(self, *args, **kwargs):
        if self.error:
            raise FittingDatabaseError("down")
        from app.virtual_fitting.repositories.async_job_state import ChangeReason, StateChange

        return StateChange(False, ChangeReason.LEASE_INVALID, None)


def test_sent_but_not_recorded_is_not_assumed_published(db):  # noqa: F811
    saved_by_webhook(db)
    sqs = FakeSqs()
    assert publisher(db, sqs, NotRecording(db["state"])).publish(JOB) is PublishOutcome.RETRY_LATER
    assert (
        job(db).result_message_status is ResultMessageStatus.PENDING
    )  # 임의로 PUBLISHED 로 만들지 않음
    expire_lease(db)
    assert publisher(db, sqs).publish(JOB) is PublishOutcome.PUBLISHED
    assert len(sqs.sent) == 2  # 중복 전달 가능(BE 는 fitting_job_id 로 중복을 견뎌야 한다)


def test_db_error_after_sending_keeps_everything_and_does_not_reset(db):  # noqa: F811
    saved_by_webhook(db)
    sqs = FakeSqs()
    with pytest.raises(FittingDatabaseError):
        publisher(db, sqs, NotRecording(db["state"], error=True)).publish(JOB)
    current = job(db)
    assert (
        current.result_message_attempt_count == 1
        and current.result_message_status is ResultMessageStatus.PENDING
    )


def test_last_attempt_sent_but_not_recorded_is_exhausted_not_published(db):  # noqa: F811
    saved_by_webhook(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET result_message_attempt_count = 2 WHERE fitting_job_id = %s",
        (JOB,),
    )
    sqs = FakeSqs()
    assert publisher(db, sqs, NotRecording(db["state"])).publish(JOB) is PublishOutcome.EXHAUSTED
    current = job(db)
    assert current.result_message_status is ResultMessageStatus.RETRY_EXHAUSTED
    assert current.result_message_attempt_count == 3 and len(sqs.sent) == 1


# --- 보내기 전 확인(횟수를 쓰지 않음) ---


def test_payload_mismatch_is_held_without_sending(db):  # noqa: F811
    saved_by_postprocess(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET result_image_key = 'virtual-fitting/results/123/other.jpg' "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    sqs = FakeSqs()
    assert publisher(db, sqs).publish(JOB) is PublishOutcome.HELD
    assert sqs.sent == [] and job(db).result_message_attempt_count == 0


def test_missing_queue_config_is_held_without_using_an_attempt(db, monkeypatch):  # noqa: F811
    saved_by_webhook(db)
    monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", "")
    sqs = FakeSqs()
    assert publisher(db, sqs).publish(JOB) is PublishOutcome.HELD
    assert sqs.sent == [] and job(db).result_message_attempt_count == 0


def test_lock_timeout_rolls_back_and_sends_nothing(db):  # noqa: F811
    saved_by_webhook(db)
    connection = db["connect"]()
    try:
        connection.execute(
            "SELECT 1 FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s FOR UPDATE", (JOB,)
        )
        sqs = FakeSqs()
        short = AsyncJobStateRepository(db["pool"], lock_timeout_seconds=0.3)
        with pytest.raises(FittingDatabaseError):
            publisher(db, sqs, short).publish(JOB)
    finally:
        connection.rollback()
        connection.close()
    current = job(db)
    assert sqs.sent == [] and current.worker_lease_token is None
    assert current.result_message_attempt_count == 0
    assert publisher(db, FakeSqs()).publish(JOB) is PublishOutcome.PUBLISHED


def test_payload_status_and_final_status_cannot_disagree(db):  # noqa: F811
    saved_by_webhook(db)
    token = db["state"].acquire_lease(JOB, NextAction.PUBLISH_RESULT, 60).token
    assert db["state"].claim_result_message_attempt(JOB, token).applied
    wrong = db["state"].mark_result_published(JOB, token, AsyncJobStatus.COMPLETED)
    assert not wrong.applied and job(db).result_message_status is ResultMessageStatus.PENDING


# --- FollowUpPaths·Webhook·메시지 삭제 ---


class Alive:
    def __init__(self, alive=True):
        self.alive = alive


def test_publish_paths_are_ready_only_with_a_running_publisher_and_pending_result(db):  # noqa: F811
    saved_by_webhook(db)
    pending = job(db)
    assert not ResultPublishPaths().ready_for(pending, NextStep.PUBLISH_RESULT)  # 등록된 루프 없음
    assert not ResultPublishPaths(lambda: Alive(False)).ready_for(pending, NextStep.PUBLISH_RESULT)
    ready = ResultPublishPaths(lambda: Alive())
    assert ready.ready_for(pending, NextStep.PUBLISH_RESULT)
    assert not any(
        ready.ready_for(pending, step) for step in NextStep if step is not NextStep.PUBLISH_RESULT
    )
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET result_message_status = 'RETRY_EXHAUSTED' WHERE fitting_job_id = %s",
        (JOB,),
    )
    assert not ready.ready_for(job(db), NextStep.PUBLISH_RESULT)  # 소진은 운영 복구 대상
    assert not deletion_allowed(job(db), ready)


def test_real_worker_status_drives_readiness(db):  # noqa: F811
    saved_by_webhook(db)
    pending = job(db)

    async def scenario():
        worker = ResultPublisherWorker(list, None, poll_seconds=0.01)
        register_result_publisher(worker)
        paths = rp.result_publish_paths()
        before = paths.ready_for(pending, NextStep.PUBLISH_RESULT)  # 아직 시작 전
        worker.start()
        await asyncio.sleep(0.05)
        running = paths.ready_for(pending, NextStep.PUBLISH_RESULT)
        stopping = worker.stop(1)
        task = asyncio.ensure_future(stopping)
        await asyncio.sleep(0)
        during_stop = paths.ready_for(pending, NextStep.PUBLISH_RESULT)
        await task
        return before, running, during_stop, paths.ready_for(pending, NextStep.PUBLISH_RESULT)

    assert asyncio.run(scenario()) == (False, True, False, False)


from tests.test_virtual_fitting_webhook import (  # noqa: F401
    api,
    error_body,
    post,
    queue_settings,
    submitted,
)


def test_webhook_permanent_failure_is_200_when_a_publisher_runs_else_503(api):  # noqa: F811
    task = submitted(api["db"])
    assert post(api, error_body(task, code="someNewCode")).status_code == 503  # 루프 없음
    register_result_publisher(Alive())
    response = post(api, error_body(task, code="someNewCode"))
    assert (
        response.status_code == 200
        and response.json()["data"]["results"][0]["outcome"] == "duplicate"
    )
    assert job(api["db"]).result_payload == FAILED_PAYLOAD
    register_result_publisher(Alive(False))  # 멈췄거나 비정상 종료
    assert post(api, error_body(task, code="someNewCode")).status_code == 503


def test_webhook_auth_error_hold_is_not_solved_by_the_publisher(api):  # noqa: F811
    register_result_publisher(Alive())
    task = submitted(api["db"])
    response = post(api, error_body(task, code="invalidApiKey"))
    assert (
        response.status_code == 503
        and response.json()["data"]["pending"][0]["next_step"] == "operator_review"
    )


def test_request_and_postprocess_messages_are_deleted_only_with_a_live_path(db):  # noqa: F811
    from app.virtual_fitting.request_worker import RequestProcessor
    from tests.test_virtual_fitting_pruna_submit import FixedCheck
    from tests.test_virtual_fitting_request_worker import FakeSqs as RequestSqs
    from tests.test_virtual_fitting_request_worker import valid_message

    saved_by_request_worker(db)

    def processor(paths):
        return RequestProcessor(
            db["jobs"],
            db["state"],
            FixedCheck(),
            RequestSqs(),
            lease_seconds=60,
            visibility_seconds=120,
            follow_up=paths,
        )

    assert processor(ResultPublishPaths()).process(valid_message()).delete is False
    assert processor(ResultPublishPaths(lambda: Alive())).process(valid_message()).delete is True
    sqs = FakeSqs()
    publisher(db, sqs).publish(JOB)
    # 발행 완료 뒤 중복 메시지(이전 삭제 실패 후 재수신): 재생성·재전송 없이 삭제
    assert processor(ResultPublishPaths()).process(valid_message()).delete is True
    assert len(sqs.sent) == 1 and job(db).result_message_attempt_count == 1


def test_postprocess_message_after_publish_is_deleted_without_rerun(db):  # noqa: F811
    from app.virtual_fitting.postprocess_worker import PostprocessHandlers
    from tests.test_virtual_fitting_postprocess_worker import Handler, item
    from tests.test_virtual_fitting_postprocess_worker import processor as postprocess_processor

    saved_by_postprocess(db)
    publisher(db, FakeSqs()).publish(JOB)
    handler = Handler()
    outcome = postprocess_processor(db, PostprocessHandlers(handler, handler)).process(
        item(JOB, job(db).task_uuid)
    )
    assert (outcome.action, outcome.delete) == ("already_published_deleted", True)
    assert handler.calls == []


def test_late_webhook_before_publish_preserves_the_saved_result(api):  # noqa: F811
    # 발행 전(PENDING) 저장된 결과: 늦은 Webhook 이 결과를 덮지 않는다(이전 이름의 테스트가 실제로 확인하던 상황).
    task = submitted(api["db"])
    register_result_publisher(Alive())
    post(api, error_body(task, code="someNewCode"))
    saved = job(api["db"])
    assert saved.result_message_status is ResultMessageStatus.PENDING
    assert saved.next_action is NextAction.PUBLISH_RESULT
    response = post(api, error_body(task))
    assert response.status_code == 200 and job(api["db"]) == saved


def test_late_webhook_after_publish_preserves_the_result(api):  # noqa: F811
    db_ = api["db"]
    task = submitted(db_)
    register_result_publisher(Alive())
    post(api, error_body(task, code="someNewCode"))
    # 준비 단계(submitted)의 요청 Worker 실행 권한을 정상 반납한다(실제 Worker 는 처리 뒤 반납한다).
    db_["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_token = NULL, worker_lease_until = NULL "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    sqs = FakeSqs()
    assert (
        publisher(db_, sqs).publish(JOB) is PublishOutcome.PUBLISHED
    )  # 실제 전송 Processor·가짜 SQS
    assert sqs.sent == [(FittingQueue.BE_FITTING_RESPONSE, FAILED_PAYLOAD)]
    final = job(db_)
    assert final.result_message_status is ResultMessageStatus.PUBLISHED
    assert final.status.value == final.result_payload["status"] == "FAILED"
    assert final.next_action is None and final.result_message_attempt_count == 1
    internal_before = len(api["boto"].bodies)

    for loops in (True, False):  # 결과 전송 루프가 돌거나 꺼져 있어도 발행 완료 중복은 안전하게 200
        register_result_publisher(Alive() if loops else None)
        for body in (error_body(task), error_body(task, code="someNewCode")):
            response = post(api, body)
            assert response.status_code == 200
            assert job(db_) == final  # payload·최종 이미지·상태·전송 횟수 그대로
    assert len(sqs.sent) == 1  # BE 결과를 다시 보내지 않음
    assert len(api["boto"].bodies) == internal_before  # POSTPROCESS·RUN_FALLBACK 재발행 없음
    assert api["forbidden"] == []  # 이미지 생성·LLM·S3 를 만들지 않음


# --- 반복 루프 ---


def test_worker_publishes_due_results_and_stops(db):  # noqa: F811
    saved_by_webhook(db)
    sqs = FakeSqs()
    worker = ResultPublisherWorker(
        lambda: db["jobs"].find_result_publish_due(20), publisher(db, sqs), poll_seconds=0.01
    )

    async def scenario():
        worker.start()
        deadline = time.monotonic() + 5
        while job(db).result_message_status is not ResultMessageStatus.PUBLISHED:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        await worker.stop(1)

    asyncio.run(scenario())
    assert len(sqs.sent) == 1 and worker.status.outcomes["published"] == 1 and worker.status.stopped


def test_worker_backs_off_on_db_errors_and_reports_crashes():
    calls = []

    def failing():
        calls.append(time.monotonic())
        raise FittingDatabaseError("down")

    async def backoff():
        worker = ResultPublisherWorker(
            failing, None, poll_seconds=0.01, backoff_initial_seconds=0.05
        )
        worker.start()
        await asyncio.sleep(0.4)
        await worker.stop(1)
        return worker

    worker = asyncio.run(backoff())
    assert 2 <= len(calls) <= 5 and not worker.status.crashed

    def bug():
        raise RuntimeError("bug")

    async def crash():
        worker = ResultPublisherWorker(bug, None)
        with pytest.raises(RuntimeError):
            await worker.start()
        return worker

    crashed = asyncio.run(crash())
    assert crashed.status.crashed and not crashed.alive
    assert not ResultPublishPaths(lambda: crashed).ready_for(None, NextStep.PUBLISH_RESULT)


# --- 실제 조립·lifespan ---


class FakeBoto:
    def __init__(self):
        self.bodies, self.urls = [], []

    def send_message(self, QueueUrl, MessageBody):
        self.urls.append(QueueUrl)
        self.bodies.append(json.loads(MessageBody))
        return {"MessageId": "be-1"}


def test_real_assembly_sends_through_the_real_sqs_client(db, monkeypatch):  # noqa: F811
    import app.clients.sqs as sqs_module
    from app.config import database

    saved_by_fallback(db)
    boto = FakeBoto()
    monkeypatch.setattr(database, "get_connection_pool", lambda: db["pool"])
    monkeypatch.setattr(sqs_module, "get_sqs_client", lambda: boto)
    monkeypatch.setattr(
        sqs_module,
        "get_webhook_sqs_client",
        lambda: pytest.fail("짧은 timeout client 를 쓰지 않는다"),
    )
    worker = rp.create_result_publisher()

    async def scenario():
        worker.start()
        deadline = time.monotonic() + 5
        while job(db).status is not AsyncJobStatus.COMPLETED:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        await worker.stop(1)

    asyncio.run(scenario())
    assert boto.urls == [QUEUE_URL] and boto.bodies == [job(db).result_payload]
    assert job(db).result_message_status is ResultMessageStatus.PUBLISHED


def test_sdk_retries_stay_disabled():
    from app.clients.sqs import get_sqs_client

    get_sqs_client.cache_clear()
    try:
        client = get_sqs_client()
        assert client.meta.config.retries == {"mode": "standard", "total_max_attempts": 1}
    finally:
        get_sqs_client.cache_clear()


from tests.test_virtual_fitting_request_worker import app_env  # noqa: F401


def test_lifespan_disabled_creates_nothing(app_env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(
        app_env, "create_result_publisher", lambda: pytest.fail("꺼져 있으면 만들지 않는다")
    )
    with TestClient(app_env.app) as client:
        assert client.app.state.result_publisher is None
        assert not ResultPublishPaths().ready_for(None, NextStep.PUBLISH_RESULT)
    assert settings.VIRTUAL_FITTING_RESULT_PUBLISHER_ENABLED is False


def test_lifespan_enabled_runs_registers_and_stops_before_the_pool(app_env, monkeypatch):  # noqa: F811
    order, entered, release = [], threading.Event(), threading.Event()

    def find_due():
        entered.set()
        release.wait(5)  # 전송 중(thread) — 다른 HTTP 요청은 처리된다
        return []

    worker = ResultPublisherWorker(find_due, None, poll_seconds=0.01)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_RESULT_PUBLISHER_ENABLED", True)
    monkeypatch.setattr(settings, "VIRTUAL_FITTING_RESULT_PUBLISHER_SHUTDOWN_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(app_env, "create_result_publisher", lambda: worker)
    monkeypatch.setattr(app_env, "close_connection_pool", lambda: order.append("close_pool"))
    try:
        with TestClient(app_env.app) as client:
            assert entered.wait(5)
            started = time.monotonic()
            assert client.get("/health").status_code == 200
            assert time.monotonic() - started < 1
            assert client.app.state.result_publisher is worker and worker.alive
            assert rp._registered is worker
            release.set()
        assert rp._registered is None and worker.status.stopped and order == ["close_pool"]
    finally:
        release.set()


# --- 14단계 보완: 결과 큐 설정과 준비 여부 ---

BAD_QUEUE_URLS = [
    "",
    "not-a-queue-url",
    "https://sqs.us-east-1.amazonaws.com/123456789012/be-fitting-response",  # 리전 불일치
    "https://sqs.ap-northeast-2.amazonaws.com/123456789012/be-fitting-response.fifo",  # FIFO
]


def test_ready_only_with_a_live_publisher_and_a_valid_queue_config(db, monkeypatch):  # noqa: F811
    saved_by_webhook(db)
    pending = job(db)
    live = ResultPublishPaths(lambda: Alive())
    assert live.ready_for(pending, NextStep.PUBLISH_RESULT)  # 루프 실행 + 올바른 설정
    for bad in BAD_QUEUE_URLS:
        monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", bad)
        assert not live.ready_for(pending, NextStep.PUBLISH_RESULT), bad
    monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", QUEUE_URL)
    for worker in (None, Alive(False)):  # 비활성화 / 종료 중·비정상 종료
        assert not ResultPublishPaths(lambda w=worker: w).ready_for(
            pending, NextStep.PUBLISH_RESULT
        )
    assert not live.ready_for(pending, NextStep.RECOVER_UNKNOWN)  # 결과 발행 준비와 무관하게 미준비


@pytest.mark.parametrize("bad", BAD_QUEUE_URLS)
def test_bad_queue_config_keeps_webhook_503_messages_results_and_counts(api, monkeypatch, bad):  # noqa: F811
    from app.virtual_fitting.request_worker import RequestProcessor
    from tests.test_virtual_fitting_pruna_submit import FixedCheck
    from tests.test_virtual_fitting_request_worker import FakeSqs as RequestSqs
    from tests.test_virtual_fitting_request_worker import valid_message

    db_ = api["db"]
    register_result_publisher(Alive())
    monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", bad)
    task = submitted(db_)
    response = post(api, error_body(task, code="someNewCode"))
    assert response.status_code == 503
    assert response.json()["data"]["pending"][0]["next_step"] == "publish_result"
    assert bad not in response.text or bad == ""  # 큐 URL 을 응답에 싣지 않는다
    saved = job(db_)
    assert saved.result_payload == FAILED_PAYLOAD
    kept = RequestProcessor(
        db_["jobs"],
        db_["state"],
        FixedCheck(),
        RequestSqs(),
        lease_seconds=60,
        visibility_seconds=120,
        follow_up=ResultPublishPaths(),
    ).process(valid_message())
    assert kept.delete is False  # 결과 저장만으로 지우지 않는다
    # 요청 Worker 는 처리 뒤 자기 실행 권한을 반납한다(submitted 도우미는 쥔 채 끝나므로 그 반납을 흉내 낸다).
    db_["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_token = NULL, worker_lease_until = NULL "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    sqs = FakeSqs()
    assert publisher(db_, sqs).publish(JOB) is PublishOutcome.HELD
    assert sqs.sent == [] and job(db_).result_message_attempt_count == 0
    assert job(db_).result_payload == FAILED_PAYLOAD


def test_bad_queue_config_keeps_the_postprocess_message(db, monkeypatch):  # noqa: F811
    from app.clients.s3 import S3ObjectState
    from app.virtual_fitting.postprocess_worker import PostprocessHandlers
    from tests.test_virtual_fitting_postprocess_handler import env as postprocess_env
    from tests.test_virtual_fitting_postprocess_worker import item, postprocessing
    from tests.test_virtual_fitting_postprocess_worker import processor as postprocess_processor

    register_result_publisher(Alive())
    task = postprocessing(db)
    handler = postprocess_env(db, [S3ObjectState.EXISTS]).handler

    def run_with(url):
        monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", url)
        return postprocess_processor(
            db, PostprocessHandlers(postprocess=handler), follow_up=ResultPublishPaths()
        ).process(item(JOB, task))

    first = run_with(BAD_QUEUE_URLS[2])
    assert (first.action, first.delete) == (
        "handler_done",
        False,
    )  # 결과는 저장했지만 지우지 않는다
    assert job(db).next_action is NextAction.PUBLISH_RESULT
    again = run_with(QUEUE_URL)  # 올바른 설정 + 실제 실행 경로 → 기존처럼 지운다
    assert (again.action, again.delete) == ("state_mismatch", True)


def test_good_config_keeps_the_existing_200_and_deletion(api):  # noqa: F811
    register_result_publisher(Alive())
    task = submitted(api["db"])
    assert post(api, error_body(task, code="someNewCode")).status_code == 200


# --- 14단계 보완: 잠금 대기 초과의 발생 시점 ---


class LockAt:
    """지정한 시점에 다른 연결로 작업 행을 잠근다(실제 PostgreSQL 행 잠금). close() 로 푼다."""

    def __init__(self, db):  # noqa: F811
        self.db, self.connection = db, None

    def lock(self):
        if self.connection is None:
            self.connection = self.db["connect"]()
            self.connection.execute(
                "SELECT 1 FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s FOR UPDATE",
                (JOB,),
            )

    def close(self):
        if self.connection is not None:
            self.connection.rollback()
            self.connection.close()
            self.connection = None


class LockBeforeClaim:
    """실행 권한 확보 뒤, 전송 횟수 확보 직전에 행이 잠긴다(A: 전송 전 DB 처리)."""

    def __init__(self, state, locker):
        self.state, self.locker = state, locker

    def __getattr__(self, name):
        return getattr(self.state, name)

    def claim_result_message_attempt(self, *args):
        self.locker.lock()
        return self.state.claim_result_message_attempt(*args)


class LockingSqs(FakeSqs):
    """보내는 순간 행을 잠근다 → 이어지는 완료 기록(mark_result_published)이 잠금 대기 초과(B: 전송 후)."""

    def __init__(self, locker):
        super().__init__()
        self.locker = locker

    def send_json(self, queue, payload):
        result = super().send_json(queue, payload)
        self.locker.lock()
        return result


def short_state(db):  # noqa: F811
    return AsyncJobStateRepository(db["pool"], lock_timeout_seconds=0.3)


def lock_timeout_shown(db):  # noqa: F811
    with db["pool"].connection() as connection:
        return connection.execute("SHOW lock_timeout").fetchone()[0]


def test_lock_timeout_before_sending_sends_nothing_and_rolls_back(db):  # noqa: F811
    saved_by_webhook(db)
    locker, sqs = LockAt(db), FakeSqs()
    try:
        with pytest.raises(FittingDatabaseError):
            publisher(db, sqs, LockBeforeClaim(short_state(db), locker)).publish(JOB)
    finally:
        locker.close()
    current = job(db)
    assert sqs.sent == []  # SQS 호출 0회
    assert current.result_message_attempt_count == 0  # 횟수 확보는 되돌려졌다
    assert (
        current.result_payload == FAILED_PAYLOAD
        and current.result_message_status is ResultMessageStatus.PENDING
    )
    assert current.status is AsyncJobStatus.WAITING_PRUNA  # 업무 실패로 바꾸지 않음(결과 그대로)
    assert lock_timeout_shown(db) == "0"
    expire_lease(db)
    assert publisher(db, FakeSqs()).publish(JOB) is PublishOutcome.PUBLISHED  # 연결 정상 재사용


def test_lock_timeout_after_sending_is_not_published_and_resends_the_same_payload(db):  # noqa: F811
    saved_by_postprocess(db)
    stored = job(db).result_payload
    locker = LockAt(db)
    sqs = LockingSqs(locker)
    try:
        with pytest.raises(FittingDatabaseError):
            publisher(db, sqs, short_state(db)).publish(JOB)
    finally:
        locker.close()
    current = job(db)
    assert len(sqs.sent) == 1  # 이미 SQS 에 전달됐을 수 있다
    assert current.result_message_status is ResultMessageStatus.PENDING  # PUBLISHED 기록 없음
    assert (
        current.status is AsyncJobStatus.POSTPROCESSING
        and current.result_message_attempt_count == 1
    )
    assert lock_timeout_shown(db) == "0"
    expire_lease(db)
    again = FakeSqs()
    assert (
        publisher(db, again).publish(JOB) is PublishOutcome.PUBLISHED
    )  # 남은 횟수로 같은 payload 재전송
    assert again.sent[0][1] == sqs.sent[0][1] == stored  # 중복 전달 가능
    final = job(db)
    assert final.result_message_attempt_count == 2 and final.status is AsyncJobStatus.COMPLETED
    assert final.result_payload == stored


def test_lock_timeout_after_the_last_send_ends_exhausted_without_a_fourth(db):  # noqa: F811
    saved_by_webhook(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET result_message_attempt_count = 2 WHERE fitting_job_id = %s",
        (JOB,),
    )
    locker = LockAt(db)
    sqs = LockingSqs(locker)
    try:
        with pytest.raises(FittingDatabaseError):
            publisher(db, sqs, short_state(db)).publish(JOB)  # 3번째 전송 뒤 완료 기록 실패
    finally:
        locker.close()
    assert len(sqs.sent) == 1 and job(db).result_message_attempt_count == 3
    assert job(db).result_message_status is ResultMessageStatus.PENDING
    expire_lease(db)
    after = FakeSqs()
    assert publisher(db, after).publish(JOB) is PublishOutcome.EXHAUSTED
    assert after.sent == []  # 4번째 자동 전송 없음
    current = job(db)
    assert current.result_message_status is ResultMessageStatus.RETRY_EXHAUSTED
    assert current.result_payload == FAILED_PAYLOAD and current.result_message_attempt_count == 3
