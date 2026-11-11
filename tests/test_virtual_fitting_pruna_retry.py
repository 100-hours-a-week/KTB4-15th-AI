"""보완 2-2: 확실한 미접수의 영속 기록·대기·최대 4회 재요청.

실제 RequestProcessor·PrunaSubmitter·RunwareWebhookService·Watchdog·PrunaResubmitter·AsyncJobStateRepository·PrunaLimiter
(격리 PostgreSQL)를 쓴다. Runware 전송·상태 조회·SQS·S3 는 가짜다. 시간은 DB 기록을 옮겨 확인한다(실제로 기다리지 않음).

확실한 미접수 = 현재 전송 시도(task)에 연결된 serviceOverloaded·inferenceRateLimitExceeded(공통 분류). 같은 작업 전체에서
최초 1회 + 재요청 3회. 미접수 기록·다음 시각 예약(또는 소진 결과)·그 시도의 자리 반환은 한 트랜잭션이고, 실제 재요청은
Watchdog 이 예약 시각 뒤 같은 Submitter 로 새 task 를 보낸다.
"""

import threading
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.config import settings
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.providers.runware_async import AsyncSubmission, SubmissionOutcome
from app.virtual_fitting.providers.runware_status import TaskStatus, TaskStatusKind
from app.virtual_fitting.pruna_resubmit import PrunaResubmitter
from app.virtual_fitting.pruna_submitter import (
    NextStep,
    PrunaSubmitter,
    deletion_allowed,
    next_step_for,
)
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import (
    PRUNA_RETRY_BACKOFF_SECONDS,
    PRUNA_SUBMIT_LIMIT,
    AsyncJobStateRepository,
    NotAcceptedOutcome,
    NotAcceptedSource,
)
from app.virtual_fitting.repositories.pruna_limiter import SubmissionReason
from app.virtual_fitting.runware_webhook import (
    EventKind,
    EventOutcome,
    WebhookEvent,
    WebhookFollowUpNotReadyError,
    parse_webhook,
)
from app.virtual_fitting.watchdog import WatchdogPaths
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_pruna_submit import (  # noqa: F401 (pytest fixture)
    JOB,
    WEBHOOK,
    FakeStorage,
    FixedBalance,
    FixedCheck,
    db,
    job,
    limiter,
    valid_message,
)
from tests.test_virtual_fitting_request_worker import FakeBudget, FakeSqs
from tests.test_virtual_fitting_v2_scenarios import env  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_watchdog import Sink, age, reasons, watchdog, wdb  # noqa: F401
from tests.test_virtual_fitting_webhook import (  # noqa: F401 (pytest fixture)
    FakeQueue,
    error_body,
    queue_settings,
    service,
    success_body,
    task_costs,
)

REJECTED = {"FAILED", "FITTING_SUBMISSION_REJECTED", "이미지 생성 요청이 접수되지 않았습니다."}


# --- 가짜 전송·조립 ---


def not_accepted(code="serviceOverloaded", **fields):
    def make(task):
        return AsyncSubmission(
            SubmissionOutcome.REJECTED,
            task,
            "not_accepted",
            http_status=503,
            error_codes=(code,),
            retryable=True,
            **fields,
        )

    return make


def rejected(reason, status, codes=()):
    def make(task):
        return AsyncSubmission(
            SubmissionOutcome.REJECTED, task, reason, http_status=status, error_codes=codes
        )

    return make


class Scripted:
    """전송마다 정해 둔 결과(없으면 접수). 보낸 task·저장 계약을 기록한다."""

    def __init__(self, *steps, during=None):
        self.steps, self.during, self.calls = list(steps), during, []

    def submit(
        self,
        fitting_input,
        *,
        task_uuid,
        webhook_url,
        upload_endpoint,
        output_format,
        ai_stores_result=False,
    ):
        self.calls.append(
            {
                "task": task_uuid,
                "upload": upload_endpoint,
                "format": output_format,
                "ai": ai_stores_result,
            }
        )
        if self.during:
            self.during(task_uuid)
        step = self.steps.pop(0) if self.steps else "accept"
        if isinstance(step, BaseException):
            raise step
        if step == "accept":
            return AsyncSubmission(SubmissionOutcome.ACCEPTED, task_uuid, "accepted", 200)
        if step == "timeout":
            return AsyncSubmission(SubmissionOutcome.UNKNOWN, task_uuid, "timeout")
        return step(task_uuid)

    def tasks(self):
        return [call["task"] for call in self.calls]


def submitter(db, provider, storage=None, **kwargs):  # noqa: F811
    from app.virtual_fitting.providers.runware_balance import MinimumBalanceCheck

    return PrunaSubmitter(
        jobs=db["jobs"],
        state=kwargs.pop("state", db["state"]),
        limiter=kwargs.pop("limiter_", None) or limiter(db, interval=0),
        provider=provider,
        storage=storage or FakeStorage(),
        budget=FakeBudget(),
        balance=MinimumBalanceCheck(FixedBalance("10"), Decimal("1.00")),
        webhook_url=WEBHOOK,
        keepalive_seconds=5,
        **kwargs,
    )


def request_processor(db, provider, follow_up=None, storage=None):  # noqa: F811
    from app.virtual_fitting.request_worker import RequestProcessor

    return RequestProcessor(
        db["jobs"],
        db["state"],
        FixedCheck(),
        FakeSqs(),
        lease_seconds=60,
        visibility_seconds=120,
        heartbeat_interval_seconds=0,
        submit=submitter(db, provider, storage, follow_up=follow_up),
        follow_up=follow_up,
    )


def resubmitter(db, provider, storage=None, check=None, **kwargs):  # noqa: F811
    # Watchdog 조립과 같다: 공유 제한을 기다리지 않는다(max_wait 0).
    return PrunaResubmitter(
        jobs=db["jobs"],
        state=db["state"],
        precheck=check or FixedCheck(),
        submit=submitter(db, provider, storage, max_wait_seconds=0, **kwargs),
        lease_seconds=60,
        heartbeat_interval_seconds=0,
    )


def first_submit(db, provider, follow_up=None, storage=None):  # noqa: F811
    return request_processor(db, provider, follow_up, storage).process(valid_message())


def make_due(db, job_id=JOB):  # noqa: F811
    """예약 시각이 지난 것처럼(실제로 기다리지 않음). 공유 최소 간격도 비운다."""
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET pruna_retry_at = statement_timestamp() "
        "- interval '1 second' WHERE fitting_job_id = %s AND pruna_retry_at IS NOT NULL",
        (job_id,),
    )
    db["sql"]("UPDATE virtual_fitting_pruna_limiter SET next_send_at = 'epoch'")
    db["sql"]("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")


def attempts(db, job_id=JOB):  # noqa: F811
    return db["sql"](
        "SELECT attempt_no, task_uuid::text, not_accepted_code, not_accepted_source, "
        "not_accepted_at IS NOT NULL, conflict_reason FROM virtual_fitting_pruna_attempts "
        "WHERE fitting_job_id = %s ORDER BY attempt_no",
        (job_id,),
    )


def wait_seconds(db, job_id=JOB):  # noqa: F811
    (row,) = db["sql"](
        "SELECT EXTRACT(EPOCH FROM pruna_retry_at - pruna_not_accepted_at) "
        "FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s",
        (job_id,),
    )
    return None if row[0] is None else float(row[0])


def in_flight(db):  # noqa: F811
    return db["sql"]("SELECT fitting_job_id, task_uuid::text FROM virtual_fitting_pruna_in_flight")


def ready_watchdog(resubmits=True):
    return WatchdogPaths(lambda: SimpleNamespace(alive=True, resubmits=resubmits))


@pytest.fixture
def queues(monkeypatch):
    monkeypatch.setattr(settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(
        settings,
        "AI_FITTING_RESPONSE_QUEUE_URL",
        "https://sqs.ap-northeast-2.amazonaws.com/123456789012/ai-fitting-response",
    )
    monkeypatch.setattr(
        settings,
        "BE_FITTING_RESPONSE_QUEUE_URL",
        "https://sqs.ap-northeast-2.amazonaws.com/123456789012/be-fitting-response",
    )


# --- 1. 제출·Webhook·조회에서 같은 시도의 미접수 멱등 기록 ---


def test_submit_response_records_not_accepted_once_and_frees_the_slot(db, queue_settings):  # noqa: F811
    provider = Scripted(not_accepted())
    first_submit(db, provider)
    current = job(db)
    task = current.task_uuid
    assert (current.status, current.next_action, current.result_payload) == (
        AsyncJobStatus.RECEIVED,
        NextAction.SUBMIT_PRUNA,
        None,
    )
    assert current.pruna_not_accepted_at is not None and current.pruna_retry_at is not None
    assert (
        next_step_for(current) is NextStep.RESUBMIT_PRUNA
    )  # 접수 불명확(recover_unknown)이 아니다
    assert attempts(db) == [(1, task, "serviceOverloaded", "submit_response", True, None)]
    assert in_flight(db) == []
    # 같은 시도의 미접수가 Webhook·조회로 다시 와도 횟수·예약은 그대로
    for event in (
        parse_webhook(error_body(task, code="serviceOverloaded"))[0],
        WebhookEvent(task, EventKind.FAILED, "inferenceRateLimitExceeded", from_status_query=True),
    ):
        try:
            results = service(db).handle([event])
        except WebhookFollowUpNotReadyError as error:
            results = error.results
        assert [r.outcome for r in results] == [EventOutcome.DUPLICATE]
    assert job(db).pruna_retry_at == current.pruna_retry_at
    assert attempts(db) == [(1, task, "serviceOverloaded", "submit_response", True, None)]


def test_webhook_before_the_accept_response_uses_the_same_record(db, queue_settings):  # noqa: F811
    seen = {}

    def during(task):
        try:
            service(db).handle(parse_webhook(error_body(task, code="inferenceRateLimitExceeded")))
        except WebhookFollowUpNotReadyError as error:
            seen["pending"] = error.pending
        seen["state"] = job(db).pruna_not_accepted_at is not None

    provider = Scripted("timeout", during=during)  # 제출 응답은 못 받았다(접수 불명확)
    first_submit(db, provider)
    task = job(db).task_uuid
    assert seen == {"state": True, "pending": [(task, "resubmit_pruna")]}
    assert attempts(db) == [(1, task, "inferenceRateLimitExceeded", "webhook", True, None)]
    assert next_step_for(job(db)) is NextStep.RESUBMIT_PRUNA and in_flight(db) == []


def test_status_query_of_an_unclear_submit_uses_the_same_record(wdb, queue_settings):  # noqa: F811
    first_submit(wdb, Scripted("timeout"))
    task = job(wdb).task_uuid
    assert next_step_for(job(wdb)) is NextStep.RECOVER_UNKNOWN and in_flight(wdb) == [(JOB, task)]
    age(wdb, "pruna_submit_started_at", 200)
    status = TaskStatus(TaskStatusKind.FAILED, task, error_code="serviceOverloaded")
    from tests.test_virtual_fitting_watchdog import FakeStatus

    watchdog(wdb, FakeStatus(status)).run_once()
    assert attempts(wdb) == [(1, task, "serviceOverloaded", "status_query", True, None)]
    assert next_step_for(job(wdb)) is NextStep.RESUBMIT_PRUNA and in_flight(wdb) == []


# --- 2·12. 최초 1회 + 재요청 3회, 소진 결과 저장·발행·중복 방지 ---


def test_four_not_accepted_attempts_end_with_the_submission_rejected_result(
    wdb,  # noqa: F811
    queues,
):
    from tests.test_virtual_fitting_result_publisher import FakeSqs as ResultSqs
    from tests.test_virtual_fitting_result_publisher import publisher

    provider = Scripted(*[not_accepted()] * 6)
    first_submit(wdb, provider)
    w = watchdog(wdb, resubmit=resubmitter(wdb, provider))
    for expected in range(2, PRUNA_SUBMIT_LIMIT + 1):
        make_due(wdb)
        w.run_once()
        assert len(provider.calls) == expected
    tasks = provider.tasks()
    assert len(set(tasks)) == PRUNA_SUBMIT_LIMIT  # 시도마다 새 task, 이전 task 기록 보존
    assert [row[:2] for row in attempts(wdb)] == list(enumerate(tasks, start=1))
    assert all(row[4] for row in attempts(wdb))
    final = job(wdb)
    assert final.task_uuid == tasks[-1]
    assert set(final.result_payload.values()) - {JOB} == REJECTED
    assert set(final.result_payload) == {"fitting_job_id", "status", "error_code", "error_message"}
    assert (final.status, final.next_action, final.result_message_status) == (
        AsyncJobStatus.RECEIVED,
        NextAction.PUBLISH_RESULT,
        ResultMessageStatus.PENDING,
    )
    assert final.fallback_attempt_count == 0 and in_flight(wdb) == []
    # 다섯 번째 전송 없음(예약을 흉내 내도)
    make_due(wdb)
    w.run_once()
    assert resubmitter(wdb, provider)(JOB).startswith("lease_")  # SUBMIT_PRUNA 가 아니다
    assert len(provider.calls) == PRUNA_SUBMIT_LIMIT
    # 기존 결과 전송 경로로 한 번만 발행한다(BE 미차감 판단용 코드)
    sqs = ResultSqs()
    assert publisher(wdb, sqs).publish(JOB).value == "published"
    assert publisher(wdb, sqs).publish(JOB).value != "published"
    assert len(sqs.sent) == 1 and sqs.sent[0][1]["error_code"] == "FITTING_SUBMISSION_REJECTED"
    assert job(wdb).status is AsyncJobStatus.FAILED


# --- 3. 대기: 기본 10·20·40초, 유효한 서버 안내, 표현 불가 ---


def test_default_waits_are_10_20_40_seconds(wdb):  # noqa: F811
    provider = Scripted(*[not_accepted()] * 3)
    first_submit(wdb, provider)
    waits = [wait_seconds(wdb)]
    rs = resubmitter(wdb, provider)
    for _ in range(2):
        make_due(wdb)
        assert rs(JOB) == "sent"
        waits.append(wait_seconds(wdb))
    assert waits == list(PRUNA_RETRY_BACKOFF_SECONDS)


@pytest.mark.parametrize("guidance", [2.0, 90.0, 3600.0])
def test_valid_server_guidance_is_used_instead_of_the_default(db, guidance):  # noqa: F811
    first_submit(db, Scripted(not_accepted(retry_after=guidance)))
    assert wait_seconds(db) == guidance
    left = db["sql"](
        "SELECT EXTRACT(EPOCH FROM next_send_at - now()) FROM virtual_fitting_pruna_limiter"
    )
    assert (
        float(left[0][0]) > guidance - 5
    )  # 공유 대기 시각에도 같은 안내(다른 작업도 그 전에는 보내지 않음)


@pytest.mark.parametrize("fields", [{"retry_after_unrepresentable": True}, {"retry_after": 1e15}])
def test_unrepresentable_guidance_holds_without_shortening(wdb, fields):  # noqa: F811
    provider = Scripted(not_accepted(**fields))
    first_submit(wdb, provider)
    current = job(wdb)
    assert current.pruna_not_accepted_at is not None and current.pruna_retry_at is None
    assert next_step_for(current) is NextStep.OPERATOR_REVIEW and in_flight(wdb) == []
    sink = Sink()
    watchdog(wdb, resubmit=resubmitter(wdb, provider), sink=sink).run_once()
    assert len(provider.calls) == 1 and ("pruna_resubmit_held", "pruna", JOB) in reasons(sink)


def test_invalid_guidance_values_are_rejected_before_any_record(db):  # noqa: F811
    task = first_unclear(db)
    for bad in (-1.0, float("nan"), float("inf"), True, "10"):
        with pytest.raises(ValueError):
            db["state"].record_not_accepted(
                task, "serviceOverloaded", NotAcceptedSource.WEBHOOK, retry_after=bad
            )
    assert [row[4] for row in attempts(db)] == [
        False
    ]  # 1번 시도(전송 시작 기록)만, 미접수 기록 없음


def first_unclear(db):  # noqa: F811
    first_submit(db, Scripted("timeout"))
    return job(db).task_uuid


def test_nothing_is_sent_before_the_scheduled_time(wdb):  # noqa: F811
    provider = Scripted(not_accepted())
    first_submit(wdb, provider)
    rs = resubmitter(wdb, provider)
    actions = [a for r in watchdog(wdb, resubmit=rs).run_once() for a in r.actions]
    assert "resubmit_sent" not in actions
    assert rs(JOB) == "not_sent"  # 직접 불러도 자리 확보 트랜잭션이 예약 시각을 다시 확인한다
    grant = limiter(wdb, interval=0).begin_retry_submission(
        JOB,
        wdb["state"].acquire_lease(JOB, NextAction.SUBMIT_PRUNA, 60).token,
        job(wdb).task_uuid,
        "0b8f8a3e-2c55-4c1e-9d5e-7c0f3b0a9e11",
    )
    assert grant.reason is SubmissionReason.RETRY_WAIT and grant.wait_seconds > 0
    assert len(provider.calls) == 1 and len(attempts(wdb)) == 1


def test_shared_interval_is_respected_at_the_scheduled_time(wdb):  # noqa: F811
    provider = Scripted(not_accepted())
    first_submit(wdb, provider)
    make_due(wdb)
    wdb["sql"]("UPDATE virtual_fitting_pruna_limiter SET next_send_at = now() + interval '1 hour'")
    assert resubmitter(wdb, provider)(JOB) == "not_sent"
    assert len(provider.calls) == 1 and len(attempts(wdb)) == 1  # 기다림만으로 횟수를 쓰지 않음
    assert job(wdb).pruna_retry_at is not None


# --- 4. 재수신·재시작·중복 알림에도 횟수·다음 시각 유지 ---


def test_redelivery_restart_and_duplicates_keep_count_and_time(db, queue_settings):  # noqa: F811
    provider = Scripted(not_accepted())
    first_submit(db, provider)
    before = job(db)
    for attempt in range(2, 4):
        restarted = request_processor(db, provider)  # 새 프로세스·Worker
        outcome = restarted.process(valid_message(message_id=f"m-{attempt}", receive_count=attempt))
        assert outcome.action == "submit_already_started"
        try:
            service(db).handle(
                parse_webhook(error_body(before.task_uuid, code="serviceOverloaded"))
            )
        except WebhookFollowUpNotReadyError:
            pass
    after = job(db)
    assert (after.pruna_retry_at, after.pruna_not_accepted_at, after.task_uuid) == (
        before.pruna_retry_at,
        before.pruna_not_accepted_at,
        before.task_uuid,
    )
    assert len(provider.calls) == 1 and len(attempts(db)) == 1


# --- 5. 대기 중 메시지 인계 후 실제 스케줄러 재개 ---


@pytest.mark.parametrize("guidance", [None, 3600.0])
def test_request_message_is_handed_off_only_to_a_real_resubmit_path(db, guidance):  # noqa: F811
    fields = {} if guidance is None else {"retry_after": guidance}
    kept = first_submit(db, Scripted(not_accepted(**fields)), follow_up=ready_watchdog(False))
    assert kept.delete is False  # 재요청 경로가 없으면 지우지 않는다
    current = job(db)
    assert deletion_allowed(
        current, ready_watchdog(True)
    )  # 긴 대기여도 예약을 DB 가 잇는다(요청 DLQ 로 보내지 않음)
    assert not deletion_allowed(current, WatchdogPaths(lambda: None))


def test_watchdog_resumes_the_scheduled_request_after_the_message_is_gone(wdb):  # noqa: F811
    provider = Scripted(not_accepted(), "accept")
    outcome = first_submit(wdb, provider, follow_up=ready_watchdog(True))
    assert outcome.delete is True
    first_task = job(wdb).task_uuid
    make_due(wdb)
    reports = watchdog(wdb, resubmit=resubmitter(wdb, provider)).run_once()
    assert "resubmit_sent" in [a for r in reports for a in r.actions]
    current = job(wdb)
    assert current.task_uuid != first_task and current.status is AsyncJobStatus.WAITING_PRUNA
    assert (current.pruna_not_accepted_at, current.pruna_retry_at) == (None, None)
    assert next_step_for(current) is NextStep.AWAIT_WEBHOOK
    assert in_flight(wdb) == [(JOB, current.task_uuid)]
    assert [row[:2] for row in attempts(wdb)] == [(1, first_task), (2, current.task_uuid)]


# --- 6. 여러 Worker·Watchdog 경합 ---


def test_competing_resubmitters_send_one_attempt(wdb):  # noqa: F811
    gate = threading.Event()
    provider = Scripted(not_accepted(), "accept", "accept", during=lambda task: gate.wait(5))
    gate.set()
    first_submit(wdb, provider)
    gate.clear()
    make_due(wdb)
    results, barrier = [], threading.Barrier(4)

    def run(callable_):
        barrier.wait()
        results.append(callable_())

    workers = [resubmitter(wdb, provider) for _ in range(2)]
    dogs = [watchdog(wdb, resubmit=resubmitter(wdb, provider)) for _ in range(2)]
    threads = [threading.Thread(target=run, args=(lambda w=w: w(JOB),)) for w in workers]
    threads += [threading.Thread(target=run, args=(d.run_once,)) for d in dogs]
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join(20)
    assert len(provider.calls) == 2  # 최초 1 + 재요청 1
    assert len(attempts(wdb)) == 2


def test_a_started_retry_cannot_be_started_again(wdb):  # noqa: F811
    first_submit(wdb, Scripted(not_accepted()))
    make_due(wdb)
    token = wdb["state"].acquire_lease(JOB, NextAction.SUBMIT_PRUNA, 60).token
    previous = job(wdb).task_uuid
    lim = limiter(wdb, interval=0)
    new = "0b8f8a3e-2c55-4c1e-9d5e-7c0f3b0a9e11"
    assert lim.begin_retry_submission(JOB, token, previous, new).granted
    again = lim.begin_retry_submission(JOB, token, previous, "1b8f8a3e-2c55-4c1e-9d5e-7c0f3b0a9e11")
    assert again.reason is SubmissionReason.STATE_MISMATCH and len(attempts(wdb)) == 2


# --- 7. DB 실패·commit 응답 유실·전송 직전 종료 ---


def test_not_accepted_record_failure_keeps_the_message_and_never_resends(db, monkeypatch):  # noqa: F811
    def fail(self, *args, **kwargs):
        raise FittingDatabaseError("injected")

    monkeypatch.setattr(AsyncJobStateRepository, "record_not_accepted", fail)
    provider = Scripted(not_accepted())
    outcome = first_submit(db, provider, follow_up=ready_watchdog(True))
    assert outcome.delete is False  # 기록이 확인되지 않아 인계하지 않는다
    current = job(db)
    assert current.pruna_not_accepted_at is None and in_flight(db) == [(JOB, current.task_uuid)]
    assert (
        next_step_for(current) is NextStep.RECOVER_UNKNOWN
    )  # 접수 불명확으로 남는다(조회 복구, 재전송 없음)
    monkeypatch.undo()
    request_processor(db, provider).process(valid_message(message_id="m-2", receive_count=2))
    assert len(provider.calls) == 1


def test_retry_start_failure_sends_nothing_and_keeps_the_schedule(wdb, monkeypatch):  # noqa: F811
    from app.virtual_fitting.repositories.pruna_limiter import PrunaLimiter

    provider = Scripted(not_accepted())
    first_submit(wdb, provider)
    before = job(wdb)
    make_due(wdb)

    def fail(self, *args, **kwargs):
        raise FittingDatabaseError("injected")

    monkeypatch.setattr(PrunaLimiter, "begin_retry_submission", fail)
    assert resubmitter(wdb, provider)(JOB) == "error"
    assert len(provider.calls) == 1 and len(attempts(wdb)) == 1
    assert (job(wdb).task_uuid, job(wdb).pruna_not_accepted_at) == (
        before.task_uuid,
        before.pruna_not_accepted_at,
    )


def test_lost_commit_response_is_an_unclear_attempt_never_resent(wdb, monkeypatch):  # noqa: F811
    from app.virtual_fitting.repositories.pruna_limiter import PrunaLimiter

    provider = Scripted(not_accepted())
    first_submit(wdb, provider)
    make_due(wdb)
    real = PrunaLimiter.begin_retry_submission

    def committed_then_lost(self, *args, **kwargs):
        real(self, *args, **kwargs)  # 커밋은 됐다
        raise FittingDatabaseError("connection lost after commit")

    monkeypatch.setattr(PrunaLimiter, "begin_retry_submission", committed_then_lost)
    assert resubmitter(wdb, provider)(JOB) == "error"
    monkeypatch.undo()
    current = job(wdb)
    # 논리적 시도(2번)는 기록됐지만 HTTP 는 보내지 않았다. 보냈는지 추측하지 않고 접수 불명확으로 둔다(재전송 없음).
    assert len(attempts(wdb)) == 2 and len(provider.calls) == 1
    assert next_step_for(current) is NextStep.RECOVER_UNKNOWN
    make_due(wdb)
    watchdog(wdb, resubmit=resubmitter(wdb, provider)).run_once()
    assert resubmitter(wdb, provider)(JOB) == "not_scheduled" and len(provider.calls) == 1


def test_stopped_right_after_the_start_record_is_not_guessed(wdb):  # noqa: F811
    first_submit(wdb, Scripted(not_accepted()))
    make_due(wdb)
    token = wdb["state"].acquire_lease(JOB, NextAction.SUBMIT_PRUNA, 60).token
    new = "0b8f8a3e-2c55-4c1e-9d5e-7c0f3b0a9e11"
    assert (
        limiter(wdb, interval=0).begin_retry_submission(JOB, token, job(wdb).task_uuid, new).granted
    )  # 시작 기록 직후 Worker 종료(전송 여부 모름)
    current = job(wdb)
    assert current.task_uuid == new and next_step_for(current) is NextStep.RECOVER_UNKNOWN
    assert in_flight(wdb) == [(JOB, new)] and len(attempts(wdb)) == 2


def test_certainly_unsent_retry_is_rolled_back_to_the_previous_schedule(wdb):  # noqa: F811
    provider = Scripted(not_accepted(), ValueError("input rejected before sending"))
    first_submit(wdb, provider)
    before = job(wdb)
    make_due(wdb)
    assert resubmitter(wdb, provider)(JOB) == "not_sent"
    after = job(wdb)
    assert (after.task_uuid, after.pruna_not_accepted_at, after.pruna_submit_started_at) == (
        before.task_uuid,
        before.pruna_not_accepted_at,
        before.pruna_submit_started_at,
    )
    assert after.pruna_retry_at is not None and len(attempts(wdb)) == 1 and in_flight(wdb) == []


# --- 8. 미접수 기록·자리 정리 사이 장애의 일관성 ---


def test_record_and_slot_release_roll_back_together(db):  # noqa: F811
    task = first_unclear(db)
    state = AsyncJobStateRepository(db["pool"], lock_timeout_seconds=0.2)
    holder = db["connect"]()
    try:
        holder.execute("SELECT 1 FROM virtual_fitting_pruna_in_flight FOR UPDATE")
        with pytest.raises(FittingDatabaseError):
            state.record_not_accepted(task, "serviceOverloaded", NotAcceptedSource.WEBHOOK)
    finally:
        holder.rollback()
        holder.close()
    current = job(db)
    assert current.pruna_not_accepted_at is None and current.pruna_retry_at is None
    assert in_flight(db) == [(JOB, task)]  # 아무것도 반만 남지 않았다
    assert [row[4] for row in attempts(db)] == [False]
    change = db["state"].record_not_accepted(task, "serviceOverloaded", NotAcceptedSource.WEBHOOK)
    assert change.outcome is NotAcceptedOutcome.SCHEDULED and in_flight(db) == []


# --- 9. 과거 task 의 늦은 알림·비용·모순 결과 ---


def test_late_results_of_a_past_attempt_are_not_applied(wdb, queue_settings):  # noqa: F811
    provider = Scripted(not_accepted(), not_accepted())
    first_submit(wdb, provider)
    make_due(wdb)
    resubmitter(wdb, provider)(JOB)
    old, current = provider.tasks()
    before = job(wdb)
    assert before.task_uuid == current and before.pruna_retry_at is not None
    # 이전 시도의 미접수 반복: 중복, 아무것도 바뀌지 않음
    (dup,) = service(wdb).handle(parse_webhook(error_body(old, code="serviceOverloaded")))
    assert dup.outcome is EventOutcome.DUPLICATE and job(wdb) == before
    # 이전 시도의 성공(모순): 현재 시도에 적용하지 않고, 비용은 그 task 로, 예약된 재요청은 보류(운영 확인)
    (late,) = service(wdb).handle(parse_webhook(success_body(old, cost=0.02)))
    assert late.outcome is EventOutcome.IGNORED
    after = job(wdb)
    assert after.task_uuid == current and after.status is AsyncJobStatus.RECEIVED
    assert after.pruna_retry_at is None and next_step_for(after) is NextStep.OPERATOR_REVIEW
    assert [(t, c) for t, c, _ in task_costs(wdb)] == [(old, Decimal("0.02"))]
    assert attempts(wdb)[0][5] == "result_after_not_accepted" and attempts(wdb)[1][5] is None


def test_success_for_the_current_not_accepted_attempt_is_followed_and_flagged(db, queue_settings):  # noqa: F811
    first_submit(db, Scripted(not_accepted()))
    task = job(db).task_uuid
    (result,) = service(db).handle(parse_webhook(success_body(task)))
    assert result.outcome is EventOutcome.APPLIED
    current = job(db)
    assert current.status is AsyncJobStatus.POSTPROCESSING  # 완성 결과를 따른다(재요청은 멈춤)
    assert next_step_for(current) is NextStep.POSTPROCESS
    assert attempts(db)[0][5] == "result_after_not_accepted"


def test_not_accepted_after_progress_never_reverts_the_job(db, queue_settings):  # noqa: F811
    first_submit(db, Scripted("accept"))
    task = job(db).task_uuid
    service(db).handle(parse_webhook(success_body(task)))
    before = job(db)
    change = db["state"].record_not_accepted(task, "serviceOverloaded", NotAcceptedSource.WEBHOOK)
    assert change.outcome is NotAcceptedOutcome.CONTRADICTED and job(db) == before


# --- 10·11. 중간 접수 성공은 예약 중단, 중간 접수 불명확은 재전송·소진 결과 없음 ---


def test_accepted_retry_stops_any_further_request(wdb):  # noqa: F811
    provider = Scripted(not_accepted(), "accept")
    first_submit(wdb, provider)
    make_due(wdb)
    assert resubmitter(wdb, provider)(JOB) == "sent"
    make_due(wdb)
    watchdog(wdb, resubmit=resubmitter(wdb, provider)).run_once()
    assert resubmitter(wdb, provider)(JOB).startswith("lease_")  # SUBMIT_PRUNA 가 아니다
    assert len(provider.calls) == 2 and job(wdb).status is AsyncJobStatus.WAITING_PRUNA


def test_unclear_attempt_is_recovered_by_query_never_by_a_new_task(wdb, queue_settings):  # noqa: F811
    from tests.test_virtual_fitting_watchdog import FakeStatus

    provider = Scripted(*[not_accepted()] * 3, "timeout")
    first_submit(wdb, provider)
    rs = resubmitter(wdb, provider)
    for _ in range(3):
        make_due(wdb)
        assert rs(JOB) == "sent"
    current = job(wdb)
    assert len(attempts(wdb)) == PRUNA_SUBMIT_LIMIT
    assert current.result_payload is None  # 4회째가 불명확이면 소진 결과를 내지 않는다
    assert next_step_for(current) is NextStep.RECOVER_UNKNOWN
    make_due(wdb)
    age(wdb, "pruna_submit_started_at", 200)
    status = FakeStatus(TaskStatus(TaskStatusKind.PROCESSING, current.task_uuid))
    watchdog(wdb, status, resubmit=rs).run_once()
    assert len(provider.calls) == PRUNA_SUBMIT_LIMIT and status.calls == [current.task_uuid]
    # 같은 task 조회로 확실한 미접수가 확인되면 그때 소진 결과
    wdb["sql"]("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")
    failed = TaskStatus(TaskStatusKind.FAILED, current.task_uuid, error_code="serviceOverloaded")
    watchdog(wdb, FakeStatus(failed), resubmit=rs).run_once()
    assert job(wdb).result_payload["error_code"] == "FITTING_SUBMISSION_REJECTED"
    assert len(provider.calls) == PRUNA_SUBMIT_LIMIT


def test_unclear_middle_attempt_does_not_schedule(wdb):  # noqa: F811
    provider = Scripted(not_accepted(), "timeout")
    first_submit(wdb, provider)
    make_due(wdb)
    assert resubmitter(wdb, provider)(JOB) == "sent"
    current = job(wdb)
    assert current.pruna_not_accepted_at is None and current.pruna_retry_at is None
    assert resubmitter(wdb, provider)(JOB) == "not_scheduled" and len(provider.calls) == 2


# --- 13. AI 저장·기존 직접 업로드 작업의 계약 유지 ---


def test_ai_storage_job_keeps_its_contract_on_every_attempt(wdb):  # noqa: F811
    storage = FakeStorage()
    provider = Scripted(not_accepted(), "accept")
    first_submit(wdb, provider, storage=storage)
    key = job(wdb).async_image_key
    make_due(wdb)
    resubmitter(wdb, provider, storage)(JOB)
    assert [(c["upload"], c["ai"], c["format"]) for c in provider.calls] == [
        (None, True, "WEBP")
    ] * 2
    assert job(wdb).async_image_key == key and job(wdb).ai_stores_result is True
    assert [kind for kind, _ in storage.calls] == ["GET", "GET"]  # 결과 PUT URL 은 발급하지 않음


def test_direct_upload_job_keeps_its_key_and_put_on_every_attempt(wdb, monkeypatch):  # noqa: F811
    from tests.v2_integration_harness import prepared_by_previous_version

    prepared_by_previous_version(monkeypatch, JOB)
    storage = FakeStorage()
    provider = Scripted(not_accepted(), "accept")
    first_submit(wdb, provider, storage=storage)
    key = job(wdb).async_image_key
    make_due(wdb)
    resubmitter(wdb, provider, storage)(JOB)
    assert [c["ai"] for c in provider.calls] == [False, False]
    assert all(c["upload"] and key in c["upload"] for c in provider.calls)
    assert [k for kind, k in storage.calls if kind == "PUT"] == [
        key,
        key,
    ]  # 시도 직전에 기존 방식으로 발급
    assert job(wdb).async_image_key == key and job(wdb).ai_stores_result is False


# --- 14. 인증·잔액·생성 실패 정책 유지 ---


@pytest.mark.parametrize(
    ("step", "check"),
    [
        (rejected("auth_error", 401), lambda j: j.auth_hold_task_uuid == j.task_uuid),
        (
            rejected("balance_exhausted", 402),
            lambda j: j.result_payload["error_code"] == "FITTING_BALANCE_EXHAUSTED",
        ),
        (
            rejected("generation_fallback", 200, ("timeoutProvider",)),
            lambda j: j.next_action is NextAction.RUN_FALLBACK,
        ),
    ],
)
def test_retry_keeps_auth_balance_and_fallback_policies(wdb, step, check):  # noqa: F811
    provider = Scripted(not_accepted(), step, "accept")
    first_submit(wdb, provider)
    make_due(wdb)
    assert resubmitter(wdb, provider)(JOB) == "sent"
    current = job(wdb)
    assert check(current) and current.pruna_retry_at is None
    make_due(wdb)
    watchdog(wdb, resubmit=resubmitter(wdb, provider)).run_once()
    assert len(provider.calls) == 2  # 같은 자격증명·실패로 자동 반복하지 않음


def test_retry_precheck_failure_saves_its_result_without_sending(wdb):  # noqa: F811
    from app.virtual_fitting.request_worker import PrecheckOutcome, PrecheckStatus

    provider = Scripted(not_accepted())
    first_submit(wdb, provider)
    make_due(wdb)
    failed = FixedCheck(PrecheckOutcome(PrecheckStatus.FAILED, "PRODUCT_NOT_FOUND", "상품 없음"))
    assert resubmitter(wdb, provider, check=failed)(JOB) == "precheck_failed_PRODUCT_NOT_FOUND"
    assert job(wdb).result_payload["error_code"] == "PRODUCT_NOT_FOUND"
    assert resubmitter(wdb, provider)(JOB).startswith(
        "lease_"
    )  # 결과가 저장돼 재요청 대상이 아니다
    assert len(provider.calls) == 1


def test_retry_precheck_postponement_keeps_the_schedule(wdb):  # noqa: F811
    from app.virtual_fitting.request_worker import PrecheckOutcome, PrecheckStatus

    provider = Scripted(not_accepted())
    first_submit(wdb, provider)
    make_due(wdb)
    before = job(wdb)
    retry = FixedCheck(PrecheckOutcome(PrecheckStatus.RETRY, reason="budget_lookup"))
    assert resubmitter(wdb, provider, check=retry)(JOB) == "precheck_retry_budget_lookup"
    assert job(wdb).pruna_retry_at == before.pruna_retry_at and len(provider.calls) == 1


def test_watchdog_without_a_resubmit_path_alerts_and_sends_nothing(wdb):  # noqa: F811
    provider = Scripted(not_accepted())
    first_submit(wdb, provider)
    make_due(wdb)
    sink = Sink()
    watchdog(wdb, sink=sink).run_once()
    assert ("pruna_resubmit_not_connected", "pruna", JOB) in reasons(sink)
    assert len(provider.calls) == 1


def test_legacy_started_job_counts_as_one_attempt(db):  # noqa: F811
    task = first_unclear(db)
    db["sql"](
        "DELETE FROM virtual_fitting_pruna_attempts"
    )  # 표 생성 전에 시작한 작업(시도 줄 없음)
    change = db["state"].record_not_accepted(task, "serviceOverloaded", NotAcceptedSource.WEBHOOK)
    assert (change.outcome, change.attempt_no) == (NotAcceptedOutcome.SCHEDULED, 1)
    assert wait_seconds(db) == PRUNA_RETRY_BACKOFF_SECONDS[0]


# --- 실제 조립(create_request_worker·create_watchdog·Router) 경로 ---


def _harness_due(env):  # noqa: F811
    env.sql(
        "UPDATE virtual_fitting_async_jobs SET pruna_retry_at = statement_timestamp() "
        "- interval '1 second' WHERE pruna_retry_at IS NOT NULL"
    )
    env.sql("UPDATE virtual_fitting_watchdog_checks SET next_check_at = now()")


def _overloaded(submission):
    return ("errors", [{"code": "serviceOverloaded", "taskUUID": submission.task_uuid}])


def test_assembled_path_resubmits_after_hand_off_and_completes(env):  # noqa: F811
    from tests.test_virtual_fitting_v2_scenarios import client, finish
    from tests.v2_integration_harness import Stack, post_webhook

    stack = Stack(env).mark_running()
    assert stack.watchdog.resubmits  # create_watchdog 이 같은 Submitter 로 재요청 경로를 조립했다
    env.runware.async_modes.extend([_overloaded, _overloaded])  # 그 다음은 접수
    env.send_request(9101)
    (outcome,) = stack.request_once()
    assert outcome.delete is True  # 미접수·예약 기록 + 실제 재요청 경로(Watchdog) → 인계
    assert (
        env.sqs.live("be-fitting-request") == []
    )  # 요청 메시지는 지웠다(긴 대기로 DLQ 에 가지 않음)
    for _ in range(2):
        _harness_due(env)
        stack.watchdog_once()
    tasks = [task for kind, task in env.runware.calls if kind == "async_submit"]
    assert len(tasks) == 3 and len(set(tasks)) == 3
    current = env.job(9101)
    assert current.task_uuid == tasks[-1] and current.status is AsyncJobStatus.WAITING_PRUNA
    assert post_webhook(client(), env.runware.complete(tasks[-1])).status_code == 200
    finish(env, stack)
    final = env.job(9101)
    assert final.status is AsyncJobStatus.COMPLETED and final.fallback_attempt_count == 0
    assert env.sqs.bodies("be-fitting-response")[0]["status"] == "COMPLETED"


def test_assembled_path_publishes_submission_rejected_after_four_attempts(env):  # noqa: F811
    from tests.v2_integration_harness import Stack

    stack = Stack(env).mark_running()
    env.runware.async_modes.extend([_overloaded] * 5)
    env.send_request(9101)
    stack.request_once()
    for _ in range(5):
        _harness_due(env)
        stack.watchdog_once()
    assert env.runware.count("async_submit") == PRUNA_SUBMIT_LIMIT
    assert env.runware.count("sync_generate") == 0  # 소진만으로 fallback 하지 않음
    env.sql(
        "UPDATE virtual_fitting_async_jobs SET worker_lease_token = NULL, worker_lease_until = NULL"
    )
    stack.publish_once()
    stack.publish_once()
    (sent,) = env.sqs.bodies("be-fitting-response")
    assert sent == {
        "fitting_job_id": 9101,
        "status": "FAILED",
        "error_code": "FITTING_SUBMISSION_REJECTED",
        "error_message": "이미지 생성 요청이 접수되지 않았습니다.",
    }
    assert env.job(9101).status is AsyncJobStatus.FAILED


# --- 최종 점검: Webhook·getResponse 의 task 오류 대기 안내(retryAfter) ---


def _webhook_error(task, code="serviceOverloaded", retry_after=None):
    import json

    item = {"taskUUID": task, "code": code, "message": "x"}
    if retry_after is not None:
        item["retryAfter"] = retry_after
    return json.dumps({"errors": [item]}).encode()


def _handle(db, body):  # noqa: F811
    try:
        return service(db).handle(parse_webhook(body))
    except WebhookFollowUpNotReadyError as error:
        return error.results


@pytest.mark.parametrize(
    ("guidance", "expected"),
    [(90, 90.0), (2.5, 2.5), (None, PRUNA_RETRY_BACKOFF_SECONDS[0]), (-5, 10.0), ("90", 10.0)],
)
def test_webhook_task_error_guidance_reaches_the_schedule(db, queue_settings, guidance, expected):  # noqa: F811
    task = first_unclear(db)
    _handle(db, _webhook_error(task, retry_after=guidance))
    assert wait_seconds(db) == expected  # 무효하거나 없으면 기본 대기


def test_webhook_unrepresentable_guidance_holds(db, queue_settings):  # noqa: F811
    task = first_unclear(db)
    _handle(db, _webhook_error(task).replace(b'"x"}', b'"x", "retryAfter": 1e400}'))
    current = job(db)
    assert current.pruna_not_accepted_at is not None and current.pruna_retry_at is None
    assert next_step_for(current) is NextStep.OPERATOR_REVIEW


def test_duplicate_webhook_guidance_does_not_move_the_schedule(db, queue_settings):  # noqa: F811
    task = first_unclear(db)
    _handle(db, _webhook_error(task, retry_after=90))
    first = job(db).pruna_retry_at
    _handle(db, _webhook_error(task, code="inferenceRateLimitExceeded", retry_after=900))
    assert job(db).pruna_retry_at == first and wait_seconds(db) == 90.0


def test_get_response_task_error_guidance_reaches_the_schedule(wdb, queue_settings):  # noqa: F811
    import json

    from app.virtual_fitting.providers.runware_status import parse_status

    task = first_unclear(wdb)
    age(wdb, "pruna_submit_started_at", 200)
    body = json.dumps(
        {"data": [{"taskUUID": task, "status": "error", "code": "serviceOverloaded", "retryAfter": 120}]}
    ).encode()

    class Query:
        def query(self, task_uuid):
            return parse_status(body, task_uuid, 200)

    watchdog(wdb, Query()).run_once()
    assert attempts(wdb)[0][3] == "status_query" and wait_seconds(wdb) == 120.0


def test_get_response_http_retry_after_is_not_task_guidance(wdb, queue_settings):  # noqa: F811
    import io
    from urllib.error import HTTPError

    from app.virtual_fitting.providers.runware_status import RunwareTaskStatusClient

    first_unclear(wdb)
    age(wdb, "pruna_submit_started_at", 200)

    def opener(request, timeout):
        raise HTTPError("https://api.runware.ai/v1", 429, "x", {"Retry-After": "600"}, io.BytesIO(b"{}"))

    client = RunwareTaskStatusClient("k", opener=opener)
    watchdog(wdb, client).run_once()
    current = job(wdb)
    # 조회 요청 자체의 429·Retry-After 는 생성 task 의 미접수 근거·재요청 예약이 아니다
    assert current.pruna_not_accepted_at is None and next_step_for(current) is NextStep.RECOVER_UNKNOWN
    assert [row[4] for row in attempts(wdb)] == [False]
