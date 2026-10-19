"""15단계 보완: Runware 인증 오류 보류를 Webhook·요청 Worker·Watchdog 이 같은 DB 기록으로 같은 단계(operator_review)로 판단.

실제 Router(app.main.app)·요청 Worker(RequestProcessor + PrunaSubmitter)·Repository·FollowUpPaths 조립(follow_up_paths)을 쓴다.
Runware·SQS·S3 는 가짜, DB 는 격리 PostgreSQL 이다.
"""

import pytest

from app.config import settings
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.providers.runware_async import SubmissionOutcome
from app.virtual_fitting.pruna_submitter import NextStep, deletion_allowed, next_step_for
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobRepository,
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import (
    AsyncJobStateRepository,
    ChangeReason,
)
from app.virtual_fitting.result_publisher import register_result_publisher
from app.virtual_fitting.watchdog import follow_up_paths, register_watchdog
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_pruna_submit import (  # noqa: F401 (pytest fixture)
    JOB,
    FakeProvider,
    build,
    db,
    job,
)
from tests.test_virtual_fitting_request_worker import valid_message
from tests.test_virtual_fitting_result_publisher import QUEUE_URL, FakeSqs, publisher
from tests.test_virtual_fitting_watchdog import (
    AI_QUEUE,
    WATCHDOG_SQL,
    FakeStatus,
    Sink,
    age,
    clear_lease,
    in_flight,
    unknown,
    waiting,
)
from tests.test_virtual_fitting_watchdog import watchdog as make_watchdog
from tests.test_virtual_fitting_webhook import (  # noqa: F401 (pytest fixture)
    api,
    error_body,
    post,
    queue_settings,
    submitted,
    success_body,
    task_costs,
)


class Alive:
    alive = True


@pytest.fixture(autouse=True)
def loops(monkeypatch):
    """두 루프(결과 전송·Watchdog)가 모두 실행 중이고 설정도 올바르다 — 가장 쉽게 '준비됨'으로 잘못 판단할 수 있는 조건."""
    monkeypatch.setattr(settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(settings, "BE_FITTING_RESPONSE_QUEUE_URL", QUEUE_URL)
    monkeypatch.setattr(settings, "AI_FITTING_RESPONSE_QUEUE_URL", AI_QUEUE)
    register_result_publisher(Alive())
    register_watchdog(Alive())
    yield
    register_result_publisher(None)
    register_watchdog(None)


def hold(db, job_id=JOB):  # noqa: F811
    current = db["jobs"].find_by_fitting_job_id(job_id)
    return current.auth_hold_task_uuid, current.auth_hold_reason


def redeliver(db, provider=None):  # noqa: F811
    """같은 요청 메시지 재수신(실제 요청 Worker 조립과 같은 준비 판단 follow_up_paths())."""
    provider = provider or FakeProvider()
    processor, _ = build(db, provider, follow_up=follow_up_paths())
    return processor.process(valid_message()), provider


def snapshot(db, job_id=JOB):  # noqa: F811
    current = db["jobs"].find_by_fitting_job_id(job_id)
    return (
        current.status,
        current.next_action,
        current.result_payload,
        current.result_image_key,
        current.result_message_status,
        current.pruna_submit_started_at,
        current.fallback_attempt_count,
        current.llm_generation_attempt_count,
        current.s3_check_attempt_count,
        current.result_message_attempt_count,
    )


# --- Webhook: 정상 토큰 + invalidApiKey ---


def test_webhook_auth_error_records_a_hold_and_stays_503(api):  # noqa: F811
    db_ = api["db"]
    task = submitted(db_)
    response = post(api, error_body(task, code="invalidApiKey"))
    assert response.status_code == 503
    assert response.json()["data"]["pending"][0]["next_step"] == "operator_review"
    current = job(db_)
    assert hold(db_) == (task, "invalidApiKey") and current.auth_hold_at is not None
    # 사용자용 실패 결과·fallback 없음, 상태·next_action 그대로
    assert (current.status, current.next_action, current.result_payload) == (
        AsyncJobStatus.WAITING_PRUNA,
        None,
        None,
    )
    assert current.fallback_attempt_count == 0 and api["boto"].bodies == []
    assert next_step_for(current) is NextStep.OPERATOR_REVIEW
    assert not deletion_allowed(current, follow_up_paths())  # 두 루프가 돌아도
    assert in_flight(db_) == []  # 그 task 자리는 기존 정책대로 정리


def test_webhook_token_failure_records_no_hold(api):  # noqa: F811
    task = submitted(api["db"])
    response = post(api, error_body(task, code="invalidApiKey"), token="wrong")
    assert response.status_code == 401
    assert hold(api["db"]) == (None, None)


def test_hold_record_failure_is_not_reported_as_held(api, monkeypatch):  # noqa: F811
    db_ = api["db"]
    task = submitted(db_)

    def fail(self, task_uuid, reason):
        raise FittingDatabaseError("injected")

    original = AsyncJobStateRepository.record_auth_hold
    monkeypatch.setattr(AsyncJobStateRepository, "record_auth_hold", fail)
    response = post(api, error_body(task, code="invalidApiKey"))
    assert response.status_code == 500 and response.json()["code"] == "DATABASE_ERROR"
    # 보류 완료로 응답하지 않았고, 그 뒤 단계(비용·자리 정리)도 하지 않았다 → 재전송 때 다시 처리
    assert hold(db_) == (None, None) and in_flight(db_) == [(JOB, task)]
    monkeypatch.setattr(AsyncJobStateRepository, "record_auth_hold", original)
    assert post(api, error_body(task, code="invalidApiKey")).status_code == 503
    assert hold(db_) == (task, "invalidApiKey") and in_flight(db_) == []


# --- 요청 Worker 재수신: 같은 기록으로 operator_review ---


def test_request_redelivery_after_webhook_hold_is_kept_without_resending(api):  # noqa: F811
    db_ = api["db"]
    task = submitted(db_)
    post(api, error_body(task, code="invalidApiKey"))
    clear_lease(db_)  # 요청 Worker 는 처리 뒤 실행 권한을 반납한다
    before = snapshot(db_)
    outcome, provider = redeliver(db_)
    assert outcome.delete is False  # Watchdog·결과 전송 루프가 돌아도 지우지 않는다
    assert provider.calls == []  # Pruna 재호출 없음
    assert snapshot(db_) == before and hold(db_) == (task, "invalidApiKey")


def test_hold_survives_a_restart(api):  # noqa: F811
    db_ = api["db"]
    task = submitted(db_)
    post(api, error_body(task, code="invalidApiKey"))
    clear_lease(db_)
    # 프로세스 재시작 가정: 메모리 상태 없이 Repository·Processor 를 새로 만든다(같은 DB).
    fresh = {
        **db_,
        "jobs": AsyncJobRepository(db_["pool"]),
        "state": AsyncJobStateRepository(db_["pool"]),
    }
    outcome, provider = redeliver(fresh)
    assert outcome.delete is False and provider.calls == []
    assert next_step_for(fresh["jobs"].find_by_fitting_job_id(JOB)) is NextStep.OPERATOR_REVIEW
    assert post(api, error_body(task, code="invalidApiKey")).status_code == 503


def test_duplicate_auth_errors_and_a_late_accept_keep_the_hold_once(api):  # noqa: F811
    db_ = api["db"]
    task = submitted(db_, accepted=False)
    for _ in range(3):
        assert post(api, error_body(task, code="invalidApiKey")).status_code == 503
    first = job(db_)
    assert db_["state"].record_submission_accepted(task).applied  # 늦은 접수 응답
    current = job(db_)
    assert current.status is AsyncJobStatus.WAITING_PRUNA
    assert (current.auth_hold_task_uuid, current.auth_hold_reason, current.auth_hold_at) == (
        first.auth_hold_task_uuid,
        first.auth_hold_reason,
        first.auth_hold_at,
    )  # 처음 기록 그대로
    assert next_step_for(current) is NextStep.OPERATOR_REVIEW
    assert db_["state"].record_auth_hold(task, "invalidApiKey").reason is (
        ChangeReason.ALREADY_RECORDED
    )
    assert task_costs(db_) == [] and in_flight(db_) == []  # 비용 0 기록 없음, 자리는 한 번 정리
    clear_lease(db_)
    outcome, provider = redeliver(db_)
    assert outcome.delete is False and provider.calls == []


# --- 8단계 제출 응답 401·403 도 같은 기준 ---


@pytest.mark.parametrize("status", [401, 403])
def test_submit_auth_rejection_records_the_same_hold(db, status):  # noqa: F811
    provider = FakeProvider(SubmissionOutcome.REJECTED, reason="client_error", http_status=status)
    processor, _ = build(db, provider, follow_up=follow_up_paths())
    first = processor.process(valid_message())
    current = job(db)
    assert first.delete is False
    assert hold(db) == (current.task_uuid, f"http_{status}")
    assert (current.status, current.next_action, current.result_payload) == (
        AsyncJobStatus.RECEIVED,
        NextAction.SUBMIT_PRUNA,
        None,
    )
    assert next_step_for(current) is NextStep.OPERATOR_REVIEW  # recover_unknown 으로 우회하지 않음
    clear_lease(db)
    again, provider_again = redeliver(db)
    assert again.delete is False and provider_again.calls == [] and len(provider.calls) == 1


def test_submit_hold_record_failure_keeps_the_message(db, monkeypatch):  # noqa: F811
    def fail(self, task_uuid, reason):
        raise FittingDatabaseError("injected")

    monkeypatch.setattr(AsyncJobStateRepository, "record_auth_hold", fail)
    provider = FakeProvider(SubmissionOutcome.REJECTED, reason="client_error", http_status=401)
    processor, _ = build(db, provider, follow_up=follow_up_paths())
    assert processor.process(valid_message()).delete is False


# --- 일반 작업·다른 시도·최종 결과는 그대로 ---


def test_normal_waiting_and_unknown_jobs_keep_their_readiness(db):  # noqa: F811
    db["sql"](WATCHDOG_SQL.read_text(encoding="utf-8"))
    waiting(db, 10)
    current = job(db)
    assert next_step_for(current) is NextStep.AWAIT_WEBHOOK
    assert deletion_allowed(current, follow_up_paths())
    outcome, provider = redeliver(db)
    assert outcome.delete is True and provider.calls == []
    unknown(db, 10, job_id=2)
    other = db["jobs"].find_by_fitting_job_id(2)
    assert next_step_for(other) is NextStep.RECOVER_UNKNOWN
    assert deletion_allowed(other, follow_up_paths())


def test_hold_belongs_to_one_job_and_attempt(api):  # noqa: F811
    db_ = api["db"]
    task = submitted(db_)
    other = submitted(db_, job_id=2)
    post(api, error_body(task, code="invalidApiKey"))
    assert hold(db_) == (task, "invalidApiKey") and hold(db_, 2) == (None, None)
    assert in_flight(db_) == [(2, other)]  # 다른 작업의 자리는 그대로


def test_auth_error_after_fallback_started_records_no_hold(api):  # noqa: F811
    db_ = api["db"]
    task = submitted(db_)
    post(api, error_body(task))  # timeoutProvider → RUN_FALLBACK
    assert job(db_).next_action is NextAction.RUN_FALLBACK
    response = post(
        api, error_body(task, code="invalidApiKey")
    )  # 늦은 인증 오류(이미 다른 경로로 진행)
    assert response.status_code == 200
    assert hold(db_) == (None, None) and job(db_).next_action is NextAction.RUN_FALLBACK


def test_late_auth_error_after_publish_keeps_the_result_without_a_hold(api):  # noqa: F811
    db_ = api["db"]
    task = submitted(db_)
    post(api, error_body(task, code="someNewCode"))
    clear_lease(db_)
    assert publisher(db_, FakeSqs()).publish(JOB).value == "published"
    final = job(db_)
    assert final.result_message_status is ResultMessageStatus.PUBLISHED
    for loops_on in (True, False):
        register_result_publisher(Alive() if loops_on else None)
        register_watchdog(Alive() if loops_on else None)
        response = post(api, error_body(task, code="invalidApiKey"))
        assert response.status_code == 200  # 안전한 발행 완료 중복
        assert job(db_) == final and hold(db_) == (None, None)


def test_success_after_a_hold_keeps_the_record_but_follows_the_confirmed_result(api):  # noqa: F811
    # 같은 task 의 확인된 성공이 뒤에 오면 그 결과로 진행한다(보류 기록은 지우지 않고 남긴다).
    db_ = api["db"]
    task = submitted(db_)
    post(api, error_body(task, code="invalidApiKey"))
    assert post(api, success_body(task)).status_code == 200
    current = job(db_)
    assert current.next_action is NextAction.POSTPROCESS and hold(db_) == (task, "invalidApiKey")
    assert next_step_for(current) is NextStep.POSTPROCESS


# --- Watchdog: 찾아서 알리기만 ---


def test_watchdog_alerts_on_a_hold_without_querying_or_releasing(api):  # noqa: F811
    db_ = api["db"]
    db_["sql"](WATCHDOG_SQL.read_text(encoding="utf-8"))
    task = submitted(db_)
    post(api, error_body(task, code="invalidApiKey"))
    clear_lease(db_)
    age(db_, "runware_accepted_at", 400)
    before = snapshot(db_)
    status, sink = FakeStatus(), Sink()
    (report,) = make_watchdog(db_, status, sink=sink).run_once()
    assert report.actions == ["auth_hold"] and status.calls == []
    assert [a.reason for a in sink.alerts] == ["pruna_auth_hold_invalidApiKey"]
    assert snapshot(db_) == before and hold(db_) == (
        task,
        "invalidApiKey",
    )  # 해제·정상 대기 전환 없음
    assert not deletion_allowed(job(db_), follow_up_paths())
