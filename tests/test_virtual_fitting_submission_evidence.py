"""접수 근거가 불완전하거나 모순될 때 재생성을 막는 회귀 검증. 외부 응답은 가짜, DB는 격리 PostgreSQL."""

import json

import pytest

from app.virtual_fitting.providers.runware_async import SubmissionOutcome, classify_response
from app.virtual_fitting.pruna_submitter import NextStep, next_step_for
from app.virtual_fitting.repositories.async_job_repository import AsyncJobStatus, NextAction
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401
from tests.test_virtual_fitting_pruna_retry import (
    Scripted,
    _handle,
    _webhook_error,
    attempts,
    first_submit,
    first_unclear,
    make_due,
    not_accepted,
    resubmitter,
    wait_seconds,
)
from tests.test_virtual_fitting_pruna_submit import JOB, db, job  # noqa: F401
from tests.test_virtual_fitting_watchdog import age, watchdog, wdb  # noqa: F401
from tests.test_virtual_fitting_webhook import queue_settings  # noqa: F401

TASK = "a770f077-f413-47de-9dac-be0b26a35da6"
OTHER = "0b8f8a3e-2c55-4c1e-9d5e-7c0f3b0a9e11"


@pytest.mark.parametrize("code", ["serviceOverloaded", "timeoutProvider"])
@pytest.mark.parametrize("missing", [None, "", 0, {}, []])
def test_missing_code_cannot_be_discarded_from_evidence(code, missing):
    body = json.dumps(
        {
            "errors": [
                {"taskUUID": TASK, "code": code},
                {"taskUUID": TASK, "code": missing},
            ]
        }
    ).encode()
    result = classify_response(200, body, TASK)
    assert result.outcome is SubmissionOutcome.UNKNOWN and not result.retryable


@pytest.mark.parametrize("code", ["serviceOverloaded", "timeoutProvider"])
@pytest.mark.parametrize("status", ["processing", "success"])
def test_contradictory_submit_response_cannot_authorize_generation(code, status):
    body = json.dumps(
        {
            "data": [{"taskUUID": TASK, "status": status}],
            "errors": [{"taskUUID": TASK, "code": code}],
        }
    ).encode()
    result = classify_response(200, body, TASK)
    assert result.outcome is SubmissionOutcome.UNKNOWN and not result.retryable


def test_rejection_history_uses_only_current_task_error(db):  # noqa: F811 (pytest fixtures)
    def response(task):
        return classify_response(
            503,
            json.dumps(
                {
                    "errors": [
                        {"taskUUID": OTHER, "code": "serviceOverloaded"},
                        {"taskUUID": task, "code": "inferenceRateLimitExceeded"},
                    ]
                }
            ).encode(),
            task,
        )

    first_submit(db, Scripted(response))
    assert attempts(db)[0][2] == "inferenceRateLimitExceeded"


def test_processing_after_rejection_holds_current_retry(db, queue_settings):  # noqa: F811 (pytest fixtures)
    first_submit(db, Scripted(not_accepted()))
    task = job(db).task_uuid
    _handle(
        db,
        json.dumps(
            {"data": [{"taskType": "imageInference", "taskUUID": task, "status": "processing"}]}
        ).encode(),
    )
    assert job(db).pruna_retry_at is None
    assert next_step_for(job(db)) is NextStep.OPERATOR_REVIEW


def test_processing_for_past_rejection_holds_next_retry(wdb, queue_settings):  # noqa: F811 (pytest fixtures)
    provider = Scripted(not_accepted(), not_accepted())
    first_submit(wdb, provider)
    make_due(wdb)
    assert resubmitter(wdb, provider)(JOB) == "sent"
    old, current = provider.tasks()
    _handle(
        wdb,
        json.dumps(
            {"data": [{"taskType": "imageInference", "taskUUID": old, "status": "processing"}]}
        ).encode(),
    )
    assert job(wdb).task_uuid == current and job(wdb).pruna_retry_at is None


def test_processing_query_after_rejection_holds_next_retry(wdb, queue_settings):  # noqa: F811 (pytest fixtures)
    from app.virtual_fitting.providers.runware_status import TaskStatus, TaskStatusKind
    from app.virtual_fitting.watchdog import CheckReport
    from tests.test_virtual_fitting_watchdog import watchdog

    first_submit(wdb, Scripted(not_accepted()))

    class Query:
        def query(self, task_uuid):
            return TaskStatus(TaskStatusKind.PROCESSING, task_uuid)

    watchdog(wdb, Query())._query(job(wdb), CheckReport(JOB), "slot")
    assert job(wdb).pruna_retry_at is None


def test_conflicting_processing_and_rejection_webhook_does_not_schedule(db, queue_settings):  # noqa: F811 (pytest fixtures)
    task = first_unclear(db)
    body = json.loads(_webhook_error(task))
    body["data"] = [{"taskType": "imageInference", "taskUUID": task, "status": "processing"}]
    _handle(db, json.dumps(body).encode())
    assert job(db).pruna_not_accepted_at is None and job(db).pruna_retry_at is None


@pytest.mark.parametrize("mode", ["accept", "timeout"])
@pytest.mark.parametrize("query_kind", ["processing", "unknown", "not_found"])
def test_two_hours_alone_never_fail_or_regenerate(wdb, mode, query_kind):  # noqa: F811
    from app.virtual_fitting.providers.runware_status import TaskStatus, TaskStatusKind

    provider = Scripted(mode)
    first_submit(wdb, provider)
    before = job(wdb)
    for column in ("created_at", "requested_at", "pruna_submit_started_at", "runware_accepted_at"):
        if getattr(before, column) is not None:
            age(wdb, column, 7200)

    class Query:
        calls = 0

        def query(self, task_uuid):
            self.calls += 1
            return TaskStatus(TaskStatusKind(query_kind), task_uuid)

    query = Query()
    watchdog(wdb, query).run_once()
    current = job(wdb)
    assert current.status is before.status and current.next_action is before.next_action
    assert current.result_payload is None and current.fallback_attempt_count == 0
    assert current.pruna_not_accepted_at is None and current.task_uuid == before.task_uuid
    assert len(provider.calls) == 1 and query.calls >= 1


def test_confirmed_timeout_can_fall_back_even_after_two_hours(wdb):  # noqa: F811
    from app.virtual_fitting.providers.runware_status import TaskStatus, TaskStatusKind

    first_submit(wdb, Scripted("accept"))
    for column in ("created_at", "requested_at", "pruna_submit_started_at", "runware_accepted_at"):
        age(wdb, column, 7200)

    class Query:
        def query(self, task_uuid):
            return TaskStatus(TaskStatusKind.FAILED, task_uuid, error_code="timeoutProvider")

    watchdog(wdb, Query()).run_once()
    current = job(wdb)
    assert current.status is AsyncJobStatus.FALLBACK_RUNNING
    assert current.next_action is NextAction.RUN_FALLBACK and current.result_payload is None


@pytest.mark.parametrize(
    "guidance, expected", [(120, 120.0), (None, 10.0), (-1, 10.0), ("90", 10.0)]
)
def test_query_guidance_and_duplicate_schedule(wdb, guidance, expected):  # noqa: F811
    from app.virtual_fitting.providers.runware_status import parse_status
    from app.virtual_fitting.watchdog import CheckReport

    task = first_unclear(wdb)
    body = _webhook_error(task, retry_after=guidance)

    class Query:
        def query(self, task_uuid):
            return parse_status(body, task_uuid, 200)

    worker = watchdog(wdb, Query())
    worker._query(job(wdb), CheckReport(JOB), "submit_unknown")
    first = job(wdb).pruna_retry_at
    assert wait_seconds(wdb) == expected
    body = _webhook_error(task, retry_after=900)
    worker._query(job(wdb), CheckReport(JOB), "submit_unknown")
    assert job(wdb).pruna_retry_at == first and wait_seconds(wdb) == expected
