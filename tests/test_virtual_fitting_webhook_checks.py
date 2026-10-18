"""9단계 보완: (1) 영구 실패 저장 뒤 후속 경로 준비 여부, (2) Runware 인증 오류와 사용자용 모델 실패의 분리.

처음에는 "지금 이렇게 동작한다"는 관찰값(영구 실패에도 200, invalidApiKey 도 FITTING_MODEL_FAILED)을 담았던 파일이다.
지금은 보완한 요구사항을 검증한다. 실제 app.main.app → webhook_router → RunwareWebhookService → Repository(격리
PostgreSQL)를 쓰고, 외부 SQS 만 가짜 boto3 client 로 바꾼다. 준비된 후속 경로는 테스트에서만 주입한다
(실제 조립에는 결과 발행 14단계·Watchdog 15단계가 없다).
"""

import inspect
import logging
from decimal import Decimal

import pytest

from app.config import settings
from app.virtual_fitting import runware_webhook, webhook_router
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.pruna_submitter import NextStep, next_step_for
from app.virtual_fitting.repositories.async_job_repository import AsyncJobStatus, NextAction
from app.virtual_fitting.repositories.pruna_limiter import TaskCostRepository
from app.virtual_fitting.runware_webhook import AUTH_ERROR_CODES, FALLBACK_ERROR_CODES
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_pruna_submit import JOB, db, job  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_webhook import (  # noqa: F401 (pytest fixture)
    TOKEN,
    api,
    error_body,
    in_flight,
    post,
    queue_settings,
    submitted,
    success_body,
    task_costs,
)

MODEL_FAILED = {
    "error_code": "FITTING_MODEL_FAILED",
    "error_message": "가상 피팅 이미지 생성에 실패했습니다.",
}
FAILED_RESULT = {"fitting_job_id": JOB, "status": "FAILED", **MODEL_FAILED}


def snapshot(api, job_id=JOB):  # noqa: F811
    current = job(api["db"], job_id)
    return {
        "status": current.status,
        "next_action": current.next_action,
        "result_payload": current.result_payload,
        "result_message_status": current.result_message_status.value,
        "fallback_attempt_count": current.fallback_attempt_count,
        "task_uuid": current.task_uuid,
        "pruna_submit_started_at": current.pruna_submit_started_at,
    }


class ReadyPaths:
    """테스트 전용: 주어진 다음 일의 실제 경로가 연결됐다고 답한다(14·15단계가 구현된 상황을 흉내)."""

    def __init__(self, *steps):
        self.steps, self.asked = set(steps), []

    def ready_for(self, job, step):
        self.asked.append(step)
        return step in self.steps


def connect(monkeypatch, paths):
    monkeypatch.setattr(webhook_router, "webhook_follow_up_paths", lambda: paths)


def not_ready_body(task, step, outcome):
    return {
        "code": "WEBHOOK_FOLLOW_UP_NOT_READY",
        "data": {
            "results": [{"task_uuid": task, "outcome": outcome, "published": None}],
            "pending": [{"task_uuid": task, "next_step": step}],
        },
        "message": "알림은 기록했지만 이어서 처리할 경로가 아직 준비되지 않았습니다.",
    }


def mark_published(api):  # noqa: F811
    """14단계가 결과를 보낸 뒤의 상태(테스트 준비용 직접 UPDATE)."""
    api["db"]["sql"](
        "UPDATE virtual_fitting_async_jobs SET status = 'FAILED', next_action = NULL, "
        "result_message_status = 'PUBLISHED', result_message_attempt_count = 1 "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )


# --- 1. 영구 실패 저장 뒤 후속 경로 ---


def test_permanent_failure_without_a_publish_path_is_503_and_keeps_the_record(api):  # noqa: F811
    task = submitted(api["db"])
    response = post(api, error_body(task, code="someNewCode", cost=0.015))

    assert response.status_code == 503
    assert response.json() == not_ready_body(task, "publish_result", "applied")
    state = snapshot(api)
    # DB 에 다음 작업은 기록됐다(기록 완료) — 그러나 실행할 경로가 없어 완료로 응답하지 않았다.
    assert state["next_action"] is NextAction.PUBLISH_RESULT
    assert state["result_payload"] == FAILED_RESULT
    assert state["status"] is AsyncJobStatus.WAITING_PRUNA
    assert state["result_message_status"] == "PENDING"
    assert task_costs(api["db"])[0][1] == Decimal("0.015") and in_flight(api["db"]) == []
    assert api["boto"].bodies == []  # 결과를 어떤 큐에도 보내지 않았다(14단계 범위)


def test_redelivery_while_not_ready_keeps_everything_and_stays_503(api):  # noqa: F811
    task = submitted(api["db"])
    post(api, error_body(task, code="someNewCode", cost=0.015))
    first = snapshot(api)
    for _ in range(2):
        again = post(api, error_body(task, code="someNewCode", cost=0.5))
        assert again.status_code == 503
        assert again.json() == not_ready_body(task, "publish_result", "duplicate")
    assert snapshot(api) == first  # 결과·상태·다음 작업·생성 기록 보존, fallback 없음
    assert len(task_costs(api["db"])) == 1 and task_costs(api["db"])[0][1] == Decimal("0.015")
    assert in_flight(api["db"]) == [] and api["boto"].bodies == []


def test_redelivery_completes_a_missed_cleanup_step_before_answering(api, monkeypatch):  # noqa: F811
    task = submitted(api["db"])
    real = TaskCostRepository.record
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FittingDatabaseError("injected")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(TaskCostRepository, "record", flaky)
    assert post(api, error_body(task, code="someNewCode", cost=0.015)).status_code == 500
    assert task_costs(api["db"]) == [] and snapshot(api)["result_payload"] == FAILED_RESULT
    again = post(api, error_body(task, code="someNewCode", cost=0.015))
    assert again.status_code == 503  # 빠진 비용·자리 정리는 마쳤고, 경로는 여전히 없다
    assert len(task_costs(api["db"])) == 1 and in_flight(api["db"]) == []


def test_injected_ready_publish_path_allows_200_with_the_same_result(api, monkeypatch):  # noqa: F811
    task = submitted(api["db"])
    assert post(api, error_body(task, code="someNewCode", cost=0.015)).status_code == 503
    paths = ReadyPaths(NextStep.PUBLISH_RESULT)
    connect(monkeypatch, paths)  # 이후 14·15단계가 연결된 상황
    response = post(api, error_body(task, code="someNewCode", cost=0.015))
    assert response.status_code == 200 and response.json()["code"] == "WEBHOOK_PROCESSED"
    assert response.json()["data"]["results"][0]["outcome"] == "duplicate"
    assert paths.asked == [NextStep.PUBLISH_RESULT]
    assert snapshot(api)["result_payload"] == FAILED_RESULT
    assert len(task_costs(api["db"])) == 1 and in_flight(api["db"]) == []


def test_default_assembly_never_reports_a_publish_or_recovery_path(api, monkeypatch):  # noqa: F811
    from app.virtual_fitting.result_publisher import ResultPublishPaths

    # 14단계부터 결과 전송 루프의 실제 상태를 본다. 루프가 등록·실행되지 않으면 어떤 단계도 준비되지 않는다.
    paths = webhook_router.webhook_follow_up_paths()
    assert isinstance(paths, ResultPublishPaths)
    assert not any(paths.ready_for(None, step) for step in NextStep)
    # 결과 큐 URL·Webhook token 이 설정돼 있어도 준비됐다고 보지 않는다.
    monkeypatch.setattr(
        settings,
        "BE_FITTING_RESPONSE_QUEUE_URL",
        "https://sqs.ap-northeast-2.amazonaws.com/123456789012/be-fitting-response",
    )
    task = submitted(api["db"])
    assert post(api, error_body(task, code="someNewCode")).status_code == 503
    assert settings.VIRTUAL_FITTING_REQUEST_WORKER_ENABLED is False


def test_duplicate_after_the_result_was_published_is_200(api):  # noqa: F811
    task = submitted(api["db"])
    post(api, error_body(task, code="someNewCode"))
    mark_published(api)
    published = snapshot(api)
    for body in (error_body(task, code="someNewCode"), success_body(task), error_body(task)):
        response = post(api, body)
        assert response.status_code == 200
        assert response.json()["data"]["results"][0]["outcome"] == "ignored"
    assert snapshot(api) == published and api["boto"].bodies == []


def test_late_success_on_a_job_waiting_to_publish_is_not_answered_as_done(api):  # noqa: F811
    task = submitted(api["db"])
    post(api, error_body(task, code="someNewCode"))
    response = post(api, success_body(task))  # 상충하는 늦은 성공 알림
    assert response.status_code == 503
    assert response.json()["data"]["results"][0]["outcome"] == "ignored"
    assert snapshot(api)["result_payload"] == FAILED_RESULT and api["boto"].bodies == []


# --- 2. Runware 인증 오류(invalidApiKey) ---


def test_runware_auth_error_is_held_not_saved_as_a_model_failure(api, caplog):  # noqa: F811
    task = submitted(api["db"])
    before = snapshot(api)
    with caplog.at_level(logging.DEBUG):
        response = post(api, error_body(task, code="invalidApiKey", cost=0.015))

    assert response.status_code == 503  # Webhook token 인증 실패(401)와 다르다
    assert response.json() == not_ready_body(task, "recover_unknown", "held")
    state = snapshot(api)
    assert state == before  # 결과 저장·상태 변경·fallback·생성 기록 초기화 없음
    assert state["result_payload"] is None and state["next_action"] is None
    assert state["fallback_attempt_count"] == 0
    assert api["boto"].bodies == []  # RUN_FALLBACK 등 후속 발행 없음 → 자동 재호출 없음
    # 이 task 의 확정된 종료에 따른 비용·자리 정리는 한다.
    assert len(task_costs(api["db"])) == 1 and in_flight(api["db"]) == []
    server = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app"))
    assert "operator check" in server and "invalidApiKey" in server
    for secret in (TOKEN, "did not respond", "Bearer"):
        assert secret not in server


def test_runware_auth_error_redelivery_has_no_double_effects(api):  # noqa: F811
    task = submitted(api["db"])
    post(api, error_body(task, code="invalidApiKey", cost=0.015))
    first = snapshot(api)
    again = post(api, error_body(task, code="invalidApiKey", cost=0.5))
    assert again.status_code == 503 and again.json()["data"]["results"][0]["outcome"] == "held"
    assert snapshot(api) == first
    assert len(task_costs(api["db"])) == 1 and task_costs(api["db"])[0][1] == Decimal("0.015")
    assert in_flight(api["db"]) == []


def test_runware_auth_error_does_not_touch_another_jobs_slot(api):  # noqa: F811
    task = submitted(api["db"])
    other = submitted(api["db"], 7)
    other_before = snapshot(api, 7)
    post(api, error_body(task, code="invalidApiKey"))
    assert in_flight(api["db"]) == [(7, other)]
    assert snapshot(api, 7) == other_before


def test_runware_auth_error_keeps_an_existing_result(api):  # noqa: F811
    task = submitted(api["db"])
    post(api, error_body(task, code="someNewCode"))
    saved = snapshot(api)
    response = post(api, error_body(task, code="invalidApiKey"))
    # 늦은 인증 오류는 무시하고 기존 결과를 보존한다. 결과 발행 경로가 없으므로 여전히 503.
    assert response.status_code == 503
    assert response.json() == not_ready_body(task, "publish_result", "ignored")
    assert snapshot(api) == saved


def test_runware_auth_error_after_publish_is_200_and_preserved(api):  # noqa: F811
    task = submitted(api["db"])
    post(api, error_body(task, code="someNewCode"))
    mark_published(api)
    published = snapshot(api)
    response = post(api, error_body(task, code="invalidApiKey"))
    assert response.status_code == 200 and snapshot(api) == published


def test_runware_auth_error_with_an_injected_recovery_path_is_200(api, monkeypatch):  # noqa: F811
    paths = ReadyPaths(NextStep.RECOVER_UNKNOWN)
    connect(monkeypatch, paths)
    task = submitted(api["db"])
    response = post(api, error_body(task, code="invalidApiKey"))
    assert response.status_code == 200
    assert response.json()["data"]["results"][0]["outcome"] == "held"
    assert snapshot(api)["result_payload"] is None and paths.asked == [NextStep.RECOVER_UNKNOWN]


def test_runware_auth_error_before_the_accept_response(api):  # noqa: F811
    task = submitted(api["db"], accepted=False)
    response = post(api, error_body(task, code="invalidApiKey"))
    assert response.status_code == 503
    current = job(api["db"])
    assert (current.status, current.next_action) == (
        AsyncJobStatus.RECEIVED,
        NextAction.SUBMIT_PRUNA,
    )
    # 접수 불명 복구 대상으로 남는다(같은 task 를 다시 보내지 않음). 호출 시작 기록도 그대로다.
    assert next_step_for(current) is NextStep.RECOVER_UNKNOWN
    assert current.pruna_submit_started_at is not None


def test_webhook_token_failure_with_an_auth_error_body_is_still_401(api):  # noqa: F811
    task = submitted(api["db"])
    before = snapshot(api)
    response = post(api, error_body(task, code="invalidApiKey"), token="wrong")
    assert response.status_code == 401 and response.json()["code"] == "UNAUTHORIZED"
    assert snapshot(api) == before and task_costs(api["db"]) == []
    assert in_flight(api["db"]) == [(JOB, task)]


# --- 3. 코드별 매핑 (실제 Router, 기본 조립 = 경로 미준비) ---

# 근거: "doc" = Runware 공식 오류 문서에 이름이 나오는 코드, "assumed" = 문서·실제 계정에서 확인하지 못한 가정
# 코드(잔액·입력 오류의 Webhook 코드명은 문서에 없다. 가정 코드에 근거 없는 매핑을 추가하지 않는다), "unknown".
CODES = [
    ("timeoutProvider", "doc"),
    ("providerRateLimitExceeded", "doc"),
    ("serviceOverloaded", "doc"),
    ("invalidApiKey", "doc"),
    ("insufficientCredits", "assumed"),
    ("invalidReferenceImage", "assumed"),
    ("someNewCode", "unknown"),
]


@pytest.mark.parametrize(("code", "basis"), CODES)
def test_failure_code_mapping_through_the_router(api, code, basis):  # noqa: F811
    task = submitted(api["db"])
    response = post(api, error_body(task, code=code))
    state = snapshot(api)
    observed = (
        response.status_code,
        state["status"],
        state["next_action"],
        state["result_payload"],
        [body["action"] for body in api["boto"].bodies],
    )
    if code in FALLBACK_ERROR_CODES:
        # 후처리 큐가 영속적으로 받았으므로 200(기존 동작 유지).
        expected = (200, AsyncJobStatus.FALLBACK_RUNNING, NextAction.RUN_FALLBACK, None)
        assert observed == (*expected, ["RUN_FALLBACK"])
    elif code in AUTH_ERROR_CODES:
        expected = (503, AsyncJobStatus.WAITING_PRUNA, None, None)
        assert observed == (*expected, [])
    else:
        expected = (503, AsyncJobStatus.WAITING_PRUNA, NextAction.PUBLISH_RESULT, FAILED_RESULT)
        assert observed == (*expected, [])
    assert state["fallback_attempt_count"] == 0  # 실제 fallback 호출은 없음


def test_success_still_publishes_postprocess_with_200(api):  # noqa: F811
    task = submitted(api["db"])
    response = post(api, success_body(task))
    assert response.status_code == 200
    assert [body["action"] for body in api["boto"].bodies] == ["POSTPROCESS"]


def test_classification_reads_only_the_code_field(api):  # noqa: F811
    """message·parameter·retryAfter·status 로 원인을 추측하지 않는다(잔액 부족 문구가 있어도 잔액 코드로 보지 않음)."""
    task = submitted(api["db"])
    body = error_body(
        task,
        code="someNewCode",
        message="Your account balance is insufficient",
        parameter="balance",
        retryAfter=30,
        status="error",
    )
    response = post(api, body)
    assert response.status_code == 503
    assert job(api["db"]).result_payload["error_code"] == "FITTING_MODEL_FAILED"


def test_no_unverified_balance_mapping_was_added():
    source = inspect.getsource(runware_webhook)
    assert "insufficientCredits" not in source
    assert (
        "FITTING_BALANCE_EXHAUSTED" not in source and "FittingBalanceExhaustedError" not in source
    )
    assert FALLBACK_ERROR_CODES == frozenset({"timeoutProvider"})
    assert AUTH_ERROR_CODES == frozenset({"invalidApiKey"})
