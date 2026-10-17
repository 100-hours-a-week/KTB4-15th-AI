"""가상피팅 V2 비동기 작업의 상태 관리 (virtual_fitting_async_jobs).

여러 Worker 가 같은 작업을 동시에 잡으려 해도 DB 가 다음을 보장하게 한다.
- 실행 권한(lease)은 한 Worker 만 가진다. 토큰은 선점할 때마다 새로 만든다.
- 예상한 상태·다음 작업일 때만 바꾼다. 종료(COMPLETED/FAILED)된 작업은 바꾸지 않는다.
- 단계별 시도 횟수는 한도 안에서만 1 늘린다(확보한 횟수는 실패해도 되돌리지 않는다).
- 이전 Worker(만료·교체된 토큰)는 상태·결과를 바꾸지 못한다.

변경은 짧은 트랜잭션 하나로 한다(외부 호출 없음, 작업이 끝날 때까지 잠금을 들고 있지 않음).
1. SELECT … FOR UPDATE 로 그 행을 잠근다. 다른 연결이 잠그고 있으면 여기서 기다린다.
2. 잠금을 얻은 뒤의 조건부 UPDATE 로 바꾼다. 시각은 statement_timestamp()(이 UPDATE 문이 시작된 실제 시각)를
   쓴다. now() 는 트랜잭션 시작(잠금 대기 전) 시각에 고정돼 있어, 기다리는 동안 만료된 lease 를 유효하다고
   보거나 이미 지난 시각으로 새 lease 를 계산하게 된다.
3. 바뀐 줄이 없으면 같은 트랜잭션에서 다시 읽어 이유를 돌려준다. 이유는 참고용이고 바뀌었는지는 applied 만 믿는다.

Worker 경로는 토큰이 필요하다. 외부 이벤트(Runware Webhook) 경로는 토큰 대신 등록된 task_uuid·호출 시작 기록·
예상 상태를 조건으로 쓴다. 이 모듈은 외부 호출·SQS·반복문을 하지 않는다(이후 단계가 연결한다).
"""

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from app.clients.s3 import S3ObjectState, validate_s3_key
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.repositories.async_job_repository import (
    _SELECT_LIST,
    TABLE,
    AsyncFittingJob,
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
    _check_job_id,
    _row_to_job,
)

# 최초 포함 최대 시도 횟수. S3 는 HEAD 한 번이 아니라 0·2·7초 확인 한 묶음이 1회다.
S3_CHECK_LIMIT = 2
LLM_GENERATION_LIMIT = 3
FALLBACK_LIMIT = 1
RESULT_MESSAGE_LIMIT = 3
MAX_LEASE_SECONDS = 3600

_TERMINAL = [AsyncJobStatus.COMPLETED.value, AsyncJobStatus.FAILED.value]
_NOW = "statement_timestamp()"
_LEASE_VALID = f"worker_lease_token = %(token)s AND worker_lease_until > {_NOW}"


class ChangeReason(str, Enum):
    APPLIED = "applied"
    NOT_FOUND = "not_found"
    TERMINAL = "terminal"
    # 토큰이 다르거나 만료됐다(재선점된 이전 Worker 포함).
    LEASE_INVALID = "lease_invalid"
    # 선점: 다른 Worker 의 유효한 lease 가 있다.
    LEASE_HELD = "lease_held"
    LIMIT_REACHED = "limit_reached"
    # 이미 기록돼 있어 바꾸지 않았다(최초 값 보존). job 의 값을 재사용하면 된다.
    ALREADY_RECORDED = "already_recorded"
    STATE_MISMATCH = "state_mismatch"


@dataclass(frozen=True)
class StateChange:
    applied: bool
    reason: ChangeReason
    # 변경 후(applied) 또는 현재(applied 아님) 작업. NOT_FOUND 면 None.
    job: AsyncFittingJob | None


@dataclass(frozen=True)
class LeaseGrant:
    acquired: bool
    reason: ChangeReason
    # 선점에 성공했을 때만 있다. 이후 Worker 메서드에 넘긴다.
    token: str | None
    job: AsyncFittingJob | None


def _check_seconds(lease_seconds: float) -> float:
    if (
        isinstance(lease_seconds, bool)
        or not isinstance(lease_seconds, int | float)
        or not 0 < lease_seconds <= MAX_LEASE_SECONDS
    ):
        raise ValueError(f"lease_seconds 는 0 초과 {MAX_LEASE_SECONDS} 이하여야 합니다.")
    return float(lease_seconds)


def _check_token(token: str) -> str:
    if not isinstance(token, str) or not token:
        raise ValueError("lease token 이 비어 있습니다.")
    return token


def _check_uuid(value: str, name: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"{name} 는 UUID 문자열이어야 합니다.") from error


_SUCCESS_FIELDS = frozenset(
    {"fitting_job_id", "status", "result_image_key", "llm_title", "llm_comment"}
)
_FAILURE_FIELDS = frozenset({"fitting_job_id", "status", "error_code", "error_message"})


def _check_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """V2 결과(be-fitting-response 로 보낼 내용) 형식. 최상위 status 로 성공·실패를 나눈다.

    성공: fitting_job_id, status=COMPLETED, result_image_key, llm_title, llm_comment
    실패: fitting_job_id, status=FAILED, error_code, error_message (이미지·제목·코멘트 없음)
    fitting_job_id 가 그 작업 번호와 같은지는 저장할 때 DB 의 작업 번호와 비교한다.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("result_payload 는 JSON 객체여야 합니다.")
    payload = dict(payload)
    status = payload.get("status")
    expected = {"COMPLETED": _SUCCESS_FIELDS, "FAILED": _FAILURE_FIELDS}.get(status)
    if expected is None:
        raise ValueError("result_payload.status 는 COMPLETED 또는 FAILED 여야 합니다.")
    if set(payload) != expected:
        raise ValueError(f"{status} result_payload 의 필드는 {sorted(expected)} 이어야 합니다.")
    job_id = payload["fitting_job_id"]
    if isinstance(job_id, bool) or not isinstance(job_id, int):
        raise TypeError("result_payload.fitting_job_id 는 정수여야 합니다.")
    for name in expected - {"fitting_job_id", "status"}:
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise ValueError(f"result_payload.{name} 이 비어 있습니다.")
    if status == "COMPLETED":
        validate_s3_key(payload["result_image_key"])
    return payload


class AsyncJobStateRepository:
    """작업별 상태 변경 메서드. 쓸 때만 pool 에서 connection 을 빌리고 즉시 반환한다."""

    def __init__(self, connection_pool: Any):
        self._connection_pool = connection_pool

    # --- 실행 권한(lease) ---

    def acquire_lease(
        self, fitting_job_id: int, expected_action: NextAction, lease_seconds: float
    ) -> LeaseGrant:
        """next_action 이 expected_action 이고 유효한 다른 lease 가 없을 때만 새 토큰으로 선점한다.

        만료된 lease 는 다시 선점할 수 있다. 만료됐다고 Pruna·V1 을 다시 불러도 된다는 뜻은 아니다
        (호출 시작 기록·fallback 횟수가 따로 막는다).
        """
        token = str(uuid.uuid4())
        change = self._update(
            fitting_job_id,
            """
            worker_lease_token = %(token)s,
            worker_lease_until = statement_timestamp() + make_interval(secs => %(seconds)s)
            """,
            """
            status <> ALL(%(terminal)s) AND next_action = %(action)s
            AND (worker_lease_token IS NULL OR worker_lease_until <= statement_timestamp())
            """,
            {
                "token": token,
                "seconds": _check_seconds(lease_seconds),
                "action": NextAction(expected_action).value,
            },
            lambda job, now: (
                ChangeReason.STATE_MISMATCH
                if job.next_action is not NextAction(expected_action)
                else ChangeReason.LEASE_HELD
            ),
            lease=False,
        )
        return LeaseGrant(
            change.applied, change.reason, token if change.applied else None, change.job
        )

    def renew_lease(self, fitting_job_id: int, token: str, lease_seconds: float) -> StateChange:
        """유효한 자기 토큰이면 지금부터 lease_seconds 동안으로 연장한다. SQS Visibility 연장과 별개다."""
        return self._update(
            fitting_job_id,
            "worker_lease_until = statement_timestamp() + make_interval(secs => %(seconds)s)",
            "status <> ALL(%(terminal)s)",
            {"token": token, "seconds": _check_seconds(lease_seconds)},
        )

    def release_lease(self, fitting_job_id: int, token: str) -> StateChange:
        return self._update(
            fitting_job_id,
            "worker_lease_token = NULL, worker_lease_until = NULL",
            "TRUE",
            {"token": token},
            terminal_ok=True,
        )

    def holds_lease(self, fitting_job_id: int, token: str) -> bool:
        """지금(DB 시각) 이 토큰이 유효한 실행 권한인지. 판단용이며 변경은 각 메서드가 다시 확인한다."""
        row = self._fetch(
            f"SELECT {_LEASE_VALID} FROM {TABLE} WHERE fitting_job_id = %(job_id)s",
            {"job_id": _check_job_id(fitting_job_id), "token": _check_token(token)},
        )
        return bool(row and row[0])

    # --- Pruna 요청(RECEIVED / SUBMIT_PRUNA) ---

    def prepare_pruna_submission(
        self, fitting_job_id: int, token: str, task_uuid: str, async_image_key: str
    ) -> StateChange:
        """Runware 호출 전에 task_uuid·async_image_key 를 저장한다. 이미 저장돼 있으면 바꾸지 않는다(재사용)."""
        return self._update(
            fitting_job_id,
            "task_uuid = %(task_uuid)s, async_image_key = %(key)s",
            """
            status = 'RECEIVED' AND next_action = 'SUBMIT_PRUNA'
            AND task_uuid IS NULL AND pruna_submit_started_at IS NULL
            """,
            {
                "token": token,
                "task_uuid": _check_uuid(task_uuid, "task_uuid"),
                "key": validate_s3_key(async_image_key),
            },
            lambda job, now: (
                ChangeReason.ALREADY_RECORDED
                if job.task_uuid is not None
                else ChangeReason.STATE_MISMATCH
            ),
        )

    def mark_pruna_submit_started(self, fitting_job_id: int, token: str) -> StateChange:
        """Runware 호출 직전에 한 번만 기록한다. 이미 기록돼 있으면 ALREADY_RECORDED.

        접수 여부를 모르는 채로 기록이 남아 있으면 새 Pruna 호출을 허용하지 않는다(Watchdog 확인 대상).
        """
        return self._update(
            fitting_job_id,
            "pruna_submit_started_at = statement_timestamp()",
            """
            status = 'RECEIVED' AND next_action = 'SUBMIT_PRUNA'
            AND task_uuid IS NOT NULL AND pruna_submit_started_at IS NULL
            """,
            {"token": token},
            lambda job, now: (
                ChangeReason.ALREADY_RECORDED
                if job.pruna_submit_started_at is not None
                else ChangeReason.STATE_MISMATCH
            ),
        )

    def record_runware_accepted(
        self, fitting_job_id: int, token: str, task_uuid: str
    ) -> StateChange:
        """Worker 가 Runware 접수 응답(ACCEPTED)을 받았을 때. RECEIVED 면 WAITING_PRUNA 로 바꾸고 next_action 을 비운다.

        Webhook 이 먼저 와서 이미 진전된 상태(POSTPROCESSING·FALLBACK_RUNNING)는 되돌리지 않는다. 최초 접수
        시각은 처음 한 번만 기록한다.
        """
        return self._update(
            fitting_job_id,
            """
            runware_accepted_at = COALESCE(runware_accepted_at, statement_timestamp()),
            status = CASE WHEN status = 'RECEIVED' THEN 'WAITING_PRUNA' ELSE status END,
            next_action = CASE WHEN status = 'RECEIVED' THEN NULL ELSE next_action END
            """,
            """
            status <> ALL(%(terminal)s) AND task_uuid = %(task_uuid)s
            AND pruna_submit_started_at IS NOT NULL
            """,
            {"token": token, "task_uuid": _check_uuid(task_uuid, "task_uuid")},
        )

    # --- 외부 이벤트(Webhook) 경로: 토큰 대신 task_uuid·호출 시작·예상 상태 ---

    def record_submission_accepted(self, task_uuid: str) -> StateChange:
        """Runware 가 이 task_uuid 의 생성 요청을 접수했다는 응답(ACCEPTED)을 기록한다. 토큰을 쓰지 않는다.

        접수 근거는 task 에 대한 사실이므로, 긴 HTTP 호출 중 lease 가 만료됐어도 기록할 수 있어야 한다. 등록된
        task_uuid·호출 시작 기록이 있고 종료되지 않은 작업만 바꾼다. RECEIVED 면 WAITING_PRUNA·next_action=NULL 로
        바꾸고, 완료 알림 등으로 이미 진전된 상태는 되돌리지 않는다. 최초 접수 시각은 처음 한 번만 기록한다.
        """
        return self._event_update(
            task_uuid,
            f"""
            runware_accepted_at = COALESCE(runware_accepted_at, {_NOW}),
            status = CASE WHEN status = 'RECEIVED' THEN 'WAITING_PRUNA' ELSE status END,
            next_action = CASE WHEN status = 'RECEIVED' THEN NULL ELSE next_action END
            """,
            "TRUE",
            {},
        )

    def record_pruna_succeeded(self, task_uuid: str) -> StateChange:
        """Pruna 성공 알림. 접수 대기 중(RECEIVED·WAITING_PRUNA)인 작업만 POSTPROCESSING / POSTPROCESS 로 바꾼다."""
        return self._event_update(
            task_uuid,
            """
            status = 'POSTPROCESSING', next_action = 'POSTPROCESS',
            runware_accepted_at = COALESCE(runware_accepted_at, statement_timestamp())
            """,
            "status IN ('RECEIVED', 'WAITING_PRUNA')",
            {},
        )

    def record_pruna_failed(
        self, task_uuid: str, *, recoverable: bool, failure_payload: Mapping[str, Any] | None = None
    ) -> StateChange:
        """**확인된** Pruna 실패를 기록한다. 확인된 실패만 넘긴다(timeout·응답 없음·접수 불명확은 넘기지 않는다).

        호출부 계약(DB 는 실패 원인을 모른다):
        - Runware 가 이 task_uuid 의 실패를 알려 줬다(실패 Webhook, getResponse 의 error, 또는 제출에 대한 명확한
          거절 응답). "응답을 못 받음"·"아직 processing" 은 실패가 아니다.
        - recoverable=True 는 그 실패를 V1 로 다시 만들어 볼 수 있는 종류로 이미 분류했다는 뜻이다(예: provider
          timeout·일시적 생성 실패). 입력·상품 검증 실패, 일일 예산 초과, Runware 잔액 부족, S3 권한·설정 오류는
          recoverable=False 로 넘기고 실패 결과(failure_payload)를 함께 준다.

        DB 가 보장하는 것: 등록된 task_uuid·호출 시작 기록이 있고, RECEIVED·WAITING_PRUNA 이며, 결과·확인한
        이미지가 없고, fallback 을 아직 쓰지 않았을 때만 바꾼다. 같은 실패 알림이 다시 와도 두 번 바뀌지 않는다.
        - recoverable=True  → FALLBACK_RUNNING / RUN_FALLBACK
        - recoverable=False → 실패 결과 저장 + next_action=PUBLISH_RESULT (상태는 발행 성공 뒤 FAILED 로 확정)
        """
        guard = """
            status IN ('RECEIVED', 'WAITING_PRUNA') AND result_payload IS NULL
            AND verified_image_key IS NULL AND fallback_attempt_count < %(fallback_limit)s
        """
        if recoverable:
            if failure_payload is not None:
                raise ValueError("V1 로 넘길 실패에는 failure_payload 를 주지 않습니다.")
            return self._event_update(
                task_uuid,
                f"""
                status = 'FALLBACK_RUNNING', next_action = 'RUN_FALLBACK',
                runware_accepted_at = COALESCE(runware_accepted_at, {_NOW})
                """,
                guard,
                {"fallback_limit": FALLBACK_LIMIT},
            )
        if failure_payload is None:
            raise ValueError("복구할 수 없는 실패는 failure_payload 가 필요합니다.")
        payload = _check_result_payload(failure_payload)
        if payload["status"] != "FAILED":
            raise ValueError("Pruna 실패 결과의 status 는 FAILED 여야 합니다.")
        return self._event_update(
            task_uuid,
            f"""
            result_payload = %(payload)s, next_action = 'PUBLISH_RESULT',
            runware_accepted_at = COALESCE(runware_accepted_at, {_NOW})
            """,
            guard + " AND %(payload_job_id)s = fitting_job_id",
            {
                "payload": Jsonb(payload),
                "payload_job_id": payload["fitting_job_id"],
                "fallback_limit": FALLBACK_LIMIT,
            },
        )

    # --- 후처리(POSTPROCESSING / POSTPROCESS) ---

    def claim_s3_check_attempt(self, fitting_job_id: int, token: str) -> StateChange:
        """S3 확인 한 묶음(0·2·7초)을 시작하기 직전에 횟수를 확보한다. 최초 포함 최대 2묶음."""
        return self._claim(
            fitting_job_id,
            token,
            "s3_check_attempt_count",
            S3_CHECK_LIMIT,
            """
            status = 'POSTPROCESSING' AND next_action = 'POSTPROCESS'
            AND verified_image_key IS NULL AND result_payload IS NULL
            """,
            lambda job: job.s3_check_attempt_count,
            lambda job: job.verified_image_key is not None,
        )

    def record_image_verified(self, fitting_job_id: int, token: str, image_key: str) -> StateChange:
        """S3 에서 확인한 이미지 Key 를 저장한다. 이미 확인한 이미지가 있으면 바꾸지 않는다(재사용)."""
        return self._update(
            fitting_job_id,
            "verified_image_key = %(key)s, image_verified_at = statement_timestamp()",
            """
            status = 'POSTPROCESSING' AND next_action = 'POSTPROCESS'
            AND verified_image_key IS NULL AND result_payload IS NULL
            """,
            {"token": token, "key": validate_s3_key(image_key)},
            lambda job, now: (
                ChangeReason.ALREADY_RECORDED
                if job.verified_image_key is not None
                else ChangeReason.STATE_MISMATCH
            ),
        )

    def claim_llm_generation_attempt(self, fitting_job_id: int, token: str) -> StateChange:
        """확인한 이미지로 제목·코멘트를 생성하기 직전에 횟수를 확보한다. 최초 포함 최대 3회."""
        return self._claim(
            fitting_job_id,
            token,
            "llm_generation_attempt_count",
            LLM_GENERATION_LIMIT,
            """
            status = 'POSTPROCESSING' AND next_action = 'POSTPROCESS'
            AND verified_image_key IS NOT NULL AND result_payload IS NULL
            """,
            lambda job: job.llm_generation_attempt_count,
            lambda job: job.result_payload is not None,
        )

    # --- V1 fallback (FALLBACK_RUNNING / RUN_FALLBACK) ---

    def move_to_fallback(
        self, fitting_job_id: int, token: str, *, final_s3_check: S3ObjectState
    ) -> StateChange:
        """S3 확인 경로의 V1 fallback 전환. Pruna 실패 알림 경로는 record_pruna_failed 를 쓴다.

        호출부 계약: 이 lease 로 2번째(마지막) S3 확인 묶음을 끝까지 돌렸고, 그 결과가 final_s3_check 다.
        횟수는 묶음을 시작하기 직전에 늘리므로 s3_check_attempt_count == 2 만으로는 2번째 묶음이 끝났는지 알 수
        없다. 그래서 확인 결과를 호출부가 넘긴다. 다른 Worker(만료 뒤 재선점)는 이전 묶음의 결과를 모르므로
        이 메서드를 부르지 않는다.
        - NOT_FOUND 만 fallback 대상이다. ACCESS_DENIED·ERROR(권한·설정)·TRANSIENT_ERROR(객체가 있을 수 있음)는
          횟수를 다 썼어도 fallback 하지 않는다(ValueError, DB 에 접근하지 않음).

        DB 가 보장하는 것: 유효한 lease, POSTPROCESSING / POSTPROCESS, S3 확인 2묶음을 모두 시작했고, 확인한
        이미지·결과가 없고, fallback 을 아직 쓰지 않았을 때만 FALLBACK_RUNNING / RUN_FALLBACK 으로 바꾼다.
        RECEIVED·WAITING_PRUNA(Pruna 진행 중·접수 불명확)에서는 바꾸지 않는다.
        """
        if S3ObjectState(final_s3_check) is not S3ObjectState.NOT_FOUND:
            raise ValueError("S3 에서 이미지가 없다고 확인(NOT_FOUND)된 경우에만 fallback 합니다.")
        return self._update(
            fitting_job_id,
            "status = 'FALLBACK_RUNNING', next_action = 'RUN_FALLBACK'",
            """
            status = 'POSTPROCESSING' AND next_action = 'POSTPROCESS'
            AND s3_check_attempt_count >= %(s3_limit)s
            AND result_payload IS NULL AND verified_image_key IS NULL
            AND fallback_attempt_count < %(limit)s
            """,
            {"token": token, "limit": FALLBACK_LIMIT, "s3_limit": S3_CHECK_LIMIT},
            lambda job, now: (
                ChangeReason.LIMIT_REACHED
                if job.fallback_attempt_count >= FALLBACK_LIMIT
                else ChangeReason.STATE_MISMATCH
            ),
        )

    def claim_fallback_attempt(self, fitting_job_id: int, token: str) -> StateChange:
        """V1 을 부르기 직전에 횟수를 확보한다. 최초 포함 최대 1회 — 확보 뒤 중단돼도 다시 부르지 않는다."""
        return self._claim(
            fitting_job_id,
            token,
            "fallback_attempt_count",
            FALLBACK_LIMIT,
            """
            status = 'FALLBACK_RUNNING' AND next_action = 'RUN_FALLBACK'
            AND result_payload IS NULL AND verified_image_key IS NULL
            """,
            lambda job: job.fallback_attempt_count,
            lambda job: job.result_payload is not None,
        )

    # --- 최종 결과와 발행 ---

    def save_final_result(
        self,
        fitting_job_id: int,
        token: str,
        result_payload: Mapping[str, Any],
        result_image_key: str | None = None,
    ) -> StateChange:
        """완성된 V2 결과와 next_action=PUBLISH_RESULT 를 함께 저장한다. 이미 결과가 있으면 덮어쓰지 않는다.

        result_payload.fitting_job_id 는 이 작업 번호와 같아야 한다. 성공 결과는 result_payload.result_image_key
        와 result_image_key 가 같아야 하고, 실패 결과는 result_image_key 를 주지 않는다.
        """
        payload = _check_result_payload(result_payload)
        job_id = _check_job_id(fitting_job_id)
        if payload["fitting_job_id"] != job_id:
            raise ValueError("result_payload.fitting_job_id 가 작업 번호와 다릅니다.")
        if payload["status"] == "COMPLETED":
            if result_image_key != payload["result_image_key"]:
                raise ValueError("result_image_key 가 result_payload.result_image_key 와 다릅니다.")
        elif result_image_key is not None:
            raise ValueError("실패 결과에는 result_image_key 를 저장하지 않습니다.")
        return self._update(
            job_id,
            "result_payload = %(payload)s, result_image_key = %(image_key)s, next_action = 'PUBLISH_RESULT'",
            "status <> ALL(%(terminal)s) AND result_payload IS NULL",
            {"token": token, "payload": Jsonb(payload), "image_key": result_image_key},
            lambda job, now: (
                ChangeReason.ALREADY_RECORDED
                if job.result_payload is not None
                else ChangeReason.STATE_MISMATCH
            ),
        )

    def claim_result_message_attempt(self, fitting_job_id: int, token: str) -> StateChange:
        """저장된 결과를 be-fitting-response 로 보내기 직전에 횟수를 확보한다. 최초 포함 최대 3회."""
        return self._claim(
            fitting_job_id,
            token,
            "result_message_attempt_count",
            RESULT_MESSAGE_LIMIT,
            """
            status <> ALL(%(terminal)s) AND next_action = 'PUBLISH_RESULT'
            AND result_payload IS NOT NULL AND result_message_status = 'PENDING'
            """,
            lambda job: job.result_message_attempt_count,
            lambda job: job.result_message_status is not ResultMessageStatus.PENDING,
        )

    def mark_result_published(
        self, fitting_job_id: int, token: str, final_status: AsyncJobStatus
    ) -> StateChange:
        """전송 성공을 확인한 뒤 PUBLISHED·최종 상태·next_action=NULL·lease 정리를 한 번에 저장한다.

        final_status 는 저장된 result_payload 의 최상위 status 와 같아야 한다(다르거나 status 가 없으면 아무것도
        바꾸지 않는다). 확보한 전송 시도가 있어야 하고, 마지막(3번째) 허용 시도의 성공도 기록할 수 있다.
        """
        final_status = AsyncJobStatus(final_status)
        if final_status.value not in _TERMINAL:
            raise ValueError("final_status 는 COMPLETED 또는 FAILED 여야 합니다.")
        return self._update(
            fitting_job_id,
            """
            result_message_status = 'PUBLISHED', status = %(final)s, next_action = NULL,
            worker_lease_token = NULL, worker_lease_until = NULL
            """,
            """
            status <> ALL(%(terminal)s) AND next_action = 'PUBLISH_RESULT'
            AND result_payload IS NOT NULL AND result_message_status = 'PENDING'
            AND result_message_attempt_count >= 1
            AND result_payload ->> 'status' = %(final)s
            """,
            {"token": token, "final": final_status.value},
        )

    def mark_result_retry_exhausted(self, fitting_job_id: int, token: str) -> StateChange:
        """성공 확인 없이 자동 전송 3회를 다 쓴 경우. 결과·복구 정보는 그대로 두고 lease 만 정리한다."""
        return self._update(
            fitting_job_id,
            """
            result_message_status = 'RETRY_EXHAUSTED',
            worker_lease_token = NULL, worker_lease_until = NULL
            """,
            """
            status <> ALL(%(terminal)s) AND next_action = 'PUBLISH_RESULT'
            AND result_payload IS NOT NULL AND result_message_status = 'PENDING'
            AND result_message_attempt_count >= %(limit)s
            """,
            {"token": token, "limit": RESULT_MESSAGE_LIMIT},
        )

    # --- 내부 ---

    def _claim(
        self,
        fitting_job_id: int,
        token: str,
        column: str,
        limit: int,
        condition: str,
        count_of: Callable[[AsyncFittingJob], int],
        already_done: Callable[[AsyncFittingJob], bool],
    ) -> StateChange:
        """한도·상태·lease 확인과 1 증가를 한 문장으로 한다. 경쟁해도 한도를 넘지 않는다(행 잠금 후 WHERE 재평가)."""

        def diagnose(job: AsyncFittingJob, now: Any) -> ChangeReason:
            if count_of(job) >= limit:
                return ChangeReason.LIMIT_REACHED
            if already_done(job):
                return ChangeReason.ALREADY_RECORDED
            return ChangeReason.STATE_MISMATCH

        return self._update(
            fitting_job_id,
            f"{column} = {column} + 1",
            f"{condition} AND {column} < %(limit)s",
            {"token": token, "limit": limit},
            diagnose,
        )

    def _update(
        self,
        fitting_job_id: int,
        assignments: str,
        condition: str,
        params: dict[str, Any],
        diagnose: Callable[[AsyncFittingJob, Any], ChangeReason] | None = None,
        *,
        lease: bool = True,
        terminal_ok: bool = False,
    ) -> StateChange:
        params = {**params, "job_id": _check_job_id(fitting_job_id), "terminal": _TERMINAL}
        where = f"fitting_job_id = %(job_id)s AND ({condition})"
        if lease:
            params["token"] = _check_token(params.get("token"))
            where += f" AND {_LEASE_VALID}"
        return self._run(
            f"SELECT fitting_job_id FROM {TABLE} WHERE fitting_job_id = %(job_id)s FOR UPDATE",
            f"""
            UPDATE {TABLE} SET {assignments}, updated_at = {_NOW}
            WHERE {where}
            RETURNING {_SELECT_LIST}
            """,
            params,
            lambda job, now: self._reason(job, now, params, lease, terminal_ok, diagnose),
        )

    def _event_update(
        self, task_uuid: str, assignments: str, condition: str, params: dict[str, Any]
    ) -> StateChange:
        """외부 이벤트 경로. 등록된 task_uuid 와 호출 시작 기록이 있어야 하고 종료된 작업은 바꾸지 않는다."""
        params = {**params, "task_uuid": _check_uuid(task_uuid, "task_uuid"), "terminal": _TERMINAL}

        def reason(job: AsyncFittingJob, now: Any) -> ChangeReason:
            if job.status.value in _TERMINAL:
                return ChangeReason.TERMINAL
            return ChangeReason.STATE_MISMATCH

        return self._run(
            f"SELECT fitting_job_id FROM {TABLE} WHERE task_uuid = %(task_uuid)s FOR UPDATE",
            f"""
            UPDATE {TABLE} SET {assignments}, updated_at = {_NOW}
            WHERE fitting_job_id = %(job_id)s AND task_uuid = %(task_uuid)s
              AND pruna_submit_started_at IS NOT NULL
              AND status <> ALL(%(terminal)s) AND ({condition})
            RETURNING {_SELECT_LIST}
            """,
            params,
            reason,
        )

    @staticmethod
    def _reason(job, now, params, lease, terminal_ok, diagnose) -> ChangeReason:
        if job.status.value in _TERMINAL and not terminal_ok:
            return ChangeReason.TERMINAL
        if lease and (
            job.worker_lease_token != params["token"]
            or job.worker_lease_until is None
            or job.worker_lease_until <= now
        ):
            return ChangeReason.LEASE_INVALID
        return (diagnose or (lambda job, now: ChangeReason.STATE_MISMATCH))(job, now)

    def _run(
        self,
        lock_sql: str,
        update_sql: str,
        params: dict[str, Any],
        reason: Callable[[AsyncFittingJob, Any], ChangeReason],
    ) -> StateChange:
        """한 트랜잭션: 행 잠금(여기서 대기) → 잠금을 쥔 채 실제 시각으로 조건부 UPDATE → 실패 이유 조회."""
        try:
            with self._connection_pool.connection() as connection:
                connection.autocommit = True
                with connection.transaction(), connection.cursor() as cursor:
                    cursor.execute(lock_sql, params)
                    locked = cursor.fetchone()
                    if locked is None:
                        return StateChange(False, ChangeReason.NOT_FOUND, None)
                    params = {**params, "job_id": locked[0]}
                    cursor.execute(update_sql, params)
                    row = cursor.fetchone()
                    if row is not None:
                        return StateChange(True, ChangeReason.APPLIED, _row_to_job(row))
                    cursor.execute(
                        f"SELECT {_SELECT_LIST}, {_NOW} FROM {TABLE} WHERE fitting_job_id = %(job_id)s",
                        params,
                    )
                    current = cursor.fetchone()
        except psycopg.Error as error:
            raise FittingDatabaseError(
                "AI PostgreSQL 비동기 작업 상태 변경에 실패했습니다."
            ) from error
        job = _row_to_job(current[:-1])
        return StateChange(False, reason(job, current[-1]), job)

    def _fetch(self, sql: str, params: dict[str, Any]) -> Any:
        try:
            with self._connection_pool.connection() as connection:
                connection.autocommit = True
                with connection.cursor() as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchone()
        except psycopg.Error as error:
            raise FittingDatabaseError("AI PostgreSQL 비동기 작업 조회에 실패했습니다.") from error
