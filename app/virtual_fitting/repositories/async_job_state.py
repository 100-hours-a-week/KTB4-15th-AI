import math
import re
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
    ATTEMPTS,
    TABLE,
    AsyncFittingJob,
    AsyncJobStatus,
    NextAction,
    ResultMessageStatus,
    _check_job_id,
    _row_to_job,
    check_runware_image_url,
)

# 최초 포함 최대 시도 횟수. S3 는 HEAD 한 번이 아니라 0·2·7초 확인 한 묶음이 1회다.
S3_CHECK_LIMIT = 2
# V2 코멘트·제목: 항목별 최초 1회 + 재시도 1회. llm_generation_attempt_count 는 LLM 호출 1번마다 1씩(코멘트·제목 합계).
LLM_COMMENT_ATTEMPT_LIMIT = 2
LLM_TITLE_ATTEMPT_LIMIT = 2
LLM_GENERATION_LIMIT = LLM_COMMENT_ATTEMPT_LIMIT + LLM_TITLE_ATTEMPT_LIMIT
FALLBACK_LIMIT = 1
# 비동기 Pruna 전송: 같은 fitting_job_id 전체에서 최초 1회 + 확실한 미접수 뒤 재요청 3회.
PRUNA_SUBMIT_LIMIT = 4
# 확실한 미접수 뒤 다음 전송까지 기다릴 시간(초). 서버 대기 안내가 없거나 무효일 때만 쓴다(1→2, 2→3, 3→4번째 시도).
PRUNA_RETRY_BACKOFF_SECONDS = (10.0, 20.0, 40.0)
# 최대 횟수 모두 확실한 미접수일 때 BE 로 보낼 최종 실패(BE 는 이 코드로 사용 횟수를 차감하지 않는다고 판단한다).
SUBMISSION_REJECTED_CODE = "FITTING_SUBMISSION_REJECTED"
SUBMISSION_REJECTED_MESSAGE = "이미지 생성 요청이 접수되지 않았습니다."
RESULT_MESSAGE_LIMIT = 3
MAX_LEASE_SECONDS = 3600
# 행 잠금 대기 제한(초). Pruna limiter 와 같은 5초. 트랜잭션 범위로만 건다(set_config(..., true) — 트랜잭션이 끝나면
# 사라져 pool 의 다음 사용자에게 남지 않는다). 메시지 처리 전체·외부 HTTP timeout 과는 별개다.
DEFAULT_LOCK_TIMEOUT_SECONDS = 5.0
_SET_LOCK_TIMEOUT = "SELECT set_config('lock_timeout', %(lock_timeout)s, true)"


def lock_timeout_setting(seconds: float) -> str:
    """PostgreSQL lock_timeout 값(밀리초 문자열). 0 은 '무한 대기'라 받지 않는다."""
    if isinstance(seconds, bool) or not isinstance(seconds, int | float) or not 0 < seconds <= 60:
        raise ValueError("lock_timeout_seconds 는 0 초과 60 이하여야 합니다.")
    return f"{max(1, round(seconds * 1000))}ms"


_TERMINAL = [AsyncJobStatus.COMPLETED.value, AsyncJobStatus.FAILED.value]
_SAFE_CODE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
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


class NotAcceptedSource(str, Enum):
    """확실한 미접수를 확인한 경로. 셋 다 같은 기록 경로(record_not_accepted)를 쓴다."""

    SUBMIT_RESPONSE = "submit_response"
    WEBHOOK = "webhook"
    STATUS_QUERY = "status_query"


class NotAcceptedOutcome(str, Enum):
    SCHEDULED = "scheduled"  # 미접수 기록·자리 정리·다음 시각 예약
    EXHAUSTED = "exhausted"  # 마지막(4번째) 시도도 미접수: 실패 결과 저장·발행 대기
    HELD = "held"  # 미접수 기록·자리 정리는 했지만 대기 시각을 정할 수 없다(자동 재요청 보류, 운영 확인)
    ALREADY_RECORDED = "already_recorded"  # 같은 시도의 미접수가 이미 기록돼 있다(횟수·예약 그대로)
    PAST_ATTEMPT = "past_attempt"  # 이전 시도의 늦은 알림. 현재 시도에 적용하지 않는다
    CONTRADICTED = (
        "contradicted"  # 이 시도에 이미 접수·생성 진행 근거가 있다(바꾸지 않고 운영 확인)
    )
    STATE_MISMATCH = "state_mismatch"  # 이미 다른 경로(결과·인증 보류 등)로 진행했다
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class NotAcceptedChange:
    outcome: NotAcceptedOutcome
    job: AsyncFittingJob | None
    attempt_no: int | None = None

    @property
    def applied(self) -> bool:
        return self.outcome in (
            NotAcceptedOutcome.SCHEDULED,
            NotAcceptedOutcome.EXHAUSTED,
            NotAcceptedOutcome.HELD,
        )


def retry_delay_seconds(attempt_no: int, retry_after: float | None) -> float:
    """미접수로 끝난 attempt_no 번째 시도 뒤 다음 전송까지의 대기. 유효한 서버 안내가 있으면 그 값을, 없으면 10·20·40초."""
    if retry_after is not None:
        return float(retry_after)
    return PRUNA_RETRY_BACKOFF_SECONDS[attempt_no - 1]


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

    def __init__(
        self, connection_pool: Any, *, lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS
    ):
        self._connection_pool = connection_pool
        self._lock_timeout = lock_timeout_setting(lock_timeout_seconds)

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
        self,
        fitting_job_id: int,
        token: str,
        task_uuid: str,
        async_image_key: str,
        *,
        ai_stores_result: bool = False,
    ) -> StateChange:
        """Runware 호출 전에 task_uuid·async_image_key 를 저장한다. 이미 저장돼 있으면 바꾸지 않는다(재사용).

        ai_stores_result(결과 저장 방식)도 같은 UPDATE 로 이때 한 번만 정한다(이미 준비된 작업은 바꾸지 않음).
        기본 False 는 기존 uploadEndpoint 직접 업로드 방식이다.
        """
        if not isinstance(ai_stores_result, bool):
            raise TypeError("ai_stores_result 는 bool 이어야 합니다.")
        return self._update(
            fitting_job_id,
            "task_uuid = %(task_uuid)s, async_image_key = %(key)s, ai_stores_result = %(ai)s",
            """
            status = 'RECEIVED' AND next_action = 'SUBMIT_PRUNA'
            AND task_uuid IS NULL AND pruna_submit_started_at IS NULL
            """,
            {
                "token": token,
                "task_uuid": _check_uuid(task_uuid, "task_uuid"),
                "key": validate_s3_key(async_image_key),
                "ai": ai_stores_result,
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

        Webhook 이 먼저 와서 이미 진전된 상태(POSTPROCESSING·FALLBACK_RUNNING, 또는 RECEIVED 인 채 실패 결과와
        PUBLISH_RESULT 가 저장된 작업)는 되돌리지 않는다(next_action 이 SUBMIT_PRUNA 일 때만 옮김). 최초 접수
        시각은 처음 한 번만 기록한다.
        """
        return self._update(
            fitting_job_id,
            """
            runware_accepted_at = COALESCE(runware_accepted_at, statement_timestamp()),
            status = CASE WHEN status = 'RECEIVED' AND next_action = 'SUBMIT_PRUNA'
                          THEN 'WAITING_PRUNA' ELSE status END,
            next_action = CASE WHEN status = 'RECEIVED' AND next_action = 'SUBMIT_PRUNA'
                               THEN NULL ELSE next_action END
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
        task_uuid·호출 시작 기록이 있고 종료되지 않은 작업만 바꾼다. RECEIVED / SUBMIT_PRUNA 면 WAITING_PRUNA·
        next_action=NULL 로 바꾸고, 완료 알림 등으로 이미 진전된 상태(RECEIVED 인 채 실패 결과·PUBLISH_RESULT 가
        저장된 경우 포함)는 되돌리지 않는다. 최초 접수 시각은 처음 한 번만 기록한다.
        """
        return self._event_update(
            task_uuid,
            f"""
            runware_accepted_at = COALESCE(runware_accepted_at, {_NOW}),
            status = CASE WHEN status = 'RECEIVED' AND next_action = 'SUBMIT_PRUNA'
                          THEN 'WAITING_PRUNA' ELSE status END,
            next_action = CASE WHEN status = 'RECEIVED' AND next_action = 'SUBMIT_PRUNA'
                               THEN NULL ELSE next_action END
            """,
            "TRUE",
            {},
        )

    def record_auth_hold(self, task_uuid: str, reason: str) -> StateChange:
        """Runware 가 이 task_uuid 의 생성 시도에서 우리 API 키를 거절했다는 확인된 사실을 남긴다(운영 확인 보류).

        정상 Webhook 토큰으로 받은 invalidApiKey 또는 생성 요청의 HTTP 401·403 만 넘긴다. 아직 Pruna 결과를 기다리는 작업
        (RECEIVED·WAITING_PRUNA, 결과 없음, next_action 이 NULL·SUBMIT_PRUNA)만 바꾼다. 상태·next_action·결과·횟수는
        바꾸지 않는다. 처음 기록만 남기고(중복 수신은 ALREADY_RECORDED), 결과가 이미 있거나 다른 경로로 진행된 작업은
        바꾸지 않는다(STATE_MISMATCH·TERMINAL). 원인은 안전한 코드만 받는다.
        """
        if not isinstance(reason, str) or not _SAFE_CODE.match(reason):
            raise ValueError("auth_hold_reason 은 안전한 원인 코드여야 합니다.")
        change = self._event_update(
            task_uuid,
            "auth_hold_task_uuid = task_uuid, auth_hold_reason = %(reason)s, "
            f"auth_hold_at = {_NOW}",
            "status IN ('RECEIVED', 'WAITING_PRUNA') AND result_payload IS NULL "
            "AND (next_action IS NULL OR next_action = 'SUBMIT_PRUNA') AND auth_hold_task_uuid IS NULL",
            {"reason": reason},
        )
        current = change.job
        if (
            not change.applied
            and current is not None
            and current.auth_hold_task_uuid is not None
            and current.auth_hold_task_uuid == current.task_uuid
        ):
            return StateChange(False, ChangeReason.ALREADY_RECORDED, current)
        return change

    def record_pruna_succeeded(
        self, task_uuid: str, *, image_url: str | None = None, image_uuid: str | None = None
    ) -> StateChange:
        """Pruna 성공 알림. 접수 대기 중(RECEIVED·WAITING_PRUNA)이고 결과가 아직 없는 작업만 POSTPROCESSING /
        POSTPROCESS 로 바꾼다. 실패 결과가 먼저 저장됐으면(PUBLISH_RESULT) 늦은 성공 알림이 그 다음 작업을 덮어쓰지 않는다.

        image_url·image_uuid: 그 알림이 준 결과 참조(Runware imageURL·imageUUID). 상태 전환과 같은 UPDATE 로 저장하므로
        전환이 일어날 때만, 한 번만 남는다. 전환되지 않은 알림(중복·늦은 알림)은 기존 값을 바꾸지 않는다. 없으면 NULL.
        """
        return self._event_update(
            task_uuid,
            """
            status = 'POSTPROCESSING', next_action = 'POSTPROCESS',
            runware_accepted_at = COALESCE(runware_accepted_at, statement_timestamp()),
            runware_image_url = %(image_url)s, runware_image_uuid = %(image_uuid)s
            """,
            "status IN ('RECEIVED', 'WAITING_PRUNA') AND result_payload IS NULL",
            {
                "image_url": check_runware_image_url(image_url) if image_url is not None else None,
                "image_uuid": (
                    _check_uuid(image_uuid, "image_uuid") if image_uuid is not None else None
                ),
            },
        )

    def fill_pruna_image_ref(
        self,
        task_uuid: str,
        *,
        image_url: str | None = None,
        image_uuid: str | None = None,
        confirmed_by_query: bool = False,
    ) -> StateChange:
        """성공 전환 뒤 늦게 온 같은 결과의 참조로 비어 있는 runware_image_url·runware_image_uuid 만 채운다.

        잠금을 얻은 뒤의 UPDATE 조건으로만 판단한다(호출부가 읽은 상태는 믿지 않는다):
        - 등록된 task_uuid·호출 시작 기록, 후처리 중(POSTPROCESSING / POSTPROCESS)이고 결과 없음. 완료·실패·fallback 은 제외.
        - 충돌 없음: 이미 있는 값과 받은 값이 둘 다 있으면 같아야 한다(다르면 아무것도 쓰지 않아 참조를 섞지 않는다).
        - 같은 결과 근거: 이미 있는 참조 중 하나가 받은 값과 일치한다(근거 없이 연결하지 않음). 예외: confirmed_by_query 는
          호출부가 우리 인증으로 이 task 를 직접 조회(getResponse)해 결과 하나를 확인한 값이라는 뜻이다(외부 본문의 필드로
          정하지 않음). AI 저장 작업(ai_stores_result)에서만 이 근거로 빈 참조(둘 다 NULL 포함)를 채운다. 충돌 검사는 그대로다.
        - 채울 것이 있음: 비어 있는 컬럼에 받은 값이 있다.
        참조 컬럼 외에는 상태·next_action·시각·lease·횟수·updated_at 을 바꾸지 않는다(Watchdog 지연 판단에 영향 없음).
        """
        if image_url is None and image_uuid is None:
            raise ValueError("보완할 결과 참조가 없습니다.")
        params = {
            "task_uuid": _check_uuid(task_uuid, "task_uuid"),
            "image_url": check_runware_image_url(image_url) if image_url is not None else None,
            "image_uuid": _check_uuid(image_uuid, "image_uuid") if image_uuid is not None else None,
            "confirmed": bool(confirmed_by_query),
        }
        return self._run(
            f"SELECT fitting_job_id FROM {TABLE} WHERE task_uuid = %(task_uuid)s FOR UPDATE",
            f"""
            UPDATE {TABLE} SET
                runware_image_url = COALESCE(runware_image_url, %(image_url)s::text),
                runware_image_uuid = COALESCE(runware_image_uuid, %(image_uuid)s::uuid)
            WHERE fitting_job_id = %(job_id)s AND task_uuid = %(task_uuid)s
              AND pruna_submit_started_at IS NOT NULL
              AND status = 'POSTPROCESSING' AND next_action = 'POSTPROCESS' AND result_payload IS NULL
              AND (runware_image_url IS NULL OR %(image_url)s::text IS NULL
                   OR runware_image_url = %(image_url)s::text)
              AND (runware_image_uuid IS NULL OR %(image_uuid)s::uuid IS NULL
                   OR runware_image_uuid = %(image_uuid)s::uuid)
              AND (runware_image_url = %(image_url)s::text OR runware_image_uuid = %(image_uuid)s::uuid
                   OR (%(confirmed)s AND ai_stores_result))
              AND ((runware_image_url IS NULL AND %(image_url)s::text IS NOT NULL)
                   OR (runware_image_uuid IS NULL AND %(image_uuid)s::uuid IS NOT NULL))
            RETURNING {_SELECT_LIST}
            """,
            params,
            lambda job, now: (
                ChangeReason.TERMINAL
                if job.status.value in _TERMINAL
                else ChangeReason.STATE_MISMATCH
            ),
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

    # --- 확실한 미접수(재요청) ---

    def record_not_accepted(
        self,
        task_uuid: str,
        code: str,
        source: NotAcceptedSource,
        *,
        retry_after: float | None = None,
        retry_after_unrepresentable: bool = False,
    ) -> NotAcceptedChange:
        """현재 전송 시도(task)의 확실한 미접수를 한 트랜잭션으로 기록한다. 제출 응답·Webhook·상태 조회가 같이 쓴다.

        호출부 계약: 공통 분류(runware_errors)가 이 task 에 연결된 미접수 코드로 판단한 경우만 넘긴다(접수 불명확은 넘기지 않음).
        같은 트랜잭션에서: 시도 줄 확보(기존 작업은 현재 task 를 1번 시도로) → 미접수 근거 기록 → 다음 시각 예약(또는 마지막
        시도면 FITTING_SUBMISSION_REJECTED 결과·PUBLISH_RESULT 저장) → 그 시도의 진행 중 자리 반환. 하나라도 실패하면 모두
        되돌린다(FittingDatabaseError). 같은 시도를 다시 기록하면 아무것도 바꾸지 않는다(다음 시각을 다시 계산하지 않음).
        retry_after 는 Provider 가 고른 유효한 서버 대기 안내(초)다. 표현할 수 없으면 retry_after_unrepresentable=True(보류).
        """
        task = _check_uuid(task_uuid, "task_uuid")
        if not isinstance(code, str) or not _SAFE_CODE.match(code):
            raise ValueError("미접수 코드는 안전한 원인 코드여야 합니다.")
        source = NotAcceptedSource(source)
        if retry_after is not None and (
            isinstance(retry_after, bool)
            or not isinstance(retry_after, int | float)
            or not math.isfinite(retry_after)
            or retry_after < 0
        ):
            raise ValueError("retry_after 는 0 이상의 유한한 초여야 합니다.")
        from app.virtual_fitting.repositories.pruna_limiter import IN_FLIGHT, OUT_OF_RANGE_ERRORS

        def work(cursor) -> NotAcceptedChange:
            cursor.execute(
                f"SELECT fitting_job_id FROM {TABLE} WHERE task_uuid = %s FOR UPDATE", (task,)
            )
            locked = cursor.fetchone()
            if locked is None:
                cursor.execute(
                    f"SELECT fitting_job_id FROM {ATTEMPTS} WHERE task_uuid = %s", (task,)
                )
                past = cursor.fetchone()
                if past is None:
                    return NotAcceptedChange(NotAcceptedOutcome.NOT_FOUND, None)
                return NotAcceptedChange(
                    NotAcceptedOutcome.PAST_ATTEMPT, self._job_in(cursor, past[0])
                )
            job_id = locked[0]
            job = self._job_in(cursor, job_id)
            if job.pruna_submit_started_at is None:
                return NotAcceptedChange(NotAcceptedOutcome.STATE_MISMATCH, job)
            cursor.execute("SELECT statement_timestamp()")
            (now,) = cursor.fetchone()
            # 시도 줄: 이 표 이전에 시작한 작업은 줄이 하나도 없다 → 현재 task 를 1번 시도로(호출 시작 기록 = 1회 전송).
            # 이 표 이후의 시도는 전송 시작 트랜잭션이 줄을 이미 만들었다.
            cursor.execute(
                f"""
                INSERT INTO {ATTEMPTS} (fitting_job_id, attempt_no, task_uuid, started_at)
                SELECT %(job)s, 1, %(task)s, %(started)s
                WHERE NOT EXISTS (SELECT 1 FROM {ATTEMPTS} WHERE fitting_job_id = %(job)s)
                ON CONFLICT (task_uuid) DO NOTHING
                """,
                {"job": job_id, "task": task, "started": job.pruna_submit_started_at},
            )
            cursor.execute(
                f"SELECT attempt_no, not_accepted_at FROM {ATTEMPTS} WHERE task_uuid = %s", (task,)
            )
            row = cursor.fetchone()
            if row is None:  # 시도 기록이 현재 task 와 맞지 않는다(추측하지 않음)
                return NotAcceptedChange(NotAcceptedOutcome.STATE_MISMATCH, job)
            attempt_no, recorded_at = row
            if recorded_at is not None:
                return NotAcceptedChange(NotAcceptedOutcome.ALREADY_RECORDED, job, attempt_no)
            if job.status.value in _TERMINAL:
                return NotAcceptedChange(NotAcceptedOutcome.STATE_MISMATCH, job, attempt_no)
            if (
                job.runware_accepted_at is not None
                or job.verified_image_key is not None
                or job.status is not AsyncJobStatus.RECEIVED
            ):
                # 이 시도에 접수·생성 진행 근거가 이미 있다. 되돌리지 않고 모순만 남긴다(자동 재요청 없음, 운영 확인).
                self._mark_conflict(cursor, task, "not_accepted_after_progress")
                return NotAcceptedChange(NotAcceptedOutcome.CONTRADICTED, job, attempt_no)
            if (
                job.next_action is not NextAction.SUBMIT_PRUNA
                or job.result_payload is not None
                or job.auth_hold_task_uuid is not None
            ):
                return NotAcceptedChange(NotAcceptedOutcome.STATE_MISMATCH, job, attempt_no)
            cursor.execute(
                f"""
                UPDATE {ATTEMPTS} SET not_accepted_code = %s, not_accepted_source = %s,
                       not_accepted_at = %s
                WHERE task_uuid = %s
                """,
                (code, source.value, now, task),
            )
            outcome, retry_at = NotAcceptedOutcome.HELD, None
            if attempt_no >= PRUNA_SUBMIT_LIMIT:
                outcome = NotAcceptedOutcome.EXHAUSTED
                payload = {
                    "fitting_job_id": job_id,
                    "status": "FAILED",
                    "error_code": SUBMISSION_REJECTED_CODE,
                    "error_message": SUBMISSION_REJECTED_MESSAGE,
                }
                cursor.execute(
                    f"""
                    UPDATE {TABLE} SET result_payload = %s, next_action = 'PUBLISH_RESULT',
                           pruna_not_accepted_at = %s, pruna_retry_at = NULL, updated_at = %s
                    WHERE fitting_job_id = %s
                    """,
                    (Jsonb(_check_result_payload(payload)), now, now, job_id),
                )
            else:
                if not retry_after_unrepresentable:
                    delay = retry_delay_seconds(attempt_no, retry_after)
                    try:
                        with (
                            cursor.connection.transaction()
                        ):  # savepoint: 범위 오류면 이 UPDATE 만 되돌린다
                            cursor.execute(
                                f"""
                                UPDATE {ATTEMPTS}
                                SET next_attempt_at = %s::timestamptz + make_interval(secs => %s)
                                WHERE task_uuid = %s RETURNING next_attempt_at
                                """,
                                (now, delay, task),
                            )
                            (retry_at,) = cursor.fetchone()
                            outcome = NotAcceptedOutcome.SCHEDULED
                    except OUT_OF_RANGE_ERRORS:
                        retry_at = None  # 시각으로 표현할 수 없다: 줄이지 않고 보류
                cursor.execute(
                    f"""
                    UPDATE {TABLE} SET pruna_not_accepted_at = %s, pruna_retry_at = %s, updated_at = %s
                    WHERE fitting_job_id = %s
                    """,
                    (now, retry_at, now, job_id),
                )
            # 처리되지 않은 시도이므로 그 시도의 진행 중 자리만 반환한다(같은 트랜잭션).
            cursor.execute(
                f"DELETE FROM {IN_FLIGHT} WHERE fitting_job_id = %s AND task_uuid = %s",
                (job_id, task),
            )
            return NotAcceptedChange(outcome, self._job_in(cursor, job_id), attempt_no)

        return self._transaction(work, "확실한 미접수 기록")

    def record_attempt_contradiction(self, task_uuid: str, reason: str) -> bool:
        """미접수로 기록한 시도(현재·이전)에 접수·생성 결과 근거가 뒤늦게 왔다. 모순을 남기고, 아직 예약만 된 재요청이면
        자동 재요청을 보류(pruna_retry_at = NULL → 운영 확인)한다. 결과·상태는 바꾸지 않는다. 기록했으면 True."""
        task = _check_uuid(task_uuid, "task_uuid")
        if not isinstance(reason, str) or not _SAFE_CODE.match(reason):
            raise ValueError("모순 원인은 안전한 원인 코드여야 합니다.")

        def work(cursor) -> bool:
            cursor.execute(
                f"SELECT fitting_job_id, not_accepted_at FROM {ATTEMPTS} WHERE task_uuid = %s",
                (task,),
            )
            attempt = cursor.fetchone()
            if attempt is None or attempt[1] is None:
                return False
            cursor.execute(
                f"SELECT fitting_job_id FROM {TABLE} WHERE fitting_job_id = %s FOR UPDATE",
                (attempt[0],),
            )
            self._mark_conflict(cursor, task, reason)
            cursor.execute(
                f"""
                UPDATE {TABLE} SET pruna_retry_at = NULL, updated_at = statement_timestamp()
                WHERE fitting_job_id = %s AND pruna_retry_at IS NOT NULL
                  AND status = 'RECEIVED' AND next_action = 'SUBMIT_PRUNA' AND result_payload IS NULL
                """,
                (attempt[0],),
            )
            return True

        return self._transaction(work, "미접수 시도의 모순 기록")

    @staticmethod
    def _mark_conflict(cursor, task: uuid.UUID, reason: str) -> None:
        cursor.execute(
            f"""
            UPDATE {ATTEMPTS} SET conflict_reason = COALESCE(conflict_reason, %s),
                   conflict_at = COALESCE(conflict_at, statement_timestamp())
            WHERE task_uuid = %s
            """,
            (reason, task),
        )

    @staticmethod
    def _job_in(cursor, job_id: int) -> AsyncFittingJob:
        cursor.execute(f"SELECT {_SELECT_LIST} FROM {TABLE} WHERE fitting_job_id = %s", (job_id,))
        return _row_to_job(cursor.fetchone())

    def _transaction(self, work: Callable[[Any], Any], purpose: str) -> Any:
        try:
            with self._connection_pool.connection() as connection:
                connection.autocommit = True
                with connection.transaction(), connection.cursor() as cursor:
                    cursor.execute(_SET_LOCK_TIMEOUT, {"lock_timeout": self._lock_timeout})
                    return work(cursor)
        except psycopg.Error as error:
            raise FittingDatabaseError(f"AI PostgreSQL {purpose}에 실패했습니다.") from error

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
        """확인한 이미지로 코멘트 또는 제목을 생성하기(LLM 호출 1번) 직전에 횟수를 확보한다. 합계 최대 4번
        (코멘트 2 + 제목 2, 항목별 한도는 호출하는 쪽 title_and_comment 가 지킨다). 되돌리지 않는다.

        이미지가 있는 두 경로만 허용한다: POSTPROCESS(S3 에서 확인한 verified_image_key) 또는 RUN_FALLBACK(13단계,
        저장을 확인한 fallback_image_key). 같은 llm_generation_attempt_count 를 쓰므로 fallback 으로 넘어가도 초기화되지 않는다.
        """
        return self._claim(
            fitting_job_id,
            token,
            "llm_generation_attempt_count",
            LLM_GENERATION_LIMIT,
            """
            result_payload IS NULL AND (
                (status = 'POSTPROCESSING' AND next_action = 'POSTPROCESS'
                 AND verified_image_key IS NOT NULL)
                OR (status = 'FALLBACK_RUNNING' AND next_action = 'RUN_FALLBACK'
                    AND fallback_image_key IS NOT NULL)
            )
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

    def prepare_fallback(
        self, fitting_job_id: int, token: str, fallback_task_uuid: str
    ) -> StateChange:
        """fallback 생성 요청의 taskUUID 를 외부 호출 전에 저장한다. 이미 있으면 바꾸지 않는다(재수신 때 재사용).

        비동기 task_uuid 는 그대로 둔다(늦은 Webhook·작업 조회용). FALLBACK_RUNNING / RUN_FALLBACK, 결과 없음, 유효한 lease.
        """
        return self._update(
            fitting_job_id,
            "fallback_task_uuid = %(fallback_task)s",
            """
            status = 'FALLBACK_RUNNING' AND next_action = 'RUN_FALLBACK'
            AND result_payload IS NULL AND fallback_task_uuid IS NULL
            AND (task_uuid IS NULL OR task_uuid <> %(fallback_task)s)
            """,
            {
                "token": token,
                "fallback_task": _check_uuid(fallback_task_uuid, "fallback_task_uuid"),
            },
            lambda job, now: (
                ChangeReason.ALREADY_RECORDED
                if job.fallback_task_uuid is not None
                else ChangeReason.STATE_MISMATCH
            ),
        )

    def record_fallback_image(self, fitting_job_id: int, token: str, image_key: str) -> StateChange:
        """fallback 생성 결과를 S3 에 저장했다는 기록. 이 값이 있으면 생성·저장은 끝났다(제목·코멘트부터 이어 간다).

        생성 요청을 시작한 기록(fallback_started_at)이 있어야 하고, 이미 있으면 바꾸지 않는다(ALREADY_RECORDED).
        """
        return self._update(
            fitting_job_id,
            "fallback_image_key = %(key)s",
            """
            status = 'FALLBACK_RUNNING' AND next_action = 'RUN_FALLBACK'
            AND result_payload IS NULL AND fallback_started_at IS NOT NULL
            AND fallback_image_key IS NULL
            """,
            {"token": token, "key": validate_s3_key(image_key)},
            lambda job, now: (
                ChangeReason.ALREADY_RECORDED
                if job.fallback_image_key is not None
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

    def claim_idm_vton_attempt(self, fitting_job_id: int, token: str) -> StateChange:
        """IDM-VTON(Modal) 생성 함수를 부르기 직전에 작업당 1회의 호출 기회를 원자적으로 선점한다.

        한 UPDATE 로 fallback_attempt_count + 1 과 fallback_started_at(DB 시각)을 함께 기록한다. 유효한 lease,
        FALLBACK_RUNNING / RUN_FALLBACK, 결과·확인 이미지·fallback 이미지 없음, 시작 기록 없음, 횟수 < FALLBACK_LIMIT 일 때만.
        선점 뒤 호출 전에 Worker 가 멈춰도 되돌리지 않는다(중복 호출 방지가 우선, 복구는 S3 확인만).
        Pruna 동기 fallback 의 claim_fallback_attempt·PrunaLimiter.begin_fallback 과 별개이며 공유 제한 자리를 쓰지 않는다.
        """

        def diagnose(job: AsyncFittingJob, now: Any) -> ChangeReason:
            if job.fallback_attempt_count >= FALLBACK_LIMIT or job.fallback_started_at is not None:
                return ChangeReason.LIMIT_REACHED
            if job.result_payload is not None or job.fallback_image_key is not None:
                return ChangeReason.ALREADY_RECORDED
            return ChangeReason.STATE_MISMATCH

        return self._update(
            fitting_job_id,
            "fallback_attempt_count = fallback_attempt_count + 1, "
            f"fallback_started_at = {_NOW}",
            """
            status = 'FALLBACK_RUNNING' AND next_action = 'RUN_FALLBACK'
            AND result_payload IS NULL AND verified_image_key IS NULL
            AND fallback_image_key IS NULL AND fallback_started_at IS NULL
            AND fallback_attempt_count < %(limit)s
            """,
            {"token": token, "limit": FALLBACK_LIMIT},
            diagnose,
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
        """한 트랜잭션: 행 잠금(여기서 대기) → 잠금을 쥔 채 실제 시각으로 조건부 UPDATE → 실패 이유 조회.

        잠금 대기는 lock_timeout(기본 5초, 이 트랜잭션에만)으로 제한한다. 넘으면 PostgreSQL 이 트랜잭션을 되돌리고
        (아무것도 바뀌지 않음) FittingDatabaseError 로 올린다. 여기서 다시 시도하지 않는다(재수신 때 최신 기록으로 이어 간다).
        """
        try:
            with self._connection_pool.connection() as connection:
                connection.autocommit = True
                with connection.transaction(), connection.cursor() as cursor:
                    cursor.execute(_SET_LOCK_TIMEOUT, {"lock_timeout": self._lock_timeout})
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
