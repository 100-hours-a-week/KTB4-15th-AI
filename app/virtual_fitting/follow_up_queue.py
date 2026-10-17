"""가상피팅 V2: ai-fitting-response(내부 후처리 큐) 메시지 형식 검증과 한 번의 수신 (10단계).

9단계 Webhook(runware_webhook.follow_up_message)이 보내는 형식 그대로다(모든 필드 필수, 다른 필드 없음):
    {"fitting_job_id": 123, "task_uuid": "a770f077-f413-47de-9dac-be0b26a35da6", "action": "POSTPROCESS"}
  - fitting_job_id: BIGINT 범위의 양의 정수(_check_job_id 재사용, bool·문자열·실수 거절)
  - task_uuid: UUID 문자열(소문자 정규화)
  - action: POSTPROCESS 또는 RUN_FALLBACK 만(PUBLISH_RESULT 는 이 큐로 보내지 않는다, 14단계)
메시지는 작업을 깨우는 신호다. 원본 입력·이미지 Key·URL 은 담지 않고, 처리 정보와 판단 기준은 DB 에서 읽는다.
그래서 약속하지 않은 필드가 있으면 형식 오류로 본다(큐 메시지로 입력을 넘기는 확장을 막는다).

수신은 SqsQueueClient.receive 를 한 번 부른다(SDK 재시도 없음). 삭제·Visibility 연장·DB 기록은 하지 않는다.
ReceiptHandle·본문은 로그·repr 에 남기지 않는다.
"""

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.clients.sqs import FittingQueue, ReceivedMessage, SqsQueueClient, SqsResult
from app.virtual_fitting.repositories.async_job_repository import NextAction, _check_job_id

logger = logging.getLogger(__name__)

FIELDS = frozenset({"fitting_job_id", "task_uuid", "action"})
ACTIONS = (NextAction.POSTPROCESS, NextAction.RUN_FALLBACK)


class FollowUpErrorKind(str, Enum):
    INVALID_JSON = "invalid_json"
    NOT_OBJECT = "not_object"
    MISSING_FIELD = "missing_field"
    UNEXPECTED_FIELD = "unexpected_field"
    INVALID_FIELD = "invalid_field"


class InvalidFollowUpMessage(ValueError):
    """형식 오류. 원본 값은 담지 않는다(필드 이름만)."""

    def __init__(self, kind: FollowUpErrorKind, field: str | None) -> None:
        super().__init__(f"{kind.value}: {field or '-'}")
        self.kind = kind
        self.field = field


@dataclass(frozen=True)
class FollowUpMessage:
    fitting_job_id: int
    task_uuid: str
    action: NextAction


def parse_follow_up_message(payload: Any) -> FollowUpMessage:
    if not isinstance(payload, Mapping):
        raise InvalidFollowUpMessage(FollowUpErrorKind.NOT_OBJECT, None)
    for name in sorted(FIELDS):
        if name not in payload:
            raise InvalidFollowUpMessage(FollowUpErrorKind.MISSING_FIELD, name)
    extra = sorted(set(payload) - FIELDS)
    if extra:
        raise InvalidFollowUpMessage(FollowUpErrorKind.UNEXPECTED_FIELD, extra[0])
    try:
        job_id = _check_job_id(payload["fitting_job_id"])
    except ValueError as error:
        raise InvalidFollowUpMessage(FollowUpErrorKind.INVALID_FIELD, "fitting_job_id") from error
    task = payload["task_uuid"]
    if not isinstance(task, str):
        raise InvalidFollowUpMessage(FollowUpErrorKind.INVALID_FIELD, "task_uuid")
    try:
        task = str(uuid.UUID(task))
    except ValueError as error:
        raise InvalidFollowUpMessage(FollowUpErrorKind.INVALID_FIELD, "task_uuid") from error
    action = payload["action"]
    if not isinstance(action, str) or action not in {a.value for a in ACTIONS}:
        raise InvalidFollowUpMessage(FollowUpErrorKind.INVALID_FIELD, "action")
    return FollowUpMessage(job_id, task, NextAction(action))


def readable_job_id(payload: Any) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        return _check_job_id(payload.get("fitting_job_id"))
    except ValueError:
        return None


@dataclass(frozen=True, repr=False)
class _Delivery:
    message_id: str
    receipt_handle: str
    # 수신 기록용. 실행 횟수·최종 실패 판단에 쓰지 않는다.
    approximate_receive_count: int | None


@dataclass(frozen=True, repr=False)
class ValidFollowUp(_Delivery):
    message: FollowUpMessage

    def __repr__(self) -> str:  # ReceiptHandle 은 남기지 않는다.
        return f"ValidFollowUp(message_id={self.message_id!r}, message={self.message!r})"


@dataclass(frozen=True, repr=False)
class InvalidFollowUp(_Delivery):
    error: InvalidFollowUpMessage
    fitting_job_id: int | None

    def __repr__(self) -> str:
        return (
            f"InvalidFollowUp(message_id={self.message_id!r}, kind={self.error.kind.value!r}, "
            f"field={self.error.field!r}, fitting_job_id={self.fitting_job_id!r})"
        )


@dataclass(frozen=True)
class FollowUpBatch:
    """한 번의 수신 결과. RequestWorker 루프가 쓰는 모양(received·receive_result·messages)과 같다."""

    receive_result: SqsResult
    messages: tuple[ValidFollowUp | InvalidFollowUp, ...] = ()

    @property
    def received(self) -> bool:
        return self.receive_result.succeeded


def _classify(message: ReceivedMessage) -> ValidFollowUp | InvalidFollowUp:
    delivery = {
        "message_id": message.message_id,
        "receipt_handle": message.receipt_handle,
        "approximate_receive_count": message.approximate_receive_count,
    }
    if message.parse_error is not None:
        kind = (
            FollowUpErrorKind.NOT_OBJECT
            if message.parse_error == "not_object"
            else FollowUpErrorKind.INVALID_JSON
        )
        return InvalidFollowUp(
            **delivery, error=InvalidFollowUpMessage(kind, None), fitting_job_id=None
        )
    try:
        return ValidFollowUp(**delivery, message=parse_follow_up_message(message.payload))
    except InvalidFollowUpMessage as error:
        return InvalidFollowUp(
            **delivery, error=error, fitting_job_id=readable_job_id(message.payload)
        )


def receive_follow_ups(
    sqs: SqsQueueClient,
    *,
    max_messages: int = 1,
    wait_time_seconds: int = 20,
    visibility_timeout: int | None = None,
) -> FollowUpBatch:
    """ai-fitting-response 를 한 번 받아(long polling) 메시지마다 검증한다. 큐 URL 설정 오류는 예외로 올라간다."""
    result = sqs.receive(
        FittingQueue.AI_FITTING_RESPONSE,
        max_messages=max_messages,
        wait_time_seconds=wait_time_seconds,
        visibility_timeout=visibility_timeout,
    )
    if not result.succeeded:
        return FollowUpBatch(receive_result=result)
    messages = tuple(_classify(message) for message in result.messages)
    if messages:
        logger.info(
            "ai-fitting-response received=%s invalid=%s",
            len(messages),
            sum(isinstance(item, InvalidFollowUp) for item in messages),
        )
    return FollowUpBatch(receive_result=result, messages=messages)
