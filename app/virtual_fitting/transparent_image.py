"""배경 제거: 저장한 Pruna 원본 → Runware RemBG → 투명 WebP 를 별도 Key 에 저장 (V2 비동기·보존된 V1 fallback 공용).

실제 RUN_FALLBACK 은 IDM-VTON(idm_fallback_handler)이고 그 결과(idm-vton.webp)에는 배경 제거를 하지 않는다.

원본(V2 pruna-async.webp / V1 fallback pruna-sync.webp, 이전 작업은 .jpg)은 그대로 두고, 같은 작업 폴더의 transparent.webp 에
결과를 저장한다. 최종 결과(result_image_key)와 LLM 입력은 이 투명 이미지다.

RemBG 는 작업당 최대 1회만 부른다(자동 재시도 없음). DB 스키마를 바꾸지 않고 두 근거로 1회를 지킨다.
  - S3 호출 표시(rembg-call.json): 호출 직전에 조건부 PUT(같은 Key 가 없을 때만)으로 만든다. 동시에 들어온 Worker 중
    하나만 만들 수 있고, 실패 원인(state=failed)을 남긴다.
  - DB 호출 자리(virtual_fitting_task_costs, 작업별로 정해진 RemBG taskUUID 의 cost 0 행): 표시를 만든 뒤 호출 전에 만든다.
    표시를 지워도 이 행이 남아 다시 부르지 않는다(표시를 지워 재실행하는 복구는 막는다).
  0. transparent.webp 에 이 작업·원본 task 의 Metadata·sha256·Alpha 검증이 맞는 객체가 있으면 재사용한다.
  1. 호출 표시가 "failed" 면 다시 부르지 않고 최종 실패(BackgroundRemovalFailed)로 이어 간다.
     "called" 인데 결과가 없으면(응답 전 중단·결과 저장 실패 등) 호출 여부·결과가 불명확하다. 다시 부르지 않고
     운영 확인으로 보류한다(BackgroundRemovalPending). 표시가 없는데 DB 호출 자리가 있어도(표시 삭제·유실) 보류한다.
  2. 호출 표시를 만든 실행만 한 번 부른다. 실패(timeout·429·5xx·오류 응답 포함)는 다시 부르지 않고 표시를 "failed" 로 바꾼 뒤
     최종 실패. 받은 결과를 쓸 수 없을 때(Alpha 없음 등)도 같다.
  3. 결과 내려받기·S3 저장의 일시 오류는 같은 결과 URL 로만 짧게 다시 해 본다(RemBG 재호출 아님).
비용은 호출 taskUUID(작업·원본 task 로 정해지는 값)의 자리에 한 번만 넣고 하루 사용액에 더한다. 원본 객체는 읽기만 하고,
URL·token 은 로그·표시 객체에 남기지 않는다. DB 실행 권한(lease)은 호출부가 쥔 채 부른다.
"""

import hashlib
import json
import logging
import secrets
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

from PIL import Image

from app.clients.s3 import (
    MAX_REMOTE_IMAGE_BYTES,
    ConditionalPutOutcome,
    ImageStorageError,
    S3ImageStorage,
    S3ObjectState,
    validate_s3_key,
)
from app.virtual_fitting.async_storage import result_image_get_url
from app.virtual_fitting.exceptions import VirtualFittingError
from app.virtual_fitting.providers.runware_rembg import RembgError, RembgResult
from app.virtual_fitting.result_image import (
    WEBP_CONTENT_TYPE,
    ResultImageError,
    StoredResultImage,
    _existing,
    _metadata,
    validate_static_webp,
)

logger = logging.getLogger(__name__)

TRANSPARENT_RESULT_FILENAME = "transparent.webp"
REMBG_CALL_FILENAME = "rembg-call.json"
CALLED, FAILED = "called", "failed"
# 받은 결과의 내려받기·저장 일시 오류를 같은 URL 로 다시 해 보는 횟수(최초 포함)와 간격(초)
STORE_TRIES = 3
STORE_RETRY_SECONDS = 1.0
# 받은 결과 자체를 쓸 수 없는 경우(최종 실패). 그 밖의 다운로드·S3 오류는 일시 오류로 본다.
_UNUSABLE_RESULT = frozenset(
    {
        "redirect",
        "address_not_allowed",
        "too_large",
        "empty",
        "not_webp",
        "animated",
        "too_many_pixels",
        "corrupt",
        "no_alpha",
        "no_transparency",
        "no_foreground",
    }
)
# RemBG taskUUID 는 작업·원본 task 로 정한다(같은 작업이면 늘 같다 → 비용 기록이 겹치지 않고 로그로 추적된다).
_REMBG_TASK_NAMESPACE = uuid.UUID("6f1d3c52-2b7e-4f0a-9d61-7c2a4e8b9f10")
_clock = time.perf_counter


class BackgroundRemovalFailed(Exception):
    """RemBG 를 한 번 불렀고 실패했다(또는 받은 결과를 쓸 수 없다). 다시 부르지 않는다. 최종 실패 대상이다."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BackgroundRemovalPending(Exception):
    """지금은 이어 갈 수 없다. 다시 부르지 않고 보류한다(호출 여부·결과 불명확이면 운영 확인 대상)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class AlphaStats:
    width: int
    height: int
    mode: str
    alpha_min: int
    alpha_max: int
    # Alpha < 255(투명·반투명) 픽셀 비율과 Alpha == 0(완전 투명) 픽셀 비율
    translucent_ratio: float
    clear_ratio: float


@dataclass(frozen=True)
class TransparentImage:
    key: str
    byte_count: int
    sha256: str
    # True 면 이번 실행에서 RemBG 를 부르지 않고 저장된 객체를 확인해 재사용했다.
    reused: bool
    alpha: AlphaStats | None = None
    rembg: RembgResult | None = None
    rembg_task_uuid: str | None = None
    # 받은 결과의 내려받기·저장 시도 수(RemBG 호출 수가 아님, 최초 포함)
    store_tries: int = 0
    # perf_counter 구간(ms): 기존 결과 확인(S3 GET) / RemBG API 호출 / 결과 다운로드·검증·S3 저장
    # (그중 마지막 시도의 다운로드, 조건부 PUT·확인)
    check_ms: int | None = None
    rembg_ms: int | None = None
    store_ms: int | None = None
    download_ms: int | None = None
    put_ms: int | None = None


def transparent_result_key(fitting_job_id: int) -> str:
    """투명 결과 Key. 원본(pruna-async.webp·pruna-sync.webp)과 같은 작업 폴더, 같은 작업이면 항상 같다."""
    return f"virtual-fitting/results/{fitting_job_id}/{TRANSPARENT_RESULT_FILENAME}"


def rembg_call_key(fitting_job_id: int) -> str:
    """RemBG 호출 표시 객체 Key(작업당 1회 호출 기록)."""
    return f"virtual-fitting/results/{fitting_job_id}/{REMBG_CALL_FILENAME}"


def rembg_task_uuid(fitting_job_id: int, source_task: str) -> str:
    return str(uuid.uuid5(_REMBG_TASK_NAMESPACE, f"{fitting_job_id}:{source_task}"))


def validate_transparent_webp(body: bytes) -> AlphaStats:
    """정상 정지 WebP 이고, Alpha 채널이 있고, 완전 투명 픽셀과 완전 불투명 픽셀이 모두 있어야 한다.

    완전 투명이 없으면 배경이 지워지지 않은 것(no_transparency), 완전 불투명이 없으면 인물까지 지워진 것(no_foreground)이다.
    """
    validate_static_webp(body)
    try:
        with Image.open(BytesIO(body)) as image:
            if "A" not in image.getbands():
                raise ResultImageError("no_alpha")
            alpha = image.getchannel("A")
            low, high = alpha.getextrema()
            histogram = alpha.histogram()
            width, height = image.size
            mode = image.mode
    except ResultImageError:
        raise
    except (OSError, ValueError) as error:
        raise ResultImageError("corrupt", detail=type(error).__name__) from None
    total = width * height
    stats = AlphaStats(
        width,
        height,
        mode,
        low,
        high,
        round((total - histogram[255]) / total, 4),
        round(histogram[0] / total, 4),
    )
    if low > 0:
        raise ResultImageError("no_transparency")
    if high < 255:
        raise ResultImageError("no_foreground")
    return stats


def _ms(started: float) -> int:
    return int((_clock() - started) * 1000)


def _marker(job_id: int, source_task: str, rembg_task: str, state: str, claim: str, reason=None):
    body = {
        "fitting_job_id": job_id,
        "source_task_uuid": source_task,
        "rembg_task_uuid": rembg_task,
        "state": state,
        "claim": claim,
        "at": datetime.now(UTC).isoformat(),
    }
    if reason is not None:
        body["reason"] = reason
    return json.dumps(body, sort_keys=True).encode()


def _read_marker(storage, key: str, job_id: int, source_task: str) -> dict | None:
    """호출 표시를 읽는다. 없으면 None. 읽지 못하거나 다른 작업·원본의 표시면 보류(다시 부르지 않음)."""
    check, found = storage.get_object_limited(key, 4096)
    if check.state is S3ObjectState.NOT_FOUND:
        return None
    if check.state is not S3ObjectState.EXISTS or found is None:
        raise BackgroundRemovalPending(f"rembg_call_check_{check.state.value}")
    try:
        marker = json.loads(found.body)
    except (ValueError, UnicodeDecodeError):
        raise BackgroundRemovalPending("rembg_call_marker_unreadable") from None
    if (
        not isinstance(marker, dict)
        or marker.get("fitting_job_id") != job_id
        or marker.get("source_task_uuid") != source_task
        or marker.get("state") not in (CALLED, FAILED)
    ):
        raise BackgroundRemovalPending("rembg_call_marker_conflict")
    return marker


def store_transparent_image(
    storage: S3ImageStorage,
    *,
    fitting_job_id: int,
    task_uuid: str,
    source_key: str,
    rembg: Any,
    costs: Any = None,
    download: Callable[[str], bytes] | None = None,
    source_url_expires_seconds: int | None = None,
    store_tries: int | None = None,
    sleep: Callable[[float], None] | None = None,
    max_bytes: int = MAX_REMOTE_IMAGE_BYTES,
) -> TransparentImage:
    """source_key(저장한 Pruna 원본)의 배경을 지워 transparent_result_key 에 저장하거나, 저장된 결과를 재사용한다.

    task_uuid 는 원본을 만든 생성 task(V2 task_uuid, V1 fallback 은 fallback_task_uuid)로 결과 Metadata 에 남긴다.
    costs 는 TaskCostRepository. 입력이 잘못되면 ValueError(외부 호출 전). 반환 전 예외:
      BackgroundRemovalFailed(최종 실패 대상) / BackgroundRemovalPending(보류, 다시 부르지 않음) /
      ResultImageError(기존 결과 확인 실패·받은 결과의 저장 일시 오류, 보류) / ImageStorageError(원본 URL 서명 실패, 호출 전).
    """
    try:
        task = str(uuid.UUID(task_uuid))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("task_uuid 는 UUID 문자열이어야 합니다.") from error
    key = transparent_result_key(fitting_job_id)
    marker_key = rembg_call_key(fitting_job_id)
    source = validate_s3_key(source_key)
    if source in (key, marker_key):
        raise ValueError("source_key 가 배경 제거 결과 Key 와 같습니다.")
    rembg_task = rembg_task_uuid(fitting_job_id, task)

    # 0. 저장된 결과 재사용
    started = _clock()
    try:
        existing = _existing(
            storage,
            key,
            fitting_job_id,
            task,
            max_bytes=max_bytes,
            validate=validate_transparent_webp,
        )
    except ResultImageError as error:
        if error.reason != "s3_access_denied":
            raise
        existing = None  # 403: 있는지 모른다. 아래 호출 표시가 1회를 지킨다
    check_ms = _ms(started)
    if existing is not None:
        _log(fitting_job_id, task, "reused_existing", existing)
        return TransparentImage(
            existing.key, existing.byte_count, existing.sha256, reused=True, check_ms=check_ms
        )

    # 1. 이미 부른 적이 있으면 다시 부르지 않는다
    marker = _read_marker(storage, marker_key, fitting_job_id, task)
    if marker is not None:
        if marker["state"] == FAILED:
            raise BackgroundRemovalFailed(str(marker.get("reason") or "failed"))
        logger.error(
            "rembg called before without a stored result fitting_job_id=%s rembg_task_uuid=%s; "
            "not calling again, operator check required",
            fitting_job_id,
            rembg_task,
        )
        raise BackgroundRemovalPending("rembg_call_unconfirmed")
    if costs is not None and costs.exists(rembg_task):
        # DB 에는 호출 자리가 있는데 S3 표시가 없다(지워졌거나 유실). 다시 부르지 않는다.
        logger.error(
            "rembg call marker missing but the call was recorded fitting_job_id=%s "
            "rembg_task_uuid=%s; not calling again, operator check required",
            fitting_job_id,
            rembg_task,
        )
        raise BackgroundRemovalPending("rembg_call_marker_missing")

    # 2. 호출 표시(조건부 PUT)를 만든 실행만 한 번 부른다. 원본 URL 서명은 표시 전에 한다(실패해도 호출 기회를 쓰지 않음).
    source_url = result_image_get_url(storage, source, expires_in=source_url_expires_seconds)
    claim = secrets.token_hex(8)
    put = storage.put_object_if_absent(
        _marker(fitting_job_id, task, rembg_task, CALLED, claim),
        marker_key,
        content_type="application/json",
        metadata={"fitting-job-id": str(fitting_job_id), "task-uuid": task},
    )
    if put.outcome is ConditionalPutOutcome.FAILED:
        raise BackgroundRemovalPending("rembg_call_marker_put_failed")  # 만들지 못했다(부르지 않음)
    if put.outcome is not ConditionalPutOutcome.CREATED:
        # 다른 실행이 먼저 만들었거나 응답을 잃었다. 우리 표시로 확인될 때만 이어 간다.
        current = _read_marker(storage, marker_key, fitting_job_id, task)
        if current is None or current.get("claim") != claim or current["state"] != CALLED:
            raise BackgroundRemovalPending("rembg_call_unconfirmed")
    # DB 호출 자리. 이미 있으면(표시를 지운 뒤 다시 만든 경우 등) 부르지 않는다. DB 오류면 부르지 않고 예외를 올린다
    # (표시는 "called" 로 남아 이후 실행은 보류한다 — 보내지 않은 호출을 두 번 보낼 위험보다 운영 확인을 택한다).
    if costs is not None and not costs.claim(rembg_task, fitting_job_id, datetime.now(UTC).date()):
        raise BackgroundRemovalPending("rembg_call_marker_missing")

    def fail(reason: str) -> BackgroundRemovalFailed:
        try:
            storage.upload_bytes(
                _marker(fitting_job_id, task, rembg_task, FAILED, claim, reason),
                marker_key,
                "application/json",
            )
        except ImageStorageError:
            # 표시를 못 바꿔도 최종 실패 저장은 이어 간다(그마저 실패하면 다음 실행은 "called" 로 보고 보류).
            logger.exception("rembg failure marker not saved fitting_job_id=%s", fitting_job_id)
        return BackgroundRemovalFailed(reason)

    started = _clock()
    try:
        result = rembg.remove_background(source_url, task_uuid=rembg_task)
    except RembgError as error:
        logger.error(
            "rembg call failed fitting_job_id=%s rembg_task_uuid=%s reason=%s http_status=%s "
            "code=%s; not retried",
            fitting_job_id,
            rembg_task,
            error.reason,
            error.http_status,
            error.code,
        )
        raise fail(error.error_code) from None
    rembg_ms = _ms(started)
    _record_cost(costs, fitting_job_id, rembg_task, result.cost)

    # 3. 받은 결과 내려받기·검증·저장(일시 오류만 같은 URL 로 다시)
    if download is None:
        # 호출 때 모듈에서 찾는다(통합 테스트가 외부 경계만 바꿀 수 있게).
        from app.virtual_fitting import result_image

        download = result_image.download_runware_image
    tries = STORE_TRIES if store_tries is None else store_tries
    pause = sleep or time.sleep
    started = _clock()
    for attempt in range(1, tries + 1):
        try:
            step = _clock()
            body = download(result.image_url)
            download_ms = _ms(step)
            alpha = validate_transparent_webp(body)
            digest = hashlib.sha256(body).hexdigest()
            step = _clock()
            stored, outcome = _put(storage, body, key, fitting_job_id, task, digest, max_bytes)
            put_ms = _ms(step)
            break
        except ResultImageError as error:
            unusable = error.reason in _UNUSABLE_RESULT or (
                error.reason == "http_error" and not error.retryable
            )
            if unusable:
                reason = (
                    f"result_http_{error.http_status}"
                    if error.reason == "http_error"
                    else f"result_{error.reason}"
                )
                logger.error(
                    "rembg result unusable fitting_job_id=%s rembg_task_uuid=%s reason=%s",
                    fitting_job_id,
                    rembg_task,
                    reason,
                )
                raise fail(reason) from None
            logger.warning(
                "transparent result store failed fitting_job_id=%s try=%s/%s reason=%s",
                fitting_job_id,
                attempt,
                tries,
                error.reason,
            )
            if not error.retryable or attempt == tries:
                raise
            pause(STORE_RETRY_SECONDS * attempt)
    _log(fitting_job_id, task, outcome, stored, alpha)
    return TransparentImage(
        stored.key,
        stored.byte_count,
        stored.sha256,
        reused=False,
        alpha=alpha,
        rembg=result,
        rembg_task_uuid=rembg_task,
        store_tries=attempt,
        check_ms=check_ms,
        rembg_ms=rembg_ms,
        store_ms=_ms(started),
        download_ms=download_ms,
        put_ms=put_ms,
    )


def _put(storage, body, key, job_id, task, digest, max_bytes) -> tuple[StoredResultImage, str]:
    put = storage.put_object_if_absent(
        body, key, content_type=WEBP_CONTENT_TYPE, metadata=_metadata(job_id, task, digest)
    )
    if put.outcome is ConditionalPutOutcome.CREATED:
        return StoredResultImage(key, len(body), digest, reused=False), "created"
    if put.outcome is ConditionalPutOutcome.FAILED:
        raise ResultImageError("s3_put_failed", http_status=put.http_status, detail=put.error_code)
    # EXISTS·CONFLICT·UNKNOWN: 응답을 잃었거나 같은 결과가 이미 있다. 이 작업·원본 task 의 정상 결과면 그것을 쓴다.
    confirmed = _existing(
        storage, key, job_id, task, max_bytes=max_bytes, validate=validate_transparent_webp
    )
    if confirmed is None:
        raise ResultImageError(
            "s3_put_unconfirmed", http_status=put.http_status, detail=put.outcome.value
        )
    stored = StoredResultImage(
        key, confirmed.byte_count, confirmed.sha256, reused=confirmed.sha256 != digest
    )
    return stored, f"confirmed_after_{put.outcome.value}"


def _record_cost(costs: Any, job_id: int, rembg_task: str, cost: Any) -> None:
    """호출 자리(cost 0)에 응답 비용을 한 번만 넣고 하루 사용액에 더한다. 실패해도 결과 처리는 계속한다."""
    if cost is None:
        logger.warning("rembg cost missing fitting_job_id=%s; not recorded", job_id)
        return
    if costs is None:
        return
    try:
        costs.settle(rembg_task, cost)
    except ValueError:
        logger.warning("rembg cost invalid fitting_job_id=%s; not recorded", job_id)
    except VirtualFittingError:
        # 기록하지 못한 비용은 Runware 사용량 보정(runware:109@1 포함)이 메운다.
        logger.exception("rembg cost record failed fitting_job_id=%s", job_id)


def _log(
    fitting_job_id: int,
    task: str,
    result: str,
    stored: StoredResultImage,
    alpha: AlphaStats | None = None,
) -> None:
    logger.info(
        "transparent result image stored fitting_job_id=%s task_uuid=%s result=%s key=%s bytes=%s "
        "mode=%s alpha_min=%s alpha_max=%s translucent_ratio=%s",
        fitting_job_id,
        task,
        result,
        stored.key,
        stored.byte_count,
        alpha.mode if alpha else None,
        alpha.alpha_min if alpha else None,
        alpha.alpha_max if alpha else None,
        alpha.translucent_ratio if alpha else None,
    )
