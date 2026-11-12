"""가상피팅 V2 RUN_FALLBACK 핸들러(IDM-VTON): 입력 검증 → 호출 기회 1회 선점 → Modal 1회 → S3 WebP 확인 → 제목·코멘트 → 결과 저장.

10단계 후처리 Worker 가 실행 권한(token)을 확보한 뒤 FALLBACK_RUNNING / RUN_FALLBACK 작업의 최신 기록으로 부른다.
기존 RunFallbackHandler(Pruna V1 동기 생성, fallback_handler.py)는 그대로 두고 실제 연결만 이 핸들러로 바꾼다. Pruna 동기
Provider·PrunaLimiter·Runware 잔액·비용 기록을 쓰지 않고, IDM-VTON 이 실패해도 Pruna 동기 API 로 넘어가지 않는다.

순서(Modal 호출 전 실패는 호출 기회를 쓰지 않는다):
  0. 결과가 있으면 끝. fallback_image_key 가 있으면 4 부터. 시작 기록(fallback_started_at)이 있으면 생성하지 않고 복구(5).
  1. 입력: 저장된 상품 코드로 Pruna 와 같은 상품 검증(select_fitting_products) → IDM 조합(상의 1개) → 상품 이미지 분류
     (GARMENT_ONLY 만) → 사용자 원본 HEAD. 지원하지 않는 입력은 FITTING_MODEL_FAILED 실패 결과(Modal 미호출).
     상품 이미지 내려받기 일시 오류·원본 HEAD 일시 오류·Modal 함수 조회 실패·서명 실패는 보류(횟수 미사용).
  2. 시도 ID(fallback_task_uuid, prepare_fallback 재사용)를 저장하고 Presigned GET(원본)·PUT(결과, Content-Type 과
     x-amz-meta-fitting-job-id·attempt-id 를 서명에 포함)을 만든다.
  3. claim_idm_vton_attempt: fallback_attempt_count + 1·fallback_started_at 을 한 트랜잭션으로(작업당 1회).
     → spawn 1회(재시도 없음) → 결과 대기(실행 상태 조회만, 다시 부르지 않음).
     - 성공 보고: 고정 Key 객체를 HEAD·GET 으로 확인(WebP·메타데이터·sha256 일치) → record_fallback_image.
     - 함수가 끝난 실패: 고정 Key 를 한 번 확인(올라와 있으면 사용) → 없으면 FITTING_MODEL_FAILED 실패 결과.
     - 결과 불명확(spawn·연결 오류)·대기 시간 초과·Worker 종료·lease 상실: 보류. 실패로 확정하지 않는다.
  4. 저장한 이미지로 기존 title_and_comment(LLM 횟수·기본 문구 정책 그대로) → save_completed(COMPLETED + PUBLISH_RESULT).
  5. 복구(재수신·Worker 재시작·Watchdog 재발행): 고정 Key 확인만 한다. 올바른 객체 → 기록 후 4. 없음 + 업로드 URL 만료
     (시작 + IDM_VTON_UPLOAD_URL_EXPIRES_SECONDS + 여유) 전 → 아직 실행 중일 수 있어 보류. 만료 뒤에도 없음 → 실패 결과.
     형식·메타데이터가 다른 객체 → 실패 결과. HEAD·GET 일시 오류 → 보류.
결과 발행·최종 상태 확정은 결과 전송 루프(14단계)가 한다. Presigned URL 은 저장·로그하지 않는다.
"""

import dataclasses
import hashlib
import io
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from app.clients.s3 import ImageStorageError, S3ObjectState
from app.virtual_fitting.exceptions import (
    FittingDatabaseError,
    FittingModelError,
    InvalidFittingCombinationError,
    ProductImageMissingError,
    ProductNotFoundError,
    UnsupportedSubCategoryError,
)
from app.virtual_fitting.fallback_handler import fallback_stage
from app.virtual_fitting.idm_image_validator import (
    IdmImageUnavailable,
    check_idm_products,
    garment_description,
)
from app.virtual_fitting.idm_modal_client import IdmCallKind, IdmModalUnavailable
from app.virtual_fitting.postprocess_handler import save_completed, title_and_comment
from app.virtual_fitting.postprocess_worker import HandlerContext, HandlerResult, HandlerStatus
from app.virtual_fitting.pruna_submitter import failure_payload
from app.virtual_fitting.repositories.async_job_repository import AsyncFittingJob
from app.virtual_fitting.repositories.async_job_state import FALLBACK_LIMIT, ChangeReason

logger = logging.getLogger(__name__)

IDM_RESULT_FILENAME = "idm-vton.webp"
IDM_CONTENT_TYPE = "image/webp"
# x-amz-meta-<이름>. S3 HEAD 의 Metadata 는 소문자 이름으로 돌아온다.
META_JOB = "fitting-job-id"
META_ATTEMPT = "attempt-id"
MAX_RESULT_BYTES = 20 * 1024 * 1024
# 업로드 URL 만료 뒤 이만큼 더 지나야 "더 올라올 수 없다"고 본다(만료 직전에 시작된 PUT·시계 차이).
RECOVERY_MARGIN_SECONDS = 120
# Modal 에 넘기는 시작 마감은 업로드 URL 만료보다 이만큼 이르다(추론을 시작해도 업로드 전에 만료되지 않게).
SERVICE_DEADLINE_MARGIN_SECONDS = 60
_INPUT_FAILURES = (
    ProductNotFoundError,
    InvalidFittingCombinationError,
    ProductImageMissingError,
    UnsupportedSubCategoryError,
)


def idm_result_key(fitting_job_id: int) -> str:
    """IDM-VTON 결과 Key. Pruna 비동기(pruna-async.webp)·배경 제거(transparent.webp)·동기(pruna-sync.*) 결과와 분리한다."""
    return f"virtual-fitting/results/{fitting_job_id}/{IDM_RESULT_FILENAME}"


# --- 결과 객체 확인(핸들러·Watchdog 공용, 생성 함수를 부르지 않는다) ---


class ResultState(str, Enum):
    FOUND = "found"  # 이 작업·이 시도의 올바른 결과가 있다
    PENDING = "pending"  # 아직 없지만 업로드가 끝나지 않았을 수 있다
    ABSENT = "absent"  # 없고, 더 올라올 수 없다(업로드 URL 만료 + 여유가 지남)
    INVALID = "invalid"  # 고정 Key 에 객체는 있지만 형식·무결성·작업이 맞지 않는다
    UNKNOWN = "unknown"  # HEAD·GET 일시 오류·권한 오류 등으로 판단할 수 없다


@dataclasses.dataclass(frozen=True)
class ResultCheck:
    state: ResultState
    reason: str
    key: str | None = None


def verify_result_object(
    storage: Any, fitting_job_id: int, attempt_id: str | None, *, sha256: str | None = None
) -> ResultCheck:
    """고정 Key 의 IDM-VTON 결과를 확인한다: 메타데이터(작업·시도) → Content-Type·크기 → 본문 sha256(있으면) → WebP 디코딩.

    NOT_FOUND 는 ABSENT 로 돌려준다(시간 판단은 inspect_fallback_result 가 한다).
    """
    key = idm_result_key(fitting_job_id)
    head = storage.head_object(key)
    if head.state is S3ObjectState.NOT_FOUND:
        return ResultCheck(ResultState.ABSENT, "not_found", key)
    if head.state is not S3ObjectState.EXISTS:
        return ResultCheck(ResultState.UNKNOWN, f"head_{head.state.value}", key)
    metadata = {str(k).lower(): v for k, v in (head.metadata or {}).items()}
    if metadata.get(META_JOB) != str(fitting_job_id) or (
        attempt_id is None or metadata.get(META_ATTEMPT) != attempt_id
    ):
        return ResultCheck(ResultState.INVALID, "metadata_mismatch", key)
    if head.content_type != IDM_CONTENT_TYPE:
        return ResultCheck(ResultState.INVALID, "content_type", key)
    if not isinstance(head.content_length, int) or not 0 < head.content_length <= MAX_RESULT_BYTES:
        return ResultCheck(ResultState.INVALID, "content_length", key)
    try:
        body = storage.read_object(key, max_bytes=MAX_RESULT_BYTES)
    except ImageStorageError:
        return ResultCheck(ResultState.UNKNOWN, "read_failed", key)
    if sha256 is not None and hashlib.sha256(body).hexdigest() != sha256:
        return ResultCheck(ResultState.INVALID, "sha256_mismatch", key)
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(body)) as decoded:
            decoded.load()
            if decoded.format != "WEBP" or min(decoded.size) < 1:
                return ResultCheck(ResultState.INVALID, "not_webp", key)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return ResultCheck(ResultState.INVALID, "decode_failed", key)
    return ResultCheck(ResultState.FOUND, "verified", key)


def upload_window_closed(
    started_at: datetime | None, upload_expires_seconds: int, now: datetime | None = None
) -> bool | None:
    """업로드 URL 만료 + 여유가 지났는지. 시작 시각이 없으면 None(판단 불가)."""
    if started_at is None:
        return None
    deadline = started_at + timedelta(seconds=upload_expires_seconds + RECOVERY_MARGIN_SECONDS)
    return (now or datetime.now(UTC)) >= deadline


def inspect_fallback_result(
    storage: Any,
    job: AsyncFittingJob,
    *,
    upload_expires_seconds: int | None = None,
    now: datetime | None = None,
) -> ResultCheck:
    """시작 기록은 있고 이미지 기록이 없는 fallback 의 결과를 S3 로만 확인한다(생성 함수를 부르지 않는다).

    IDM-VTON 고정 Key(idm-vton.webp)만 본다. Pruna 동기 fallback 결과(pruna-sync.*)는 조회하지 않는다. 없으면 업로드 URL
    만료 전에는 PENDING, 뒤에는 ABSENT. HEAD 403·5xx·timeout 은 없다는 근거가 아니다(UNKNOWN).
    """
    if upload_expires_seconds is None:
        from app.config import settings

        upload_expires_seconds = settings.IDM_VTON_UPLOAD_URL_EXPIRES_SECONDS
    check = verify_result_object(storage, job.fitting_job_id, job.fallback_task_uuid)
    if check.state is not ResultState.ABSENT:
        return check
    closed = upload_window_closed(job.fallback_started_at, upload_expires_seconds, now)
    if closed is None:
        return ResultCheck(ResultState.UNKNOWN, "no_started_at", check.key)
    if not closed:
        return ResultCheck(ResultState.PENDING, "upload_window_open", check.key)
    return ResultCheck(ResultState.ABSENT, "upload_window_closed", check.key)


# --- 핸들러 ---


@dataclasses.dataclass(frozen=True)
class _Deps:
    jobs: Any
    state: Any
    products: Any
    storage: Any
    comments: Any
    modal: Any
    classifier: Any
    colors: Any


class ProductColorRepository:
    """IDM-VTON 상품 설명용 색상(products.color) 조회. Pruna 의 ProductRepository 는 바꾸지 않는다."""

    def __init__(self, connection_pool: Any) -> None:
        self._pool = connection_pool

    def find_color(self, product_code: str) -> str | None:
        import psycopg

        try:
            with self._pool.connection() as connection:
                connection.autocommit = True
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT color FROM products WHERE product_code = %s", (int(product_code),)
                    )
                    row = cursor.fetchone()
        except psycopg.Error as error:
            raise FittingDatabaseError("AI PostgreSQL 상품 색상 조회에 실패했습니다.") from error
        return row[0] if row and isinstance(row[0], str) else None


class IdmVtonFallbackHandler:
    """10단계 ActionHandler 구현. 의존성은 처음 부를 때 만든다(연결만으로 DB·S3·Modal client 를 만들지 않음)."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        poll_seconds: float = 10.0,
        max_wait_seconds: float | None = None,
        upload_expires_seconds: int | None = None,
        result_url_expires_seconds: int | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        new_attempt_id: Callable[[], str] = lambda: str(uuid.uuid4()),
        **deps: Any,
    ) -> None:
        self._given = deps
        self._deps: _Deps | None = None
        self._clock, self._poll, self._now = clock, poll_seconds, now
        self._max_wait = max_wait_seconds
        self._upload_expires = upload_expires_seconds
        self._expires = result_url_expires_seconds
        self._new_attempt_id = new_attempt_id

    # --- 의존성 ---

    def _resolve(self) -> _Deps:
        if self._deps is not None:
            return self._deps
        from app.config import settings

        if self._max_wait is None:
            self._max_wait = settings.IDM_VTON_WAIT_SECONDS
        if self._upload_expires is None:
            self._upload_expires = settings.IDM_VTON_UPLOAD_URL_EXPIRES_SECONDS
        names = [f.name for f in dataclasses.fields(_Deps)]
        given = dict(self._given)
        if any(given.get(name) is None for name in names):
            given = {**_real_deps(given), **{k: v for k, v in given.items() if v is not None}}
        self._deps = _Deps(**{name: given[name] for name in names})
        return self._deps

    # --- 처리 ---

    def __call__(self, job: AsyncFittingJob, token: str, context: HandlerContext) -> HandlerResult:
        deps = self._resolve()
        job_id = job.fitting_job_id
        if job.result_payload is not None:
            return HandlerResult(HandlerStatus.DONE, "result_already_saved")
        image = job.fallback_image_key
        if image is None:
            if job.fallback_started_at is not None or job.fallback_attempt_count >= FALLBACK_LIMIT:
                # 생성 함수를 이미 불렀을 수 있다. 다시 부르지 않고 S3 결과로만 판단한다.
                image, stop = self._recover(deps, job, token, context)
            else:
                image, stop = self._generate(deps, job, token, context)
            if stop is not None:
                return stop
        title, comment, stop = title_and_comment(
            deps,
            job,
            token,
            context,
            image,
            stage_ok=fallback_stage(image),
            expires_in=self._expires,
        )
        if stop is not None:
            return stop
        if not context.may_continue():
            return HandlerResult(HandlerStatus.DEFERRED, "stopped_before_save")
        return save_completed(deps.state, job_id, token, image, title, comment)

    def _fail(self, deps, job_id: int, token: str, code: str, message: str, why: str):
        saved = deps.state.save_final_result(job_id, token, failure_payload(job_id, code, message))
        if saved.applied or saved.reason is ChangeReason.ALREADY_RECORDED:
            logger.info(
                "idm fallback failure saved fitting_job_id=%s code=%s reason=%s", job_id, code, why
            )
            return HandlerResult(HandlerStatus.DONE, f"failure_saved_{why}")
        return HandlerResult(HandlerStatus.DEFERRED, f"failure_not_saved_{saved.reason.value}")

    def _model_failed(self, deps, job_id: int, token: str, why: str) -> HandlerResult:
        return self._fail(
            deps, job_id, token, FittingModelError.code, FittingModelError.message, why
        )

    def _record(self, deps, job_id: int, token: str, key: str):
        recorded = deps.state.record_fallback_image(job_id, token, key)
        if recorded.applied:
            return key, None
        if recorded.reason is ChangeReason.ALREADY_RECORDED and recorded.job is not None:
            return recorded.job.fallback_image_key, None
        logger.warning(
            "idm fallback image verified but not recorded fitting_job_id=%s reason=%s",
            job_id,
            recorded.reason.value,
        )
        return None, HandlerResult(
            HandlerStatus.DEFERRED, f"fallback_image_record_{recorded.reason.value}"
        )

    def _recover(self, deps, job, token, context) -> tuple[str | None, HandlerResult | None]:
        job_id = job.fitting_job_id
        if not context.may_continue():
            return None, HandlerResult(HandlerStatus.DEFERRED, "stopped_before_recovery")
        check = inspect_fallback_result(
            deps.storage, job, upload_expires_seconds=self._upload_expires, now=self._now()
        )
        logger.info(
            "idm fallback recovery check fitting_job_id=%s state=%s reason=%s",
            job_id,
            check.state.value,
            check.reason,
        )
        if check.state is ResultState.FOUND:
            return self._record(deps, job_id, token, check.key)
        if check.state in (ResultState.ABSENT, ResultState.INVALID):
            return None, self._model_failed(deps, job_id, token, f"idm_result_{check.reason}")
        return None, HandlerResult(HandlerStatus.DEFERRED, f"idm_result_{check.reason}")

    def _generate(self, deps, job, token, context) -> tuple[str | None, HandlerResult | None]:
        from app.virtual_fitting.async_storage import user_image_get_url
        from app.virtual_fitting.product_selection import select_fitting_products
        from app.virtual_fitting.request_worker import (
            USER_IMAGE_NOT_FOUND_CODE,
            USER_IMAGE_NOT_FOUND_MESSAGE,
        )

        job_id = job.fitting_job_id
        if not context.may_continue():
            return None, HandlerResult(HandlerStatus.DEFERRED, "stopped_before_input")
        if not job.user_image_key or not job.products:
            logger.error("idm fallback job has no original input fitting_job_id=%s", job_id)
            return None, HandlerResult(HandlerStatus.DEFERRED, "no_original_input")
        # 1. 입력 검증(호출 기회를 쓰지 않는다)
        try:
            products = select_fitting_products(list(job.products), deps.products)
        except _INPUT_FAILURES as error:
            return None, self._fail(deps, job_id, token, error.code, error.message, error.code)
        support = check_idm_products(products)
        if not support.supported:
            logger.info(
                "idm fallback unsupported products fitting_job_id=%s reason=%s",
                job_id,
                support.reason,
            )
            return None, self._model_failed(deps, job_id, token, f"idm_{support.reason}")
        product = support.product
        try:
            garment = deps.classifier.classify_url(product.image_url)
        except IdmImageUnavailable as error:
            logger.warning(
                "idm fallback garment image unavailable fitting_job_id=%s reason=%s", job_id, error
            )
            return None, HandlerResult(HandlerStatus.DEFERRED, "garment_image_unavailable")
        logger.info(
            "idm fallback garment image fitting_job_id=%s product_code=%s class=%s reason=%s",
            job_id,
            product.product_code,
            garment.classification.value,
            garment.reason,
        )
        if not garment.allowed:
            return None, self._model_failed(
                deps, job_id, token, garment.classification.value.lower()
            )
        head = deps.storage.head_object(job.user_image_key)
        if head.state is S3ObjectState.NOT_FOUND:
            return None, self._fail(
                deps,
                job_id,
                token,
                USER_IMAGE_NOT_FOUND_CODE,
                USER_IMAGE_NOT_FOUND_MESSAGE,
                "user_image_not_found",
            )
        if head.state is not S3ObjectState.EXISTS:
            return None, HandlerResult(HandlerStatus.DEFERRED, f"user_image_{head.state.value}")
        try:
            deps.modal.resolve()
        except IdmModalUnavailable:
            logger.exception("idm fallback modal function unavailable fitting_job_id=%s", job_id)
            return None, HandlerResult(HandlerStatus.DEFERRED, "idm_function_unavailable")
        try:
            color = deps.colors.find_color(product.product_code)
        except FittingDatabaseError:
            color = None  # 색상은 설명 보조 정보다. 없어도 진행한다.
        description = garment_description(support.garment_name, color)

        # 2. 시도 ID·URL(외부 호출 전, 횟수를 쓰지 않는다)
        prepared = deps.state.prepare_fallback(job_id, token, self._new_attempt_id())
        if not (prepared.applied or prepared.reason is ChangeReason.ALREADY_RECORDED):
            return None, HandlerResult(
                HandlerStatus.DEFERRED, f"fallback_task_{prepared.reason.value}"
            )
        attempt_id = prepared.job.fallback_task_uuid
        key = idm_result_key(job_id)
        metadata = {META_JOB: str(job_id), META_ATTEMPT: attempt_id}
        try:
            person_url = user_image_get_url(deps.storage, job.user_image_key)
            upload_url = deps.storage.presigned_put_url(
                key,
                expires_in=self._upload_expires,
                content_type=IDM_CONTENT_TYPE,
                metadata=metadata,
            )
        except ImageStorageError:
            logger.exception("idm fallback presign failed fitting_job_id=%s", job_id)
            return None, HandlerResult(HandlerStatus.DEFERRED, "presign_failed")

        # 3. 호출 기회 1회 선점 → spawn 1회
        if not context.may_continue():
            return None, HandlerResult(HandlerStatus.DEFERRED, "stopped_before_claim")
        claim = deps.state.claim_idm_vton_attempt(job_id, token)
        if not claim.applied:
            if claim.reason is ChangeReason.LIMIT_REACHED and claim.job is not None:
                # 다른 Worker 가 먼저 선점했다. 부르지 않고 결과만 확인한다.
                return self._recover(deps, claim.job, token, context)
            if (
                claim.reason is ChangeReason.ALREADY_RECORDED
                and claim.job is not None
                and claim.job.fallback_image_key is not None
            ):
                return claim.job.fallback_image_key, None
            return None, HandlerResult(HandlerStatus.DEFERRED, f"idm_claim_{claim.reason.value}")
        started_at = claim.job.fallback_started_at
        request = {
            "fitting_job_id": job_id,
            "attempt_id": attempt_id,
            "person_image_url": person_url,
            "garment_image_url": product.image_url,
            "garment_description": description,
            "upload_url": upload_url,
            "upload_headers": {
                "Content-Type": IDM_CONTENT_TYPE,
                **{f"x-amz-meta-{name}": value for name, value in metadata.items()},
            },
            "not_after_epoch": started_at.timestamp()
            + self._upload_expires
            - SERVICE_DEADLINE_MARGIN_SECONDS,
        }
        logger.info(
            "idm fallback generation started fitting_job_id=%s attempt_id=%s product_code=%s",
            job_id,
            attempt_id,
            product.product_code,
        )
        wall = self._clock()
        call, unknown = deps.modal.spawn(request)
        if unknown is not None:
            logger.warning(
                "idm fallback spawn unconfirmed fitting_job_id=%s reason=%s; held (no re-call)",
                job_id,
                unknown.reason,
            )
            return None, HandlerResult(HandlerStatus.DEFERRED, "idm_result_unknown")
        result, stop = self._wait(deps, call, job_id, attempt_id, context)
        elapsed = round(self._clock() - wall, 1)
        if stop is not None:
            logger.warning(
                "idm fallback wait stopped fitting_job_id=%s reason=%s elapsed_s=%s",
                job_id,
                stop.reason,
                elapsed,
            )
            return None, stop
        logger.info(
            "idm fallback generation finished fitting_job_id=%s kind=%s reason=%s elapsed_s=%s",
            job_id,
            result.kind.value,
            result.reason,
            elapsed,
        )
        if result.kind is IdmCallKind.UNKNOWN:
            return None, HandlerResult(HandlerStatus.DEFERRED, "idm_result_unknown")
        if not context.may_continue():
            return None, HandlerResult(HandlerStatus.DEFERRED, "stopped_before_result_check")
        sha = result.output.get("sha256") if result.kind is IdmCallKind.SUCCEEDED else None
        check = verify_result_object(deps.storage, job_id, attempt_id, sha256=sha)
        if check.state is ResultState.FOUND:
            return self._record(deps, job_id, token, key)
        if check.state is ResultState.INVALID:
            return None, self._model_failed(deps, job_id, token, f"idm_result_{check.reason}")
        if check.state is ResultState.ABSENT and result.kind is IdmCallKind.FAILED:
            # 함수가 끝났고 결과가 없다. 다시 부르지 않고 실패로 확정한다.
            return None, self._model_failed(deps, job_id, token, f"idm_{result.reason}")
        # 성공 보고인데 아직 보이지 않음, 또는 S3 확인 일시 오류: 보류(복구가 이어 간다).
        return None, HandlerResult(HandlerStatus.DEFERRED, f"idm_result_{check.reason}")

    def _wait(self, deps, call, job_id, attempt_id, context):
        """결과를 나눠 기다린다(그 사이 종료·연장 실패 확인). 실행 상태 조회일 뿐 다시 부르지 않는다."""
        started = self._clock()
        while True:
            result = deps.modal.wait(call, self._poll, job_id, attempt_id)
            if result is not None:
                return result, None
            if not context.may_continue():
                # Modal 은 계속 돈다. 결과는 다음 수신·Watchdog 이 S3 로 확인한다.
                return None, HandlerResult(HandlerStatus.DEFERRED, "idm_detached_running")
            if self._clock() - started >= self._max_wait:
                return None, HandlerResult(HandlerStatus.DEFERRED, "idm_wait_exceeded")


def _real_deps(given: dict[str, Any]) -> dict[str, Any]:
    from app.clients.s3 import S3ImageStorage
    from app.config.database import get_connection_pool
    from app.virtual_fitting.idm_image_validator import IdmGarmentImageClassifier
    from app.virtual_fitting.idm_modal_client import IdmVtonModalClient
    from app.virtual_fitting.providers.runware_comment import RunwareCommentProvider
    from app.virtual_fitting.repositories.async_job_repository import AsyncJobRepository
    from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository
    from app.virtual_fitting.repositories.product_repository import ProductRepository

    pool = get_connection_pool()

    def need(name, build):
        return given.get(name) if given.get(name) is not None else build()

    return {
        "jobs": need("jobs", lambda: AsyncJobRepository(pool)),
        "state": need("state", lambda: AsyncJobStateRepository(pool)),
        "products": need("products", lambda: ProductRepository(pool)),
        "storage": need("storage", S3ImageStorage),
        "comments": need("comments", RunwareCommentProvider),
        "modal": need("modal", IdmVtonModalClient),
        "classifier": need("classifier", IdmGarmentImageClassifier),
        "colors": need("colors", lambda: ProductColorRepository(pool)),
    }
