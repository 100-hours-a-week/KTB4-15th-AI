import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.clients.s3 import ImageStorageError, S3ObjectState
from app.virtual_fitting.exceptions import FittingPostprocessError
from app.virtual_fitting.postprocess_worker import HandlerContext, HandlerResult, HandlerStatus
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncFittingJob,
    AsyncJobStatus,
    NextAction,
)
from app.virtual_fitting.repositories.async_job_state import (
    LLM_COMMENT_ATTEMPT_LIMIT,
    LLM_GENERATION_LIMIT,
    LLM_TITLE_ATTEMPT_LIMIT,
    ChangeReason,
)
from app.virtual_fitting.service import FALLBACK_COMMENT, FALLBACK_TITLE

logger = logging.getLogger(__name__)

COMMENT_ATTEMPT_LIMIT = LLM_COMMENT_ATTEMPT_LIMIT  # 최초 1회 + 재시도 1회
TITLE_ATTEMPT_LIMIT = LLM_TITLE_ATTEMPT_LIMIT  # 최초 1회 + 재시도 1회(같은 코멘트로)
# 저장할 제목의 최대 길이(앞뒤 공백 제거 후). BE fitting_temp_result.outfit_name(varchar 100)과 같다.
# 프롬프트의 목표(10~20자)를 벗어났다는 이유로는 실패시키지 않는다.
TITLE_MAX_LENGTH = 100

# 한 묶음 안에서 HEAD 를 보내는 시점(묶음 시작 기준, 초).
S3_CHECK_OFFSETS = (0.0, 2.0, 7.0)
# 긴 대기를 나눠 그 사이마다 종료·연장 실패를 확인한다.
WAIT_SLICE_SECONDS = 0.5


@dataclass(frozen=True)
class _Deps:
    jobs: Any
    state: Any
    products: Any
    storage: Any
    comments: Any


class PostprocessHandler:
    """10단계 ActionHandler 구현. 의존성은 처음 부를 때 만든다(연결만으로 DB·S3·LLM client 를 만들지 않음)."""

    def __init__(
        self,
        *,
        jobs: Any = None,
        state: Any = None,
        products: Any = None,
        storage: Any = None,
        comments: Any = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        offsets: Sequence[float] = S3_CHECK_OFFSETS,
        result_url_expires_seconds: int | None = None,
        store_result: Callable[..., Any] | None = None,
    ) -> None:
        self._given = (jobs, state, products, storage, comments)
        self._deps: _Deps | None = None
        self._clock, self._sleep, self._offsets = clock, sleep, tuple(offsets)
        self._expires = result_url_expires_seconds
        # AI 저장 작업의 결과 저장(기본 store_runware_result_image). 처음 쓸 때 가져온다.
        self._store_result = store_result

    # --- 의존성 ---

    def _resolve(self) -> _Deps:
        if self._deps is None:
            jobs, state, products, storage, comments = self._given
            if None in self._given:
                from app.clients.s3 import S3ImageStorage
                from app.config.database import get_connection_pool
                from app.virtual_fitting.providers.runware_comment import RunwareCommentProvider
                from app.virtual_fitting.repositories.async_job_repository import (
                    AsyncJobRepository,
                )
                from app.virtual_fitting.repositories.async_job_state import (
                    AsyncJobStateRepository,
                )
                from app.virtual_fitting.repositories.product_repository import ProductRepository

                pool = (
                    get_connection_pool()
                    if jobs is None or state is None or products is None
                    else None
                )
                jobs = jobs or AsyncJobRepository(pool)
                state = state or AsyncJobStateRepository(pool)
                products = products or ProductRepository(pool)
                storage = storage or S3ImageStorage()
                comments = comments or RunwareCommentProvider()
            self._deps = _Deps(jobs, state, products, storage, comments)
        return self._deps

    def __call__(self, job: AsyncFittingJob, token: str, context: HandlerContext) -> HandlerResult:
        deps = self._resolve()
        job_id = job.fitting_job_id
        if job.result_payload is not None:
            # 이미 결과가 있다(재수신). 이미지 확인·LLM 호출을 반복하지 않는다.
            return HandlerResult(HandlerStatus.DONE, "result_already_saved")
        verified = job.verified_image_key
        if verified is None:
            if job.ai_stores_result:
                verified, stop = self._store_image(deps, job, token, context)
            else:
                verified, stop = self._verify_image(deps, job, token, context)
            if stop is not None:
                return stop
        title, comment, stop = title_and_comment(
            deps,
            job,
            token,
            context,
            verified,
            stage_ok=postprocess_stage(verified),
            expires_in=self._expires,
        )
        if stop is not None:
            return stop
        if not context.may_continue():
            return HandlerResult(HandlerStatus.DEFERRED, "stopped_before_save")
        return save_completed(deps.state, job_id, token, verified, title, comment)

    # --- 1'. AI 저장 작업: 결과 다운로드·저장 ---

    def _store_image(self, deps, job, token, context) -> tuple[str | None, HandlerResult | None]:
        from app.virtual_fitting.result_image import ResultImageError, store_runware_result_image

        job_id, key = job.fitting_job_id, job.async_image_key
        if key is None:
            logger.error("postprocess job has no async_image_key fitting_job_id=%s", job_id)
            return None, HandlerResult(HandlerStatus.DEFERRED, "no_async_image_key")
        if job.runware_image_url is None:
            # 다운로드할 URL 이 없다. 있다고 가정하지 않고, Watchdog 의 같은 task 조회(참조 복구)를 기다린다.
            logger.warning(
                "postprocess result url missing fitting_job_id=%s; waiting for reference recovery",
                job_id,
            )
            return None, HandlerResult(HandlerStatus.DEFERRED, "result_url_missing")
        if not context.may_continue():
            return None, HandlerResult(HandlerStatus.DEFERRED, "stopped_before_result_store")
        store = self._store_result or store_runware_result_image
        try:
            stored = store(
                deps.storage,
                fitting_job_id=job_id,
                task_uuid=job.task_uuid,
                image_url=job.runware_image_url,
                result_key=key,
            )
        except ResultImageError as error:
            # 생성 실패가 아니다. fallback·FAILED 로 넘기지 않고 메시지를 남긴다(재수신·Watchdog 이 이어 감).
            log = logger.warning if error.retryable else logger.error
            log(
                "postprocess result store deferred fitting_job_id=%s reason=%s retryable=%s "
                "http_status=%s",
                job_id,
                error.reason,
                error.retryable,
                error.http_status,
            )
            return None, HandlerResult(HandlerStatus.DEFERRED, f"result_store_{error.reason}")
        except ValueError:
            # 이 작업의 예정 WebP Key·URL 이 아니다(기록 이상). 운영 확인 대상이며 다른 Key 로 저장하지 않는다.
            logger.error("postprocess result store input rejected fitting_job_id=%s", job_id)
            return None, HandlerResult(HandlerStatus.DEFERRED, "result_store_invalid_input")
        if not context.may_continue():
            # 저장은 끝났을 수 있지만 실행 권한을 잃었다. 기록하지 않는다(다음 실행이 같은 객체를 확인해 재사용).
            return None, HandlerResult(HandlerStatus.DEFERRED, "stopped_before_verify_save")
        recorded = deps.state.record_image_verified(job_id, token, stored.key)
        if recorded.applied:
            return stored.key, None
        if recorded.reason is ChangeReason.ALREADY_RECORDED and recorded.job is not None:
            return recorded.job.verified_image_key, None
        return None, HandlerResult(HandlerStatus.DEFERRED, f"verify_save_{recorded.reason.value}")

    # --- 1. S3 확인 ---

    def _wait_until(self, start: float, offset: float, context: HandlerContext) -> bool:
        """묶음 시작 뒤 offset 초가 될 때까지 나눠 기다린다. 중간에 종료·연장 실패가 있으면 False."""
        while True:
            remaining = start + offset - self._clock()
            if remaining <= 0:
                return context.may_continue()
            if not context.may_continue():
                return False
            self._sleep(min(WAIT_SLICE_SECONDS, remaining))

    def _verify_image(self, deps, job, token, context) -> tuple[str | None, HandlerResult | None]:
        job_id, key = job.fitting_job_id, job.async_image_key
        if key is None:
            logger.error("postprocess job has no async_image_key fitting_job_id=%s", job_id)
            return None, HandlerResult(HandlerStatus.DEFERRED, "no_async_image_key")
        while True:
            if not context.may_continue():
                return None, HandlerResult(HandlerStatus.DEFERRED, "stopped_before_s3_check")
            claim = deps.state.claim_s3_check_attempt(job_id, token)
            if not claim.applied:
                if claim.reason is ChangeReason.ALREADY_RECORDED and claim.job is not None:
                    return claim.job.verified_image_key, None  # 그사이 확인 기록이 생겼다
                if claim.reason is ChangeReason.LIMIT_REACHED:
                    # 횟수는 다 썼지만 이 lease 로 끝까지 확인한 결과가 없다(이전 Worker 가 중단했을 수 있음).
                    logger.warning(
                        "postprocess s3 checks exhausted without a confirmed result "
                        "fitting_job_id=%s; held for recovery (stage 15)",
                        job_id,
                    )
                    return None, HandlerResult(HandlerStatus.DEFERRED, "s3_checks_unconfirmed")
                return None, HandlerResult(HandlerStatus.DEFERRED, f"s3_claim_{claim.reason.value}")
            bundle = claim.job.s3_check_attempt_count
            last = None
            start = self._clock()
            for offset in self._offsets:
                if not self._wait_until(start, offset, context):
                    return None, HandlerResult(HandlerStatus.DEFERRED, "stopped_during_s3_check")
                last = deps.storage.head_object(key)
                if last.state is S3ObjectState.EXISTS:
                    break
                if last.state in (S3ObjectState.ACCESS_DENIED, S3ObjectState.ERROR):
                    # 권한·설정 문제. 객체가 없다고 단정하지 않고 V1 fallback 하지 않는다. 운영 확인 대상.
                    logger.error(
                        "postprocess s3 check needs operator check fitting_job_id=%s state=%s "
                        "error_code=%s",
                        job_id,
                        last.state.value,
                        last.error_code,
                    )
                    return None, HandlerResult(HandlerStatus.DEFERRED, f"s3_{last.state.value}")
            if last is not None and last.state is S3ObjectState.EXISTS:
                if not context.may_continue():
                    return None, HandlerResult(HandlerStatus.DEFERRED, "stopped_before_verify_save")
                recorded = deps.state.record_image_verified(job_id, token, key)
                if recorded.applied:
                    return key, None
                if recorded.reason is ChangeReason.ALREADY_RECORDED and recorded.job is not None:
                    return recorded.job.verified_image_key, None
                return None, HandlerResult(
                    HandlerStatus.DEFERRED, f"verify_save_{recorded.reason.value}"
                )
            logger.info(
                "postprocess s3 bundle finished without image fitting_job_id=%s bundle=%s last=%s",
                job_id,
                bundle,
                last.state.value if last else None,
            )
            if bundle >= 2:
                if last is not None and last.state is S3ObjectState.NOT_FOUND:
                    if not context.may_continue():
                        return None, HandlerResult(
                            HandlerStatus.DEFERRED, "stopped_before_fallback"
                        )
                    moved = deps.state.move_to_fallback(job_id, token, final_s3_check=last.state)
                    if moved.applied:
                        # DB 를 RUN_FALLBACK 으로 넘겼다. RUN_FALLBACK 메시지 발행은 Worker 가 한다(handoff).
                        return None, HandlerResult(
                            HandlerStatus.DONE, "moved_to_fallback", NextAction.RUN_FALLBACK
                        )
                    return None, HandlerResult(
                        HandlerStatus.DEFERRED, f"fallback_{moved.reason.value}"
                    )
                # 마지막 확인이 일시 오류: 객체가 없다고 단정할 수 없다.
                return None, HandlerResult(HandlerStatus.DEFERRED, "s3_unconfirmed_transient")
            # 첫 묶음 실패 → 같은 실행 안에서 두 번째 묶음.


# --- 2. 제목·코멘트 (POSTPROCESS·RUN_FALLBACK 공용) ---


def postprocess_stage(key: str) -> Callable[[AsyncFittingJob], bool]:
    """POSTPROCESS 단계가 그대로인지(같은 S3 확인 이미지)."""
    return lambda latest: (
        latest.status is AsyncJobStatus.POSTPROCESSING
        and latest.next_action is NextAction.POSTPROCESS
        and latest.verified_image_key == key
    )


def product_summaries(products: Any, job: AsyncFittingJob) -> list[str]:
    """V1 과 같이 상품 설명 요약을 보조 근거로 쓴다(요청 순서). 상품 DB 의 product_code 는 정수다."""
    codes = [code for code in (job.products or ()) if code.isdigit()]
    found = {product.product_code: product for product in products.find_by_codes(codes)}
    summaries = []
    for code in codes:
        summary = getattr(found.get(str(int(code))), "description_summary", None)
        if isinstance(summary, str) and summary.strip():
            summaries.append(summary)
    return summaries


def title_and_comment(
    deps: Any,
    job: AsyncFittingJob,
    token: str,
    context: HandlerContext,
    key: str,
    *,
    stage_ok: Callable[[AsyncFittingJob], bool],
    expires_in: int | None = None,
) -> tuple[str | None, str | None, HandlerResult | None]:
    """확인한 이미지 key 로 코멘트 → 제목을 만든다. 항목별 최초 1회 + 재시도 1회(각 최대 2회).

    - 코멘트가 두 번 모두 실패하면 제목은 부르지 않고 기본 코멘트 + 기본 제목.
    - 코멘트가 성공하면 그 코멘트로 제목을 최대 2회(재시도에도 같은 코멘트, 코멘트를 다시 만들지 않음).
      제목이 두 번 모두 실패하면 생성 코멘트 + 기본 제목.
    - 실패: FittingPostprocessError(HTTP 오류·timeout·잘린 응답·빈 응답·콘텐츠 필터 등 Provider 판단)·ValueError·빈 문자열,
      제목은 앞뒤 공백 제거 후 TITLE_MAX_LENGTH 자 초과도 실패(잘라서 저장하지 않음).
    - 호출마다 직전에 claim_llm_generation_attempt 로 DB 횟수를 확보한다(재수신·재시작에도 초기화하지 않는 호출 한도).
    - 종료·실행 권한 상실·상태 변경·DB 오류는 기본 문구로 성공 처리하지 않는다(보류하거나 예외를 그대로 올림).
    deps 는 jobs·state·products·storage·comments 를 가진다. stage_ok 는 최신 기록이 아직 이 단계(POSTPROCESS 또는
    RUN_FALLBACK, 같은 이미지)인지 본다. 반환: (제목, 코멘트, None) 또는 (None, None, 멈출 때의 HandlerResult).
    """
    job_id = job.fitting_job_id
    summaries = product_summaries(deps.products, job)
    if not summaries:
        logger.warning(
            "postprocess has no product description summaries fitting_job_id=%s; default text",
            job_id,
        )
        return FALLBACK_TITLE, FALLBACK_COMMENT, None
    from app.virtual_fitting.async_storage import result_image_get_url

    def gate(where: str) -> tuple[AsyncFittingJob | None, HandlerResult | None]:
        """호출 직전 확인(URL 서명·횟수 확보 전). 최종 허용 판단은 claim 의 조건부 확보다."""
        if not context.may_continue():
            return None, HandlerResult(HandlerStatus.DEFERRED, f"stopped_before_{where}")
        latest = deps.jobs.find_by_fitting_job_id(job_id)
        if latest is None:
            return None, HandlerResult(HandlerStatus.DEFERRED, "job_missing")
        if latest.result_payload is not None:
            return None, HandlerResult(HandlerStatus.DONE, "result_already_saved")
        if not stage_ok(latest):
            return None, HandlerResult(HandlerStatus.DEFERRED, "llm_state_changed")
        if not deps.state.holds_lease(job_id, token):
            return None, HandlerResult(HandlerStatus.DEFERRED, "llm_lease_lost")
        return latest, None

    def claim() -> tuple[bool, HandlerResult | None]:
        """(확보됨, 멈출 결과). 한도 소진이면 (False, None) — 그 항목은 기본 문구로."""
        change = deps.state.claim_llm_generation_attempt(job_id, token)
        if change.applied:
            return True, None
        if change.reason is ChangeReason.LIMIT_REACHED:
            logger.info("postprocess llm call budget exhausted fitting_job_id=%s", job_id)
            return False, None
        return False, HandlerResult(HandlerStatus.DEFERRED, f"llm_claim_{change.reason.value}")

    def failed(item: str, attempt: int, reason: str) -> None:
        logger.warning(
            "postprocess llm %s attempt failed fitting_job_id=%s attempt=%s/%s reason=%s",
            item,
            job_id,
            attempt,
            COMMENT_ATTEMPT_LIMIT if item == "comment" else TITLE_ATTEMPT_LIMIT,
            reason,
        )

    # 1. 코멘트(최대 2회)
    comment: str | None = None
    for attempt in range(1, COMMENT_ATTEMPT_LIMIT + 1):
        latest, stop = gate("llm")
        if stop is not None:
            return None, None, stop
        if latest.llm_generation_attempt_count >= LLM_GENERATION_LIMIT:
            break  # 이미 한도면 URL 을 만들지 않는다(쓰지 않을 서명 없음)
        try:
            # 남은 횟수가 있을 때만 새 URL 을 만든다(한도 확인은 claim 이 최종 판단, URL 은 저장·로그하지 않음).
            url = result_image_get_url(deps.storage, key, expires_in=expires_in)
        except ImageStorageError:
            # S3 서명 설정 문제(LLM 오류와 구분). 시도 횟수를 쓰지 않고 보류한다.
            logger.exception("postprocess result image url failed fitting_job_id=%s", job_id)
            return None, None, HandlerResult(HandlerStatus.DEFERRED, "s3_presign_failed")
        claimed, stop = claim()
        if stop is not None:
            return None, None, stop
        if not claimed:
            break
        try:
            text = deps.comments.generate_comment(url, summaries)
        except (FittingPostprocessError, ValueError) as error:
            failed("comment", attempt, type(error).__name__)
            continue
        if _usable(text):
            comment = text.strip()
            break
        failed("comment", attempt, "empty")
    if comment is None:
        logger.info(
            "postprocess comment failed fitting_job_id=%s; default comment and title", job_id
        )
        return FALLBACK_TITLE, FALLBACK_COMMENT, None

    # 2. 제목(같은 코멘트로 최대 2회)
    for attempt in range(1, TITLE_ATTEMPT_LIMIT + 1):
        _, stop = gate("title")
        if stop is not None:
            return None, None, stop
        claimed, stop = claim()
        if stop is not None:
            return None, None, stop
        if not claimed:
            break
        try:
            text = deps.comments.generate_title(comment)
        except (FittingPostprocessError, ValueError) as error:
            failed("title", attempt, type(error).__name__)
            continue
        if not _usable(text):
            failed("title", attempt, "empty")
            continue
        title = text.strip()
        if len(title) > TITLE_MAX_LENGTH:
            failed("title", attempt, "too_long")  # 잘라서 저장하지 않는다
            continue
        return title, comment, None
    logger.info(
        "postprocess title failed fitting_job_id=%s; generated comment + default title", job_id
    )
    return FALLBACK_TITLE, comment, None


def save_completed(
    state: Any, job_id: int, token: str, key: str, title: str, comment: str
) -> HandlerResult:
    """COMPLETED V2 결과(result_image_key = 확인한 이미지)와 PUBLISH_RESULT 를 저장한다. 기존 결과는 덮어쓰지 않는다."""
    payload = {
        "fitting_job_id": job_id,
        "status": "COMPLETED",
        "result_image_key": key,
        "llm_title": title,
        "llm_comment": comment,
    }
    saved = state.save_final_result(job_id, token, payload, key)
    if saved.applied or saved.reason is ChangeReason.ALREADY_RECORDED:
        return HandlerResult(HandlerStatus.DONE, "result_saved")
    logger.warning("result not saved fitting_job_id=%s reason=%s", job_id, saved.reason.value)
    return HandlerResult(HandlerStatus.DEFERRED, f"save_{saved.reason.value}")


def _usable(text: Any) -> bool:
    return isinstance(text, str) and bool(text.strip())
