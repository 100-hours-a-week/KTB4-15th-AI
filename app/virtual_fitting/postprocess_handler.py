"""가상피팅 V2 POSTPROCESS 핸들러 (11단계): 결과 이미지 S3 확인 → 제목·코멘트 생성 → V2 결과 저장.

10단계 후처리 Worker 가 실행 권한(token)을 확보한 뒤 POSTPROCESSING / POSTPROCESS 작업의 최신 기록으로 부른다.
DB 변경은 모두 5단계의 토큰 검증 메서드로 한다(외부 호출 중에는 트랜잭션·행 잠금을 잡지 않는다). 새 외부 호출(S3 HEAD,
Presigned 발급 뒤 LLM 호출, 저장) 직전마다 context.may_continue() 로 종료·연장 실패를 확인한다.

1. S3 확인 (verified_image_key 가 없을 때만)
   - 이미지 위치는 DB 의 async_image_key 다(Webhook imageURL·사용자 원본 사진을 쓰지 않음). 내려받지 않고 HEAD 만 한다.
   - 한 묶음 = 시작 후 0·2·7초 시점 HEAD(대기 2초 → 5초). 성공하면 남은 HEAD 는 생략한다.
   - 묶음 시작 직전에 claim_s3_check_attempt 로 횟수를 확보한다(최초 포함 최대 2묶음, 되돌리지 않음).
   - EXISTS → record_image_verified(저장이 확인돼야 다음 단계). NOT_FOUND·일시 오류 → 같은 묶음의 다음 HEAD.
   - ACCESS_DENIED·ERROR(권한·설정) → 묶음을 멈추고 운영 확인으로 보류한다(V1 fallback 하지 않음).
   - 이 핸들러가 두 번째 묶음을 현재 lease 로 끝까지 돌렸고 마지막 HEAD 가 NOT_FOUND 일 때만 move_to_fallback.
     마지막이 일시 오류면 객체가 없다고 단정하지 않고 보류한다.
   - 횟수를 이미 다 썼는데(이전 Worker 가 두 번째 묶음 도중 멈췄을 수 있음) 확인 결과를 모르면 보류한다(15단계 복구 대상).
2. 제목·코멘트 (확인한 이미지로만, S3 확인으로 되돌아가지 않음)
   - 시도 1회 = V1 RunwareCommentProvider 의 comment + title 한 쌍(HTTP 2번, 내부 재시도 없음).
   - 매 시도(반복):
     a. DB 의 최신 작업 기록을 읽는다. 결과가 이미 있으면 덮어쓰지 않고 끝내고, 상태(POSTPROCESSING / POSTPROCESS,
        같은 확인 이미지)나 실행 권한(holds_lease)이 유효하지 않으면 보류한다.
     b. llm_generation_attempt_count 가 이미 한도(3)면 URL 을 만들지 않고 V1 기본 문구로 저장한다.
     c. 남은 횟수가 있으면 확인한 Key 로 Presigned GET 을 새로 발급한다(만료 VIRTUAL_FITTING_RESULT_IMAGE_URL_EXPIRES_SECONDS).
        서명 실패는 S3 오류로 보류하고 LLM 시도를 쓰지 않는다. URL 은 저장·로그하지 않는다.
     d. claim_llm_generation_attempt(최초 포함 최대 3회, < 3, 되돌리지 않음)로 실행 권한·상태·한도를 원자적으로 다시
        확인하고 횟수를 확보한다. a·b 는 불필요한 서명을 줄이기 위한 확인이고, 호출 허용은 이 조건부 확보가 최종 판단한다.
     e. 확보했을 때만 comment → 그 comment 로 title. 실패(FittingPostprocessError·잘못된 응답)면 a 부터 다시.
3. 저장: save_final_result 로 COMPLETED 결과(result_image_key = 확인한 이미지 Key)와 PUBLISH_RESULT 를 함께 저장한다.
   이미 결과가 있으면 덮어쓰지 않는다. 결과 발행·최종 상태 확정(14단계)은 하지 않는다.
LLM 이 실패해도 이미지 생성 성공은 유지한다(기본 문구로 COMPLETED, fallback·Pruna 재생성 없음).
"""

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
from app.virtual_fitting.repositories.async_job_state import LLM_GENERATION_LIMIT, ChangeReason
from app.virtual_fitting.service import FALLBACK_COMMENT, FALLBACK_TITLE

logger = logging.getLogger(__name__)

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
    ) -> None:
        self._given = (jobs, state, products, storage, comments)
        self._deps: _Deps | None = None
        self._clock, self._sleep, self._offsets = clock, sleep, tuple(offsets)
        self._expires = result_url_expires_seconds

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
            verified, stop = self._verify_image(deps, job, token, context)
            if stop is not None:
                return stop
        title, comment, stop = self._title_and_comment(deps, job, token, context, verified)
        if stop is not None:
            return stop
        if not context.may_continue():
            return HandlerResult(HandlerStatus.DEFERRED, "stopped_before_save")
        payload = {
            "fitting_job_id": job_id,
            "status": "COMPLETED",
            "result_image_key": verified,
            "llm_title": title,
            "llm_comment": comment,
        }
        saved = deps.state.save_final_result(job_id, token, payload, verified)
        if saved.applied or saved.reason is ChangeReason.ALREADY_RECORDED:
            return HandlerResult(HandlerStatus.DONE, "result_saved")
        logger.warning(
            "postprocess result not saved fitting_job_id=%s reason=%s", job_id, saved.reason.value
        )
        return HandlerResult(HandlerStatus.DEFERRED, f"save_{saved.reason.value}")

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
                        # DB 전환만 한다. RUN_FALLBACK 실행·인계는 13단계에서 연결한다.
                        return None, HandlerResult(HandlerStatus.DONE, "moved_to_fallback")
                    return None, HandlerResult(
                        HandlerStatus.DEFERRED, f"fallback_{moved.reason.value}"
                    )
                # 마지막 확인이 일시 오류: 객체가 없다고 단정할 수 없다.
                return None, HandlerResult(HandlerStatus.DEFERRED, "s3_unconfirmed_transient")
            # 첫 묶음 실패 → 같은 실행 안에서 두 번째 묶음.

    # --- 2. 제목·코멘트 ---

    def _summaries(self, deps, job) -> list[str]:
        """V1 과 같이 상품 설명 요약을 보조 근거로 쓴다(요청 순서). 상품 DB 의 product_code 는 정수다."""
        codes = [code for code in (job.products or ()) if code.isdigit()]
        found = {product.product_code: product for product in deps.products.find_by_codes(codes)}
        summaries = []
        for code in codes:
            summary = getattr(found.get(str(int(code))), "description_summary", None)
            if isinstance(summary, str) and summary.strip():
                summaries.append(summary)
        return summaries

    def _title_and_comment(self, deps, job, token, context, key):
        job_id = job.fitting_job_id
        summaries = self._summaries(deps, job)
        if not summaries:
            logger.warning(
                "postprocess has no product description summaries fitting_job_id=%s; default text",
                job_id,
            )
            return FALLBACK_TITLE, FALLBACK_COMMENT, None
        from app.virtual_fitting.async_storage import result_image_get_url

        while True:
            if not context.may_continue():
                return None, None, HandlerResult(HandlerStatus.DEFERRED, "stopped_before_llm")
            # a. 최신 기록 확인(불필요한 URL 서명을 줄이기 위한 확인, 최종 판단은 d 의 조건부 확보).
            latest = deps.jobs.find_by_fitting_job_id(job_id)
            if latest is None:
                return None, None, HandlerResult(HandlerStatus.DEFERRED, "job_missing")
            if latest.result_payload is not None:
                return None, None, HandlerResult(HandlerStatus.DONE, "result_already_saved")
            if (
                latest.status is not AsyncJobStatus.POSTPROCESSING
                or latest.next_action is not NextAction.POSTPROCESS
                or latest.verified_image_key != key
            ):
                return None, None, HandlerResult(HandlerStatus.DEFERRED, "llm_state_changed")
            if not deps.state.holds_lease(job_id, token):
                return None, None, HandlerResult(HandlerStatus.DEFERRED, "llm_lease_lost")
            # b. 이미 한도면 URL 을 만들지 않는다.
            if latest.llm_generation_attempt_count >= LLM_GENERATION_LIMIT:
                logger.info(
                    "postprocess llm attempts exhausted fitting_job_id=%s; default text", job_id
                )
                return FALLBACK_TITLE, FALLBACK_COMMENT, None
            # c. 남은 횟수가 있을 때만 새 URL.
            try:
                url = result_image_get_url(deps.storage, key, expires_in=self._expires)
            except ImageStorageError:
                # S3 서명 설정 문제(LLM 오류와 구분). 시도 횟수를 쓰지 않고 보류한다.
                logger.exception("postprocess result image url failed fitting_job_id=%s", job_id)
                return None, None, HandlerResult(HandlerStatus.DEFERRED, "s3_presign_failed")
            # d. 실행 권한·상태·한도를 원자적으로 다시 확인하고 횟수를 확보한다(호출 허용의 최종 판단).
            claim = deps.state.claim_llm_generation_attempt(job_id, token)
            if not claim.applied:
                if claim.reason is ChangeReason.LIMIT_REACHED:
                    logger.info(
                        "postprocess llm attempts exhausted fitting_job_id=%s; default text", job_id
                    )
                    return FALLBACK_TITLE, FALLBACK_COMMENT, None
                return (
                    None,
                    None,
                    HandlerResult(HandlerStatus.DEFERRED, f"llm_claim_{claim.reason.value}"),
                )
            attempt = claim.job.llm_generation_attempt_count
            try:
                comment = deps.comments.generate_comment(url, summaries)
                if not context.may_continue():
                    return None, None, HandlerResult(HandlerStatus.DEFERRED, "stopped_between_llm")
                title = deps.comments.generate_title(comment)
            except (FittingPostprocessError, ValueError) as error:
                logger.warning(
                    "postprocess llm attempt failed fitting_job_id=%s attempt=%s error=%s",
                    job_id,
                    attempt,
                    type(error).__name__,
                )
                continue
            if _usable(title) and _usable(comment):
                return title.strip(), comment.strip(), None
            logger.warning(
                "postprocess llm returned empty text fitting_job_id=%s attempt=%s", job_id, attempt
            )


def _usable(text: Any) -> bool:
    return isinstance(text, str) and bool(text.strip())
