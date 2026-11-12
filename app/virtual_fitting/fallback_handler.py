import dataclasses
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC
from typing import Any
from urllib.error import HTTPError

from app.clients.s3 import ImageStorageError, S3ObjectState
from app.virtual_fitting.exceptions import (
    FittingBalanceExhaustedError,
    FittingDailyBudgetExceededError,
    FittingImageStorageError,
    FittingModelError,
    InvalidFittingCombinationError,
    ProductImageMissingError,
    ProductNotFoundError,
    UnsupportedSubCategoryError,
    VirtualFittingError,
)
from app.virtual_fitting.postprocess_handler import (
    BackgroundRemoval,
    save_completed,
    timing,
    title_and_comment,
)
from app.virtual_fitting.postprocess_worker import HandlerContext, HandlerResult, HandlerStatus
from app.virtual_fitting.pruna_submitter import failure_payload
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncFittingJob,
    AsyncJobStatus,
    NextAction,
)
from app.virtual_fitting.repositories.async_job_state import FALLBACK_LIMIT, ChangeReason
from app.virtual_fitting.repositories.pruna_limiter import SubmissionReason

logger = logging.getLogger(__name__)

# 동기 생성 출력(WEBP, outputQuality 90) 원본. 이전 작업의 원본(pruna-sync.jpg)은 기록된 Key 를 그대로 쓴다.
FALLBACK_RESULT_FILENAME = "pruna-sync.webp"
_INPUT_FAILURES = (
    ProductNotFoundError,
    InvalidFittingCombinationError,
    ProductImageMissingError,
    UnsupportedSubCategoryError,
)
_WAIT_REASONS = (
    SubmissionReason.INTERVAL_WAIT,
    SubmissionReason.IN_FLIGHT_FULL,
    SubmissionReason.SENDS_FULL,
)


LEGACY_FALLBACK_RESULT_FILENAME = "pruna-sync.jpg"


def fallback_result_key(fitting_job_id: int) -> str:
    """fallback 원본 이미지 Key. 비동기 원본(pruna-async.webp)·투명 결과(transparent.webp)와 분리한다."""
    return f"virtual-fitting/results/{fitting_job_id}/{FALLBACK_RESULT_FILENAME}"


def fallback_result_keys(fitting_job_id: int) -> tuple[str, str]:
    """복구 때 확인할 fallback 원본 Key(현재 .webp, 이전 코드가 시작한 작업의 .jpg)."""
    folder = f"virtual-fitting/results/{fitting_job_id}"
    return (fallback_result_key(fitting_job_id), f"{folder}/{LEGACY_FALLBACK_RESULT_FILENAME}")


def fallback_stage(key: str) -> Callable[[AsyncFittingJob], bool]:
    return lambda latest: (
        latest.status is AsyncJobStatus.FALLBACK_RUNNING
        and latest.next_action is NextAction.RUN_FALLBACK
        and latest.fallback_image_key == key
    )


def classify_sync_error(error: BaseException) -> str:
    """동기 생성 예외를 나눈다. 근거(HTTP 상태)가 있을 때만 명확한 거절로 본다. 문구는 분석하지 않는다.

    balance(402) / auth(401·403) / rejected(그 밖의 4xx, 408·409·429 제외) / unknown(결과 불명확)
    """
    if isinstance(error, FittingBalanceExhaustedError):
        return "balance"
    cause = error.__cause__
    if isinstance(error, FittingModelError) and isinstance(cause, HTTPError):
        if cause.code in (401, 403):
            return "auth"
        if 400 <= cause.code < 500 and cause.code not in (408, 409, 429):
            return "rejected"
    return "unknown"


@dataclasses.dataclass(frozen=True)
class _Deps:
    jobs: Any
    state: Any
    products: Any
    storage: Any
    comments: Any
    provider: Any
    limiter: Any
    costs: Any
    budget: Any
    balance: Any
    service: Any


class RunFallbackHandler:
    """10단계 ActionHandler 구현. 의존성은 처음 부를 때 만든다(연결만으로 DB·S3·Runware client 를 만들지 않음)."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_wait_seconds: float = 60.0,
        poll_seconds: float = 2.0,
        recheck_after_seconds: float = 30.0,
        result_url_expires_seconds: int | None = None,
        new_task_uuid: Callable[[], str] = lambda: str(uuid.uuid4()),
        remove_background: Callable[..., Any] | None = None,
        remove_background_enabled: bool | None = None,
        **deps: Any,
    ) -> None:
        self._given = deps
        self._deps: _Deps | None = None
        self._clock, self._sleep = clock, sleep
        self._max_wait, self._poll, self._recheck_after = (
            max_wait_seconds,
            poll_seconds,
            recheck_after_seconds,
        )
        self._expires = result_url_expires_seconds
        self._new_task_uuid = new_task_uuid
        self._background = BackgroundRemoval(
            store=remove_background,
            enabled=remove_background_enabled,
            expires=result_url_expires_seconds,
        )

    # --- 의존성 ---

    def _resolve(self) -> _Deps:
        if self._deps is not None:
            return self._deps
        given = dict(self._given)
        names = [f.name for f in dataclasses.fields(_Deps)]
        if any(given.get(name) is None for name in names if name != "service"):
            given = {**_real_deps(given), **{k: v for k, v in given.items() if v is not None}}
        if given.get("service") is None:
            from app.virtual_fitting.service import VirtualFittingService

            given["service"] = VirtualFittingService(
                given["products"],
                given["provider"],
                given["comments"],
                given["storage"],
                given["budget"],
            )
        self._deps = _Deps(**{name: given[name] for name in names})
        return self._deps

    # --- 처리 ---

    def __call__(self, job: AsyncFittingJob, token: str, context: HandlerContext) -> HandlerResult:
        deps = self._resolve()
        job_id = job.fitting_job_id
        if job.result_payload is not None:
            return HandlerResult(HandlerStatus.DONE, "result_already_saved")
        started = time.perf_counter()
        timings: dict[str, Any] = {}
        image, source_task = job.fallback_image_key, job.fallback_task_uuid
        if image is None:
            if job.fallback_started_at is not None or job.fallback_attempt_count >= FALLBACK_LIMIT:
                # 생성 요청을 보냈을 수 있는데 결과 이미지 기록이 없다. 다시 생성하지 않는다(15단계 복구 대상).
                logger.warning(
                    "fallback already started without a recorded image fitting_job_id=%s; held",
                    job_id,
                )
                return HandlerResult(HandlerStatus.DEFERRED, "fallback_started_unconfirmed")
            image, source_task, stop = self._generate(deps, job, token, context)
            if stop is not None:
                return stop
        # 원본(image)은 fallback_image_key 로 그대로 두고, 최종 결과·LLM 입력은 배경을 지운 이미지다.
        final, stop = self._background(
            deps, job, token, context, image, timings, source_task=source_task
        )
        if stop is not None:
            return stop
        title, comment, stop = title_and_comment(
            deps,
            job,
            token,
            context,
            final,
            stage_ok=fallback_stage(image),
            expires_in=self._expires,
            timings=timings,
        )
        if stop is not None:
            return stop
        if not context.may_continue():
            return HandlerResult(HandlerStatus.DEFERRED, "stopped_before_save")
        result = save_completed(deps.state, job_id, token, final, title, comment)
        timing(
            job,
            "fallback_summary",
            result.reason,
            int((time.perf_counter() - started) * 1000),
            result_image_key=final,
            original_image_key=image,
            **timings,
        )
        return result

    def _fail(self, deps, job_id: int, token: str, code: str, message: str) -> HandlerResult:
        saved = deps.state.save_final_result(job_id, token, failure_payload(job_id, code, message))
        if saved.applied or saved.reason is ChangeReason.ALREADY_RECORDED:
            logger.info("fallback failure saved fitting_job_id=%s code=%s", job_id, code)
            return HandlerResult(HandlerStatus.DONE, f"failure_saved_{code}")
        return HandlerResult(HandlerStatus.DEFERRED, f"failure_not_saved_{saved.reason.value}")

    def _check_budget(self, deps, job_id, token) -> HandlerResult | None:
        try:
            deps.budget.ensure_available()
            deps.balance.ensure_available()
        except (FittingDailyBudgetExceededError, FittingBalanceExhaustedError) as error:
            return self._fail(deps, job_id, token, error.code, error.message)
        except (
            VirtualFittingError
        ) as error:  # 잔액 조회 실패·사용량 조회 실패·DB 오류 → 확정하지 않고 보류
            logger.warning(
                "fallback budget/balance check postponed fitting_job_id=%s reason=%s",
                job_id,
                error.code,
            )
            return HandlerResult(HandlerStatus.DEFERRED, f"check_{error.code.lower()}")
        return None

    def _user_image_url(self, deps, job) -> str:
        from app.virtual_fitting.async_storage import user_image_get_url

        return user_image_get_url(deps.storage, job.user_image_key)

    def _generate(
        self, deps, job, token, context
    ) -> tuple[str | None, str | None, HandlerResult | None]:
        """(원본 Key, fallback taskUUID, None) 또는 (None, None, 멈출 때의 HandlerResult)."""
        key, stop = self._generate_image(deps, job, token, context)
        if stop is not None:
            return None, None, stop
        latest = deps.jobs.find_by_fitting_job_id(job.fitting_job_id)
        return key, latest.fallback_task_uuid if latest is not None else None, None

    def _generate_image(self, deps, job, token, context) -> tuple[str | None, HandlerResult | None]:
        from app.virtual_fitting.request_worker import (
            USER_IMAGE_NOT_FOUND_CODE,
            USER_IMAGE_NOT_FOUND_MESSAGE,
        )

        job_id = job.fitting_job_id
        if not context.may_continue():
            return None, HandlerResult(HandlerStatus.DEFERRED, "stopped_before_input")
        if not job.user_image_key or not job.products:
            logger.error("fallback job has no original input fitting_job_id=%s", job_id)
            return None, HandlerResult(HandlerStatus.DEFERRED, "no_original_input")
        # 1. 입력 준비(외부 생성 전, 횟수를 쓰지 않는다)
        try:
            url = self._user_image_url(deps, job)
            prepared = deps.service.prepare_fitting(list(job.products), url)
        except _INPUT_FAILURES as error:
            return None, self._fail(deps, job_id, token, error.code, error.message)
        except ImageStorageError:
            logger.exception("fallback user image url failed fitting_job_id=%s", job_id)
            return None, HandlerResult(HandlerStatus.DEFERRED, "user_image_presign_failed")
        head = deps.storage.head_object(job.user_image_key)
        if head.state is S3ObjectState.NOT_FOUND:
            return None, self._fail(
                deps, job_id, token, USER_IMAGE_NOT_FOUND_CODE, USER_IMAGE_NOT_FOUND_MESSAGE
            )
        if head.state is not S3ObjectState.EXISTS:
            return None, HandlerResult(HandlerStatus.DEFERRED, f"user_image_{head.state.value}")
        stop = self._check_budget(deps, job_id, token)
        if stop is not None:
            return None, stop
        # 2. fallback taskUUID 를 외부 호출 전에 저장(있으면 재사용)
        prepared_task = deps.state.prepare_fallback(job_id, token, self._new_task_uuid())
        if not (prepared_task.applied or prepared_task.reason is ChangeReason.ALREADY_RECORDED):
            return None, HandlerResult(
                HandlerStatus.DEFERRED, f"fallback_task_{prepared_task.reason.value}"
            )
        fallback_task = prepared_task.job.fallback_task_uuid
        # 3. 자리 대기 → 확보(횟수·시작 기록) → 생성 1회
        started = last_check = self._clock()
        while True:
            if not context.may_continue():
                return None, HandlerResult(HandlerStatus.DEFERRED, "stopped_waiting_for_slot")
            if self._clock() - last_check >= self._recheck_after:
                stop = self._check_budget(deps, job_id, token)
                if stop is not None:
                    return None, stop
                last_check = self._clock()
            try:
                fitting_input = dataclasses.replace(
                    prepared.fitting_input, person_image_url=self._user_image_url(deps, job)
                )
            except ImageStorageError:
                logger.exception("fallback user image url failed fitting_job_id=%s", job_id)
                return None, HandlerResult(HandlerStatus.DEFERRED, "user_image_presign_failed")
            grant = deps.limiter.begin_fallback(job_id, token, fallback_task)
            if grant.granted:
                break
            if grant.reason not in _WAIT_REASONS:
                logger.warning(
                    "fallback slot refused fitting_job_id=%s reason=%s", job_id, grant.reason.value
                )
                return None, HandlerResult(HandlerStatus.DEFERRED, f"fallback_{grant.reason.value}")
            waited = self._clock() - started
            if waited >= self._max_wait:
                return None, HandlerResult(HandlerStatus.DEFERRED, "fallback_slot_wait")
            self._sleep(min(self._poll, grant.wait_seconds or self._poll, self._max_wait - waited))
        try:
            result = deps.provider.try_on(fitting_input, task_uuid=fallback_task, strict_cost=False)
        except Exception as error:  # noqa: BLE001 (보낸 뒤의 어떤 예외도 결과 불명확으로 다룬다, 재전송 없음)
            return None, self._after_error(deps, job_id, token, fallback_task, error)
        finally:
            try:
                deps.limiter.finish_send(job_id)
            except VirtualFittingError:
                logger.exception("fallback send slot release failed fitting_job_id=%s", job_id)
        # 생성 성공: 진행 중 자리 반환 → 비용 → 이미지 저장 → 저장 기록
        deps.limiter.release_in_flight(job_id, task_uuid=fallback_task)
        self._record_cost(deps, job_id, fallback_task, result.cost, grant.started_at)
        key = fallback_result_key(job_id)
        try:
            deps.service.store_result(result, key=key)
        except FittingImageStorageError:
            # 생성은 끝났고 결과 URL 은 저장하지 않았으므로 이어 갈 근거가 없다(15단계 복구 대상, 다시 생성하지 않음).
            logger.error("fallback image store failed fitting_job_id=%s; held", job_id)
            return None, HandlerResult(HandlerStatus.DEFERRED, "fallback_image_store_failed")
        recorded = deps.state.record_fallback_image(job_id, token, key)
        if recorded.applied or recorded.reason is ChangeReason.ALREADY_RECORDED:
            return key, None
        logger.warning(
            "fallback image stored but not recorded fitting_job_id=%s reason=%s",
            job_id,
            recorded.reason.value,
        )
        return None, HandlerResult(
            HandlerStatus.DEFERRED, f"fallback_image_record_{recorded.reason.value}"
        )

    def _after_error(self, deps, job_id, token, fallback_task, error) -> HandlerResult:
        kind = classify_sync_error(error)
        logger.warning(
            "fallback generation failed fitting_job_id=%s kind=%s error=%s",
            job_id,
            kind,
            type(error).__name__,
        )
        if kind == "unknown":
            # 원격 생성이 끝났는지 모른다. 진행 중 자리를 유지하고 다시 보내지 않는다(15단계 복구 대상).
            return HandlerResult(HandlerStatus.DEFERRED, "fallback_result_unknown")
        deps.limiter.release_in_flight(job_id, task_uuid=fallback_task)
        if kind == "balance":
            error = FittingBalanceExhaustedError("Runware 잔액이 부족합니다.")
            return self._fail(deps, job_id, token, error.code, error.message)
        if kind == "auth":
            logger.error(
                "fallback rejected by auth fitting_job_id=%s; operator check required", job_id
            )
            return HandlerResult(HandlerStatus.DEFERRED, "fallback_auth_operator_check")
        return self._fail(deps, job_id, token, FittingModelError.code, FittingModelError.message)

    def _record_cost(self, deps, job_id, fallback_task, cost, started_at) -> None:
        if cost is None:
            logger.warning("fallback cost missing fitting_job_id=%s; not recorded", job_id)
            return
        try:
            deps.costs.record(fallback_task, job_id, cost, started_at.astimezone(UTC).date())
        except ValueError:
            logger.warning("fallback cost invalid fitting_job_id=%s; not recorded", job_id)
        except VirtualFittingError:
            # 비용 기록 실패 때문에 이미 받은 이미지 저장을 건너뛰지 않는다. 기록하지 못한 비용은 Runware 보정이 메운다.
            logger.exception("fallback cost record failed fitting_job_id=%s", job_id)


def _real_deps(given: dict[str, Any]) -> dict[str, Any]:
    from datetime import timedelta

    from app.clients.s3 import S3ImageStorage
    from app.config import settings
    from app.config.database import get_connection_pool
    from app.virtual_fitting.budget import FittingBudget
    from app.virtual_fitting.providers.runware import DEFAULT_TIMEOUT, RunwarePrunaProvider
    from app.virtual_fitting.providers.runware_balance import (
        MinimumBalanceCheck,
        RunwareBalanceClient,
    )
    from app.virtual_fitting.providers.runware_comment import RunwareCommentProvider
    from app.virtual_fitting.providers.runware_usage import RunwareUsageClient
    from app.virtual_fitting.repositories.async_job_repository import AsyncJobRepository
    from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository
    from app.virtual_fitting.repositories.product_repository import ProductRepository
    from app.virtual_fitting.repositories.pruna_limiter import PrunaLimiter, TaskCostRepository
    from app.virtual_fitting.repositories.usage_repository import UsageRepository

    pool = get_connection_pool()

    def need(name, build):
        return given.get(name) if given.get(name) is not None else build()

    return {
        "jobs": need("jobs", lambda: AsyncJobRepository(pool)),
        "state": need("state", lambda: AsyncJobStateRepository(pool)),
        "products": need("products", lambda: ProductRepository(pool)),
        "storage": need("storage", S3ImageStorage),
        "comments": need("comments", RunwareCommentProvider),
        "provider": need("provider", RunwarePrunaProvider),
        "limiter": need(
            "limiter",
            lambda: PrunaLimiter(
                pool,
                max_in_flight=settings.VIRTUAL_FITTING_PRUNA_MAX_IN_FLIGHT,
                max_concurrent_sends=settings.VIRTUAL_FITTING_PRUNA_MAX_CONCURRENT_SENDS,
                min_send_interval_seconds=settings.VIRTUAL_FITTING_PRUNA_MIN_SEND_INTERVAL_SECONDS,
                send_hold_seconds=DEFAULT_TIMEOUT + 30,
            ),
        ),
        "costs": need("costs", lambda: TaskCostRepository(pool)),
        "budget": need(
            "budget",
            lambda: FittingBudget(
                UsageRepository(pool),
                RunwareUsageClient(),
                daily_limit=settings.FITTING_DAILY_BUDGET_USD,
                reconcile_interval=timedelta(seconds=settings.FITTING_USAGE_RECONCILE_SECONDS),
            ),
        ),
        "balance": need(
            "balance",
            lambda: MinimumBalanceCheck(RunwareBalanceClient(), settings.FITTING_MIN_BALANCE_USD),
        ),
    }
