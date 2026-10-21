"""가상피팅 V2 AI 요청 Worker (be-fitting-request).

FastAPI lifespan 이 시작·종료한다(별도 실행 명령 없음, VIRTUAL_FITTING_REQUEST_WORKER_ENABLED 기본 꺼짐).
한 번에 메시지 하나씩:
  수신(6단계 receive_fitting_requests) → fingerprint → 작업 등록·중복 비교 → 실행 권한 선점 → 사전 검증
  → (통과) 8단계 Pruna 제출 연결 지점 / (확정 실패) 실패 결과 저장 / (일시 장애) 그대로 두고 다음 수신 때 재처리

메시지 삭제 원칙: 접수 결과와 후속 실행·복구에 필요한 기록이 저장되고, 실제로 이어서 처리할 경로가 있을 때만
삭제한다. RECEIVED 저장, task_uuid·이미지 Key·호출 시작 기록만으로는 삭제하지 않고, 아직 없는 Watchdog·결과
발행을 복구 수단으로 가정하지 않는다.
결과 발행까지 끝난 작업(COMPLETED/FAILED + PUBLISHED)의 중복 메시지는 지운다. 그 밖에는 deletion_allowed
(pruna_submitter.py)가 최신 기록·다음 일·FollowUpPaths 로 판단한다. 기본 NoFollowUpPaths 는 이어 갈 경로가 없다고
답하므로(Webhook 9단계·결과 발행 14단계·Watchdog 15단계 미구현), 8단계에서도 접수·실패 저장·UNKNOWN 뒤 메시지를
지우지 않는다. Pruna 제출은 8단계 PrunaSubmitter 가 submit 연결 지점으로 한다.

실행 권한(DB lease)·SQS Visibility Timeout 은 처리 단계 사이(before_step)에서 각각 연장하고, Pruna 생성 요청 HTTP
전송 중에는 PrunaSubmitter 의 keepalive thread 가 주기적으로 연장한다. 잔액·사용량 조회 같은 짧은 단계 하나 안에서는
주기적으로 연장하지 않는다.

동기 작업(SQS long polling·DB·S3·Runware)은 worker thread 에서 하나씩 돈다(전용 CapacityLimiter(1)). 그래서
FastAPI 이벤트 루프와 다른 요청의 thread 를 막지 않는다. DB 트랜잭션·행 잠금은 5단계 메서드 안에서만 짧게 잡고,
외부 조회 동안 들고 있지 않는다.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from anyio import CapacityLimiter, to_thread

from app import metrics
from app.clients.s3 import S3ObjectState
from app.clients.sqs import FittingQueue, SqsQueueClient, SqsResult
from app.virtual_fitting import diagnostics as diag
from app.virtual_fitting.async_request import (
    FittingRequest,
    InvalidRequestMessage,
    RequestBatch,
    ValidRequestMessage,
)
from app.virtual_fitting.exceptions import (
    FittingBalanceExhaustedError,
    FittingDailyBudgetExceededError,
    FittingDatabaseError,
    InvalidFittingCombinationError,
    ProductImageMissingError,
    ProductNotFoundError,
    UnsupportedSubCategoryError,
    VirtualFittingError,
)
from app.virtual_fitting.models import FittingProduct
from app.virtual_fitting.pruna_submitter import (
    FollowUpPaths,
    NoFollowUpPaths,
    deletion_allowed,
    failure_payload,
)
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncFittingJob,
    AsyncJobRepository,
    AsyncJobStatus,
    NewAsyncFittingJob,
    NextAction,
    ResultMessageStatus,
)
from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository, ChangeReason

logger = logging.getLogger(__name__)

FINGERPRINT_VERSION = "v1"
USER_IMAGE_NOT_FOUND_CODE = "INVALID_REQUEST"
USER_IMAGE_NOT_FOUND_MESSAGE = "가상 피팅에 사용할 사진을 찾을 수 없습니다."
# 재시도로 풀리지 않는 업무 실패. 모델을 부르지 않고 실패 결과를 저장한다.
_PERMANENT_FAILURES = (
    ProductNotFoundError,
    InvalidFittingCombinationError,
    ProductImageMissingError,
    UnsupportedSubCategoryError,
    FittingDailyBudgetExceededError,
    FittingBalanceExhaustedError,
)


# --- fingerprint ---


def request_fingerprint(user_image_key: str, product_codes: Sequence[str]) -> str:
    """같은 fitting_job_id 로 다시 온 요청의 사진·상품이 같은지 비교하는 값.

    규칙(v1): {"product_codes": 정렬한 상품 코드 목록(중복 개수 유지, 문자열 그대로), "user_image_key": Key} 를
    키 정렬·공백 없는 UTF-8 JSON 으로 직렬화해 SHA-256 → "v1:<hex>". requested_at·수신 정보·추가 필드는 넣지
    않는다. 정렬은 비교용일 뿐, 저장하는 상품 순서는 요청 그대로다.
    """
    canonical = json.dumps(
        {"product_codes": sorted(product_codes), "user_image_key": user_image_key},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"{FINGERPRINT_VERSION}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


# --- 사전 검증 ---


class PrecheckStatus(str, Enum):
    PASSED = "passed"
    # 재시도로 풀리지 않는 업무 실패(실패 결과 저장 대상)
    FAILED = "failed"
    # 일시 장애·설정 문제 등 지금 확정할 수 없음(결과 저장 안 함)
    RETRY = "retry"


@dataclass(frozen=True)
class PrecheckOutcome:
    status: PrecheckStatus
    error_code: str | None = None
    error_message: str | None = None
    reason: str = ""
    # PASSED 일 때 상의 → 하의 순서의 상품(8단계 Pruna 입력 준비에 쓴다)
    products: tuple[FittingProduct, ...] = ()


class ProductLookup(Protocol):
    def find_by_codes(self, product_codes: Sequence[str]) -> list[FittingProduct]: ...


class ObjectHead(Protocol):
    def head_object(self, key: str) -> Any: ...


class Budget(Protocol):
    def ensure_available(self) -> None: ...


class BalanceCheck(Protocol):
    """Runware 잔액 사전 확인 연결 지점. 구현체는 잔액이 부족하면 FittingBalanceExhaustedError 를 낸다.

    8단계 구현: providers/runware_balance.py 의 MinimumBalanceCheck 가 getDetails 의 balance.amount(USD)를 조회해
    FITTING_MIN_BALANCE_USD(기본 1.00 달러) 이하이면 막는다. 조회 실패(FittingBalanceUnknownError)는 잔액 부족으로
    확정하지 않고 생성을 보류·재처리하며, 잔액 부족이면 V1 fallback 도 하지 않는다. create_request_worker 가 연결한다.
    """

    def ensure_available(self) -> None: ...


class NoBalancePrecheck:
    """RequestPrecheck 에 balance 를 주지 않았을 때의 기본값(테스트용). 실제 잔액을 조회하지 않고 막지 않는다.

    실제 Worker(create_request_worker)는 MinimumBalanceCheck 를 쓴다.

    프로젝트에 Runware 잔액 사전 조회 기능이 없다(V1 도 생성 요청에 대한 402 응답으로만 잔액 부족을 안다).
    생성 요청의 잔액 부족 응답은 Provider 가 분류한다(V1: FittingBalanceExhaustedError, V2 1단계 Provider:
    REJECTED / balance_exhausted). V2 의 실제 생성 호출 연결은 8단계다.
    """

    def ensure_available(self) -> None:
        return None


class RequestPrecheck:
    """Pruna 호출 전 확인. V1 의 상품 조회·조합 검증·세부 분류 매핑·하루 예산 확인을 그대로 쓴다.

    순서: 상품(존재·이미지·조합·중복 코드) → 세부 분류 → 원본 사진(S3 HEAD, 내려받지 않음) → 하루 예산 → 잔액.
    - 하루 예산(FittingBudget.ensure_available): 오늘 사용액이 FITTING_DAILY_BUDGET_USD(5 달러) **이상**이면 막는다.
      남은 잔액이 아니라 오늘 쓴 금액이 기준이다. 마지막 보정이 FITTING_USAGE_RECONCILE_SECONDS 보다 오래됐으면
      Runware getUsageActivity 로 하루 사용액을 조회해 AI DB usage_amount 에 보정값(큰 쪽)과 보정 시각을 저장한다.
      새 생성 비용을 더하는 기록(record)은 하지 않는다(생성 후에 하는 일, 8단계).
    - 잔액: BalanceCheck(실제 Worker 는 MinimumBalanceCheck — 사용 가능한 잔액이 1 달러 이하이면 막음). 하루 사용액
      5 달러 상한과는 별개다.
    - 유료 생성 시도 횟수는 늘리지 않는다.
    """

    def __init__(
        self,
        products: ProductLookup,
        image_storage: ObjectHead,
        budget: Budget,
        balance: BalanceCheck | None = None,
    ):
        self._products = products
        self._image_storage = image_storage
        self._budget = budget
        self._balance = balance or NoBalancePrecheck()

    def run(
        self, request: FittingRequest, before_step: Callable[[], bool] = lambda: True
    ) -> PrecheckOutcome:
        """before_step() 이 False 면(lease·Visibility 연장 실패, 종료 요청) 다음 외부 조회를 시작하지 않는다."""
        from app.virtual_fitting.product_selection import select_fitting_products
        from app.virtual_fitting.prompt import build_fitting_prompt

        try:
            if not before_step():
                return PrecheckOutcome(PrecheckStatus.RETRY, reason="stopped_before_products")
            codes = list(request.product_codes)
            not_numeric = [code for code in codes if not code.isdigit()]
            if not_numeric:  # 상품 DB 의 product_code 는 정수다. 숫자가 아닌 코드는 없는 상품이다.
                diag.event(
                    "product_lookup",
                    component="request_worker",
                    fitting_job_id=request.fitting_job_id,
                    result="invalid_code",
                    requested_count=len(codes),
                    invalid_count=len(not_numeric),
                )
                raise ProductNotFoundError(not_numeric)
            lookup_started = time.monotonic()
            try:
                products = select_fitting_products(codes, self._products)
            except ProductNotFoundError as error:
                # 기존 조회 결과(누락 코드)만 쓴다. 로그를 위한 추가 조회는 하지 않는다.
                missing = [str(code) for code in error.product_codes]
                diag.event(
                    "product_lookup",
                    component="request_worker",
                    fitting_job_id=request.fitting_job_id,
                    result="missing",
                    requested=diag.codes(codes),
                    found=diag.codes(c for c in codes if c not in missing),
                    missing=diag.codes(missing),
                    elapsed_ms=diag.elapsed_ms(lookup_started),
                )
                raise
            except FittingDatabaseError as error:
                diag.event(
                    "product_lookup_failed",
                    logging.ERROR,
                    component="request_worker",
                    fitting_job_id=request.fitting_job_id,
                    result="db_error",
                    error_code=error.code,
                    cause_type=type(error.__cause__).__name__ if error.__cause__ else "none",
                    requested=diag.codes(codes),
                    elapsed_ms=diag.elapsed_ms(lookup_started),
                )
                raise
            except VirtualFittingError as error:
                diag.event(
                    "product_lookup",
                    component="request_worker",
                    fitting_job_id=request.fitting_job_id,
                    result="rejected",
                    error_code=error.code,
                    requested=diag.codes(codes),
                    elapsed_ms=diag.elapsed_ms(lookup_started),
                )
                raise
            diag.event(
                "product_lookup",
                component="request_worker",
                fitting_job_id=request.fitting_job_id,
                result="found",
                requested=diag.codes(codes),
                found=diag.codes(p.product_code for p in products),
                missing="empty",
                elapsed_ms=diag.elapsed_ms(lookup_started),
            )
            build_fitting_prompt(products)  # 지원하지 않는 세부 분류면 UnsupportedSubCategoryError

            if not before_step():
                return PrecheckOutcome(PrecheckStatus.RETRY, reason="stopped_before_image")
            image = self._image_storage.head_object(request.user_image_key)
            state = getattr(image, "state", None)
            if state is S3ObjectState.NOT_FOUND:
                return PrecheckOutcome(
                    PrecheckStatus.FAILED,
                    USER_IMAGE_NOT_FOUND_CODE,
                    USER_IMAGE_NOT_FOUND_MESSAGE,
                    reason="user_image_not_found",
                )
            if state is not S3ObjectState.EXISTS:
                # 403·설정 오류·일시 오류는 확정하지 않는다(403 은 객체가 없을 때도 날 수 있다).
                return PrecheckOutcome(
                    PrecheckStatus.RETRY, reason=f"user_image_{getattr(state, 'value', 'unknown')}"
                )

            if not before_step():
                return PrecheckOutcome(PrecheckStatus.RETRY, reason="stopped_before_budget")
            self._budget.ensure_available()
            if not before_step():
                return PrecheckOutcome(PrecheckStatus.RETRY, reason="stopped_before_balance")
            self._balance.ensure_available()
        except _PERMANENT_FAILURES as error:
            return PrecheckOutcome(
                PrecheckStatus.FAILED, error.code, error.message, reason=type(error).__name__
            )
        except VirtualFittingError as error:  # DB 조회 실패·Runware 사용량 조회 실패 등
            return PrecheckOutcome(PrecheckStatus.RETRY, reason=error.code)
        except Exception as error:
            # 예상하지 못한 오류도 업무 실패로 확정하지 않는다.
            logger.exception("request precheck crashed fitting_job_id=%s", request.fitting_job_id)
            return PrecheckOutcome(PrecheckStatus.RETRY, reason=type(error).__name__)
        return PrecheckOutcome(PrecheckStatus.PASSED, products=tuple(products))


# --- 메시지 하나 처리 ---


@dataclass(frozen=True)
class RequestOutcome:
    """메시지 하나의 처리 결과. delete 가 True 일 때만 Worker 가 메시지를 지운다."""

    message_id: str
    fitting_job_id: int | None
    action: str
    delete: bool = False


class Heartbeat:
    """DB lease 와 SQS Visibility Timeout 을 각각 연장한다. 둘 중 하나라도 실패하면 False.

    beat() 는 불릴 때만 연장한다(마지막 연장 뒤 interval 이 지났을 때). 처리 단계 사이에서 부르고, Pruna 생성 요청
    전송 중에는 pruna_submitter.keepalive 가 별도 thread 에서 주기적으로 부른다. queue 는 그 메시지를 받은 큐다
    (요청 Worker: be-fitting-request, 후처리 Worker: ai-fitting-response).
    """

    def __init__(
        self,
        sqs: SqsQueueClient,
        state: AsyncJobStateRepository,
        fitting_job_id: int,
        token: str,
        receipt_handle: str,
        *,
        lease_seconds: float,
        visibility_seconds: int,
        interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        queue: FittingQueue = FittingQueue.BE_FITTING_REQUEST,
        name: str = "request",
    ):
        self._sqs, self._state = sqs, state
        self._queue, self._name = queue, name
        self._job_id, self._token, self._receipt = fitting_job_id, token, receipt_handle
        self._lease_seconds, self._visibility_seconds = lease_seconds, visibility_seconds
        self._interval, self._clock = interval_seconds, clock
        self._last = clock()

    def beat(self) -> bool:
        if self._clock() - self._last < self._interval:
            return True
        lease = self._state.renew_lease(self._job_id, self._token, self._lease_seconds)
        if not lease.applied:
            logger.warning(
                "%s worker lease renewal failed fitting_job_id=%s reason=%s",
                self._name,
                self._job_id,
                lease.reason.value,
            )
            return False
        visibility: SqsResult = self._sqs.change_visibility(
            self._queue, self._receipt, self._visibility_seconds
        )
        if not visibility.succeeded:
            logger.warning(
                "%s worker visibility extension failed fitting_job_id=%s reason=%s",
                self._name,
                self._job_id,
                visibility.reason,
            )
            return False
        self._last = self._clock()
        return True


# 8단계 Pruna 제출 연결 지점. (작업, lease 토큰, 요청, 사전 검증 결과, heartbeat) → 메시지를 지워도 되는지.
# True 는 접수 결과와 후속 실행·복구에 필요한 기록을 저장했고, 실제로 이어서 처리할 경로가 있을 때만 돌려준다
# (task_uuid·이미지 Key·호출 시작 기록만으로는 False). 8단계 구현은 pruna_submitter.PrunaSubmitter 다.
SubmitHandler = Callable[[AsyncFittingJob, str, FittingRequest, PrecheckOutcome, Heartbeat], bool]
ConflictAlert = Callable[[AsyncFittingJob, FittingRequest, str], None]


def log_conflict(job: AsyncFittingJob, request: FittingRequest, message_id: str) -> None:
    """기본 입력 충돌 처리: 오류 로그만 남긴다(실제 알림 전송은 구현돼 있지 않다). 기존 작업은 그대로 둔다.

    운영 알림이 필요하면 RequestProcessor(on_conflict=...) 로 다른 구현을 넣는다.
    """
    logger.error(
        "be-fitting-request input conflict fitting_job_id=%s message_id=%s existing_status=%s",
        job.fitting_job_id,
        message_id,
        job.status.value,
    )


class RequestProcessor:
    """동기. 메시지 하나를 처리하고 RequestOutcome 을 돌려준다. 예외를 밖으로 내지 않는다."""

    def __init__(
        self,
        jobs: AsyncJobRepository,
        state: AsyncJobStateRepository,
        precheck: RequestPrecheck,
        sqs: SqsQueueClient,
        *,
        lease_seconds: float,
        visibility_seconds: int,
        heartbeat_interval_seconds: float | None = None,
        submit: SubmitHandler | None = None,
        on_conflict: ConflictAlert = log_conflict,
        should_stop: Callable[[], bool] = lambda: False,
        follow_up: FollowUpPaths | None = None,
        follow_up_queue: Any = None,
    ):
        self._jobs, self._state, self._precheck, self._sqs = jobs, state, precheck, sqs
        # RUN_FALLBACK 을 기다리는 작업의 재수신 때 내부 큐 발행을 다시 시도한다(SqsQueueClient). 없으면 하지 않는다.
        self._follow_up_queue = follow_up_queue
        # 이미 진행된 작업의 중복 메시지를 지워도 되는지(이어 갈 실제 경로가 있는지). 기본은 "없음".
        self._follow_up = follow_up or NoFollowUpPaths()
        self._lease_seconds = lease_seconds
        self._visibility_seconds = visibility_seconds
        self._heartbeat_interval = (
            lease_seconds / 3 if heartbeat_interval_seconds is None else heartbeat_interval_seconds
        )
        self._submit = submit
        self._on_conflict = on_conflict
        self._should_stop = should_stop

    def process(self, item: ValidRequestMessage | InvalidRequestMessage) -> RequestOutcome:
        if isinstance(item, InvalidRequestMessage):
            return self._invalid(item)
        try:
            return self._valid(item)
        except Exception:
            # 메시지는 남겨 두고 다음 수신 때 다시 본다.
            logger.exception(
                "request worker failed fitting_job_id=%s message_id=%s",
                item.request.fitting_job_id,
                item.message_id,
            )
            return RequestOutcome(item.message_id, item.request.fitting_job_id, "error")

    @staticmethod
    def _invalid(item: InvalidRequestMessage) -> RequestOutcome:
        # 저장할 수 있는 값이 없다(가짜 사진·상품·시각을 만들지 않는다). 읽힌 작업 번호가 있어도 기존 작업에
        # 실패를 쓰지 않는다. 삭제하지 않으므로 재전달되다가 Redrive Policy 에 따라 DLQ 로 간다(운영 복구 대상).
        return RequestOutcome(item.message_id, item.fitting_job_id, "invalid_kept")

    def _valid(self, item: ValidRequestMessage) -> RequestOutcome:
        request = item.request
        job_id = request.fitting_job_id

        def outcome(action: str, delete: bool = False) -> RequestOutcome:
            return RequestOutcome(item.message_id, job_id, action, delete)

        fingerprint = request_fingerprint(request.user_image_key, request.product_codes)
        diag.event(
            "job_save_started",
            component="request_worker",
            fitting_job_id=job_id,
            message_id=item.message_id,
        )
        try:
            created = self._jobs.create(
                NewAsyncFittingJob(
                    fitting_job_id=job_id,
                    request_fingerprint=fingerprint,
                    user_image_key=request.user_image_key,
                    products=request.product_codes,
                    requested_at=request.requested_at,
                )
            )
        except Exception as error:
            diag.event(
                "job_save_failed",
                logging.ERROR,
                component="request_worker",
                fitting_job_id=job_id,
                message_id=item.message_id,
                error_type=type(error).__name__,
                error_code=getattr(error, "code", "none"),
            )
            raise
        job = created.job
        # create 는 INSERT·기존 행 조회를 한 트랜잭션으로 끝낸 뒤 돌아온다(여기서는 이미 commit 됐다).
        diag.event(
            "job_saved" if created.created else "job_existing",
            component="request_worker",
            fitting_job_id=job_id,
            message_id=item.message_id,
            status=job.status.value,
            next_action=job.next_action.value if job.next_action else "none",
            result_message_status=job.result_message_status.value,
            fingerprint_match=job.request_fingerprint == fingerprint,
        )
        if not created.created and job.request_fingerprint != fingerprint:
            self._on_conflict(job, request, item.message_id)
            return outcome("input_conflict_kept")
        if (
            job.status in (AsyncJobStatus.COMPLETED, AsyncJobStatus.FAILED)
            and job.result_message_status is ResultMessageStatus.PUBLISHED
        ):
            # 결과 발행까지 끝난 작업의 중복 메시지. 다시 할 일이 없으므로 삭제해도 잃는 것이 없다.
            return outcome("already_published_deleted", delete=True)
        if job.next_action is not NextAction.SUBMIT_PRUNA:
            from app.virtual_fitting.follow_up_queue import publish_follow_up, waiting_for_fallback

            latest = self._jobs.find_by_fitting_job_id(job_id)
            if waiting_for_fallback(latest) and self._follow_up_queue is not None:
                # 제출 거절로 RUN_FALLBACK 이 됐는데 발행이 실패했거나 발행 전에 멈췄다. Pruna 제출·사전 검증은 다시 하지
                # 않고 RUN_FALLBACK 발행만 다시 한다. 등록이 확인될 때만 요청 메시지를 지운다.
                published = publish_follow_up(
                    self._follow_up_queue, latest, NextAction.RUN_FALLBACK
                ).succeeded
                return outcome(
                    "run_fallback_published" if published else "run_fallback_publish_failed_kept",
                    delete=published,
                )
            # 실패 결과 저장 뒤(PUBLISH_RESULT)나 Pruna 이후 단계. 이어서 할 처리는 해당 단계가 맡는다.
            # 그 일을 이어 갈 실제 경로가 있을 때만 메시지를 지운다(최신 기록으로 판단).
            return outcome(
                f"waiting_{(job.next_action.value if job.next_action else 'none').lower()}",
                delete=deletion_allowed(self._jobs.find_by_fitting_job_id(job_id), self._follow_up),
            )
        if self._should_stop():
            return outcome("stopping_kept")

        grant = self._state.acquire_lease(job_id, NextAction.SUBMIT_PRUNA, self._lease_seconds)
        if not grant.acquired:
            # 다른 Worker 가 처리 중(LEASE_HELD)이거나 상태가 바뀌었다. 추가로 실행하지 않는다.
            return outcome(f"lease_{grant.reason.value}")
        token = grant.token
        try:
            heartbeat = Heartbeat(
                self._sqs,
                self._state,
                job_id,
                token,
                item.receipt_handle,
                lease_seconds=self._lease_seconds,
                visibility_seconds=self._visibility_seconds,
                interval_seconds=self._heartbeat_interval,
            )
            if grant.job is not None and grant.job.pruna_submit_started_at is not None:
                # 보냈을 수 있는 작업(접수 여부 불명). 사전 검증·재전송 없이 복구 경로 유무만 본다.
                return outcome(
                    "submit_already_started",
                    delete=deletion_allowed(
                        self._jobs.find_by_fitting_job_id(job_id), self._follow_up
                    ),
                )
            check = self._precheck.run(
                request, before_step=lambda: not self._should_stop() and heartbeat.beat()
            )
            if check.status is PrecheckStatus.RETRY:
                return outcome(f"retry_{check.reason}")
            if check.status is PrecheckStatus.FAILED:
                try:
                    saved = self._state.save_final_result(
                        job_id,
                        token,
                        failure_payload(job_id, check.error_code, check.error_message),
                    )
                except Exception as error:
                    diag.event(
                        "failure_result_save_failed",
                        logging.ERROR,
                        component="request_worker",
                        fitting_job_id=job_id,
                        message_id=item.message_id,
                        error_code=check.error_code,
                        error_type=type(error).__name__,
                    )
                    raise
                if saved.reason not in (ChangeReason.APPLIED, ChangeReason.ALREADY_RECORDED):
                    diag.event(
                        "failure_result_not_saved",
                        logging.WARNING,
                        component="request_worker",
                        fitting_job_id=job_id,
                        message_id=item.message_id,
                        error_code=check.error_code,
                        reason=saved.reason.value,
                    )
                    return outcome(f"failure_not_saved_{saved.reason.value}")
                # save_final_result 는 commit 뒤 돌아온다. 저장만 했고 결과 큐 발행은 결과 전송 루프가 따로 한다.
                diag.event(
                    "failure_result_saved",
                    component="request_worker",
                    fitting_job_id=job_id,
                    message_id=item.message_id,
                    error_code=check.error_code,
                    change=saved.reason.value,
                )
                # 결과 발행(14단계)이 아직 없다. 저장만으로 삭제하지 않는다.
                return outcome(f"failure_saved_{check.error_code}")
            if self._submit is None:
                # submit 이 연결되지 않은 조립(테스트 등). 제출·삭제하지 않는다. 실제 Worker 는 PrunaSubmitter 를 쓴다.
                return outcome("prechecked_awaiting_submit")
            if not heartbeat.beat():
                return outcome("heartbeat_failed_before_submit")
            return outcome("submitted", delete=self._submit(job, token, request, check, heartbeat))
        finally:
            released = self._state.release_lease(job_id, token)
            if not released.applied and released.reason is not ChangeReason.LEASE_INVALID:
                logger.warning(
                    "request worker lease release failed fitting_job_id=%s reason=%s",
                    job_id,
                    released.reason.value,
                )


# --- 반복 수신 Worker ---


@dataclass
class WorkerStatus:
    running: bool = False
    stopped: bool = False
    crashed: bool = False
    last_error: str | None = None
    consecutive_errors: int = 0
    received: int = 0
    deleted: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)


class RequestWorker:
    """FastAPI 이벤트 루프에서 도는 반복 수신 Task. 동기 단계는 전용 thread 하나에서 한다.

    예기치 않은 종료는 오류 로그와 status(crashed·last_error)로만 드러난다(자동 재시작·알림 전송 없음).
    수신·처리·삭제 함수만 받으므로 후처리 Worker(ai-fitting-response, 10단계)도 같은 루프를 쓴다(name 으로 구분).
    """

    def __init__(
        self,
        receive: Callable[[], RequestBatch],
        process: Callable[[ValidRequestMessage | InvalidRequestMessage], RequestOutcome],
        delete: Callable[[str], SqsResult],
        *,
        idle_seconds: float = 1.0,
        backoff_initial_seconds: float = 1.0,
        backoff_max_seconds: float = 60.0,
        name: str = "request",
    ):
        self._receive, self._process, self._delete = receive, process, delete
        self.name = name
        self._idle = idle_seconds
        self._backoff_initial, self._backoff_max = backoff_initial_seconds, backoff_max_seconds
        self._limiter = CapacityLimiter(1)
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.status = WorkerStatus()

    @property
    def queue_name(self) -> str:
        return "be-fitting-request" if self.name == "request" else "ai-fitting-response"

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def start(self) -> asyncio.Task:
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(), name=f"virtual-fitting-{self.name}-worker"
            )
            self._task.add_done_callback(self._finished)
        return self._task

    async def stop(self, timeout: float) -> None:
        """새 수신을 멈추고 진행 중인 단계를 정리한다. 전체 종료 시간의 상한이 아니다.

        1. stop 표시 → 진행 중인 수신·메시지 처리가 끝나기를 최대 timeout 초 기다린다.
        2. timeout 이 지나면 Task 를 취소한다.
           - long polling 수신 중이면 그 대기를 버리고 바로 끝난다(abandon_on_cancel). 버린 thread 는 남은 수신
             시간 동안 SQS 를 계속 기다릴 수 있고, 그때 받은 메시지는 처리·삭제하지 않아 Visibility Timeout 뒤 다시 보인다.
           - 메시지 처리·삭제 단계(DB·S3·Runware 를 쓰는 thread)는 버리지 않는다. 그 단계가 끝날 때까지 추가로 기다리며,
             이 시간은 timeout 에 포함되지 않고 제한도 없다(각 DB·HTTP 호출 자체의 timeout 만 적용된다).
        그래서 DB pool 은 처리 중인 단계가 끝난 뒤에 닫힌다.
        """
        self._stop.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout)
        except TimeoutError:
            logger.warning(
                "%s worker did not stop within %ss; cancelling receive wait", self.name, timeout
            )
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _finished(self, task: asyncio.Task) -> None:
        self.status.running = False
        if task.cancelled():
            self.status.stopped = True
            metrics.worker_stopped(self.name, self.queue_name, crashed=False)
            return
        error = task.exception()
        if error is not None:  # 예상하지 못한 종료를 숨기지 않는다.
            self.status.crashed = True
            self.status.last_error = type(error).__name__
            metrics.worker_stopped(self.name, self.queue_name, crashed=True)
            logger.error("%s worker crashed", self.name, exc_info=error)
            diag.event(
                "component_crashed",
                logging.ERROR,
                component=f"{self.name}_worker",
                error_type=type(error).__name__,
            )
        else:
            self.status.stopped = True
            metrics.worker_stopped(self.name, self.queue_name, crashed=False)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    def _backoff(self) -> float:
        return min(
            self._backoff_max,
            self._backoff_initial * 2 ** max(0, self.status.consecutive_errors - 1),
        )

    async def _run(self) -> None:
        self.status.running = True
        metrics.worker_started(self.name, self.queue_name)
        logger.info("%s worker started", self.name)
        diag.event("component_started", component=f"{self.name}_worker")
        while not self._stop.is_set():
            try:
                # long polling 은 thread 에서 기다린다. 종료 때 이 대기는 버릴 수 있다(받은 메시지는 지우지 않으므로
                # Visibility Timeout 뒤 다시 보인다).
                batch = await to_thread.run_sync(
                    self._receive, limiter=self._limiter, abandon_on_cancel=True
                )
            except Exception as error:
                # 설정 오류 등. 빠르게 반복하지 않고 물러난다.
                self.status.consecutive_errors += 1
                self.status.last_error = type(error).__name__
                metrics.worker_receive_failed(
                    self.name, self.queue_name, self.status.consecutive_errors
                )
                logger.exception(
                    "%s worker receive crashed; backing off %.1fs", self.name, self._backoff()
                )
                await self._sleep(self._backoff())
                continue
            if not batch.received:
                self.status.consecutive_errors += 1
                self.status.last_error = batch.receive_result.reason
                metrics.worker_receive_failed(
                    self.name, self.queue_name, self.status.consecutive_errors
                )
                logger.warning(
                    "%s worker receive failed reason=%s; backing off %.1fs",
                    self.name,
                    batch.receive_result.reason,
                    self._backoff(),
                )
                await self._sleep(self._backoff())
                continue
            self.status.consecutive_errors = 0
            metrics.worker_receive_succeeded(self.name, self.queue_name)
            if not batch.messages:
                await self._sleep(self._idle)
                continue
            for item in batch.messages:
                if self._stop.is_set():
                    break  # 남은 메시지는 처리·삭제하지 않는다(다시 보인다).
                self.status.received += 1
                metrics.message_received(self.name, self.queue_name)
                # 진단: 이 프로세스(run_id·hostname·pid)의 이 Worker 가 이 메시지를 처리 대상으로 골랐다.
                # 이후 처리 결과와 실제 삭제 결과를 같은 worker·message_id 로 연결한다.
                selected = time.monotonic()
                count = getattr(item, "approximate_receive_count", None)
                diag.event(
                    "message_selected",
                    component=f"{self.name}_worker",
                    worker=self.name,
                    queue=(
                        "be-fitting-request" if self.name == "request" else "ai-fitting-response"
                    ),
                    **diag.process_fields(),
                    message_id=getattr(item, "message_id", None),
                    receive_count=count if count is not None else "unknown",
                    **_verified_ids(item),
                )
                metrics.message_processing_started(self.name, self.queue_name)
                processing_started = time.monotonic()
                try:
                    result = await to_thread.run_sync(self._process, item, limiter=self._limiter)
                except BaseException:
                    metrics.message_processing_finished(
                        self.name,
                        self.queue_name,
                        "exception",
                        time.monotonic() - processing_started,
                    )
                    raise
                metrics.message_processing_finished(
                    self.name,
                    self.queue_name,
                    result.action,
                    time.monotonic() - processing_started,
                )
                self.status.outcomes[result.action] = self.status.outcomes.get(result.action, 0) + 1
                identity = _verified_ids(item)
                diag.event(
                    "message_processing_result",
                    component=f"{self.name}_worker",
                    worker=self.name,
                    queue=(
                        "be-fitting-request" if self.name == "request" else "ai-fitting-response"
                    ),
                    message_id=getattr(item, "message_id", None),
                    **identity,
                    action=getattr(result, "action", None),
                    delete_decided=result.delete,
                    processing_elapsed_ms=diag.elapsed_ms(selected),
                )
                if result.delete:
                    delete_started = time.monotonic()
                    deleted = await to_thread.run_sync(
                        self._delete, item.receipt_handle, limiter=self._limiter
                    )
                    if deleted.succeeded:
                        metrics.message_delete_finished(self.name, self.queue_name, "succeeded")
                        self.status.deleted += 1
                        # SQS DeleteMessage 응답을 받았다는 뜻이다(중복 재전달이 없다는 보장은 아니다).
                        diag.event(
                            "message_delete_result",
                            component=f"{self.name}_worker",
                            worker=self.name,
                            queue=(
                                "be-fitting-request"
                                if self.name == "request"
                                else "ai-fitting-response"
                            ),
                            message_id=getattr(item, "message_id", None),
                            **identity,
                            action=getattr(result, "action", None),
                            sqs_result="succeeded",
                            delete_elapsed_ms=diag.elapsed_ms(delete_started),
                        )
                    else:
                        metrics.message_delete_finished(self.name, self.queue_name, "failed")
                        diag.event(
                            "message_delete_result",
                            logging.WARNING,
                            component=f"{self.name}_worker",
                            worker=self.name,
                            queue=(
                                "be-fitting-request"
                                if self.name == "request"
                                else "ai-fitting-response"
                            ),
                            message_id=getattr(item, "message_id", None),
                            **identity,
                            action=getattr(result, "action", None),
                            sqs_result=deleted.outcome.value,
                            reason=deleted.reason,
                            delete_elapsed_ms=diag.elapsed_ms(delete_started),
                        )
        logger.info("%s worker stopped", self.name)
        diag.event("component_stopped", component=f"{self.name}_worker")


def _verified_ids(item: Any) -> dict[str, Any]:
    """진단용 작업 식별자. 수신 단계에서 이미 검증된 값만 꺼낸다(새로 파싱하지 않음).

    요청(ValidRequestMessage.request)·후처리(ValidFollowUp.message)의 fitting_job_id 는 양의 정수, task_uuid 는
    UUID 로 검증돼 있다. 형식 오류 메시지의 "읽을 수 있었던" 값은 신뢰하지 않아 남기지 않는다.
    """
    try:
        if getattr(item, "error", None) is not None:
            return {"message_valid": False}
        parsed = getattr(item, "request", None) or getattr(item, "message", None)
        if parsed is None:
            return {}
        ids: dict[str, Any] = {"message_valid": True}
        job_id = getattr(parsed, "fitting_job_id", None)
        if isinstance(job_id, int) and not isinstance(job_id, bool):
            ids["fitting_job_id"] = job_id
        task = getattr(parsed, "task_uuid", None)
        if isinstance(task, str):
            ids["task_uuid"] = task
        return ids
    except Exception:  # noqa: BLE001 — 진단 값만 포기
        return {}


def create_request_worker() -> RequestWorker:
    """실제 의존성으로 조립한다(blocking: DB pool·boto3 client). lifespan 이 thread 에서 부른다."""
    from datetime import timedelta

    from app.clients.s3 import S3ImageStorage
    from app.config import settings
    from app.config.database import get_connection_pool
    from app.virtual_fitting.async_request import receive_fitting_requests
    from app.virtual_fitting.budget import FittingBudget
    from app.virtual_fitting.providers.runware_async import (
        DEFAULT_SUBMIT_TIMEOUT,
        RunwarePrunaAsyncProvider,
    )
    from app.virtual_fitting.providers.runware_balance import (
        MinimumBalanceCheck,
        RunwareBalanceClient,
    )
    from app.virtual_fitting.providers.runware_usage import RunwareUsageClient
    from app.virtual_fitting.pruna_submitter import PrunaSubmitter
    from app.virtual_fitting.repositories.product_repository import ProductRepository
    from app.virtual_fitting.repositories.pruna_limiter import PrunaLimiter
    from app.virtual_fitting.repositories.usage_repository import UsageRepository
    from app.virtual_fitting.runware_webhook import webhook_url_matches_token

    # Runware 에 넘길 Webhook URL 의 token 이 이 서버의 검증값과 다르면 모든 완료 알림이 401 이 된다. 시작 전에 멈춘다.
    if not webhook_url_matches_token(
        settings.VIRTUAL_FITTING_RUNWARE_WEBHOOK_URL, settings.WEBHOOK_TOKEN
    ):
        raise ValueError(
            "VIRTUAL_FITTING_RUNWARE_WEBHOOK_URL 이 https://<host>/api/v2/webhooks/runware?token=<WEBHOOK_TOKEN> "
            "형식이 아니거나 token 이 WEBHOOK_TOKEN 과 다릅니다."
        )
    from app.virtual_fitting.watchdog import follow_up_paths

    # 실제로 도는 경로만 준비됐다고 본다: 결과 발행(14단계 결과 전송 루프) + 접수 대기·접수 불명확 확인(15단계 Watchdog).
    follow_up = follow_up_paths()
    pool = get_connection_pool()
    sqs = SqsQueueClient()
    storage = S3ImageStorage()
    budget = FittingBudget(
        UsageRepository(pool),
        RunwareUsageClient(),
        daily_limit=settings.FITTING_DAILY_BUDGET_USD,
        reconcile_interval=timedelta(seconds=settings.FITTING_USAGE_RECONCILE_SECONDS),
    )
    balance = MinimumBalanceCheck(RunwareBalanceClient(), settings.FITTING_MIN_BALANCE_USD)
    precheck = RequestPrecheck(ProductRepository(pool), storage, budget, balance)
    jobs, state = AsyncJobRepository(pool), AsyncJobStateRepository(pool)
    worker: RequestWorker
    # 기본 NoFollowUpPaths 를 쓴다. Webhook(9단계)은 구현됐지만 Webhook 이 오지 않을 때의 복구(Watchdog 15단계)와
    # 후처리(10~11단계)·결과 발행(14단계)이 없어, 접수 뒤 요청 메시지를 지워도 되는 경로는 아직 없다.
    submitter = PrunaSubmitter(
        jobs=jobs,
        state=state,
        limiter=PrunaLimiter(
            pool,
            max_in_flight=settings.VIRTUAL_FITTING_PRUNA_MAX_IN_FLIGHT,
            max_concurrent_sends=settings.VIRTUAL_FITTING_PRUNA_MAX_CONCURRENT_SENDS,
            min_send_interval_seconds=settings.VIRTUAL_FITTING_PRUNA_MIN_SEND_INTERVAL_SECONDS,
            send_hold_seconds=DEFAULT_SUBMIT_TIMEOUT + 30,
        ),
        provider=RunwarePrunaAsyncProvider(),
        storage=storage,
        budget=budget,
        balance=balance,
        webhook_url=settings.VIRTUAL_FITTING_RUNWARE_WEBHOOK_URL,
        follow_up=follow_up,
        follow_up_queue=sqs,
        should_stop=lambda: worker.stopping,
        keepalive_seconds=settings.VIRTUAL_FITTING_REQUEST_LEASE_SECONDS / 3,
    )

    processor = RequestProcessor(
        jobs,
        state,
        precheck,
        sqs,
        lease_seconds=settings.VIRTUAL_FITTING_REQUEST_LEASE_SECONDS,
        visibility_seconds=settings.VIRTUAL_FITTING_REQUEST_VISIBILITY_SECONDS,
        should_stop=lambda: worker.stopping,
        submit=submitter,
        follow_up=follow_up,
        follow_up_queue=sqs,
    )
    worker = RequestWorker(
        lambda: receive_fitting_requests(
            sqs,
            max_messages=1,
            visibility_timeout=settings.VIRTUAL_FITTING_REQUEST_VISIBILITY_SECONDS,
        ),
        processor.process,
        lambda receipt: sqs.delete(FittingQueue.BE_FITTING_REQUEST, receipt),
    )
    return worker
