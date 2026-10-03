import logging
from dataclasses import dataclass

from anyio import to_thread

from app.clients.s3 import ImageStorage, ImageStorageError
from app.virtual_fitting.budget import FittingBudget
from app.virtual_fitting.concurrency import FittingLimiters
from app.virtual_fitting.exceptions import FittingImageStorageError, FittingPostprocessError
from app.virtual_fitting.models import FittingProduct, VirtualFittingResult
from app.virtual_fitting.product_selection import select_fitting_products
from app.virtual_fitting.prompt import build_fitting_prompt
from app.virtual_fitting.providers.base import FittingInput, FittingProvider, FittingResult
from app.virtual_fitting.providers.comment import CommentProvider
from app.virtual_fitting.repositories.product_repository import ProductRepository
from app.virtual_fitting.schemas import SyncFittingRequest

logger = logging.getLogger(__name__)

# LLM 후처리가 실패해도 가상피팅 결과는 돌려준다. 이미지 내용을 모르므로 일반 문구만 쓴다.
FALLBACK_COMMENT = "선택하신 상품으로 완성한 가상 피팅 결과입니다."
FALLBACK_TITLE = "가상 피팅 결과"


@dataclass(frozen=True)
class PreparedFitting:
    """상품 조회·조합 검증을 마치고 VTON 입력까지 만든 요청. 상의 → 하의 순서다.

    하루 사용액 확인은 여기서 하지 않는다. VTON 직전(fit_with_limiters 에서는 VTON 슬롯 안)에 한다.
    """

    products: list[FittingProduct]
    fitting_input: FittingInput


def _fallback_comment_and_title() -> tuple[str, str]:
    logger.exception("virtual fitting LLM postprocess failed; using fallback comment/title")
    return FALLBACK_COMMENT, FALLBACK_TITLE


class VirtualFittingService:
    """v1 동기 가상피팅. VirtualFittingError 는 변환하지 않고 그대로 전파한다.

    단계 메서드는 모두 동기다. fit 은 한 thread 에서 차례로 부르고, fit_with_limiters 는 같은 단계를
    VTON / LLM 슬롯 안에서 하나씩 worker thread 로 넘긴다. 단계 순서와 fallback 은 둘이 같다.
    """

    def __init__(
        self,
        repository: ProductRepository,
        fitting_provider: FittingProvider,
        comment_provider: CommentProvider,
        image_storage: ImageStorage,
        budget: FittingBudget,
    ):
        self.repository = repository
        self.fitting_provider = fitting_provider
        self.comment_provider = comment_provider
        self.image_storage = image_storage
        self.budget = budget

    def fit(self, request: SyncFittingRequest) -> VirtualFittingResult:
        prepared = self.prepare(request)
        self.ensure_budget_available()
        fitting_result = self.run_vton(prepared)
        self.record_cost(fitting_result)

        # comment/title 은 한 세트다. 하나라도 실패하면 둘 다 fallback 으로 바꾸고 계속 진행한다.
        try:
            llm_comment = self.generate_comment(prepared, fitting_result)
            llm_title = self.generate_title(llm_comment)
        except FittingPostprocessError:
            llm_comment, llm_title = _fallback_comment_and_title()

        return VirtualFittingResult(
            result_image_key=self.store_result(fitting_result),
            llm_comment=llm_comment,
            llm_title=llm_title,
        )

    async def fit_with_limiters(
        self, request: SyncFittingRequest, limiters: FittingLimiters
    ) -> VirtualFittingResult:
        """fit 과 같은 흐름. 슬롯은 실제 외부 요청 하나만 감싸고, 끝나면 바로 반환한다.

        슬롯 대기는 이벤트 루프에서 하므로 기다리는 동안 worker thread 를 잡지 않는다.
        VTON 슬롯을 못 얻으면(429) 사용액 확인도, try_on 도, budget.record 도 부르지 않는다.
        """
        prepared = await to_thread.run_sync(self.prepare, request)
        # 사용액 확인과 try_on 을 같은 슬롯 안에서 한다. 확인을 통과하고 VTON 에 들어갈 수 있는 요청이
        # 실행 슬롯 수로 묶이므로, 동시 요청 때문에 상한을 넘는 폭도 그만큼으로 줄어든다.
        # 상한에 닿으면(402) try_on 없이 슬롯을 나가므로 슬롯은 바로 다음 대기자에게 간다.
        async with limiters.vton.slot():
            await to_thread.run_sync(self.ensure_budget_available)
            fitting_result = await to_thread.run_sync(self.run_vton, prepared)
        await to_thread.run_sync(self.record_cost, fitting_result)

        # comment 와 title 은 각자 LLM 슬롯을 얻는다. 둘 사이에 슬롯을 다른 요청에 넘겨준다.
        # 슬롯 대기 시간 초과는 LLM limiter 가 FittingPostprocessError 로 내므로 fallback 으로 이어진다.
        try:
            async with limiters.llm.slot():
                llm_comment = await to_thread.run_sync(
                    self.generate_comment, prepared, fitting_result
                )
            async with limiters.llm.slot():
                llm_title = await to_thread.run_sync(self.generate_title, llm_comment)
        except FittingPostprocessError:
            llm_comment, llm_title = _fallback_comment_and_title()

        return VirtualFittingResult(
            result_image_key=await to_thread.run_sync(self.store_result, fitting_result),
            llm_comment=llm_comment,
            llm_title=llm_title,
        )

    def prepare(self, request: SyncFittingRequest) -> PreparedFitting:
        # 상의 → 하의 순서. garment_image_urls 와 prompt 의 N번째 garment image 가 같은 순서를 쓴다.
        products = select_fitting_products(
            [product.product_code for product in request.products], self.repository
        )
        return PreparedFitting(
            products=products,
            fitting_input=FittingInput(
                person_image_url=request.user_image_url,
                garment_image_urls=[product.image_url for product in products],
                prompt=build_fitting_prompt(products),
            ),
        )

    def ensure_budget_available(self) -> None:
        """오늘 사용액이 상한에 닿았으면 FittingDailyBudgetExceededError(402)."""
        self.budget.ensure_available()

    def run_vton(self, prepared: PreparedFitting) -> FittingResult:
        return self.fitting_provider.try_on(prepared.fitting_input)

    def record_cost(self, fitting_result: FittingResult) -> None:
        # 뒤의 LLM 후처리·S3 저장보다 먼저 더한다. 여기에 닿은 순간 Runware 비용은 이미 나갔다.
        if fitting_result.cost is not None:
            self.budget.record(fitting_result.cost)

    def generate_comment(self, prepared: PreparedFitting, fitting_result: FittingResult) -> str:
        # comment 는 S3 key 가 아니라 VTON 이 돌려준 원본 결과 이미지 URL 로 만든다.
        return self.comment_provider.generate_comment(
            fitting_result.result_image_url,
            [product.description_summary for product in prepared.products],
        )

    def generate_title(self, llm_comment: str) -> str:
        return self.comment_provider.generate_title(llm_comment)

    def store_result(self, fitting_result: FittingResult) -> str:
        # LLM 이 실패해도 S3 저장까지 가서 정상 응답한다. S3 실패는 숨기지 않는다.
        try:
            return self.image_storage.store_remote_image(
                fitting_result.result_image_url,
                "virtual-fitting/results",
            )
        except ImageStorageError as error:
            raise FittingImageStorageError("가상피팅 결과 S3 저장에 실패했습니다.") from error
