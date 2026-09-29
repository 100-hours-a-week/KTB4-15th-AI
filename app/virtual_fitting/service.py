import logging

from app.clients.s3 import ImageStorage, ImageStorageError
from app.virtual_fitting.budget import FittingBudget
from app.virtual_fitting.exceptions import FittingImageStorageError, FittingPostprocessError
from app.virtual_fitting.models import VirtualFittingResult
from app.virtual_fitting.product_selection import select_fitting_products
from app.virtual_fitting.prompt import build_fitting_prompt
from app.virtual_fitting.providers.base import FittingInput, FittingProvider
from app.virtual_fitting.providers.comment import CommentProvider
from app.virtual_fitting.repositories.product_repository import ProductRepository
from app.virtual_fitting.schemas import SyncFittingRequest

logger = logging.getLogger(__name__)

# LLM 후처리가 실패해도 가상피팅 결과는 돌려준다. 이미지 내용을 모르므로 일반 문구만 쓴다.
FALLBACK_COMMENT = "선택하신 상품으로 완성한 가상 피팅 결과입니다."
FALLBACK_TITLE = "가상 피팅 결과"


class VirtualFittingService:
    """v1 동기 가상피팅. VirtualFittingError 는 변환하지 않고 그대로 전파한다."""

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
        # 상의 → 하의 순서. garment_image_urls 와 prompt 의 N번째 garment image 가 같은 순서를 쓴다.
        products = select_fitting_products(
            [product.product_code for product in request.products], self.repository
        )

        self.budget.ensure_available()
        fitting_result = self.fitting_provider.try_on(
            FittingInput(
                person_image_url=request.user_image_url,
                garment_image_urls=[product.image_url for product in products],
                prompt=build_fitting_prompt(products),
            )
        )
        # 뒤의 LLM 후처리·S3 저장보다 먼저 더한다. 이 줄에 닿은 순간 Runware 비용은 이미 나갔다.
        if fitting_result.cost is not None:
            self.budget.record(fitting_result.cost)

        # comment 는 S3 key 가 아니라 VTON 이 돌려준 원본 결과 이미지 URL 로 만든다.
        # comment/title 은 한 세트다. 하나라도 실패하면 둘 다 fallback 으로 바꾸고 계속 진행한다.
        try:
            llm_comment = self.comment_provider.generate_comment(
                fitting_result.result_image_url,
                [product.description_summary for product in products],
            )
            llm_title = self.comment_provider.generate_title(llm_comment)
        except FittingPostprocessError:
            logger.exception(
                "virtual fitting LLM postprocess failed; using fallback comment/title"
            )
            llm_comment = FALLBACK_COMMENT
            llm_title = FALLBACK_TITLE

        # LLM 이 실패해도 S3 저장까지 가서 정상 응답한다. S3 실패는 숨기지 않는다.
        try:
            result_image_key = self.image_storage.store_remote_image(
                fitting_result.result_image_url,
                "virtual-fitting/results",
            )
        except ImageStorageError as error:
            raise FittingImageStorageError("가상피팅 결과 S3 저장에 실패했습니다.") from error

        return VirtualFittingResult(
            result_image_key=result_image_key,
            llm_comment=llm_comment,
            llm_title=llm_title,
        )
