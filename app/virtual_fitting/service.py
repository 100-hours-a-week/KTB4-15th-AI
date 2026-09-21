from app.virtual_fitting.models import VirtualFittingResult
from app.virtual_fitting.product_selection import select_fitting_products
from app.virtual_fitting.prompt import build_fitting_prompt
from app.virtual_fitting.providers.base import FittingInput, FittingProvider
from app.virtual_fitting.providers.comment import CommentProvider
from app.virtual_fitting.repositories.product_repository import ProductRepository
from app.virtual_fitting.schemas import SyncFittingRequest


class VirtualFittingService:
    """v1 동기 가상피팅. VirtualFittingError 는 변환하지 않고 그대로 전파한다."""

    def __init__(
        self,
        repository: ProductRepository,
        fitting_provider: FittingProvider,
        comment_provider: CommentProvider,
    ):
        self.repository = repository
        self.fitting_provider = fitting_provider
        self.comment_provider = comment_provider

    def fit(self, request: SyncFittingRequest) -> VirtualFittingResult:
        # 상의 → 하의 순서. garment_image_urls 와 prompt 의 N번째 garment image 가 같은 순서를 쓴다.
        products = select_fitting_products(
            [product.product_code for product in request.products], self.repository
        )

        fitting_result = self.fitting_provider.try_on(
            FittingInput(
                person_image_url=request.user_image_url,
                garment_image_urls=[product.image_url for product in products],
                prompt=build_fitting_prompt(products),
            )
        )

        llm_comment = self.comment_provider.generate_comment(products)
        llm_title = self.comment_provider.generate_title(llm_comment)

        return VirtualFittingResult(
            result_image_url=fitting_result.result_image_url,
            llm_comment=llm_comment,
            llm_title=llm_title,
        )
