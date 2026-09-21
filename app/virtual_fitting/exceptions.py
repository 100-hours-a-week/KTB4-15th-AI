"""가상피팅 도메인 예외. Router 에서 이 값으로 HTTP Response 의 code / message 를 구성한다."""

from collections.abc import Sequence


class VirtualFittingError(Exception):
    status_code: int = 500
    message: str = "virtual_fitting_error"


class ProductNotFoundError(VirtualFittingError):
    status_code = 404
    message = "product_not_found"

    def __init__(self, product_codes: Sequence[str]):
        self.product_codes = list(product_codes)
        super().__init__(
            f"AI PostgreSQL에서 상품을 찾을 수 없습니다: {self.product_codes}"
        )


class InvalidFittingCombinationError(VirtualFittingError):
    status_code = 422
    message = "invalid_fitting_combination"


class ProductImageMissingError(VirtualFittingError):
    status_code = 422
    message = "product_image_missing"

    def __init__(self, product_code: str):
        self.product_code = product_code
        super().__init__(f"상품 image_url이 없습니다: {product_code}")


class FittingModelError(VirtualFittingError):
    status_code = 502
    message = "fitting_model_failed"


class FittingTimeoutError(VirtualFittingError):
    status_code = 504
    message = "fitting_timeout"


class UnsupportedSubCategoryError(VirtualFittingError):
    """DB 의 sub_category 가 영어 garment 명칭 매핑에 없을 때. 요청이 아니라 데이터/매핑 문제다."""

    status_code = 500
    message = "unsupported_sub_category"

    def __init__(self, sub_category: str):
        self.sub_category = sub_category
        super().__init__(f"영어 garment 명칭 매핑이 없는 sub_category 입니다: {sub_category!r}")
