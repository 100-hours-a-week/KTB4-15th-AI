"""가상피팅 도메인 예외. Router 에서 이 값으로 HTTP Response 의 code / message 를 구성한다."""

from collections.abc import Sequence

from fastapi import status


class VirtualFittingError(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "VIRTUAL_FITTING_ERROR"
    message: str = "가상 피팅 처리 중 오류가 발생했습니다."


class ProductNotFoundError(VirtualFittingError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "PRODUCT_NOT_FOUND"
    message = "요청한 상품을 찾을 수 없습니다."

    def __init__(self, product_codes: Sequence[str]):
        self.product_codes = list(product_codes)
        super().__init__(
            f"AI PostgreSQL에서 상품을 찾을 수 없습니다: {self.product_codes}"
        )


class InvalidFittingCombinationError(VirtualFittingError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "INVALID_FITTING_COMBINATION"
    message = "가상 피팅할 수 없는 상품 조합입니다."


class ProductImageMissingError(VirtualFittingError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "PRODUCT_IMAGE_MISSING"
    message = "상품 이미지가 없어 가상 피팅할 수 없습니다."

    def __init__(self, product_code: str):
        self.product_code = product_code
        super().__init__(f"상품 image_url이 없습니다: {product_code}")


class FittingModelError(VirtualFittingError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "FITTING_MODEL_FAILED"
    message = "가상 피팅 이미지 생성에 실패했습니다."


class FittingTimeoutError(VirtualFittingError):
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    code = "FITTING_TIMEOUT"
    message = "가상 피팅 시간이 초과되었습니다."


class UnsupportedSubCategoryError(VirtualFittingError):
    """DB 의 sub_category 가 영어 garment 명칭 매핑에 없을 때. 요청이 아니라 데이터/매핑 문제다."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "UNSUPPORTED_SUB_CATEGORY"
    message = "가상 피팅을 지원하지 않는 상품 분류입니다."

    def __init__(self, sub_category: str):
        self.sub_category = sub_category
        super().__init__(f"영어 garment 명칭 매핑이 없는 sub_category 입니다: {sub_category!r}")


class FittingPostprocessError(VirtualFittingError):
    """가상피팅 이미지는 만들어졌지만 llm_comment / llm_title 생성에 실패했을 때 (명세 V1 표)."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "FITTING_POSTPROCESS_FAILED"
    message = "가상 피팅 결과 후처리에 실패했습니다."


class FittingDatabaseError(VirtualFittingError):
    """AI PostgreSQL 연결 또는 상품 조회 자체가 실패했을 때.

    조회는 성공했는데 상품이 없는 경우가 아니다. psycopg 의 원인 예외는 `from` 으로만
    연결한다. 응답과 이 메시지에는 접속 정보나 DB 상세를 싣지 않는다.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "DATABASE_ERROR"
    message = "상품 정보를 조회하지 못했습니다."


class FittingImageStorageError(VirtualFittingError):
    """가상피팅 결과 이미지를 영구 S3 저장소로 옮기지 못했을 때."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "FITTING_IMAGE_STORAGE_FAILED"
    message = "가상 피팅 결과 이미지를 저장하지 못했습니다."
