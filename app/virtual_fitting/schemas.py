from pydantic import BaseModel, Field

from app.errors import ApiResponse


class FittingProductRequest(BaseModel):
    # DB의 product_code는 BIGINT 이므로 숫자 문자열만 허용한다.
    product_code: str = Field(pattern=r"^[0-9]{1,18}$")


class SyncFittingRequest(BaseModel):
    user_image_url: str = Field(min_length=1, pattern=r"^https?://")
    products: list[FittingProductRequest] = Field(min_length=1, max_length=2)


class SyncFittingData(BaseModel):
    result_image_key: str
    llm_title: str
    llm_comment: str


class SyncFittingResponse(ApiResponse):
    code: str = "FITTING_SUCCESS"
    data: SyncFittingData
    message: str = "가상 피팅이 완료되었습니다."
