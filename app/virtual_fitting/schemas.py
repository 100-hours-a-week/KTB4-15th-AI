from pydantic import BaseModel, Field


class FittingProductRequest(BaseModel):
    # DB의 product_code는 BIGINT 이므로 숫자 문자열만 허용한다.
    product_code: str = Field(pattern=r"^[0-9]{1,18}$")


class SyncFittingRequest(BaseModel):
    user_image_url: str = Field(min_length=1, pattern=r"^https?://")
    products: list[FittingProductRequest] = Field(min_length=1, max_length=2)


class SyncFittingData(BaseModel):
    result_image_url: str
    llm_title: str
    llm_comment: str


class SyncFittingResponse(BaseModel):
    code: int = 200
    message: str = "fitting_succeeded"
    data: SyncFittingData
