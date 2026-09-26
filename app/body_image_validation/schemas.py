"""전신 이미지 검증 API 응답 schema."""

from pydantic import BaseModel


class BodyImageValidationData(BaseModel):
    s3_key: str


class BodyImageValidationResponse(BaseModel):
    code: int = 200
    message: str = "body_image_validation_success"
    data: BodyImageValidationData
