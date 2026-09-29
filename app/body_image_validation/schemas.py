"""전신 이미지 검증 API 응답 schema."""

from pydantic import BaseModel

from app.errors import ApiResponse


class BodyImageValidationData(BaseModel):
    s3_key: str


class BodyImageValidationResponse(ApiResponse):
    code: str = "BODY_IMAGE_UPLOAD_SUCCESS"
    data: BodyImageValidationData
    message: str = "전신 사진 검증에 성공했습니다."
