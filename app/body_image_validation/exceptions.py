"""전신 이미지 검증의 사용자 실패와 시스템 실패를 분리한다.

code / message 가 그대로 API 응답의 code / message 가 된다. 시스템 실패는 공통 문구를 쓴다.
"""

from fastapi import status

from app.config import body_image_validation as config

SYSTEM_FAILURE_MESSAGE = "전신 사진 처리 중 서버 오류가 발생했습니다."
SERVER_BUSY_MESSAGE = "현재 이미지 처리 요청이 많습니다. 잠시 후 다시 시도해주세요."


class BodyImageValidationError(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        self.message = message
        super().__init__(message or code)


class UserImageValidationError(BodyImageValidationError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class InvalidImageError(UserImageValidationError):
    status_code = status.HTTP_400_BAD_REQUEST


class ImageTooLargeError(UserImageValidationError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE


class BodyImageSystemError(BodyImageValidationError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class ServerBusyError(BodyImageValidationError):
    """실행 슬롯과 대기 슬롯이 모두 찼을 때. 기다리게 하지 않고 바로 거절한다."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS

    def __init__(self) -> None:
        super().__init__("SERVER_BUSY", SERVER_BUSY_MESSAGE)


USER_MESSAGES = {
    "IMAGE_EMPTY": "이미지 파일을 첨부해주세요.",
    "IMAGE_FORMAT_UNSUPPORTED": "JPG, JPEG 또는 PNG 이미지만 업로드해주세요.",
    "IMAGE_TOO_LARGE": f"이미지 크기는 {config.MAX_IMAGE_SIZE_MB}MB 이하여야 합니다.",
    "INVALID_IMAGE": "이미지 파일을 확인할 수 없습니다. 다른 사진을 업로드해주세요.",
    "IMAGE_DECODE_FAILED": "이미지가 손상되어 처리할 수 없습니다. 다른 사진을 업로드해주세요.",
    "IMAGE_RESOLUTION_TOO_LARGE": (
        "이미지 해상도가 너무 높습니다. 더 작은 해상도의 사진을 업로드해주세요."
    ),
    "IMAGE_RESOLUTION_TOO_SMALL": "짧은 변이 480px 이상인 이미지를 업로드해주세요.",
    "PERSON_NOT_FOUND": "사진에서 사람을 찾을 수 없습니다.",
    "MULTIPLE_PERSONS": "한 명만 나온 사진을 업로드해주세요.",
    "PERSON_TOO_SMALL": "전신이 더 크게 보이도록 촬영해주세요.",
    "FULL_BODY_NOT_VISIBLE": "머리부터 발끝까지 모두 나오도록 전신을 촬영해주세요.",
    "ARMS_NOT_VISIBLE": "양팔이 모두 보이도록 촬영해주세요.",
    "LEGS_NOT_VISIBLE": "양다리가 모두 보이도록 촬영해주세요.",
    "NOT_FRONTAL": "카메라를 정면으로 바라보고 촬영해주세요.",
    "IMAGE_TOO_DARK": "사진이 너무 어둡습니다. 밝은 곳에서 다시 촬영해주세요.",
}


def user_error(code: str) -> UserImageValidationError:
    error_type = {
        "IMAGE_EMPTY": InvalidImageError,
        "IMAGE_FORMAT_UNSUPPORTED": InvalidImageError,
        "INVALID_IMAGE": InvalidImageError,
        "IMAGE_DECODE_FAILED": InvalidImageError,
        "IMAGE_TOO_LARGE": ImageTooLargeError,
    }.get(code, UserImageValidationError)
    return error_type(code, USER_MESSAGES[code])
