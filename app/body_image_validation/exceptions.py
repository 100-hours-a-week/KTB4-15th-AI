"""전신 이미지 검증의 사용자 실패와 시스템 실패를 분리한다."""

from app.config import body_image_validation as config

SERVER_BUSY_REASON = "현재 이미지 처리 요청이 많습니다. 잠시 후 다시 시도해주세요."


class BodyImageValidationError(Exception):
    status_code: int = 500
    message: str = "body_image_validation_system_failed"
    reason_code: str

    def __init__(self, reason_code: str, reason: str | None = None) -> None:
        self.reason_code = reason_code
        self.reason = reason
        super().__init__(reason or reason_code)


class UserImageValidationError(BodyImageValidationError):
    status_code = 422
    message = "body_image_validation_failed"


class InvalidImageError(UserImageValidationError):
    status_code = 400


class ImageTooLargeError(UserImageValidationError):
    status_code = 413


class BodyImageSystemError(BodyImageValidationError):
    status_code = 500


class ServerBusyError(BodyImageValidationError):
    """실행 슬롯과 대기 슬롯이 모두 찼을 때. 기다리게 하지 않고 바로 거절한다."""

    status_code = 429
    message = "server_busy"

    def __init__(self) -> None:
        super().__init__("SERVER_BUSY", SERVER_BUSY_REASON)


USER_REASONS = {
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


def user_error(reason_code: str) -> UserImageValidationError:
    error_type = {
        "IMAGE_EMPTY": InvalidImageError,
        "IMAGE_FORMAT_UNSUPPORTED": InvalidImageError,
        "INVALID_IMAGE": InvalidImageError,
        "IMAGE_DECODE_FAILED": InvalidImageError,
        "IMAGE_TOO_LARGE": ImageTooLargeError,
    }.get(reason_code, UserImageValidationError)
    return error_type(reason_code, USER_REASONS[reason_code])
