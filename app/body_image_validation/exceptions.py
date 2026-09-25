"""전신 이미지 검증의 사용자 실패와 시스템 실패를 분리한다."""


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


USER_REASONS = {
    "IMAGE_EMPTY": "이미지 파일을 첨부해주세요.",
    "IMAGE_FORMAT_UNSUPPORTED": "JPG, JPEG 또는 PNG 이미지만 업로드해주세요.",
    "IMAGE_TOO_LARGE": "이미지 크기는 10MB 이하여야 합니다.",
    "IMAGE_DECODE_FAILED": "이미지 파일을 읽을 수 없습니다. 다른 이미지를 선택해주세요.",
    "IMAGE_RESOLUTION_TOO_SMALL": "짧은 변이 480px 이상인 이미지를 업로드해주세요.",
    "PERSON_NOT_FOUND": "사진에서 사람을 찾을 수 없습니다.",
    "MULTIPLE_PERSONS": "한 명만 나온 사진을 업로드해주세요.",
    "PERSON_TOO_SMALL": "전신이 더 크게 보이도록 촬영해주세요.",
    "FULL_BODY_NOT_VISIBLE": "머리부터 발끝까지 모두 나오도록 전신을 촬영해주세요.",
    "ARMS_NOT_VISIBLE": "양팔이 모두 보이도록 촬영해주세요.",
    "LEGS_NOT_VISIBLE": "양다리가 모두 보이도록 촬영해주세요.",
    "NOT_FRONTAL": "카메라를 정면으로 바라보고 촬영해주세요.",
}


def user_error(reason_code: str) -> UserImageValidationError:
    error_type = {
        "IMAGE_EMPTY": InvalidImageError,
        "IMAGE_FORMAT_UNSUPPORTED": InvalidImageError,
        "IMAGE_DECODE_FAILED": InvalidImageError,
        "IMAGE_TOO_LARGE": ImageTooLargeError,
    }.get(reason_code, UserImageValidationError)
    return error_type(reason_code, USER_REASONS[reason_code])
