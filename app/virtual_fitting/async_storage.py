from app.clients.s3 import S3ImageStorage, S3ObjectCheck
from app.config import settings

ASYNC_RESULT_FILENAME = "pruna-async.webp"


def async_result_key(fitting_job_id: int) -> str:
    """Pruna 비동기 결과 이미지 Key. 같은 작업이면 항상 같은 Key 다."""
    if (
        isinstance(fitting_job_id, bool)
        or not isinstance(fitting_job_id, int)
        or fitting_job_id < 1
    ):
        raise ValueError("fitting_job_id 는 1 이상의 정수여야 합니다.")
    return f"virtual-fitting/results/{fitting_job_id}/{ASYNC_RESULT_FILENAME}"


def async_output_format(result_key: str) -> str:
    """이미 준비한 Key 가 출력 계약이다. 구 JPG 작업은 재개해도 JPG 를 유지한다."""
    from app.clients.s3 import validate_s3_key

    key = validate_s3_key(result_key)
    if key.endswith(".webp"):
        return "WEBP"
    if key.endswith((".jpg", ".jpeg")):
        return "JPG"
    raise ValueError("비동기 결과 Key 의 출력 형식이 유효하지 않습니다.")


def user_image_get_url(
    storage: S3ImageStorage, user_image_key: str, *, expires_in: int | None = None
) -> str:
    """Pruna 호출 직전(그리고 V1 fallback 직전)에 사용자 원본 이미지를 읽을 URL 을 새로 발급한다."""
    if expires_in is None:
        expires_in = settings.VIRTUAL_FITTING_USER_IMAGE_URL_EXPIRES_SECONDS
    return storage.presigned_get_url(user_image_key, expires_in=expires_in)


def result_upload_url(
    storage: S3ImageStorage,
    result_key: str,
    *,
    expires_in: int | None = None,
    content_type: str | None = None,
) -> str:
    """Runware uploadEndpoint 로 넘길 Presigned PUT URL.

    content_type 은 기본으로 서명에 넣지 않는다. Runware 의 실제 PUT Content-Type 을 확인한 뒤에만 넣는다.
    """
    if expires_in is None:
        expires_in = settings.VIRTUAL_FITTING_RESULT_UPLOAD_URL_EXPIRES_SECONDS
    return storage.presigned_put_url(result_key, expires_in=expires_in, content_type=content_type)


def check_result_image(storage: S3ImageStorage, result_key: str) -> S3ObjectCheck:
    """결과 이미지가 올라왔는지 HEAD 한 번으로 확인한다. 재시도 일정은 호출하는 쪽이 정한다."""
    return storage.head_object(result_key)


def result_image_get_url(
    storage: S3ImageStorage, result_key: str, *, expires_in: int | None = None
) -> str:
    """OpenAI 후처리에 넘길 결과 이미지 URL. 사용자 이미지와 같은 Presigned GET 을 쓴다."""
    if expires_in is None:
        expires_in = settings.VIRTUAL_FITTING_RESULT_IMAGE_URL_EXPIRES_SECONDS
    return storage.presigned_get_url(result_key, expires_in=expires_in)
