"""V2 비동기 가상피팅이 쓰는 S3 Key 규칙과 Presigned URL 유효기간.

S3 호출 자체는 app.clients.s3.S3ImageStorage 가 한다. 여기서는 용도별 Key 형식과 만료시간만 정한다.

| 용도 | Key | 방식 | 기본 만료 |
| --- | --- | --- | --- |
| Pruna 가 사용자 원본을 읽음 | user_image_key (BE 전달) | Presigned GET | 1800초 |
| Pruna 가 결과를 업로드 | virtual-fitting/results/{fitting_job_id}/pruna-async.jpg | Presigned PUT | 3600초 |
| 결과가 올라왔는지 확인 | 같은 결과 Key | HEAD | — |
| OpenAI 후처리에 결과 전달 | 같은 결과 Key | Presigned GET | 600초 |

발급한 URL 은 그때그때 쓰고 버린다. DB 에는 Key 만 저장한다.
"""

from app.clients.s3 import S3ImageStorage, S3ObjectCheck
from app.config import settings

ASYNC_RESULT_FILENAME = "pruna-async.jpg"


def async_result_key(fitting_job_id: int) -> str:
    """Pruna 비동기 결과 이미지 Key. 같은 작업이면 항상 같은 Key 다."""
    if (
        isinstance(fitting_job_id, bool)
        or not isinstance(fitting_job_id, int)
        or fitting_job_id < 1
    ):
        raise ValueError("fitting_job_id 는 1 이상의 정수여야 합니다.")
    return f"virtual-fitting/results/{fitting_job_id}/{ASYNC_RESULT_FILENAME}"


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
