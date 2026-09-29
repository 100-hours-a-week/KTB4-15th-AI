"""S3 client와 이미지 저장소.

boto3의 표준 credential provider chain을 사용한다. 운영에서는 EC2 IAM Role로
자격증명을 공급하며 access key를 코드에 받거나 저장하지 않는다.
"""

import uuid
from collections.abc import Callable
from functools import lru_cache
from io import BytesIO
from typing import Any, Protocol, runtime_checkable
from urllib.request import Request, urlopen

from PIL import Image

from app.config import settings

DEFAULT_DOWNLOAD_TIMEOUT = 30.0
MAX_REMOTE_IMAGE_BYTES = 20 * 1024 * 1024

# Pillow 가 실제 bytes 에서 판별한 format -> (Content-Type, 확장자).
_IMAGE_TYPES = {
    "PNG": ("image/png", ".png"),
    "JPEG": ("image/jpeg", ".jpg"),
    "WEBP": ("image/webp", ".webp"),
}


class S3ConfigError(RuntimeError):
    """필수 S3 설정이 없을 때 발생한다."""

    reason_code = "S3_CONFIG_ERROR"


class ImageStorageError(RuntimeError):
    """원격 이미지 다운로드 또는 S3 업로드 실패."""


@runtime_checkable
class ImageStorage(Protocol):
    def upload_bytes(self, body: bytes, key: str, content_type: str) -> str: ...

    def store_remote_image(self, url: str, key_prefix: str) -> str: ...


def get_s3_bucket() -> str:
    bucket = settings.S3_BUCKET.strip()
    if not bucket:
        raise S3ConfigError("S3_BUCKET 환경변수가 설정되지 않았습니다.")
    return bucket


@lru_cache(maxsize=1)
def get_s3_client() -> Any:
    import boto3

    return boto3.client("s3", region_name=settings.AWS_REGION)


def _detect_image_type(body: bytes) -> tuple[str, str]:
    """실제 이미지 bytes 로 (Content-Type, 확장자)를 정한다.

    원격 응답의 Content-Type 헤더나 URL 확장자는 누락/오기(text/plain,
    application/octet-stream 등)가 흔해서 믿지 않는다. 헤더가 정상이면 결과가 같다.
    """
    try:
        with Image.open(BytesIO(body)) as image:
            detected = _IMAGE_TYPES.get(image.format)
    except (OSError, ValueError, Image.DecompressionBombError):
        detected = None
    if detected is None:
        raise ImageStorageError("가상피팅 결과 이미지가 지원하는 형식(PNG/JPEG/WEBP)이 아닙니다.")
    return detected


class S3ImageStorage:
    def __init__(
        self,
        client: Any | None = None,
        *,
        bucket: str | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._bucket = bucket if bucket is not None else get_s3_bucket()
        self._client = client if client is not None else get_s3_client()
        self._opener = opener or urlopen

    def upload_bytes(self, body: bytes, key: str, content_type: str) -> str:
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
        except Exception as error:
            raise ImageStorageError("S3 이미지 업로드에 실패했습니다.") from error
        return key

    def store_remote_image(self, url: str, key_prefix: str) -> str:
        request = Request(url, headers={"User-Agent": "LookDDak-AI/1.0"})
        try:
            with self._opener(request, timeout=DEFAULT_DOWNLOAD_TIMEOUT) as response:
                body = response.read(MAX_REMOTE_IMAGE_BYTES + 1)
        except (OSError, TimeoutError) as error:
            raise ImageStorageError("가상피팅 결과 이미지 다운로드에 실패했습니다.") from error
        if not body or len(body) > MAX_REMOTE_IMAGE_BYTES:
            raise ImageStorageError("가상피팅 결과 이미지 크기가 허용 범위를 벗어났습니다.")
        content_type, extension = _detect_image_type(body)
        key = f"{key_prefix}/{uuid.uuid4()}{extension}"
        return self.upload_bytes(body, key, content_type)
