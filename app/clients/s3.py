"""S3 client와 이미지 저장소.

boto3의 표준 credential provider chain을 사용한다. 운영에서는 EC2 IAM Role로
자격증명을 공급하며 access key를 코드에 받거나 저장하지 않는다.
"""

import mimetypes
import uuid
from collections.abc import Callable
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app.config import settings

DEFAULT_DOWNLOAD_TIMEOUT = 30.0
MAX_REMOTE_IMAGE_BYTES = 20 * 1024 * 1024


class S3ConfigError(RuntimeError):
    """필수 S3 설정이 없을 때 발생한다."""


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


def _extension_for(url: str, content_type: str) -> str:
    suffix = PurePosixPath(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    return mimetypes.guess_extension(content_type.split(";", 1)[0]) or ".png"


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
                content_type = response.headers.get_content_type()
        except (OSError, TimeoutError) as error:
            raise ImageStorageError("가상피팅 결과 이미지 다운로드에 실패했습니다.") from error
        if not body or len(body) > MAX_REMOTE_IMAGE_BYTES:
            raise ImageStorageError("가상피팅 결과 이미지 크기가 허용 범위를 벗어났습니다.")
        key = f"{key_prefix}/{uuid.uuid4()}{_extension_for(url, content_type)}"
        return self.upload_bytes(body, key, content_type)
