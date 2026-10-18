"""S3 client와 이미지 저장소.

boto3의 표준 credential provider chain을 사용한다. 운영에서는 EC2 IAM Role로
자격증명을 공급하며 access key를 코드에 받거나 저장하지 않는다.
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from io import BytesIO
from typing import Any, Protocol, runtime_checkable
from urllib.request import Request, urlopen

from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

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

    code = "S3_CONFIG_ERROR"


class ImageStorageError(RuntimeError):
    """원격 이미지 다운로드 또는 S3 업로드 실패."""


class InvalidS3KeyError(ValueError):
    """S3 Object Key 형식이 잘못됐다. S3 에 요청하기 전에 거절한다."""


# SigV4 Presigned URL 의 최대 유효기간(7일). 임시 자격증명(IAM Role)으로 서명한 URL 은
# 이보다 짧은 자격증명 만료 시각에 함께 만료된다.
MAX_PRESIGNED_EXPIRES_SECONDS = 7 * 24 * 3600
_MAX_KEY_BYTES = 1024
# HEAD 는 본문이 없어 오류 Code 가 HTTP status 문자열("404" 등)로 온다.
_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
_ACCESS_DENIED_CODES = frozenset({"403", "AccessDenied", "Forbidden"})
_TRANSIENT_CODES = frozenset(
    {
        "500",
        "502",
        "503",
        "504",
        "InternalError",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "RequestTimeout",
    }
)


class S3ObjectState(str, Enum):
    EXISTS = "exists"
    # 명확한 404. 객체가 없다.
    NOT_FOUND = "not_found"
    # 403. 권한이 없거나(GetObject 없음), 객체가 없는데 ListBucket 권한이 없어 404 대신 403 이 왔을 수 있다.
    ACCESS_DENIED = "access_denied"
    # 5xx·throttling·연결 오류·timeout. 다시 확인하면 달라질 수 있다.
    TRANSIENT_ERROR = "transient_error"
    # 자격증명 없음·잘못된 요청(400·301 등) 같은 설정 문제. 다시 확인해도 같다.
    ERROR = "error"


@dataclass(frozen=True)
class S3ObjectCheck:
    """HEAD 한 번의 결과. URL·자격증명은 담지 않는다."""

    state: S3ObjectState
    key: str
    http_status: int | None = None
    error_code: str | None = None
    content_length: int | None = None
    content_type: str | None = None
    # 객체의 마지막 수정 시각(HEAD 응답의 LastModified). Watchdog 이 "이 생성 시도 뒤에 저장된 객체인지" 확인할 때 쓴다.
    last_modified: Any = None

    @property
    def exists(self) -> bool | None:
        """True·False 는 확실할 때만. 403·일시 오류·설정 오류는 None(알 수 없음)."""
        if self.state is S3ObjectState.EXISTS:
            return True
        if self.state is S3ObjectState.NOT_FOUND:
            return False
        return None

    @property
    def retryable(self) -> bool:
        return self.state is S3ObjectState.TRANSIENT_ERROR


def validate_s3_key(key: str) -> str:
    """버킷 안의 상대 Key 만 받는다. URL·s3:// 주소·앞뒤 공백·빈 경로 조각·'..' 은 거절한다."""
    if not isinstance(key, str) or not key or key != key.strip():
        raise InvalidS3KeyError("S3 Key 가 비어 있거나 앞뒤에 공백이 있습니다.")
    if len(key.encode("utf-8")) > _MAX_KEY_BYTES:
        raise InvalidS3KeyError("S3 Key 가 너무 깁니다.")
    if any(ord(char) < 32 or ord(char) == 127 for char in key):
        raise InvalidS3KeyError("S3 Key 에 제어 문자가 있습니다.")
    if any(part in ("", ".", "..") for part in key.split("/")):
        # "/a", "a/", "a//b", "../a", "https://...", "s3://..." 가 여기서 걸린다.
        raise InvalidS3KeyError("S3 Key 형식이 올바르지 않습니다.")
    return key


def _validate_expires(expires_in: int) -> int:
    if (
        isinstance(expires_in, bool)
        or not isinstance(expires_in, int)
        or not 1 <= expires_in <= MAX_PRESIGNED_EXPIRES_SECONDS
    ):
        raise ValueError(f"expires_in 은 1~{MAX_PRESIGNED_EXPIRES_SECONDS} 사이 정수여야 합니다.")
    return expires_in


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


@lru_cache(maxsize=1)
def get_s3_presign_client() -> Any:
    """Presigned URL 서명 전용 client. 서명은 로컬 계산이라 AWS 에 요청하지 않는다.

    기본 client 로 서명하면 host 가 전역 endpoint(<bucket>.s3.amazonaws.com)가 된다. 새로 만든
    버킷은 그 host 에 307 redirect 를 돌려줄 수 있어, 외부(Runware·OpenAI)가 redirect 를 따라가지
    않으면 실패한다. 리전 endpoint(<bucket>.s3.<region>.amazonaws.com)로 고정한다.
    자격증명은 get_s3_client 와 같은 기본 credential chain 이다.
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


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
        presign_client: Any | None = None,
    ) -> None:
        self._bucket = bucket if bucket is not None else get_s3_bucket()
        self._client = client if client is not None else get_s3_client()
        self._opener = opener or urlopen
        # V1 은 Presigned URL 을 쓰지 않으므로 처음 서명할 때 만든다.
        self._presign_client = presign_client

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
        """V1: 원격 이미지를 내려받아 key_prefix 아래 새 이름(uuid + 감지한 확장자)으로 저장한다."""
        body, content_type, extension = self._download_image(url)
        key = f"{key_prefix}/{uuid.uuid4()}{extension}"
        return self.upload_bytes(body, key, content_type)

    def store_remote_image_at(self, url: str, key: str) -> str:
        """원격 이미지를 내려받아 정해진 key 에 저장한다(V2 fallback: results/{id}/pruna-sync.jpg).

        Content-Type 은 내려받은 내용으로 정한다(key 의 확장자와 다를 수 있다). URL 은 로그·예외에 남기지 않는다.
        """
        key = validate_s3_key(key)
        body, content_type, _ = self._download_image(url)
        return self.upload_bytes(body, key, content_type)

    def _download_image(self, url: str) -> tuple[bytes, str, str]:
        request = Request(url, headers={"User-Agent": "LookDDak-AI/1.0"})
        try:
            with self._opener(request, timeout=DEFAULT_DOWNLOAD_TIMEOUT) as response:
                body = response.read(MAX_REMOTE_IMAGE_BYTES + 1)
        except (OSError, TimeoutError) as error:
            raise ImageStorageError("가상피팅 결과 이미지 다운로드에 실패했습니다.") from error
        if not body or len(body) > MAX_REMOTE_IMAGE_BYTES:
            raise ImageStorageError("가상피팅 결과 이미지 크기가 허용 범위를 벗어났습니다.")
        content_type, extension = _detect_image_type(body)
        return body, content_type, extension

    # --- V2 비동기 가상피팅: Presigned URL 과 객체 확인 ---
    # Presigned URL 에는 서명·임시 자격증명이 들어 있다. 로그·예외 메시지·DB 에 남기지 않는다.

    def presigned_get_url(self, key: str, *, expires_in: int) -> str:
        """key 객체를 expires_in 초 동안 읽을 수 있는 URL. 객체가 있는지는 확인하지 않는다."""
        return self._presign("get_object", validate_s3_key(key), expires_in)

    def presigned_put_url(
        self, key: str, *, expires_in: int, content_type: str | None = None
    ) -> str:
        """key 위치에 raw binary 를 PUT 할 수 있는 URL. 업로드는 하지 않는다.

        content_type 을 주면 Content-Type 이 서명에 들어가, PUT 요청이 정확히 같은 헤더를 보내야 한다.
        Runware 가 보내는 Content-Type 이 확인되기 전까지는 None(서명에 넣지 않음)으로 둔다.
        """
        params = {"ContentType": content_type} if content_type else {}
        return self._presign("put_object", validate_s3_key(key), expires_in, params)

    def head_object(self, key: str) -> S3ObjectCheck:
        """HEAD 를 한 번 보내 결과를 분류한다. 예외를 던지지 않는다(잘못된 key 제외).

        재시도 일정(0·2·7초)은 호출하는 쪽이 정한다. boto3 client 의 기본 재시도는 그대로 적용된다.
        """
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            ConnectionError,
            HTTPClientError,
        )

        key = validate_s3_key(key)
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as error:
            check = _classify_client_error(key, error)
        except (ConnectionError, HTTPClientError) as error:
            check = S3ObjectCheck(
                S3ObjectState.TRANSIENT_ERROR, key, error_code=type(error).__name__
            )
        except BotoCoreError as error:
            # NoCredentialsError·ParamValidationError 등. 다시 해도 같다.
            check = S3ObjectCheck(S3ObjectState.ERROR, key, error_code=type(error).__name__)
        else:
            check = S3ObjectCheck(
                S3ObjectState.EXISTS,
                key,
                http_status=response.get("ResponseMetadata", {}).get("HTTPStatusCode"),
                content_length=response.get("ContentLength"),
                content_type=response.get("ContentType"),
                last_modified=response.get("LastModified"),
            )
        if check.state is not S3ObjectState.EXISTS:
            logger.info(
                "s3 head key=%s state=%s http_status=%s error_code=%s",
                key,
                check.state.value,
                check.http_status,
                check.error_code,
            )
        return check

    def _presign(
        self, method: str, key: str, expires_in: int, extra: dict[str, str] | None = None
    ) -> str:
        expires_in = _validate_expires(expires_in)
        if self._presign_client is None:
            self._presign_client = get_s3_presign_client()
        try:
            return self._presign_client.generate_presigned_url(
                method,
                Params={"Bucket": self._bucket, "Key": key, **(extra or {})},
                ExpiresIn=expires_in,
            )
        except Exception as error:
            # 원인 메시지에 자격증명 정보가 섞이지 않게 고정 문구로 감싼다.
            raise ImageStorageError("S3 Presigned URL 생성에 실패했습니다.") from error


def _classify_client_error(key: str, error: Any) -> S3ObjectCheck:
    response = getattr(error, "response", {}) or {}
    code = str(response.get("Error", {}).get("Code", "")) or None
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 404 or code in _NOT_FOUND_CODES:
        state = S3ObjectState.NOT_FOUND
    elif status == 403 or code in _ACCESS_DENIED_CODES:
        state = S3ObjectState.ACCESS_DENIED
    elif (isinstance(status, int) and (status >= 500 or status == 429)) or code in _TRANSIENT_CODES:
        state = S3ObjectState.TRANSIENT_ERROR
    else:
        state = S3ObjectState.ERROR
    return S3ObjectCheck(state, key, http_status=status, error_code=code)
