from email.message import Message
from io import BytesIO
from urllib.error import URLError

import boto3
import pytest
from botocore.stub import Stubber
from PIL import Image

from app.clients import s3 as s3_module
from app.clients.s3 import ImageStorageError, S3ConfigError, S3ImageStorage, get_s3_client
from app.config import settings


class FakeS3Client:
    def __init__(self, error=None):
        self.error = error
        self.puts = []

    def put_object(self, **kwargs):
        if self.error:
            raise self.error
        self.puts.append(kwargs)


class FakeResponse:
    def __init__(self, body, content_type="image/jpeg"):
        self.body = body
        self.headers = Message()
        if content_type is not None:
            self.headers["Content-Type"] = content_type

    def read(self, limit):
        return self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _image_bytes(image_format):
    output = BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, format=image_format)
    return output.getvalue()


PNG_BYTES = _image_bytes("PNG")
JPEG_BYTES = _image_bytes("JPEG")
WEBP_BYTES = _image_bytes("WEBP")
GIF_BYTES = _image_bytes("GIF")


def _storage_returning(body, content_type):
    client = FakeS3Client()
    storage = S3ImageStorage(
        client,
        bucket="bucket",
        opener=lambda *args, **kwargs: FakeResponse(body, content_type),
    )
    return storage, client


@pytest.mark.parametrize(
    ("header", "body", "url", "expected_type", "expected_extension"),
    [
        ("image/png", PNG_BYTES, "https://trusted.example/r.png", "image/png", ".png"),
        ("image/jpeg", JPEG_BYTES, "https://trusted.example/r.jpg", "image/jpeg", ".jpg"),
        (None, PNG_BYTES, "https://trusted.example/r", "image/png", ".png"),
        ("text/plain", PNG_BYTES, "https://trusted.example/r", "image/png", ".png"),
        (
            "application/octet-stream",
            JPEG_BYTES,
            "https://trusted.example/r",
            "image/jpeg",
            ".jpg",
        ),
        ("image/webp", WEBP_BYTES, "https://trusted.example/r.webp", "image/webp", ".webp"),
        # URL 확장자와 헤더보다 실제 이미지 bytes 가 우선한다.
        ("image/jpeg", PNG_BYTES, "https://trusted.example/r.jpg", "image/png", ".png"),
    ],
    ids=[
        "png-header",
        "jpeg-header",
        "no-header-png-bytes",
        "text-plain-png-bytes",
        "octet-stream-jpeg-bytes",
        "webp",
        "bytes-win-over-header-and-url",
    ],
)
def test_content_type_and_extension_follow_the_real_image_bytes(
    header, body, url, expected_type, expected_extension
):
    storage, client = _storage_returning(body, header)

    key = storage.store_remote_image(url, "virtual-fitting/results")

    assert key.startswith("virtual-fitting/results/")
    assert key.endswith(expected_extension)
    assert not key.endswith(".txt")
    assert client.puts == [
        {"Bucket": "bucket", "Key": key, "Body": body, "ContentType": expected_type}
    ]


@pytest.mark.parametrize("header", ["image/png", "text/plain", None])
@pytest.mark.parametrize(
    "body",
    [b"<html>not an image</html>", b"plain text", GIF_BYTES],
    ids=["html", "text", "unsupported-gif"],
)
def test_non_image_or_unsupported_bytes_are_rejected_without_upload(header, body):
    storage, client = _storage_returning(body, header)

    with pytest.raises(ImageStorageError):
        storage.store_remote_image("https://trusted.example/result.png", "virtual-fitting/results")

    assert client.puts == []


def test_s3_error_is_wrapped_without_provider_details():
    storage = S3ImageStorage(FakeS3Client(RuntimeError("secret provider detail")), bucket="bucket")

    with pytest.raises(ImageStorageError) as exc_info:
        storage.upload_bytes(b"png", "key.png", "image/png")

    assert "secret provider detail" not in str(exc_info.value)


def test_upload_bytes_puts_body_with_content_type_and_returns_the_key():
    client = FakeS3Client()
    storage = S3ImageStorage(client, bucket="bucket")

    key = storage.upload_bytes(b"png-body", "body-images/a.png", "image/png")

    # 인자를 정확히 이 네 개로 고정한다. ACL/ServerSideEncryption 을 넣으면 Object Ownership
    # (ACL 비활성)이나 SSE-S3 기본 암호화 설정과 충돌할 수 있으므로 넣지 않는 것이 계약이다.
    assert key == "body-images/a.png"
    assert client.puts == [
        {
            "Bucket": "bucket",
            "Key": "body-images/a.png",
            "Body": b"png-body",
            "ContentType": "image/png",
        }
    ]


def test_put_object_arguments_are_valid_for_the_real_boto3_client(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    client = boto3.client(
        "s3",
        region_name="ap-northeast-2",
        aws_access_key_id="unit-test",
        aws_secret_access_key="unit-test",
    )
    storage = S3ImageStorage(client, bucket="bucket")

    with Stubber(client) as stubber:
        stubber.add_response(
            "put_object",
            {},
            expected_params={
                "Bucket": "bucket",
                "Key": "body-images/a.png",
                "Body": b"png",
                "ContentType": "image/png",
            },
        )
        storage.upload_bytes(b"png", "body-images/a.png", "image/png")
        stubber.assert_no_pending_responses()

    assert client.meta.endpoint_url == "https://s3.ap-northeast-2.amazonaws.com"


@pytest.mark.parametrize("error", [URLError("down"), TimeoutError("slow")])
def test_remote_download_failure_is_wrapped_and_nothing_is_uploaded(error):
    def failing_opener(*args, **kwargs):
        raise error

    client = FakeS3Client()
    storage = S3ImageStorage(client, bucket="bucket", opener=failing_opener)

    with pytest.raises(ImageStorageError):
        storage.store_remote_image("https://trusted.example/result.jpg", "virtual-fitting/results")

    assert client.puts == []


@pytest.mark.parametrize("body", [b"", b"12345"])
def test_empty_or_oversized_remote_image_is_rejected_without_upload(monkeypatch, body):
    monkeypatch.setattr(s3_module, "MAX_REMOTE_IMAGE_BYTES", 4)
    client = FakeS3Client()
    storage = S3ImageStorage(
        client, bucket="bucket", opener=lambda *args, **kwargs: FakeResponse(body)
    )

    with pytest.raises(ImageStorageError):
        storage.store_remote_image("https://trusted.example/result.jpg", "virtual-fitting/results")

    assert client.puts == []


def test_missing_bucket_setting_fails_fast(monkeypatch):
    monkeypatch.setattr(settings, "S3_BUCKET", "  ")

    with pytest.raises(S3ConfigError):
        S3ImageStorage(FakeS3Client())


@pytest.fixture
def fresh_s3_client_cache():
    get_s3_client.cache_clear()
    yield
    get_s3_client.cache_clear()


def test_bucket_and_region_come_from_settings_and_client_is_created_once(
    monkeypatch, fresh_s3_client_cache
):
    monkeypatch.setattr(settings, "S3_BUCKET", "configured-bucket")
    monkeypatch.setattr(settings, "AWS_REGION", "ap-northeast-2")
    fake_client = FakeS3Client()
    calls = []

    def fake_boto3_client(*args, **kwargs):
        calls.append((args, kwargs))
        return fake_client

    monkeypatch.setattr(boto3, "client", fake_boto3_client)

    S3ImageStorage().upload_bytes(b"png", "a.png", "image/png")
    S3ImageStorage().upload_bytes(b"png", "b.png", "image/png")

    # credential 인자를 넘기지 않아야 boto3 기본 credential chain(IAM Role)을 그대로 쓴다.
    assert calls == [(("s3",), {"region_name": "ap-northeast-2"})]
    assert [put["Bucket"] for put in fake_client.puts] == ["configured-bucket"] * 2
