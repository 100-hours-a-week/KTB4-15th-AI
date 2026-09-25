from email.message import Message

import pytest

from app.clients.s3 import ImageStorageError, S3ImageStorage


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
        self.headers["Content-Type"] = content_type

    def read(self, limit):
        return self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_remote_image_is_downloaded_and_uploaded_under_generated_key():
    client = FakeS3Client()
    storage = S3ImageStorage(
        client,
        bucket="bucket",
        opener=lambda *args, **kwargs: FakeResponse(b"jpeg-body"),
    )

    key = storage.store_remote_image("https://trusted.example/result.jpg", "virtual-fitting/results")

    assert key.startswith("virtual-fitting/results/")
    assert key.endswith(".jpg")
    assert client.puts == [
        {
            "Bucket": "bucket",
            "Key": key,
            "Body": b"jpeg-body",
            "ContentType": "image/jpeg",
        }
    ]


def test_s3_error_is_wrapped_without_provider_details():
    storage = S3ImageStorage(FakeS3Client(RuntimeError("secret provider detail")), bucket="bucket")

    with pytest.raises(ImageStorageError) as exc_info:
        storage.upload_bytes(b"png", "key.png", "image/png")

    assert "secret provider detail" not in str(exc_info.value)
