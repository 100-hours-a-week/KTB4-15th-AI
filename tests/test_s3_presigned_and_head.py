"""V2 비동기 가상피팅의 S3 Presigned GET/PUT 과 HEAD.

Presigned URL 은 boto3 가 로컬에서 계산한다(가짜 자격증명, AWS 요청 없음).
HEAD 는 botocore Stubber 나 fake client 로 응답을 흉내 낸다. 실제 AWS 에는 요청하지 않는다.
"""

import logging
from urllib.parse import parse_qs, urlsplit

import boto3
import pytest
from botocore.exceptions import (
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
)
from botocore.stub import Stubber

from app.clients import s3 as s3_module
from app.clients.s3 import (
    MAX_PRESIGNED_EXPIRES_SECONDS,
    ImageStorageError,
    InvalidS3KeyError,
    S3ImageStorage,
    S3ObjectCheck,
    S3ObjectState,
    get_s3_client,
    get_s3_presign_client,
)
from app.config import settings
from app.virtual_fitting import async_storage

BUCKET = "lookddak-dev-bucket"
REGION = "ap-northeast-2"
ACCESS_KEY = "AKIAUNITTESTACCESSKEY"
SECRET_KEY = "unit-test-secret-access-key"
USER_KEY = "users/3.jpg"
RESULT_KEY = "virtual-fitting/results/3/pruna-async.jpg"


@pytest.fixture(autouse=True)
def no_ambient_aws(monkeypatch):
    # 개발자 PC 의 AWS 설정·자격증명이 테스트에 섞이지 않게 한다.
    for name in (
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_S3_ADDRESSING_STYLE",
    ):
        monkeypatch.delenv(name, raising=False)


def _boto_client(**kwargs):
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        **kwargs,
    )


def _presign_client():
    # 운영 presign client 와 같은 설정(리전 virtual-hosted, SigV4)에 가짜 자격증명만 넣는다.
    from botocore.config import Config

    return _boto_client(config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))


class NoCallS3Client:
    """HEAD 이외의 S3 호출이 일어나면 실패한다."""

    def __init__(self, error=None, response=None):
        self.error = error
        self.response = response or {}
        self.heads = []

    def head_object(self, **kwargs):
        self.heads.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _storage(client=None):
    return S3ImageStorage(
        client or NoCallS3Client(), bucket=BUCKET, presign_client=_presign_client()
    )


def _query(url):
    return {name: values[0] for name, values in parse_qs(urlsplit(url).query).items()}


# --- 1·3. Presigned GET (사용자 원본, 결과 이미지) ---


def test_user_image_get_url_is_regional_sigv4_with_default_1800s():
    url = async_storage.user_image_get_url(_storage(), USER_KEY)

    parts = urlsplit(url)
    assert parts.scheme == "https"
    assert parts.netloc == f"{BUCKET}.s3.{REGION}.amazonaws.com"
    assert parts.path == f"/{USER_KEY}"
    query = _query(url)
    assert query["X-Amz-Algorithm"] == "AWS4-HMAC-SHA256"
    assert query["X-Amz-Expires"] == "1800"
    assert query["X-Amz-SignedHeaders"] == "host"
    assert f"/{REGION}/s3/aws4_request" in query["X-Amz-Credential"]


def test_result_image_get_url_reuses_presigned_get_with_default_600s():
    url = async_storage.result_image_get_url(_storage(), RESULT_KEY)

    assert urlsplit(url).path == f"/{RESULT_KEY}"
    assert _query(url)["X-Amz-Expires"] == "600"


# --- 2. Presigned PUT (Runware uploadEndpoint) ---


def test_result_upload_url_does_not_sign_content_type_by_default():
    url = async_storage.result_upload_url(_storage(), RESULT_KEY)

    assert urlsplit(url).netloc == f"{BUCKET}.s3.{REGION}.amazonaws.com"
    assert urlsplit(url).path == f"/{RESULT_KEY}"
    query = _query(url)
    assert query["X-Amz-Expires"] == "3600"
    # Content-Type 을 서명하지 않으므로 Runware 가 어떤 Content-Type 으로 PUT 해도 서명이 맞는다.
    assert query["X-Amz-SignedHeaders"] == "host"


def test_result_upload_url_can_sign_content_type_when_confirmed():
    url = async_storage.result_upload_url(_storage(), RESULT_KEY, content_type="image/jpeg")

    assert _query(url)["X-Amz-SignedHeaders"] == "content-type;host"


def test_put_and_get_urls_are_different_signatures():
    storage = _storage()
    get_url = storage.presigned_get_url(RESULT_KEY, expires_in=600)
    put_url = storage.presigned_put_url(RESULT_KEY, expires_in=600)

    assert _query(get_url)["X-Amz-Signature"] != _query(put_url)["X-Amz-Signature"]


def test_presign_makes_no_s3_api_call():
    client = NoCallS3Client()
    storage = _storage(client)

    storage.presigned_get_url(USER_KEY, expires_in=60)
    storage.presigned_put_url(RESULT_KEY, expires_in=60)

    assert client.heads == []


# --- 4. 만료시간 설정 ---


@pytest.mark.parametrize(
    ("setting", "function", "key"),
    [
        (
            "VIRTUAL_FITTING_USER_IMAGE_URL_EXPIRES_SECONDS",
            async_storage.user_image_get_url,
            USER_KEY,
        ),
        (
            "VIRTUAL_FITTING_RESULT_UPLOAD_URL_EXPIRES_SECONDS",
            async_storage.result_upload_url,
            RESULT_KEY,
        ),
        (
            "VIRTUAL_FITTING_RESULT_IMAGE_URL_EXPIRES_SECONDS",
            async_storage.result_image_get_url,
            RESULT_KEY,
        ),
    ],
)
def test_expiry_comes_from_settings_and_can_be_overridden(monkeypatch, setting, function, key):
    monkeypatch.setattr(settings, setting, 1234)
    assert _query(function(_storage(), key))["X-Amz-Expires"] == "1234"
    assert _query(function(_storage(), key, expires_in=90))["X-Amz-Expires"] == "90"


def test_default_expiry_settings():
    assert settings.VIRTUAL_FITTING_USER_IMAGE_URL_EXPIRES_SECONDS == 1800
    assert settings.VIRTUAL_FITTING_RESULT_UPLOAD_URL_EXPIRES_SECONDS == 3600
    assert settings.VIRTUAL_FITTING_RESULT_IMAGE_URL_EXPIRES_SECONDS == 600


@pytest.mark.parametrize(
    "expires_in", [0, -1, MAX_PRESIGNED_EXPIRES_SECONDS + 1, 1.5, "1800", True, None]
)
def test_invalid_expiry_is_rejected(expires_in):
    with pytest.raises(ValueError):
        _storage().presigned_get_url(USER_KEY, expires_in=expires_in)


def test_max_expiry_is_allowed():
    url = _storage().presigned_put_url(RESULT_KEY, expires_in=MAX_PRESIGNED_EXPIRES_SECONDS)
    assert _query(url)["X-Amz-Expires"] == str(MAX_PRESIGNED_EXPIRES_SECONDS)


# --- 5. 잘못된 S3 Key ---


BAD_KEYS = [
    None,
    "",
    " users/3.jpg",
    "users/3.jpg ",
    "/users/3.jpg",
    "users/",
    "users//3.jpg",
    "../users/3.jpg",
    "users/../admin.jpg",
    "users/./3.jpg",
    "users/3\n.jpg",
    "https://bucket.s3.amazonaws.com/users/3.jpg",
    "s3://bucket/users/3.jpg",
    "a" * 1025,
    "가" * 342,  # UTF-8 1026 bytes
]


@pytest.mark.parametrize("key", BAD_KEYS)
def test_bad_key_is_rejected_before_any_s3_work(key):
    client = NoCallS3Client()
    storage = _storage(client)

    with pytest.raises(InvalidS3KeyError):
        storage.presigned_get_url(key, expires_in=60)
    with pytest.raises(InvalidS3KeyError):
        storage.presigned_put_url(key, expires_in=60)
    with pytest.raises(InvalidS3KeyError):
        storage.head_object(key)
    assert client.heads == []


def test_unicode_and_space_inside_key_are_allowed_and_encoded():
    url = _storage().presigned_get_url("users/홍 길동.jpg", expires_in=60)
    assert "%ED%99%8D%20" in urlsplit(url).path


@pytest.mark.parametrize("job_id", [0, -3, True, "3", 3.0, None])
def test_async_result_key_requires_positive_int(job_id):
    with pytest.raises(ValueError):
        async_storage.async_result_key(job_id)


def test_async_result_key_format():
    assert async_storage.async_result_key(3) == RESULT_KEY


# --- 6~9. HEAD ---


def _stubbed_storage():
    client = _boto_client()
    return S3ImageStorage(client, bucket=BUCKET, presign_client=_presign_client()), client


def test_head_existing_object():
    storage, client = _stubbed_storage()
    with Stubber(client) as stubber:
        stubber.add_response(
            "head_object",
            {
                "ContentLength": 2048,
                "ContentType": "image/jpeg",
                "ResponseMetadata": {"HTTPStatusCode": 200},
            },
            expected_params={"Bucket": BUCKET, "Key": RESULT_KEY},
        )
        check = async_storage.check_result_image(storage, RESULT_KEY)
        stubber.assert_no_pending_responses()

    assert check == S3ObjectCheck(
        S3ObjectState.EXISTS,
        RESULT_KEY,
        http_status=200,
        content_length=2048,
        content_type="image/jpeg",
    )
    assert check.exists is True and not check.retryable


@pytest.mark.parametrize(
    ("status", "code", "state", "exists", "retryable"),
    [
        (404, "404", S3ObjectState.NOT_FOUND, False, False),
        (404, "NoSuchKey", S3ObjectState.NOT_FOUND, False, False),
        (403, "403", S3ObjectState.ACCESS_DENIED, None, False),
        (403, "AccessDenied", S3ObjectState.ACCESS_DENIED, None, False),
        (500, "500", S3ObjectState.TRANSIENT_ERROR, None, True),
        (503, "SlowDown", S3ObjectState.TRANSIENT_ERROR, None, True),
        (400, "RequestTimeout", S3ObjectState.TRANSIENT_ERROR, None, True),
        (400, "400", S3ObjectState.ERROR, None, False),
        (301, "301", S3ObjectState.ERROR, None, False),
    ],
)
def test_head_errors_are_classified(status, code, state, exists, retryable):
    storage, client = _stubbed_storage()
    with Stubber(client) as stubber:
        stubber.add_client_error(
            "head_object",
            service_error_code=code,
            http_status_code=status,
            expected_params={"Bucket": BUCKET, "Key": RESULT_KEY},
        )
        check = storage.head_object(RESULT_KEY)
        stubber.assert_no_pending_responses()

    assert (check.state, check.http_status, check.error_code) == (state, status, code)
    assert check.exists is exists
    assert check.retryable is retryable


def test_403_is_not_treated_as_missing():
    storage, client = _stubbed_storage()
    with Stubber(client) as stubber:
        stubber.add_client_error("head_object", service_error_code="403", http_status_code=403)
        check = storage.head_object(RESULT_KEY)

    # ListBucket 권한이 없으면 없는 객체도 403 이다. "없음(False)"으로 단정하지 않는다.
    assert check.state is S3ObjectState.ACCESS_DENIED
    assert check.exists is None


@pytest.mark.parametrize(
    ("error", "state"),
    [
        (EndpointConnectionError(endpoint_url="https://s3"), S3ObjectState.TRANSIENT_ERROR),
        (ConnectTimeoutError(endpoint_url="https://s3"), S3ObjectState.TRANSIENT_ERROR),
        (ReadTimeoutError(endpoint_url="https://s3"), S3ObjectState.TRANSIENT_ERROR),
        (NoCredentialsError(), S3ObjectState.ERROR),
    ],
)
def test_head_transport_and_credential_errors(error, state):
    client = NoCallS3Client(error=error)
    check = _storage(client).head_object(RESULT_KEY)

    assert check.state is state
    assert check.http_status is None
    assert check.error_code == type(error).__name__
    assert check.exists is None
    assert len(client.heads) == 1  # 이 함수는 HEAD 를 한 번만 부른다


# --- 10. URL·자격증명 로그 노출 방지 ---


def test_urls_and_credentials_never_reach_logs(caplog):
    storage, client = _stubbed_storage()
    with caplog.at_level(logging.DEBUG, logger="app"):
        urls = [
            async_storage.user_image_get_url(storage, USER_KEY),
            async_storage.result_upload_url(storage, RESULT_KEY),
            async_storage.result_image_get_url(storage, RESULT_KEY),
        ]
        with Stubber(client) as stubber:
            stubber.add_client_error("head_object", service_error_code="403", http_status_code=403)
            storage.head_object(RESULT_KEY)

    signatures = [_query(url)["X-Amz-Signature"] for url in urls]
    for secret in (ACCESS_KEY, SECRET_KEY, "X-Amz-Signature", "X-Amz-Credential", *signatures):
        assert secret not in caplog.text
    assert f"key={RESULT_KEY} state=access_denied" in caplog.text


def test_presign_failure_message_hides_cause():
    class BrokenPresigner:
        def generate_presigned_url(self, *args, **kwargs):
            raise RuntimeError(f"cannot sign with {SECRET_KEY}")

    storage = S3ImageStorage(NoCallS3Client(), bucket=BUCKET, presign_client=BrokenPresigner())

    with pytest.raises(ImageStorageError) as exc_info:
        storage.presigned_get_url(USER_KEY, expires_in=60)
    assert SECRET_KEY not in str(exc_info.value)


# --- 설정·client 생성 (기본 credential chain, 단일 버킷) ---


@pytest.fixture
def fresh_clients():
    get_s3_client.cache_clear()
    get_s3_presign_client.cache_clear()
    yield
    get_s3_client.cache_clear()
    get_s3_presign_client.cache_clear()


def test_presign_client_uses_default_credential_chain_and_region(monkeypatch, fresh_clients):
    monkeypatch.setattr(settings, "S3_BUCKET", BUCKET)
    monkeypatch.setattr(settings, "AWS_REGION", REGION)
    calls = []

    class Presigner:
        def generate_presigned_url(self, method, Params, ExpiresIn):
            return f"https://signed/{method}/{Params['Bucket']}/{Params['Key']}?e={ExpiresIn}"

    def fake_boto3_client(*args, **kwargs):
        calls.append((args, kwargs))
        return Presigner() if "config" in kwargs else NoCallS3Client()

    monkeypatch.setattr(boto3, "client", fake_boto3_client)

    storage = S3ImageStorage()
    # V1 처럼 Presigned URL 을 쓰지 않으면 presign client 를 만들지 않는다.
    assert len(calls) == 1 and calls[0] == (("s3",), {"region_name": REGION})

    url = storage.presigned_get_url(USER_KEY, expires_in=60)
    S3ImageStorage().presigned_put_url(RESULT_KEY, expires_in=60)

    assert url == f"https://signed/get_object/{BUCKET}/{USER_KEY}?e=60"
    presign_calls = [call for call in calls if "config" in call[1]]
    assert len(presign_calls) == 1  # 한 번만 만든다
    (args, kwargs) = presign_calls[0]
    assert args == ("s3",) and kwargs["region_name"] == REGION
    # 자격증명을 넘기지 않아 기본 credential chain(IAM Role)을 쓴다.
    assert not {"aws_access_key_id", "aws_secret_access_key", "aws_session_token"} & kwargs.keys()
    assert kwargs["config"].s3 == {"addressing_style": "virtual"}
    assert kwargs["config"].signature_version == "s3v4"


def test_missing_bucket_still_fails_fast(monkeypatch):
    monkeypatch.setattr(settings, "S3_BUCKET", "")
    with pytest.raises(s3_module.S3ConfigError):
        S3ImageStorage(NoCallS3Client())
