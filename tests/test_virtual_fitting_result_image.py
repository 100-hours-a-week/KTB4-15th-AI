"""Runware 결과 이미지 다운로드·WebP 검증·S3 조건부 저장(2단계, 서비스 미연결)을 가짜 HTTP·S3 로 검증한다.

외부 네트워크·실제 AWS 는 쓰지 않는다. 실제 연결 클래스는 127.0.0.1 의 임시 TLS 서버(자체 서명 인증서)로만 확인한다.
"""

import hashlib
import http.server
import itertools
import logging
import shutil
import socket
import ssl
import subprocess
import threading
import time
import traceback
from datetime import UTC, datetime
from io import BytesIO

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from botocore.response import StreamingBody
from botocore.stub import Stubber
from PIL import Image

from app.clients.s3 import ConditionalPutOutcome, S3ImageStorage
from app.virtual_fitting import result_image as ri
from app.virtual_fitting.async_storage import async_result_key
from app.virtual_fitting.result_image import (
    ResultImageError,
    StoredResultImage,
    download_runware_image,
    resolve_public_addresses,
    store_runware_result_image,
    validate_static_webp,
)

JOB = 41
TASK = "a770f077-f413-47de-9dac-be0b26a35da6"
OTHER_TASK = "7d444840-9dc0-11d1-b245-5ffdce74fad2"
KEY = async_result_key(JOB)
BUCKET = "test-bucket"
URL = (
    "https://im.runware.ai/image/ws/2/ii/RESULT-IMG.webp"
    "?b=2&a=%2Fx%20y&token=tok-SECRET-123%3D%3D&X-Amz-Signature=sig-SECRET-456&Expires=1700000000"
)
TARGET = URL[len("https://im.runware.ai") :]
SECRETS = ("tok-SECRET-123", "sig-SECRET-456", "RESULT-IMG", "Expires=1700000000", "im.runware.ai/")
PUBLIC = "104.18.0.10"


# --- 이미지 ---


def webp(mode="RGB", size=(48, 64), **options):
    color = (10, 120, 200) if mode == "RGB" else (10, 120, 200, 90)
    buffer = BytesIO()
    Image.new(mode, size, color).save(buffer, "WEBP", **(options or {"quality": 90}))
    return buffer.getvalue()


def encoded(fmt, mode="RGB"):
    buffer = BytesIO()
    Image.new(mode, (40, 40), "red").save(buffer, fmt)
    return buffer.getvalue()


def animated_webp():
    frames = [Image.new("RGB", (32, 32), color) for color in ("red", "blue")]
    buffer = BytesIO()
    frames[0].save(buffer, "WEBP", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return buffer.getvalue()


RGB = webp()
RGBA = webp("RGBA", lossless=True)


# --- 가짜 HTTP ---


class FakeResponse:
    def __init__(
        self, status=200, body=b"", headers=None, chunks=None, read_error=None, on_read=None
    ):
        self.status = status
        self._headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._chunks = list(chunks) if chunks is not None else [body]
        self._read_error = read_error
        self.on_read = on_read  # 읽기 1번에 걸린 시간 흉내(가짜 시계 진행)
        self.reads, self.closed = 0, False

    def close(self):
        self.closed = True

    def getheader(self, name, default=None):
        return self._headers.get(name.lower(), default)

    def read(self, amount):
        self.reads += 1
        if self.on_read is not None:
            self.on_read()
        if self._read_error is not None:
            raise self._read_error
        if not self._chunks:
            return b""
        chunk = self._chunks.pop(0)
        if len(chunk) > amount:
            self._chunks.insert(0, chunk[amount:])
            chunk = chunk[:amount]
        return chunk


class FakeConnection:
    def __init__(self, response=None, request_error=None, on_request=None, on_headers=None):
        self.response = response or FakeResponse(body=RGB)
        self.request_error = request_error
        self.on_request, self.on_headers = on_request, on_headers
        self.requests, self.closed, self.headers_read = [], False, 0

    def request(self, method, target, headers):
        self.requests.append((method, target, dict(headers)))
        if self.on_request is not None:
            self.on_request()
        if self.request_error is not None:
            raise self.request_error

    def getresponse(self):
        self.headers_read += 1
        if self.on_headers is not None:
            self.on_headers()
        return self.response

    def close(self):
        self.closed = True


class Net:
    """download_runware_image 의 resolve·connect 자리. 어떤 연결을 열었는지 기록한다."""

    def __init__(self, connection=None, addresses=(PUBLIC,), on_resolve=None, on_connect=None):
        self.connection = connection or FakeConnection()
        self.addresses, self.resolved, self.connects = list(addresses), [], []
        self.on_resolve, self.on_connect = on_resolve, on_connect

    def resolve(self, host, port):
        self.resolved.append((host, port))
        if self.on_resolve is not None:
            self.on_resolve()
        return self.addresses

    def connect(self, host, port, address, deadline):
        self.connects.append((host, port, address, deadline))
        if self.on_connect is not None:
            self.on_connect()
        return self.connection

    def download(self, url, **kwargs):
        return download_runware_image(url, resolve=self.resolve, connect=self.connect, **kwargs)


# --- 가짜 S3 (메모리) ---


def client_error(status, code, operation):
    return ClientError(
        {"Error": {"Code": code, "Message": "x"}, "ResponseMetadata": {"HTTPStatusCode": status}},
        operation,
    )


class FakeS3:
    """조건부 PUT·GetObject 만. put_error 는 저장한 뒤(store_then_fail) 또는 저장하지 않고 던진다."""

    def __init__(self):
        self.objects, self.calls = {}, []
        self.put_error, self.store_then_fail = None, False
        self.get_errors = []  # 순서대로 한 번씩 던진다
        self.before_put = None

    def put_object(self, **kwargs):
        self.calls.append(("put_object", kwargs))
        if self.before_put is not None:
            self.before_put()
        key = kwargs["Key"]
        if self.put_error is not None and not self.store_then_fail:
            raise self.put_error
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise client_error(412, "PreconditionFailed", "PutObject")
        self.objects[key] = {
            "Body": bytes(kwargs["Body"]),
            "ContentType": kwargs.get("ContentType"),
            "Metadata": dict(kwargs.get("Metadata") or {}),
        }
        if self.put_error is not None:
            raise self.put_error  # 저장은 됐지만 응답을 못 받았다
        return {"ResponseMetadata": {"HTTPStatusCode": 200}, "ETag": '"e"'}

    def get_object(self, **kwargs):
        self.calls.append(("get_object", kwargs))
        if self.get_errors:
            raise self.get_errors.pop(0)
        stored = self.objects.get(kwargs["Key"])
        if stored is None:
            raise client_error(404, "NoSuchKey", "GetObject")
        body = stored["Body"]
        return {
            "Body": StreamingBody(BytesIO(body), len(body)),
            "ContentType": stored["ContentType"],
            "ContentLength": len(body),
            "Metadata": dict(stored["Metadata"]),
            "LastModified": datetime(2026, 11, 4, tzinfo=UTC),
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    def puts(self):
        return [kwargs for name, kwargs in self.calls if name == "put_object"]


def storage(client):
    return S3ImageStorage(client, bucket=BUCKET, presign_client=object())


def put_existing(s3, body=RGB, *, job=JOB, task=TASK, content_type="image/webp", meta=None):
    digest = hashlib.sha256(body).hexdigest()
    s3.objects[KEY] = {
        "Body": body,
        "ContentType": content_type,
        "Metadata": meta
        if meta is not None
        else {"fitting-job-id": str(job), "task-uuid": task, "sha256": digest},
    }


def no_download(url):
    raise AssertionError("기존 정상 객체가 있으면 원본 URL 을 다시 내려받지 않는다")


def store(s3, download=None, **overrides):
    values = {"fitting_job_id": JOB, "task_uuid": TASK, "image_url": URL, "result_key": KEY}
    values.update(overrides)
    return store_runware_result_image(storage(s3), download=download or Net().download, **values)


# === 1·6·7. 정상 RGB·RGBA 다운로드·저장, generic MIME, 원본 bytes·해시 ===


@pytest.mark.parametrize("body", [RGB, RGBA], ids=["rgb", "rgba"])
@pytest.mark.parametrize("mime", ["image/webp", "application/octet-stream", None])
def test_valid_webp_is_downloaded_and_stored_byte_for_byte(body, mime):
    headers = {"Content-Type": mime} if mime else {}
    net = Net(FakeConnection(FakeResponse(body=body, headers=headers)))
    s3 = FakeS3()
    result = store(s3, net.download)
    digest = hashlib.sha256(body).hexdigest()
    assert result == StoredResultImage(KEY, len(body), digest, reused=False)
    saved = s3.objects[KEY]
    assert saved["Body"] == body  # 재인코딩·변환 없이 원본 그대로
    assert hashlib.sha256(saved["Body"]).hexdigest() == digest
    assert saved["ContentType"] == "image/webp"
    assert saved["Metadata"] == {"fitting-job-id": str(JOB), "task-uuid": TASK, "sha256": digest}
    (put,) = s3.puts()
    assert put["IfNoneMatch"] == "*" and put["Key"] == KEY and put["Bucket"] == BUCKET


def test_validation_reports_mode_and_size_without_touching_bytes():
    rgba = bytes(RGBA)
    info = validate_static_webp(rgba)
    assert (info.width, info.height, info.mode) == (48, 64, "RGBA") and rgba == RGBA
    assert validate_static_webp(RGB).mode == "RGB"


# === 2. token·서명 쿼리 포함 원본 URL 유지 ===


def test_request_target_is_the_original_path_and_query():
    net = Net()
    net.download(URL)
    ((method, target, headers),) = net.connection.requests
    assert (method, target) == ("GET", TARGET)  # 삭제·재정렬·재인코딩 없음
    assert net.resolved == [("im.runware.ai", 443)]
    ((host, port, address, _deadline),) = net.connects
    assert (host, port, address) == ("im.runware.ai", 443, PUBLIC)
    assert "Authorization" not in headers and net.connection.closed


def test_fragment_is_not_sent_and_explicit_443_is_kept():
    net = Net()
    net.download("https://im.runware.ai:443/i/x.webp?token=a&token=b#frag")
    assert net.connection.requests[0][1] == "/i/x.webp?token=a&token=b"


# === 3. 비허용 주소·내부 주소·리다이렉트 ===


@pytest.mark.parametrize(
    "url",
    [
        "http://im.runware.ai/x.webp",
        "https://evil.example/x.webp?token=a",
        "https://im.runware.ai.evil.example/x.webp",
        "https://cdn.im.runware.ai/x.webp",
        "https://evil.example/im.runware.ai/x.webp",
        "https://im.runware.ai@evil.example/x.webp",
        "https://user:pw@im.runware.ai/x.webp",
        "https://im.runware.ai:8443/x.webp",
        "https://127.0.0.1/x.webp",
        "https://169.254.169.254/latest/meta-data",
    ],
)
def test_disallowed_url_is_rejected_before_any_connection(url):
    net = Net()
    with pytest.raises(ResultImageError) as error:
        net.download(url)
    assert error.value.reason == "address_not_allowed" and not error.value.retryable
    assert net.resolved == [] and net.connects == []


@pytest.mark.parametrize(
    "addresses",
    [
        ["10.0.0.5"],
        ["127.0.0.1"],
        ["169.254.169.254"],
        ["192.168.1.2"],
        ["100.64.0.1"],
        ["::1"],
        ["fd00::1"],
        ["fe80::1%en0"],
        ["224.0.0.1"],
        [PUBLIC, "10.0.0.5"],  # 하나라도 내부 주소면 거절
    ],
)
def test_internal_resolved_address_is_rejected(addresses):
    def getaddrinfo(host, port, type):
        return [(None, None, None, "", (address, port)) for address in addresses]

    with pytest.raises(ResultImageError) as error:
        resolve_public_addresses("im.runware.ai", 443, getaddrinfo=getaddrinfo)
    assert error.value.reason == "address_not_allowed"


def test_public_address_is_used_and_dns_failure_is_safe():
    def ok(host, port, type):
        return [(None, None, None, "", (PUBLIC, port)), (None, None, None, "", (PUBLIC, port))]

    assert resolve_public_addresses("im.runware.ai", 443, getaddrinfo=ok) == [PUBLIC]

    def fail(host, port, type):
        raise socket.gaierror(f"cannot resolve {URL}")

    with pytest.raises(ResultImageError) as error:
        resolve_public_addresses("im.runware.ai", 443, getaddrinfo=fail)
    assert error.value.reason == "dns_failed" and error.value.retryable
    assert_no_secret(error.value)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_is_not_followed(status):
    response = FakeResponse(status, headers={"Location": "https://evil.example/x.webp"})
    net = Net(FakeConnection(response))
    with pytest.raises(ResultImageError) as error:
        net.download(URL)
    assert (error.value.reason, error.value.http_status) == ("redirect", status)
    assert len(net.connection.requests) == 1 and len(net.connects) == 1 and response.reads == 0
    assert "evil.example" not in str(error.value)


# === 4. timeout·HTTP 오류·실제 수신 크기 ===


@pytest.mark.parametrize(
    ("status", "retryable"), [(403, False), (404, False), (410, False), (429, True), (503, True)]
)
def test_http_errors_are_distinguishable(status, retryable):
    net = Net(FakeConnection(FakeResponse(status, body=b"error page with " + URL.encode())))
    with pytest.raises(ResultImageError) as error:
        net.download(URL)
    assert (error.value.reason, error.value.http_status, error.value.retryable) == (
        "http_error",
        status,
        retryable,
    )
    assert_no_secret(error.value)


def test_timeouts_and_connection_errors_are_safe():
    for exc, reason in (
        (TimeoutError(f"timed out {URL}"), "timeout"),
        (ConnectionResetError(f"reset {URL}"), "connection_failed"),
        (ssl.SSLCertVerificationError(f"bad cert {URL}"), "connection_failed"),
    ):
        net = Net(FakeConnection(request_error=exc))
        with pytest.raises(ResultImageError) as error:
            net.download(URL)
        assert error.value.reason == reason and error.value.retryable
        assert_no_secret(error.value)
    net = Net(FakeConnection(FakeResponse(read_error=TimeoutError(URL))))
    with pytest.raises(ResultImageError) as error:
        net.download(URL)
    assert error.value.reason == "timeout"
    assert_no_secret(error.value)


# --- 하나의 30초 예산 (가짜 시계, 실제 대기 없음) ---


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def spend(self, seconds):
        return lambda: setattr(self, "now", self.now + seconds)


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(ri, "_clock", fake)  # 전역 time.monotonic 은 바꾸지 않는다
    return fake


def chunked(size=10):
    return [RGB[i : i + size] for i in range(0, len(RGB), size)]


def test_one_deadline_is_fixed_before_dns_and_shared_by_every_step(clock):
    net = Net(on_resolve=clock.spend(5), on_connect=clock.spend(3))
    net.download(URL)
    ((_, _, _, deadline),) = net.connects
    # DNS 직전 시각 기준이고, 연결 때 다시 주지 않는다.
    assert deadline == 1000.0 + ri.DEFAULT_DOWNLOAD_TIMEOUT


def test_time_spent_connecting_is_taken_from_the_read_budget(clock):
    # 연결 8초 + 읽기 1번 6초: 8→14→20→26→32. 32초에 다음 읽기를 시작하지 않는다(연결 0초였다면 5번 읽음).
    response = FakeResponse(chunks=chunked(), on_read=clock.spend(6))
    net = Net(FakeConnection(response), on_connect=clock.spend(8))
    with pytest.raises(ResultImageError) as error:
        net.download(URL)
    assert error.value.reason == "timeout" and response.reads == 4


def test_time_spent_on_headers_is_taken_from_the_body_budget(clock):
    response = FakeResponse(chunks=chunked(), on_read=clock.spend(6))
    net = Net(FakeConnection(response, on_headers=clock.spend(20)))
    with pytest.raises(ResultImageError) as error:
        net.download(URL)
    assert error.value.reason == "timeout" and response.reads == 2  # 20→26→32


def test_slow_body_drip_is_cut_at_the_total_limit_not_per_read(clock):
    # 읽기마다 1초씩만 걸려도(읽기별 timeout 에는 안 걸림) 합계 30초에서 끊는다.
    response = FakeResponse(chunks=[b"x"] * 1000, on_read=clock.spend(1))
    with pytest.raises(ResultImageError) as error:
        Net(FakeConnection(response)).download(URL)
    assert error.value.reason == "timeout" and response.reads == 30


@pytest.mark.parametrize(
    ("stage", "expect"),
    [
        ("resolve", {"connects": 0, "requests": 0, "headers": 0, "reads": 0}),
        ("connect", {"connects": 1, "requests": 0, "headers": 0, "reads": 0}),
        ("request", {"connects": 1, "requests": 1, "headers": 0, "reads": 0}),
        ("headers", {"connects": 1, "requests": 1, "headers": 1, "reads": 0}),
    ],
)
def test_exhausted_budget_starts_no_new_connection_request_or_read(clock, stage, expect):
    """DNS 가 늦게(31초) 끝나도 그 결과로 연결하지 않는다. 다른 단계도 같다. 자원은 닫는다."""
    late = clock.spend(31)
    response = FakeResponse(body=RGB)
    connection = FakeConnection(
        response,
        on_request=late if stage == "request" else None,
        on_headers=late if stage == "headers" else None,
    )
    net = Net(
        connection,
        on_resolve=late if stage == "resolve" else None,
        on_connect=late if stage == "connect" else None,
    )
    with pytest.raises(ResultImageError) as error:
        net.download(URL)
    assert error.value.reason == "timeout" and error.value.retryable
    assert {
        "connects": len(net.connects),
        "requests": len(connection.requests),
        "headers": connection.headers_read,
        "reads": response.reads,
    } == expect
    assert connection.closed or stage == "resolve"
    assert response.closed or stage != "headers"
    assert_no_secret(error.value)


def test_bytes_received_after_the_deadline_are_not_used(clock):
    # 본문은 다 받았지만 끝(EOF) 확인이 종료 시각을 넘겨 끝났다. 그 본문을 결과로 쓰지 않는다.
    response = FakeResponse(chunks=[RGB])
    response.on_read = lambda: clock.spend(31)() if response.reads == 2 else None
    with pytest.raises(ResultImageError) as error:
        Net(FakeConnection(response)).download(URL)
    assert error.value.reason == "timeout" and response.closed


def test_timeout_during_store_skips_validation_and_put(clock, monkeypatch):
    def must_not_validate(body, **kwargs):
        raise AssertionError("timeout 뒤에는 검증하지 않는다")

    monkeypatch.setattr(ri, "validate_static_webp", must_not_validate)
    s3 = FakeS3()
    response = FakeResponse(chunks=chunked(), on_read=clock.spend(10))
    with pytest.raises(ResultImageError) as error:
        store(s3, Net(FakeConnection(response)).download)
    assert error.value.reason == "timeout" and error.value.retryable
    assert s3.puts() == [] and s3.objects == {}
    assert_no_secret(error.value)


def test_actual_received_bytes_are_limited_even_if_content_length_lies():
    body = b"x" * 1000
    for headers in ({}, {"Content-Length": "10"}):
        net = Net(FakeConnection(FakeResponse(chunks=[body[:600], body[600:]], headers=headers)))
        with pytest.raises(ResultImageError) as error:
            net.download(URL, max_bytes=800)
        assert error.value.reason == "too_large"


def test_declared_oversize_is_rejected_before_reading():
    response = FakeResponse(body=RGB, headers={"Content-Length": str(10**9)})
    with pytest.raises(ResultImageError) as error:
        Net(FakeConnection(response)).download(URL)
    assert error.value.reason == "too_large" and response.reads == 0


def test_empty_body_is_rejected():
    with pytest.raises(ResultImageError) as error:
        Net(FakeConnection(FakeResponse(body=b""))).download(URL)
    assert error.value.reason == "empty"


# === 5. JPEG·PNG·손상·애니메이션 거절 ===


TRUNCATED = RGB[: len(RGB) // 2]


def noisy_webp():
    import random

    rng = random.Random(7)
    image = Image.frombytes(
        "RGB", (256, 256), bytes(rng.randrange(256) for _ in range(256 * 256 * 3))
    )
    buffer = BytesIO()
    image.save(buffer, "WEBP", quality=90)
    return buffer.getvalue()


NOISY = noisy_webp()
# RIFF 전체 크기·VP8 청크 크기를 실제보다 크게 적은 파일(잘린 전송과 같은 구조 손상)
RIFF_SIZE_BIG = NOISY[:4] + (len(NOISY) + 1000).to_bytes(4, "little") + NOISY[8:]
CHUNK_SIZE_BIG = (
    NOISY[:16] + (int.from_bytes(NOISY[16:20], "little") + 500).to_bytes(4, "little") + NOISY[20:]
)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (encoded("JPEG"), "not_webp"),
        (encoded("PNG"), "not_webp"),
        (encoded("PNG", "RGBA"), "not_webp"),
        (b"RIFF\x00\x00\x00\x00WEBPjunk", "not_webp"),  # Pillow 가 이미지로 인식하지 못한다
        (TRUNCATED, "corrupt"),
        (NOISY[: int(len(NOISY) * 0.7)], "corrupt"),
        (RIFF_SIZE_BIG, "corrupt"),
        (CHUNK_SIZE_BIG, "corrupt"),
        (b"not an image", "not_webp"),
        (animated_webp(), "animated"),
    ],
    ids=[
        "jpeg", "png", "png-rgba", "riff-junk", "truncated", "truncated-large",
        "riff-size", "chunk-size", "text", "animated",
    ],
)  # fmt: skip
def test_non_static_or_broken_webp_is_never_stored(body, reason):
    with pytest.raises(ResultImageError) as error:
        validate_static_webp(body)
    assert error.value.reason == reason and not error.value.retryable
    s3 = FakeS3()
    net = Net(FakeConnection(FakeResponse(body=body, headers={"Content-Type": "image/webp"})))
    with pytest.raises(ResultImageError):
        store(s3, net.download)
    assert s3.puts() == [] and s3.objects == {}


def test_pixel_limit_is_applied_before_decoding():
    with pytest.raises(ResultImageError) as error:
        validate_static_webp(RGB, max_pixels=48 * 64 - 1)
    assert error.value.reason == "too_many_pixels"


# === 8. .webp·image/webp·예정 Key 일치 (실제 botocore 모델로 요청 계약 확인) ===


def boto_client():
    return boto3.client(
        "s3", region_name="ap-northeast-2", aws_access_key_id="x", aws_secret_access_key="y"
    )


def test_s3_request_contract_with_stubber():
    client = boto_client()
    digest = hashlib.sha256(RGB).hexdigest()
    with Stubber(client) as stub:
        stub.add_client_error(
            "get_object",
            service_error_code="NoSuchKey",
            http_status_code=404,
            expected_params={"Bucket": BUCKET, "Key": KEY},
        )
        stub.add_response(
            "put_object",
            {"ETag": '"e"'},
            expected_params={
                "Bucket": BUCKET,
                "Key": KEY,
                "Body": RGB,
                "ContentType": "image/webp",
                "Metadata": {"fitting-job-id": str(JOB), "task-uuid": TASK, "sha256": digest},
                "IfNoneMatch": "*",
            },
        )
        result = store_runware_result_image(
            storage(client),
            fitting_job_id=JOB,
            task_uuid=TASK.upper(),
            image_url=URL,
            result_key=KEY,
            download=Net().download,
        )
        stub.assert_no_pending_responses()
    assert result == StoredResultImage(KEY, len(RGB), digest, reused=False)
    assert KEY.endswith(".webp") and KEY == f"virtual-fitting/results/{JOB}/pruna-async.webp"


@pytest.mark.parametrize(
    "overrides",
    [
        {"result_key": async_result_key(JOB + 1)},  # 다른 작업의 Key
        {"result_key": f"virtual-fitting/results/{JOB}/pruna-async.jpg"},
        {"result_key": f"virtual-fitting/results/{JOB}/pruna-sync.jpg"},
        {"result_key": f"virtual-fitting/results/{JOB}/other.webp"},
        {"task_uuid": "not-a-uuid"},
        {"fitting_job_id": 0},
        {"image_url": "https://evil.example/x.webp"},
    ],
)
def test_invalid_inputs_fail_before_any_s3_or_http_call(overrides):
    s3 = FakeS3()
    with pytest.raises(ValueError) as error:
        store(s3, no_download, **overrides)
    assert s3.calls == []
    assert "evil.example" not in str(error.value)


# === 9·12. 조건부 PUT·중복 실행 재사용(재다운로드 생략) ===


def test_second_run_reuses_the_object_without_downloading_again():
    s3 = FakeS3()
    first = store(s3)
    second = store(s3, no_download)  # 원본 URL 이 만료됐어도 된다
    assert second == StoredResultImage(KEY, first.byte_count, first.sha256, reused=True)
    assert len(s3.puts()) == 1


def test_existing_object_from_a_previous_run_is_reused_without_download():
    s3 = FakeS3()
    put_existing(s3, RGBA)
    result = store(s3, no_download)
    assert result.reused and result.sha256 == hashlib.sha256(RGBA).hexdigest()
    assert s3.puts() == [] and s3.objects[KEY]["Body"] == RGBA


def test_object_created_by_another_run_between_check_and_put_is_reused():
    s3 = FakeS3()
    # 확인 뒤·PUT 전에 다른 실행이 같은 task 결과를 저장했다.
    s3.before_put = lambda: put_existing(s3, RGBA)
    result = store(s3)
    assert result.reused and result.sha256 == hashlib.sha256(RGBA).hexdigest()
    assert s3.objects[KEY]["Body"] == RGBA  # 덮어쓰지 않았다


def test_conditional_write_conflict_409_is_confirmed_by_reading():
    s3 = FakeS3()
    s3.put_error = client_error(409, "ConditionalRequestConflict", "PutObject")
    s3.before_put = lambda: put_existing(s3, RGB)
    assert store(s3).reused is False  # 같은 bytes(같은 sha256)로 확인됐다
    s3_empty = FakeS3()
    s3_empty.put_error = client_error(409, "ConditionalRequestConflict", "PutObject")
    with pytest.raises(ResultImageError) as error:
        store(s3_empty)
    assert error.value.reason == "s3_put_unconfirmed" and error.value.retryable


# === 10. 다른 작업·출처 불명·검증 불충분 객체 덮어쓰기 방지 ===


@pytest.mark.parametrize(
    ("existing", "reason"),
    [
        ({"job": JOB + 1}, "existing_object_conflict"),
        ({"task": OTHER_TASK}, "existing_object_conflict"),
        ({"meta": {}}, "existing_object_conflict"),  # 출처 불명(예: Runware 가 직접 올린 객체)
        ({"meta": {"fitting-job-id": str(JOB), "task-uuid": TASK}}, "existing_object_unverified"),
        (
            {"meta": {"fitting-job-id": str(JOB), "task-uuid": TASK, "sha256": "0" * 64}},
            "existing_object_unverified",
        ),
        ({"content_type": "application/octet-stream"}, "existing_object_unverified"),
        ({"body": TRUNCATED}, "existing_object_unverified"),
        ({"body": animated_webp()}, "existing_object_unverified"),
        ({"body": encoded("PNG")}, "existing_object_unverified"),
    ],
)
def test_foreign_or_unverified_object_is_never_overwritten(existing, reason):
    s3 = FakeS3()
    put_existing(s3, existing.pop("body", RGB), **existing)
    before = dict(s3.objects[KEY])
    with pytest.raises(ResultImageError) as error:
        store(s3, no_download)
    assert error.value.reason == reason and not error.value.retryable
    assert s3.puts() == [] and s3.objects[KEY] == before


def test_foreign_object_appearing_before_put_is_not_overwritten():
    s3 = FakeS3()
    s3.before_put = lambda: put_existing(s3, RGB, task=OTHER_TASK)
    with pytest.raises(ResultImageError) as error:
        store(s3)
    assert error.value.reason == "existing_object_conflict"
    assert s3.objects[KEY]["Metadata"]["task-uuid"] == OTHER_TASK


def test_existing_check_errors_are_safe_and_do_not_download():
    s3 = FakeS3()
    s3.get_errors = [client_error(503, "SlowDown", "GetObject")]
    with pytest.raises(ResultImageError) as error:
        store(s3, no_download)
    assert error.value.reason == "s3_check_failed" and error.value.retryable
    assert s3.puts() == []


def test_access_denied_precheck_falls_back_to_the_conditional_put():
    # ListBucket 이 없으면 없는 객체도 403 이다. 조건부 PUT 이 덮어쓰기를 막으므로 진행한다.
    s3 = FakeS3()
    s3.get_errors = [client_error(403, "AccessDenied", "GetObject")]
    assert store(s3).reused is False and len(s3.puts()) == 1
    # 403 인데 실제로는 객체가 있었다면 412 → 확인(이번엔 읽힘) → 다른 task 면 덮어쓰지 않는다.
    s3 = FakeS3()
    put_existing(s3, RGB, task=OTHER_TASK)
    s3.get_errors = [client_error(403, "AccessDenied", "GetObject")]
    with pytest.raises(ResultImageError) as error:
        store(s3)
    assert error.value.reason == "existing_object_conflict"


def test_put_permission_error_is_reported():
    s3 = FakeS3()
    s3.put_error = client_error(403, "AccessDenied", "PutObject")
    with pytest.raises(ResultImageError) as error:
        store(s3)
    assert (error.value.reason, error.value.http_status) == ("s3_put_failed", 403)
    assert s3.objects == {}


# === 11. PUT 응답 유실 후 기존 객체 확인 ===


@pytest.mark.parametrize(
    "lost",
    [
        EndpointConnectionError(endpoint_url="https://s3"),
        ReadTimeoutError(endpoint_url="https://s3"),
        client_error(500, "InternalError", "PutObject"),
    ],
    ids=["connection", "read-timeout", "500"],
)
def test_lost_put_response_is_confirmed_by_reading_the_object(lost):
    s3 = FakeS3()
    s3.put_error, s3.store_then_fail = lost, True
    result = store(s3)
    assert result == StoredResultImage(KEY, len(RGB), hashlib.sha256(RGB).hexdigest(), False)
    assert len(s3.puts()) == 1  # 다시 PUT 하지 않았다


def test_lost_put_without_object_is_unconfirmed_not_retried():
    s3 = FakeS3()
    s3.put_error = EndpointConnectionError(endpoint_url="https://s3")
    with pytest.raises(ResultImageError) as error:
        store(s3)
    assert error.value.reason == "s3_put_unconfirmed" and error.value.retryable
    assert len(s3.puts()) == 1 and s3.objects == {}


def test_storage_method_outcomes():
    s3 = FakeS3()
    st = storage(s3)
    assert st.put_object_if_absent(RGB, KEY, content_type="image/webp", metadata={}).outcome is (
        ConditionalPutOutcome.CREATED
    )
    again = st.put_object_if_absent(b"other", KEY, content_type="image/webp", metadata={})
    assert (again.outcome, again.http_status) == (ConditionalPutOutcome.EXISTS, 412)
    assert s3.objects[KEY]["Body"] == RGB
    check, found = st.get_object_limited(KEY, max_bytes=10)
    assert check.exists and found.truncated and len(found.body) == 10
    assert "RIFF" not in repr(found)  # 본문 bytes 는 repr 에 없다


# === 13. URL·token·서명 비노출 ===


def assert_no_secret(error):
    text = "\n".join(
        [str(error), repr(error), "".join(traceback.format_exception(error)), str(vars(error))]
    )
    for secret in SECRETS:
        assert secret not in text, secret


def test_nothing_exposes_the_url(caplog):
    s3 = FakeS3()
    with caplog.at_level(logging.DEBUG):
        result = store(s3)
        store(s3, no_download)
        net = Net(FakeConnection(request_error=OSError(f"boom {URL}")))
        with pytest.raises(ResultImageError) as error:
            net.download(URL)
    assert_no_secret(error.value)
    text = "\n".join(r.getMessage() for r in caplog.records) + repr(result)
    text += repr(s3.objects[KEY]["Metadata"])
    for secret in SECRETS:
        assert secret not in text
    assert "runware result image stored" in text  # 처리 로그는 남는다(작업·task·Key·크기만)


# === 실제 연결 클래스: 127.0.0.1 임시 TLS 서버 ===


@pytest.fixture(scope="module")
def tls_cert(tmp_path_factory):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl 이 없다")
    folder = tmp_path_factory.mktemp("tls")
    cert, key = folder / "cert.pem", folder / "key.pem"
    done = subprocess.run(
        [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
         "-keyout", str(key), "-out", str(cert), "-subj", "/CN=im.runware.ai",
         "-addext", "subjectAltName=DNS:im.runware.ai"],
        capture_output=True,
        check=False,
    )  # fmt: skip
    if done.returncode != 0:
        pytest.skip("자체 서명 인증서를 만들 수 없다")
    return cert, key


@pytest.fixture
def tls_server(tls_cert):
    cert, key = tls_cert
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def drip(self, head, parts, gap):
            """head 를 보낸 뒤 parts 를 gap 초 간격으로 하나씩 보낸다(최대 3초, 클라이언트가 끊으면 멈춤)."""
            try:
                self.wfile.write(head)
                for part in parts:
                    time.sleep(gap)
                    self.wfile.write(part)
            except OSError:
                pass

        def do_GET(self):
            seen.append((self.path, self.headers.get("Host")))
            if self.path.startswith("/slow-headers"):
                lines = [f"X-Pad-{i}: x\r\n".encode() for i in range(30)]
                return self.drip(b"HTTP/1.1 200 OK\r\n", lines, 0.1)
            if self.path.startswith("/slow-body"):
                head = f"HTTP/1.1 200 OK\r\nContent-Length: {len(RGB)}\r\n\r\n".encode()
                return self.drip(head, [bytes([b]) for b in RGB][:30], 0.1)
            if self.path.startswith("/slow-start"):
                time.sleep(0.3)
            if self.path.startswith("/redirect"):
                self.send_response(302)
                self.send_header("Location", "https://im.runware.ai/final.webp")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(RGB)))
            self.end_headers()
            self.wfile.write(RGB)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client_context = ssl.create_default_context(cafile=str(cert))
    port = server.server_address[1]

    def connect(host, _port, address, deadline):
        return ri._PinnedHTTPSConnection(host, port, address, deadline, client_context)

    try:
        yield {"seen": seen, "connect": connect, "context": client_context, "port": port}
    finally:
        server.shutdown()
        server.server_close()


def test_pinned_connection_sends_original_target_and_host_and_verifies_tls(tls_server):
    body = download_runware_image(
        URL, resolve=lambda host, port: ["127.0.0.1"], connect=tls_server["connect"]
    )
    assert body == RGB
    # Host 는 원래 호스트명(연결한 IP 가 아님). 이 테스트 서버만 443 이 아니라 포트가 붙는다(실제 443 은 붙지 않음).
    assert tls_server["seen"] == [(TARGET, f"im.runware.ai:{tls_server['port']}")]


def test_pinned_connection_does_not_follow_redirects(tls_server):
    with pytest.raises(ResultImageError) as error:
        download_runware_image(
            "https://im.runware.ai/redirect?token=a",
            resolve=lambda host, port: ["127.0.0.1"],
            connect=tls_server["connect"],
        )
    assert error.value.reason == "redirect"
    assert [path for path, _ in tls_server["seen"]] == ["/redirect?token=a"]


def test_pinned_connection_rejects_a_certificate_for_another_host(tls_server):
    port = tls_server["port"]

    def connect(host, _port, address, deadline):
        # 인증서는 im.runware.ai 용인데 다른 이름으로 확인하면 실패해야 한다(호스트명 검증이 켜져 있다).
        return ri._PinnedHTTPSConnection(
            "other.example", port, address, deadline, tls_server["context"]
        )

    with pytest.raises(ResultImageError) as error:
        download_runware_image(URL, resolve=lambda host, port: ["127.0.0.1"], connect=connect)
    assert error.value.reason == "connection_failed"
    assert tls_server["seen"] == []


# --- 실제 소켓의 남은 시간 (127.0.0.1 TLS 서버, 1초 미만 대기) ---


@pytest.fixture
def socket_spy(monkeypatch):
    """_DeadlineSSLSocket 이 실제로 건 timeout 값과 소켓을 기록한다."""
    record = {"timeouts": [], "sockets": []}
    original_settimeout = ssl.SSLSocket.settimeout
    original_handshake = ri._DeadlineSSLSocket.do_handshake

    def settimeout(self, value):
        record["timeouts"].append(value)
        return original_settimeout(self, value)

    def do_handshake(self, block=False):
        record["sockets"].append(self)
        return original_handshake(self, block)

    monkeypatch.setattr(ri._DeadlineSSLSocket, "settimeout", settimeout, raising=False)
    monkeypatch.setattr(ri._DeadlineSSLSocket, "do_handshake", do_handshake)
    return record


def local(tls_server, path, timeout):
    started = time.monotonic()
    try:
        return download_runware_image(
            f"https://im.runware.ai{path}",
            resolve=lambda host, port: ["127.0.0.1"],
            connect=tls_server["connect"],
            timeout=timeout,
        )
    finally:
        local.elapsed = time.monotonic() - started


@pytest.mark.parametrize(
    "path", ["/slow-headers?token=tok-SECRET-123", "/slow-body?token=tok-SECRET-123"]
)
def test_dripping_headers_or_body_stop_at_the_total_limit(tls_server, socket_spy, path):
    """헤더 줄·본문 bytes 를 0.1초마다 보내면 recv 마다는 timeout 에 안 걸린다. 합계 0.6초에서 끊는다."""
    with pytest.raises(ResultImageError) as error:
        local(tls_server, path, timeout=0.6)
    assert error.value.reason == "timeout"
    # 정리·스케줄링 오차만 더해진다(수정 전: 헤더·본문을 3초 내내 받았다).
    assert 0.55 <= local.elapsed < 0.9
    (sock,) = socket_spy["sockets"]
    assert sock.fileno() == -1  # 응답이 넘겨받은 소켓까지 닫혔다
    assert_no_secret(error.value)


def test_real_reads_use_the_shrinking_remaining_time(tls_server, socket_spy):
    """응답이 0.3초 늦게 시작해도 이어지는 실제 recv 들의 timeout 은 남은 시간(줄어듦)이고 처음 예산을 넘지 않는다."""
    assert local(tls_server, "/slow-start", timeout=2.0) == RGB
    timeouts = socket_spy["timeouts"]
    assert len(timeouts) >= 3 and all(0 < value <= 2.0 for value in timeouts)
    # 단계·읽기마다 다시 늘어나지 않는다(TLS 소켓이 TCP 소켓 timeout 을 ns 로 반올림해 복사하는 오차만 허용).
    assert all(later <= earlier + 1e-6 for earlier, later in itertools.pairwise(timeouts))
    assert timeouts[-1] < 2.0 - 0.25  # 응답 대기 0.3초가 이후 읽기 예산에서 빠졌다
    (sock,) = socket_spy["sockets"]
    assert sock.fileno() == -1
