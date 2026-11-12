import hashlib
import http.client
import ipaddress
import logging
import socket
import ssl
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.parse import urlsplit

from PIL import Image

from app.clients.s3 import (
    DEFAULT_DOWNLOAD_TIMEOUT,
    MAX_REMOTE_IMAGE_BYTES,
    ConditionalPutOutcome,
    ImageStorageError,
    S3ImageStorage,
    S3ObjectState,
    validate_s3_key,
)
from app.config.body_image_validation import MAX_IMAGE_PIXELS
from app.virtual_fitting.async_storage import async_result_key
from app.virtual_fitting.repositories.async_job_repository import check_runware_image_url

logger = logging.getLogger(__name__)

WEBP_CONTENT_TYPE = "image/webp"
_CHUNK_BYTES = 64 * 1024
# 경과시간 시계(벽시계 변경 영향 없음). 테스트는 이것만 바꾼다(전역 time.monotonic 은 건드리지 않음).
_clock = time.monotonic
# S3 일반 Metadata(x-amz-meta-*). 이 함수가 만든 객체인지·어느 작업·task 의 결과인지·내용이 그대로인지 확인할 최소 정보.
META_JOB = "fitting-job-id"
META_TASK = "task-uuid"
META_SHA256 = "sha256"

# 오류 원인 코드 → 다시 시도하면 달라질 수 있는지. 메시지·속성에 URL·token 을 담지 않는다.
_RETRYABLE = {
    "dns_failed": True,
    "connection_failed": True,
    "timeout": True,
    "http_error": False,  # 5xx·429 는 http_status 로 따로 본다(retryable 속성)
    "redirect": False,
    "address_not_allowed": False,
    "too_large": False,
    "empty": False,
    "not_webp": False,
    "animated": False,
    "too_many_pixels": False,
    "corrupt": False,
    "s3_put_failed": False,
    "s3_put_unconfirmed": True,
    "s3_check_failed": True,
    "s3_access_denied": False,
    "existing_object_conflict": False,
    "existing_object_unverified": False,
    # 배경 제거 결과 검증(transparent_image). 같은 입력으로 다시 만들어도 달라진다고 볼 근거가 없다.
    "no_alpha": False,
    "no_transparency": False,
    "no_foreground": False,
}


class ResultImageError(ImageStorageError):
    """결과 이미지 다운로드·검증·저장 실패. reason 은 안전한 원인 코드다(URL·token·서명·응답 본문 없음)."""

    def __init__(
        self, reason: str, *, http_status: int | None = None, detail: str | None = None
    ) -> None:
        super().__init__(f"가상피팅 결과 이미지 처리에 실패했습니다: {reason}")
        self.reason = reason
        self.http_status = http_status
        # 하위 예외의 타입 이름 등(값·메시지는 담지 않음).
        self.detail = detail

    @property
    def retryable(self) -> bool:
        if self.reason == "http_error":
            return self.http_status is not None and (
                self.http_status >= 500 or self.http_status == 429
            )
        return _RETRYABLE.get(self.reason, False)


@dataclass(frozen=True)
class StoredResultImage:
    """저장(또는 재사용)한 결과 이미지의 확인 정보. URL 은 담지 않는다."""

    key: str
    byte_count: int
    sha256: str
    # True 면 이번에 PUT 하지 않고 같은 작업·task 의 기존 객체를 확인해 재사용했다.
    reused: bool


@dataclass(frozen=True)
class ValidatedWebp:
    width: int
    height: int
    mode: str


# --- 다운로드 ---


def resolve_public_addresses(
    host: str, port: int, *, getaddrinfo: Callable[..., Any] = socket.getaddrinfo
) -> list[str]:
    """host 의 주소 목록. 하나라도 공인(global) 주소가 아니면(사설·loopback·link-local·예약·multicast) 거절한다."""
    try:
        infos = getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        raise ResultImageError("dns_failed") from None
    addresses = list(dict.fromkeys(info[4][0] for info in infos))
    if not addresses:
        raise ResultImageError("dns_failed")
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if not ip.is_global or ip.is_multicast:
            raise ResultImageError("address_not_allowed")
    return addresses


def _remaining(deadline: float) -> float:
    """종료 시각까지 남은 초. 남지 않았으면 TimeoutError(새 연결·요청·읽기를 시작하지 않는다)."""
    remaining = deadline - _clock()
    if remaining <= 0:
        raise TimeoutError("download deadline exceeded")
    return remaining


class _DeadlineSSLSocket(ssl.SSLSocket):
    """recv·send·TLS 핸드셰이크 직전마다 소켓 timeout 을 종료 시각까지 남은 시간으로 다시 건다.

    http.client 는 헤더를 줄마다, 본문을 BufferedReader 로 여러 번 recv 한다. 소켓 timeout 은 recv 1번마다 새로 적용되므로
    고정값이면 조금씩 보내는 응답에 제한이 계속 연장된다. 응답 객체로 넘어간 뒤(Connection: close 면 연결 객체의 sock 은
    None 이 된다)에도 같은 소켓이 읽으므로 여기서 걸어야 실제 읽기에 적용된다.
    """

    _deadline: float | None = None

    def _arm(self) -> None:
        if self._deadline is not None:
            self.settimeout(_remaining(self._deadline))

    def do_handshake(self, block: bool = False) -> None:
        self._arm()
        super().do_handshake(block)

    def recv(self, buflen: int = 1024, flags: int = 0) -> bytes:
        self._arm()
        return super().recv(buflen, flags)

    def recv_into(self, buffer: Any, nbytes: int | None = None, flags: int = 0) -> int:
        self._arm()
        return super().recv_into(buffer, nbytes, flags)

    def send(self, data: Any, flags: int = 0) -> int:  # sendall 도 이 send 를 반복해 부른다
        self._arm()
        return super().send(data, flags)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """검증한 IP 로만 연결한다. TLS 인증서 확인·SNI·Host 헤더는 원래 호스트명으로 한다(DNS 재조회 없음).

    TCP 연결·TLS 핸드셰이크·요청 전송·헤더·본문 수신이 모두 하나의 종료 시각(deadline)을 나눠 쓴다.
    """

    def __init__(self, host: str, port: int, address: str, deadline: float, context: Any) -> None:
        super().__init__(host, port, timeout=_remaining(deadline), context=context)
        self._address, self._deadline = address, deadline
        self._context.sslsocket_class = _DeadlineSSLSocket

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), _remaining(self._deadline))
        try:
            tls = self._context.wrap_socket(
                sock, server_hostname=self.host, do_handshake_on_connect=False
            )
        except BaseException:
            sock.close()
            raise
        try:
            tls._deadline = self._deadline
            tls.do_handshake()
        except BaseException:
            tls.close()
            raise
        self.sock = tls


def _connect(host: str, port: int, address: str, deadline: float) -> Any:
    return _PinnedHTTPSConnection(host, port, address, deadline, ssl.create_default_context())


def _request_target(url: str) -> str:
    """URL 의 경로·쿼리 원문(token·서명 포함). 삭제·재정렬·재인코딩하지 않는다. fragment 는 보내지 않는다."""
    rest = url[len("https://") + len(urlsplit(url).netloc) :]
    target = rest.split("#", 1)[0]
    return target if target.startswith("/") else "/" + target


def download_runware_image(
    url: str,
    *,
    resolve: Callable[[str, int], list[str]] = resolve_public_addresses,
    connect: Callable[[str, int, str, float], Any] = _connect,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    max_bytes: int = MAX_REMOTE_IMAGE_BYTES,
) -> bytes:
    """허용된 Runware 결과 URL 을 한 번 GET 한다(리다이렉트·재시도 없음). 200 의 본문 bytes 를 돌려준다.

    timeout 은 DNS 조회 직전부터 본문 끝까지 하나의 예산이다(단계마다 다시 주지 않음). connect(host, port, address,
    deadline) 는 같은 종료 시각을 받아 TCP·TLS·전송·수신에 남은 시간만 쓴다. DNS(getaddrinfo)는 중단할 수 없어 끝난 뒤
    시간을 확인한다: 넘었으면 연결을 시작하지 않는다.
    """
    try:
        check_runware_image_url(url)
    except ValueError:
        raise ResultImageError("address_not_allowed") from None
    parts = urlsplit(url)
    host, port = parts.hostname, parts.port or 443
    target = _request_target(url)
    deadline = _clock() + timeout
    address = resolve(host, port)[0]
    connection = response = None
    try:
        try:
            _remaining(deadline)  # 늦게 끝난 DNS 결과로 연결을 시작하지 않는다
            connection = connect(host, port, address, deadline)
            _remaining(deadline)
            connection.request(
                "GET",
                target,
                headers={"User-Agent": "LookDDak-AI/1.0", "Accept": "image/webp,*/*"},
            )
            _remaining(deadline)
            response = connection.getresponse()
        except TimeoutError:
            raise ResultImageError("timeout") from None
        except (OSError, http.client.HTTPException, UnicodeError) as error:
            raise ResultImageError("connection_failed", detail=type(error).__name__) from None
        status = response.status
        if 300 <= status < 400:
            # Location 은 따라가지도 남기지도 않는다.
            raise ResultImageError("redirect", http_status=status)
        if status != 200:
            raise ResultImageError("http_error", http_status=status)
        declared = response.getheader("Content-Length")
        if declared is not None and declared.isdigit() and int(declared) > max_bytes:
            raise ResultImageError("too_large")
        return _read_limited(response, max_bytes, deadline)
    finally:
        # timeout·오류에도 응답·연결(소켓)을 닫는다. 응답이 소켓을 넘겨받았으면 응답을 닫아야 소켓이 닫힌다.
        for resource in (response, connection):
            if resource is not None:
                resource.close()


def _read_limited(response: Any, max_bytes: int, deadline: float) -> bytes:
    """Content-Length 와 관계없이 실제로 받은 bytes 로 상한을 본다. 읽기마다 남은 시간을 확인하고(실제 소켓은
    _DeadlineSSLSocket 이 recv 마다 남은 시간으로 제한), 종료 시각을 넘겨 받은 본문은 결과로 쓰지 않는다."""
    chunks, received = [], 0
    while True:
        try:
            _remaining(deadline)
            chunk = response.read(min(_CHUNK_BYTES, max_bytes + 1 - received))
        except TimeoutError:
            raise ResultImageError("timeout") from None
        except (OSError, http.client.HTTPException) as error:
            raise ResultImageError("connection_failed", detail=type(error).__name__) from None
        if not chunk:
            break
        chunks.append(chunk)
        received += len(chunk)
        if received > max_bytes:
            raise ResultImageError("too_large")
    if _clock() > deadline:
        raise ResultImageError("timeout")
    body = b"".join(chunks)
    if not body:
        raise ResultImageError("empty")
    return body


# --- 검증 ---


def validate_static_webp(body: bytes, *, max_pixels: int = MAX_IMAGE_PIXELS) -> ValidatedWebp:
    """실제 bytes 가 정상 정지 WebP 인지. 전체 픽셀을 디코딩해 손상을 확인한다. body 는 바꾸지 않는다."""
    if not body:
        raise ResultImageError("empty")
    try:
        with Image.open(BytesIO(body)) as image:
            if image.format != "WEBP":
                raise ResultImageError("not_webp")
            if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1:
                raise ResultImageError("animated")
            width, height = image.size
            if width < 1 or height < 1 or width * height > max_pixels:
                raise ResultImageError("too_many_pixels")
            image.load()
            return ValidatedWebp(width, height, image.mode)
    except ResultImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ResultImageError("too_many_pixels") from None
    except (OSError, ValueError, SyntaxError, EOFError) as error:
        # UnidentifiedImageError(OSError) 포함: 형식을 알 수 없거나 본문이 깨졌다.
        reason = "not_webp" if type(error).__name__ == "UnidentifiedImageError" else "corrupt"
        raise ResultImageError(reason, detail=type(error).__name__) from None


# --- 저장 ---


def _check_inputs(fitting_job_id: int, task_uuid: str, result_key: str) -> tuple[str, str]:
    expected = async_result_key(fitting_job_id)  # 양의 정수가 아니면 ValueError
    try:
        task = str(uuid.UUID(task_uuid))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("task_uuid 는 UUID 문자열이어야 합니다.") from error
    key = validate_s3_key(result_key)
    if key != expected or not key.endswith(".webp"):
        raise ValueError("result_key 는 이 작업의 예정 WebP 결과 Key 여야 합니다.")
    return task, key


def _metadata(fitting_job_id: int, task: str, digest: str) -> dict[str, str]:
    return {META_JOB: str(fitting_job_id), META_TASK: task, META_SHA256: digest}


def _existing(
    storage: S3ImageStorage,
    key: str,
    fitting_job_id: int,
    task: str,
    *,
    max_bytes: int,
    validate: Callable[[bytes], Any] = validate_static_webp,
) -> StoredResultImage | None:
    """같은 Key 의 기존 객체를 확인한다. 없으면 None. 이 작업·task 의 정상 결과로 확인되면 그 정보.

    객체 존재·확장자·LastModified 만으로는 판단하지 않는다. Metadata(작업·task·sha256)·실제 bytes 의 sha256·Content-Type·
    길이·정상 정지 WebP(validate, 투명 결과는 Alpha 검증까지)가 모두 맞아야 한다. 다른 작업·출처 불명이면
    existing_object_conflict, 그 밖에 근거가 부족하면 existing_object_unverified. 객체를 고치거나 지우지 않는다.
    """
    check, found = storage.get_object_limited(key, max_bytes)
    if check.state is S3ObjectState.NOT_FOUND:
        return None
    if check.state is S3ObjectState.ACCESS_DENIED:
        # ListBucket 이 없으면 없는 객체도 403 이다. 있는지 모르는 상태로 둔다(호출부가 정한다).
        raise ResultImageError("s3_access_denied", http_status=check.http_status)
    if check.state is not S3ObjectState.EXISTS or found is None:
        raise ResultImageError(
            "s3_check_failed", http_status=check.http_status, detail=check.error_code
        )
    meta = {name.lower(): value for name, value in found.metadata.items()}
    if meta.get(META_JOB) != str(fitting_job_id) or meta.get(META_TASK) != task:
        raise ResultImageError("existing_object_conflict")
    digest = hashlib.sha256(found.body).hexdigest()
    if (
        found.truncated
        or found.content_type != WEBP_CONTENT_TYPE
        or found.content_length != len(found.body)
        or meta.get(META_SHA256) != digest
    ):
        raise ResultImageError("existing_object_unverified")
    try:
        validate(found.body)
    except ResultImageError:
        raise ResultImageError("existing_object_unverified") from None
    return StoredResultImage(key, len(found.body), digest, reused=True)


def store_runware_result_image(
    storage: S3ImageStorage,
    *,
    fitting_job_id: int,
    task_uuid: str,
    image_url: str,
    result_key: str,
    download: Callable[[str], bytes] | None = None,
    max_bytes: int = MAX_REMOTE_IMAGE_BYTES,
) -> StoredResultImage:
    """Runware 결과 이미지를 result_key 에 원본 bytes 그대로 저장하거나, 이미 저장된 같은 결과를 재사용한다.

    입력이 잘못되면 ValueError(외부 호출 전). 다운로드·검증·저장 실패는 ResultImageError(reason·retryable).
    DB·lease 는 다루지 않는다. 호출부(3단계)가 실행 권한을 쥔 채 부르고 결과를 기록한다.
    """
    task, key = _check_inputs(fitting_job_id, task_uuid, result_key)
    try:
        check_runware_image_url(image_url)
    except ValueError:
        raise ValueError("image_url 이 허용된 Runware 결과 URL 이 아닙니다.") from None

    try:
        existing = _existing(storage, key, fitting_job_id, task, max_bytes=max_bytes)
    except ResultImageError as error:
        if error.reason != "s3_access_denied":
            raise
        existing = None  # 403: 없는지 있는지 모른다. 아래 조건부 PUT 이 덮어쓰기를 막는다
    if existing is not None:
        _log(fitting_job_id, task, "reused_existing", existing)
        return existing

    # 기본은 download_runware_image. 호출 때 모듈에서 찾는다(통합 테스트가 외부 경계만 바꿀 수 있게).
    body = (download or download_runware_image)(image_url)
    validate_static_webp(body)
    digest = hashlib.sha256(body).hexdigest()
    put = storage.put_object_if_absent(
        body,
        key,
        content_type=WEBP_CONTENT_TYPE,
        metadata=_metadata(fitting_job_id, task, digest),
    )
    if put.outcome is ConditionalPutOutcome.CREATED:
        stored = StoredResultImage(key, len(body), digest, reused=False)
        _log(fitting_job_id, task, "created", stored)
        return stored
    if put.outcome is ConditionalPutOutcome.FAILED:
        raise ResultImageError("s3_put_failed", http_status=put.http_status, detail=put.error_code)
    # EXISTS(412)·CONFLICT(409)·UNKNOWN(응답 유실): 다시 확인한다. 우리 요청이 만든 것이든 다른 실행이 만든 것이든
    # 이 작업·task 의 정상 결과면 재사용한다. 확인 자체가 안 되면 그 원인(s3_check_failed·s3_access_denied 등)을 올린다.
    confirmed = _existing(storage, key, fitting_job_id, task, max_bytes=max_bytes)
    if confirmed is None:  # 409·응답 유실 뒤에도 객체가 없다(아직 안 보이거나 저장되지 않음)
        raise ResultImageError(
            "s3_put_unconfirmed", http_status=put.http_status, detail=put.outcome.value
        )
    stored = StoredResultImage(
        key, confirmed.byte_count, confirmed.sha256, reused=confirmed.sha256 != digest
    )
    _log(fitting_job_id, task, f"confirmed_after_{put.outcome.value}", stored)
    return stored


def _log(fitting_job_id: int, task: str, result: str, stored: StoredResultImage) -> None:
    logger.info(
        "runware result image stored fitting_job_id=%s task_uuid=%s result=%s key=%s bytes=%s "
        "reused=%s",
        fitting_job_id,
        task,
        result,
        stored.key,
        stored.byte_count,
        stored.reused,
    )
