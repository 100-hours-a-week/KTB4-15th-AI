"""16단계 로컬 통합 검증용 외부 경계 대체(테스트 전용, pytest 가 수집하지 않는 파일).

실제 구현(app.main.app·lifespan·Router·요청/후처리 Worker·Submitter·Provider·Webhook·결과 전송 루프·Watchdog·
Repository·SqsQueueClient·S3ImageStorage)은 그대로 쓰고, 그 바깥 경계만 바꾼다.

- SQS: boto3.client("sqs") 자리에 FakeSqsBackend 의 client. 수신·발행·삭제·Visibility·재전달(receive count)·고장 주입만.
- S3: boto3.client("s3") 자리에 FakeS3Backend 의 client(put/head/presign). Presigned URL 은 로컬 문자열.
- Runware(생성 비동기·동기, 잔액, 사용량, getResponse, LLM)·결과 이미지 다운로드: 각 모듈의 urlopen 자리에 FakeRunware.
  실제 Provider 가 요청 JSON 을 만들고 응답 JSON 을 해석한다(Provider·Service 전체를 성공 Mock 으로 바꾸지 않음).
- 네트워크: AF_INET·AF_INET6 소켓 연결을 막는다(실수로 실제 AWS·Runware 로 나가지 않게). DB 는 Unix socket.
"""

import io
import json
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

REGION = "ap-northeast-2"
BUCKET = "test-bucket"
ACCOUNT = "123456789012"
QUEUE_URLS = {
    "be-fitting-request": f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT}/be-fitting-request",
    "ai-fitting-response": f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT}/ai-fitting-response",
    "be-fitting-response": f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT}/be-fitting-response",
}
WEBHOOK_TOKEN = "TESTWEBHOOKTOKEN"
WEBHOOK_URL = f"https://ai.example.test/api/v2/webhooks/runware?token={WEBHOOK_TOKEN}"
USER_IMAGE_KEY = "users/10/body.png"
PRODUCTS = (
    # product_code, image_url, main_category, sub_category, description_summary
    (
        "123456",
        "https://image.example.test/top.jpg",
        "상의",
        "반소매 티셔츠",
        "네이비 반소매 티셔츠",
    ),
    ("789012", "https://image.example.test/bottom.jpg", "하의", "슬림 팬츠", "중청 슬림 데님 팬츠"),
)
COMMENT = (
    "네이비 티셔츠와 중청 데님이 어우러진 캐주얼한 코디입니다.\n편안한 데일리룩으로 잘 어울립니다."
)
TITLE = "편안한 데일리 캐주얼"


def jpeg_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (20, 40, 80)).save(buffer, format="JPEG")
    return buffer.getvalue()


# --- 네트워크 차단 ---


class NetworkBlocked(OSError):
    pass


def block_network(monkeypatch) -> list:
    """AF_INET/AF_INET6 연결 시도를 막고 기록한다. Unix socket(PostgreSQL)·in-process TestClient 는 그대로."""
    attempts: list = []
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guard(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append(address)
            raise NetworkBlocked(f"network blocked in integration test: {address!r}")

    def connect(sock, address):
        guard(sock, address)
        return real_connect(sock, address)

    def connect_ex(sock, address):
        guard(sock, address)
        return real_connect_ex(sock, address)

    def create_connection(address, *args, **kwargs):
        attempts.append(address)
        raise NetworkBlocked(f"network blocked in integration test: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    return attempts


# --- SQS ---


def _client_error(code: str, status: int, operation: str):
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "Error": {"Code": code, "Message": "fake"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation,
    )


@dataclass
class _Message:
    message_id: str
    body: str
    sent_at: float
    visible_at: float = 0.0
    receive_count: int = 0
    receipt: str | None = None
    deleted: bool = False


class FakeSqsBackend:
    """세 큐의 메시지 저장소. 기본 Visibility 는 요청한 값(없으면 30초)을 실제 시각으로 따른다.

    faults[(큐 이름, 작업)] 에 차례로 쓸 고장을 넣는다:
      "throttled"        → ClientError(ThrottlingException, 400) — 저장 안 함(FAILED)
      "server_error"     → ClientError(InternalError, 500) — 저장 안 함(UNKNOWN 으로 분류됨)
      "stored_timeout"   → 저장한 뒤 ReadTimeoutError(UNKNOWN: 실제로는 들어감)
      "receipt_invalid"  → ClientError(ReceiptHandleIsInvalid, 400)
    """

    def __init__(self) -> None:
        self.queues: dict[str, list[_Message]] = {url: [] for url in QUEUE_URLS.values()}
        self.faults: dict[tuple[str, str], deque] = {}
        self.log: list[tuple[str, str, str]] = []  # (작업, 큐 이름, message_id)
        self._lock = threading.Lock()

    @staticmethod
    def name(url: str) -> str:
        return url.rsplit("/", 1)[-1]

    def fault(self, queue: str, operation: str, *items: str) -> None:
        self.faults.setdefault((queue, operation), deque()).extend(items)

    def _take_fault(self, url: str, operation: str) -> str | None:
        items = self.faults.get((self.name(url), operation))
        return items.popleft() if items else None

    # 테스트 조작
    def put(self, queue: str, payload: Any) -> str:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        message = _Message(str(uuid.uuid4()), body, time.time())
        with self._lock:
            self.queues[QUEUE_URLS[queue]].append(message)
        return message.message_id

    def live(self, queue: str) -> list[_Message]:
        with self._lock:
            return [m for m in self.queues[QUEUE_URLS[queue]] if not m.deleted]

    def bodies(self, queue: str) -> list[dict]:
        with self._lock:
            return [json.loads(m.body) for m in self.queues[QUEUE_URLS[queue]]]

    def expire_visibility(self, queue: str) -> None:
        """받아 간 메시지의 Visibility Timeout 이 지난 것처럼 다시 보이게 한다(재전달)."""
        with self._lock:
            for message in self.queues[QUEUE_URLS[queue]]:
                message.visible_at = 0.0

    def client(self) -> "FakeSqsClient":
        return FakeSqsClient(self)


class FakeSqsClient:
    def __init__(self, backend: FakeSqsBackend) -> None:
        self._b = backend

    def send_message(self, QueueUrl, MessageBody):
        from botocore.exceptions import ReadTimeoutError

        b = self._b
        fault = b._take_fault(QueueUrl, "send")
        if fault == "throttled":
            raise _client_error("ThrottlingException", 400, "SendMessage")
        if fault == "server_error":
            raise _client_error("InternalError", 500, "SendMessage")
        message = _Message(str(uuid.uuid4()), MessageBody, time.time())
        with b._lock:
            b.queues[QueueUrl].append(message)
            b.log.append(("send", b.name(QueueUrl), message.message_id))
        if fault == "stored_timeout":
            raise ReadTimeoutError(endpoint_url=QueueUrl)
        return {"MessageId": message.message_id}

    def receive_message(self, QueueUrl, MaxNumberOfMessages=1, WaitTimeSeconds=0, **kwargs):
        b = self._b
        visibility = kwargs.get("VisibilityTimeout", 30)
        deadline = time.time() + min(WaitTimeSeconds, 0.2)  # long polling 은 짧게만 흉내 낸다
        while True:
            now = time.time()
            out = []
            with b._lock:
                for message in b.queues[QueueUrl]:
                    if len(out) >= MaxNumberOfMessages:
                        break
                    if message.deleted or message.visible_at > now:
                        continue
                    message.receive_count += 1
                    message.receipt = f"rh-{uuid.uuid4().hex}"
                    message.visible_at = now + visibility
                    out.append(
                        {
                            "MessageId": message.message_id,
                            "ReceiptHandle": message.receipt,
                            "Body": message.body,
                            "Attributes": {"ApproximateReceiveCount": str(message.receive_count)},
                        }
                    )
                    b.log.append(("receive", b.name(QueueUrl), message.message_id))
            if out or now >= deadline:
                return {"Messages": out} if out else {}
            time.sleep(0.02)

    def _find(self, url, receipt):
        for message in self._b.queues[url]:
            if message.receipt == receipt:
                return message
        return None

    def delete_message(self, QueueUrl, ReceiptHandle):
        b = self._b
        if b._take_fault(QueueUrl, "delete") == "receipt_invalid":
            raise _client_error("ReceiptHandleIsInvalid", 400, "DeleteMessage")
        with b._lock:
            message = self._find(QueueUrl, ReceiptHandle)
            if message is not None:
                message.deleted = True
                b.log.append(("delete", b.name(QueueUrl), message.message_id))
        return {}

    def change_message_visibility(self, QueueUrl, ReceiptHandle, VisibilityTimeout):
        b = self._b
        if b._take_fault(QueueUrl, "visibility") == "receipt_invalid":
            raise _client_error("ReceiptHandleIsInvalid", 400, "ChangeMessageVisibility")
        with b._lock:
            message = self._find(QueueUrl, ReceiptHandle)
            if message is None or message.deleted:
                raise _client_error("ReceiptHandleIsInvalid", 400, "ChangeMessageVisibility")
            message.visible_at = time.time() + VisibilityTimeout
        return {}


# --- S3 ---


class FakeS3Backend:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str, datetime]] = {}
        self.heads: list[str] = []
        self.presigned: list[tuple[str, str]] = []  # (method, key)
        self.head_faults: deque = deque()  # "forbidden" / "server_error"
        self._lock = threading.Lock()

    def store(self, key: str, body: bytes | None = None, content_type: str = "image/jpeg") -> None:
        with self._lock:
            self.objects[key] = (body or jpeg_bytes(), content_type, datetime.now(UTC))

    def client(self) -> "FakeS3Client":
        return FakeS3Client(self)


class FakeS3Client:
    def __init__(self, backend: FakeS3Backend) -> None:
        self._b = backend

    def put_object(self, Bucket, Key, Body, ContentType=None, **kwargs):
        assert Bucket == BUCKET
        self._b.store(Key, Body, ContentType or "application/octet-stream")
        return {"ETag": '"fake"'}

    def head_object(self, Bucket, Key):
        assert Bucket == BUCKET
        b = self._b
        b.heads.append(Key)
        if b.head_faults:
            fault = b.head_faults.popleft()
            raise _client_error(
                "403" if fault == "forbidden" else "500",
                403 if fault == "forbidden" else 500,
                "HeadObject",
            )
        with b._lock:
            item = b.objects.get(Key)
        if item is None:
            raise _client_error("404", 404, "HeadObject")
        body, content_type, modified = item
        return {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "ContentLength": len(body),
            "ContentType": content_type,
            "LastModified": modified,
        }

    def generate_presigned_url(self, ClientMethod, Params, ExpiresIn):
        self._b.presigned.append((ClientMethod, Params["Key"]))
        return (
            f"https://{Params['Bucket']}.s3.{REGION}.amazonaws.com/{Params['Key']}"
            f"?X-Amz-Signature=FAKESIGNATURE&X-Amz-Expires={ExpiresIn}&method={ClientMethod}"
        )


def key_of(presigned_url: str) -> str:
    return urlparse(presigned_url).path.lstrip("/")


# --- Runware HTTP ---


class _Response(io.BytesIO):
    def __init__(self, payload: Any, status: int = 200) -> None:
        super().__init__(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        self.status = status
        self.headers: dict = {}

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(url: str, code: int, body: Any = None, headers: dict | None = None) -> HTTPError:
    raw = json.dumps(body if body is not None else {"errors": [{"code": "fake"}]}).encode()
    return HTTPError(url, code, "fake", headers or {}, io.BytesIO(raw))


@dataclass
class Submission:
    task_uuid: str
    payload: dict
    upload_key: str
    webhook_url: str
    person_key: str


@dataclass
class FakeRunware:
    """api.runware.ai·im.runware.ai 자리. 각 모드는 deque 로 차례로 쓰고 비면 기본 동작이다."""

    s3: FakeS3Backend
    balance: str = "10.00"
    usage_total: str = "0"
    async_modes: deque = field(
        default_factory=deque
    )  # "accept" | ("http", code) | "timeout" | ("errors", [...]) | callable
    sync_modes: deque = field(
        default_factory=deque
    )  # "success" | ("http", code) | "timeout" | ("no_cost",)
    comment_modes: deque = field(default_factory=deque)  # str | ("http", code)
    title_modes: deque = field(default_factory=deque)
    statuses: dict = field(
        default_factory=dict
    )  # task_uuid → getResponse 응답 dict 또는 ("http", code)
    sync_cost: float = 0.015
    calls: list = field(default_factory=list)  # (kind, detail)
    submissions: dict = field(default_factory=dict)  # task_uuid → Submission
    llm_requests: list = field(default_factory=list)
    gate: threading.Event | None = None  # 비동기 접수 응답 전에 기다릴 때(종료 검증)
    entered: threading.Event = field(default_factory=threading.Event)

    def __call__(self, request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        host = urlparse(url).netloc
        if host == "im.runware.ai":
            self.calls.append(("download", url))
            return _Response(jpeg_bytes())
        if url.endswith("/chat/completions"):
            return self._llm(url, json.loads(request.data))
        if host != "api.runware.ai":
            raise URLError(f"unexpected host in integration test: {host}")
        tasks = json.loads(request.data)
        task = tasks[0]
        kind = task.get("taskType")
        if kind == "accountManagement" and task.get("operation") == "getDetails":
            self.calls.append(("balance", None))
            item = {
                "taskType": "accountManagement",
                "operation": "getDetails",
                "balance": {"amount": float(self.balance), "freeBalance": 0, "currency": "USD"},
            }
            return _Response({"data": [item]})
        if kind == "accountManagement" and task.get("operation") == "getUsageActivity":
            self.calls.append(("usage", None))
            spend = float(self.usage_total)
            return _Response(
                {"data": [{"usage": {"timeseries": {"data": [], "meta": {"totalSpend": spend}}}}]}
            )
        if kind == "getResponse":
            return self._status(url, task["taskUUID"])
        if kind == "imageInference" and task.get("deliveryMethod") == "async":
            return self._async(url, task)
        if kind == "imageInference":
            return self._sync(url, task)
        raise URLError(f"unexpected Runware task in integration test: {kind}")

    # 비동기 접수
    def _async(self, url, task):
        submission = Submission(
            task["taskUUID"],
            task,
            key_of(task["uploadEndpoint"]),
            task["webhookURL"],
            key_of(task["inputs"]["referenceImages"][0]["image"]),
        )
        self.submissions[submission.task_uuid] = submission
        self.calls.append(("async_submit", submission.task_uuid))
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        mode = self.async_modes.popleft() if self.async_modes else "accept"
        if callable(mode):
            mode = mode(submission) or "accept"
        if mode == "accept":
            return _Response(
                {
                    "data": [
                        {
                            "taskType": "imageInference",
                            "taskUUID": submission.task_uuid,
                            "status": "processing",
                        }
                    ]
                }
            )
        if mode == "timeout":
            raise TimeoutError("fake timeout")
        if mode[0] == "http":
            raise _http_error(url, mode[1])
        if mode[0] == "errors":
            return _Response({"errors": mode[1]})
        raise AssertionError(mode)

    # V1 동기(fallback)
    def _sync(self, url, task):
        self.calls.append(("sync_generate", task.get("taskUUID")))
        mode = self.sync_modes.popleft() if self.sync_modes else "success"
        if mode == "timeout":
            raise TimeoutError("fake timeout")
        if isinstance(mode, tuple) and mode[0] == "http":
            raise _http_error(url, mode[1])
        item = {
            "taskType": "imageInference",
            "taskUUID": task.get("taskUUID"),
            "imageURL": f"https://im.runware.ai/image/fake/{uuid.uuid4().hex}.jpg",
        }
        if mode != ("no_cost",):
            item["cost"] = self.sync_cost
        return _Response({"data": [item]})

    def _status(self, url, task_uuid):
        self.calls.append(("get_response", task_uuid))
        result = self.statuses.get(task_uuid)
        if result is None:
            result = {
                "data": [
                    {"taskType": "imageInference", "taskUUID": task_uuid, "status": "processing"}
                ]
            }
        if isinstance(result, tuple) and result[0] == "http":
            raise _http_error(url, result[1])
        return _Response(result)

    def _llm(self, url, payload):
        self.llm_requests.append(payload)
        user = payload["messages"][1]["content"]
        is_comment = isinstance(user, list)
        self.calls.append(("llm_comment" if is_comment else "llm_title", None))
        modes = self.comment_modes if is_comment else self.title_modes
        mode = modes.popleft() if modes else (COMMENT if is_comment else TITLE)
        if isinstance(mode, tuple):
            raise _http_error(url, mode[1], {"error": {"message": "fake"}})
        return _Response(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": mode},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    # Runware 쪽 완료 흉내
    def complete(self, task_uuid: str, *, put: bool = True, cost: float | None = 0.023) -> bytes:
        """결과를 uploadEndpoint 로 PUT 하고(put=True) 성공 Webhook 본문을 돌려준다."""
        submission = self.submissions[task_uuid]
        if put:
            self.s3.store(submission.upload_key)
        body = {
            "taskType": "imageInference",
            "taskUUID": task_uuid,
            "imageUUID": str(uuid.uuid4()),
            "imageURL": "https://im.runware.ai/image/fake/result.jpg",
        }
        if cost is not None:
            body["cost"] = cost
        return json.dumps(body).encode()

    @staticmethod
    def failure(task_uuid: str, code: str) -> bytes:
        return json.dumps(
            {
                "errors": [
                    {
                        "taskUUID": task_uuid,
                        "code": code,
                        "status": "error",
                        "message": "fake external message",
                    }
                ]
            }
        ).encode()

    def success_status(self, task_uuid: str, cost: float = 0.023) -> None:
        self.statuses[task_uuid] = {
            "data": [
                {
                    "taskType": "imageInference",
                    "taskUUID": task_uuid,
                    "status": "success",
                    "cost": cost,
                }
            ]
        }

    def count(self, kind: str) -> int:
        return sum(1 for k, _ in self.calls if k == kind)


def webhook_token_from(url: str) -> str:
    return parse_qs(urlparse(url).query)["token"][0]


RUNWARE_MODULES = (
    "app.virtual_fitting.providers.runware",
    "app.virtual_fitting.providers.runware_async",
    "app.virtual_fitting.providers.runware_balance",
    "app.virtual_fitting.providers.runware_usage",
    "app.virtual_fitting.providers.runware_status",
    "app.virtual_fitting.providers.runware_comment",
    "app.clients.s3",
)


# --- 실행 환경(격리 PostgreSQL schema + 설정 + 외부 경계 교체) ---

ROOT_SQL = "scripts/db"
MINIMAL_USAGE_SQL = """
-- schema.sql 의 usage_amount 초기 정의와 같은 컬럼(테스트 DB 에 pgvector 가 없어 schema.sql 전체는 실행하지 못한다)
CREATE TABLE IF NOT EXISTS usage_amount (
    usage_date date PRIMARY KEY, amount numeric NOT NULL DEFAULT 0, reconciled_at timestamptz
)
"""


def minimal_products_sql() -> str:
    """schema.sql 의 products CREATE 를 그대로 쓰고, ALTER 중 vector(pgvector) 컬럼만 뺀다(최소 테이블 대체)."""
    text = Path(f"{ROOT_SQL}/schema.sql").read_text(encoding="utf-8")
    create = text[text.index("CREATE TABLE IF NOT EXISTS products") :]
    create = create[: create.index(");") + 2]
    alter = text[text.index("ALTER TABLE products") :]
    alter = alter[: alter.index(";") + 1]
    alter = "\n".join(line for line in alter.splitlines() if "vector(" not in line).rstrip(",\n ")
    if not alter.endswith(";"):
        alter = alter.rstrip(",") + ";"
    return create + "\n" + alter


class V2Env:
    """한 테스트의 외부 상태(DB schema·가짜 SQS/S3/Runware)는 유지하고, 프로세스(=pool·조립 객체)는 새로 만들 수 있다."""

    def __init__(self, monkeypatch, url: str, schema: str) -> None:
        import psycopg

        self.monkeypatch, self.url, self.schema = monkeypatch, url, schema
        self.options = f"-c search_path={schema}"
        self.sqs, self.s3 = FakeSqsBackend(), FakeS3Backend()
        self.runware = FakeRunware(self.s3)
        self.pools: list = []
        self.app_pools: list = []  # 앱(프로세스)이 쓰는 pool 만, 만든 순서대로
        self._psycopg = psycopg

    # DB
    def sql(self, statement: str, params: Any = None):
        with self._psycopg.connect(self.url, autocommit=True, options=self.options) as connection:
            cursor = connection.execute(statement, params)
            return cursor.fetchall() if cursor.description else None

    def new_process_pool(self):
        """프로세스 재시작 가정: 새 connection pool 을 app 전역 pool 로 둔다(이전 객체·메모리 상태 없음)."""
        from psycopg_pool import ConnectionPool

        from app.config import database

        pool = ConnectionPool(
            self.url, min_size=1, max_size=12, kwargs={"options": self.options}, open=True
        )
        self.pools.append(pool)
        self.app_pools.append(pool)
        self.monkeypatch.setattr(database, "_pool", pool)
        self.clear_clients()
        return pool

    @staticmethod
    def clear_clients() -> None:
        from app.clients import s3 as s3_module
        from app.clients import sqs as sqs_module

        for fn in (
            sqs_module.get_sqs_client,
            sqs_module.get_webhook_sqs_client,
            s3_module.get_s3_client,
            s3_module.get_s3_presign_client,
        ):
            fn.cache_clear()

    def job(self, job_id: int):
        """검증용 조회. 앱 pool 이 아니라 테스트 전용 pool 을 쓴다(앱 종료로 pool 이 닫혀도 확인할 수 있게)."""
        from psycopg_pool import ConnectionPool

        from app.virtual_fitting.repositories.async_job_repository import AsyncJobRepository

        if getattr(self, "_inspect_pool", None) is None:
            self._inspect_pool = ConnectionPool(
                self.url, min_size=1, max_size=2, kwargs={"options": self.options}, open=True
            )
            self.pools.append(self._inspect_pool)
        return AsyncJobRepository(self._inspect_pool).find_by_fitting_job_id(job_id)

    def request(self, job_id: int, codes=("123456", "789012"), image_key=USER_IMAGE_KEY) -> dict:
        return {
            "fitting_job_id": job_id,
            "requested_at": "2026-10-18T03:00:00Z",
            "user_image_key": image_key,
            "products": [{"product_code": code} for code in codes],
        }

    def send_request(self, job_id: int, **kwargs) -> str:
        return self.sqs.put("be-fitting-request", self.request(job_id, **kwargs))


def make_env(monkeypatch, url: str, schema: str) -> V2Env:
    import boto3

    from app.config import settings
    from app.virtual_fitting.result_publisher import register_result_publisher
    from app.virtual_fitting.watchdog import register_watchdog

    env = V2Env(monkeypatch, url, schema)
    for name in (
        "virtual_fitting_async_jobs.sql",
        "virtual_fitting_fallback.sql",
        "virtual_fitting_auth_hold.sql",
        "virtual_fitting_pruna_limits.sql",
    ):
        env.sql(Path(f"{ROOT_SQL}/{name}").read_text(encoding="utf-8"))
    env.sql(MINIMAL_USAGE_SQL)
    env.sql(Path(f"{ROOT_SQL}/usage_amount_split_totals.sql").read_text(encoding="utf-8"))
    env.sql(Path(f"{ROOT_SQL}/virtual_fitting_watchdog.sql").read_text(encoding="utf-8"))
    env.sql(minimal_products_sql())
    for code, image_url, main, sub, summary in PRODUCTS:
        env.sql(
            "INSERT INTO products (product_code, id, product_name, price, detail_url, image_url, "
            "main_category, sub_category, is_sold_out, crawled_at, description_summary) "
            "VALUES (%s, 1, %s, 10000, %s, %s, %s, %s, false, now(), %s)",
            (int(code), sub, f"https://shop.example.test/{code}", image_url, main, sub, summary),
        )
    env.s3.store(USER_IMAGE_KEY, content_type="image/png")

    # 설정: 테스트에서만. 기본값(비활성화 등)은 settings 파일에서 바꾸지 않는다.
    values = {
        "AWS_REGION": REGION,
        "S3_BUCKET": BUCKET,
        "BE_FITTING_REQUEST_QUEUE_URL": QUEUE_URLS["be-fitting-request"],
        "AI_FITTING_RESPONSE_QUEUE_URL": QUEUE_URLS["ai-fitting-response"],
        "BE_FITTING_RESPONSE_QUEUE_URL": QUEUE_URLS["be-fitting-response"],
        "WEBHOOK_TOKEN": WEBHOOK_TOKEN,
        "VIRTUAL_FITTING_RUNWARE_WEBHOOK_URL": WEBHOOK_URL,
        "CHECKPOINT_DSN": "",
        # 한 테스트 안에서 여러 작업을 빨리 보내려고 최소 간격만 0(공유 제한 검증은 따로 실제 값으로).
        "VIRTUAL_FITTING_PRUNA_MIN_SEND_INTERVAL_SECONDS": 0.0,
        "VIRTUAL_FITTING_RESULT_PUBLISHER_POLL_SECONDS": 0.05,
        "VIRTUAL_FITTING_WATCHDOG_INTERVAL_SECONDS": 0.2,
        "VIRTUAL_FITTING_WATCHDOG_STATUS_MIN_INTERVAL_SECONDS": 0.0,
    }
    for name, value in values.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "test-vton-key")
    monkeypatch.setenv("RUNWARE_LLM_API_KEY", "test-llm-key")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    real_client = boto3.client

    def fake_boto_client(service, *args, **kwargs):
        if service == "sqs":
            return env.sqs.client()
        if service == "s3":
            return env.s3.client()
        raise AssertionError(f"unexpected boto3 client in integration test: {service}")

    monkeypatch.setattr(boto3, "client", fake_boto_client)
    import importlib

    for module_name in RUNWARE_MODULES:
        monkeypatch.setattr(importlib.import_module(module_name), "urlopen", env.runware)
    env.network_attempts = block_network(monkeypatch)
    env.real_boto_client = real_client
    register_result_publisher(None)
    register_watchdog(None)
    env.new_process_pool()
    return env


def close_env(env: V2Env) -> None:
    from app.virtual_fitting.result_publisher import register_result_publisher
    from app.virtual_fitting.watchdog import register_watchdog

    register_result_publisher(None)
    register_watchdog(None)
    for pool in env.pools:
        if not pool.closed:
            pool.close()
    env.clear_clients()


# --- 실제 조립 객체를 한 단계씩 돌리기(재시작마다 새로 만든다) ---


class Stack:
    """create_request_worker·create_postprocess_worker·create_result_publisher·create_watchdog 로 만든 실제 객체.

    루프(asyncio Task)는 시작하지 않고, 루프가 한 번에 하는 일(수신 → 처리 → 허용될 때만 삭제 / 결과 전송 cycle /
    Watchdog run_once)을 테스트가 차례로 부른다. mark_running() 은 결과 전송 루프·Watchdog 객체를 '실행 중'으로
    등록해 실제 준비 판단(FollowUpPaths)이 그 상태를 보게 한다(실제 루프 실행 검증은 lifespan 테스트가 따로 한다).
    """

    def __init__(self, env: V2Env, handlers=None) -> None:
        from app.virtual_fitting.postprocess_worker import (
            connected_postprocess_handlers,
            create_postprocess_worker,
        )
        from app.virtual_fitting.request_worker import create_request_worker
        from app.virtual_fitting.result_publisher import create_result_publisher
        from app.virtual_fitting.watchdog import create_watchdog

        self.env = env
        self.request = create_request_worker()
        self.postprocess = create_postprocess_worker(handlers or connected_postprocess_handlers())
        self.publisher = create_result_publisher()
        self.watchdog = create_watchdog(dict)

    def mark_running(self, publisher: bool = True, watchdog: bool = True) -> "Stack":
        from app.virtual_fitting.result_publisher import register_result_publisher
        from app.virtual_fitting.watchdog import register_watchdog

        self.publisher.status.running = publisher
        self.watchdog.status.running = watchdog
        register_result_publisher(self.publisher if publisher else None)
        register_watchdog(self.watchdog if watchdog else None)
        return self

    @staticmethod
    def _drain(worker) -> list:
        batch = worker._receive()
        outcomes = []
        for item in batch.messages:
            outcome = worker._process(item)
            if outcome.delete:
                worker._delete(item.receipt_handle)
            outcomes.append(outcome)
        return outcomes

    def request_once(self) -> list:
        return self._drain(self.request)

    def postprocess_once(self) -> list:
        return self._drain(self.postprocess)

    def publish_once(self) -> int:
        return self.publisher._cycle()

    def watchdog_once(self):
        return self.watchdog._run_once()


def post_webhook(client, body: bytes, token: str = WEBHOOK_TOKEN):
    return client.post(
        "/api/v2/webhooks/runware",
        params={"token": token},
        content=body,
        headers={"Content-Type": "application/json"},
    )


def wait_until(predicate, timeout: float = 15.0, interval: float = 0.05, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}")
