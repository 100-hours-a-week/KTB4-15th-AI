"""Prometheus HTTP 및 가상피팅 Worker 지표."""

from fastapi import FastAPI, Request
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

WORKER_RUNNING = Gauge(
    "ai_worker_running",
    "Worker가 현재 실행 중이면 1, 아니면 0.",
    ("worker", "queue"),
)
WORKER_CRASHED = Gauge(
    "ai_worker_crashed",
    "Worker가 비정상 종료됐으면 1, 아니면 0.",
    ("worker", "queue"),
)
WORKER_IN_PROGRESS = Gauge(
    "ai_worker_in_progress",
    "현재 처리 중인 메시지 수.",
    ("worker", "queue"),
)
WORKER_MESSAGES_RECEIVED = Counter(
    "ai_worker_messages_received",
    "Worker가 처리 대상으로 받은 메시지 수.",
    ("worker", "queue"),
)
WORKER_MESSAGES_PROCESSED = Counter(
    "ai_worker_messages_processed",
    "Worker의 action별 메시지 처리 결과 수.",
    ("worker", "queue", "action"),
)
WORKER_MESSAGE_PROCESSING_SECONDS = Histogram(
    "ai_worker_message_processing_seconds",
    "메시지 하나를 처리하는 데 걸린 시간(초).",
    ("worker", "queue"),
)
WORKER_RECEIVE_ERRORS = Counter(
    "ai_worker_receive_errors",
    "SQS 메시지 수신 실패 횟수.",
    ("worker", "queue"),
)
WORKER_DELETE = Counter(
    "ai_worker_delete",
    "SQS 메시지 삭제 결과 수.",
    ("worker", "queue", "result"),
)
WORKER_CONSECUTIVE_ERRORS = Gauge(
    "ai_worker_consecutive_errors",
    "Worker의 연속 수신 오류 횟수.",
    ("worker", "queue"),
)
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_inprogress",
    "현재 처리 중인 HTTP 요청 수.",
)


def setup_metrics(app: FastAPI) -> None:
    """FastAPI HTTP 자동 계측과 인증 없는 ``/metrics``를 등록한다."""
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=[r"^/metrics$"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    @app.middleware("http")
    async def track_requests_in_progress(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        HTTP_REQUESTS_IN_PROGRESS.inc()
        try:
            return await call_next(request)
        finally:
            HTTP_REQUESTS_IN_PROGRESS.dec()


def worker_started(worker: str, queue: str) -> None:
    WORKER_RUNNING.labels(worker, queue).set(1)
    WORKER_CRASHED.labels(worker, queue).set(0)
    WORKER_CONSECUTIVE_ERRORS.labels(worker, queue).set(0)


def worker_stopped(worker: str, queue: str, *, crashed: bool) -> None:
    WORKER_RUNNING.labels(worker, queue).set(0)
    WORKER_CRASHED.labels(worker, queue).set(1 if crashed else 0)


def worker_receive_failed(worker: str, queue: str, consecutive_errors: int) -> None:
    WORKER_RECEIVE_ERRORS.labels(worker, queue).inc()
    WORKER_CONSECUTIVE_ERRORS.labels(worker, queue).set(consecutive_errors)


def worker_receive_succeeded(worker: str, queue: str) -> None:
    WORKER_CONSECUTIVE_ERRORS.labels(worker, queue).set(0)


def message_received(worker: str, queue: str) -> None:
    WORKER_MESSAGES_RECEIVED.labels(worker, queue).inc()


def message_processing_started(worker: str, queue: str) -> None:
    WORKER_IN_PROGRESS.labels(worker, queue).inc()


def message_processing_finished(
    worker: str,
    queue: str,
    action: str,
    elapsed_seconds: float,
) -> None:
    WORKER_IN_PROGRESS.labels(worker, queue).dec()
    WORKER_MESSAGES_PROCESSED.labels(worker, queue, action).inc()
    WORKER_MESSAGE_PROCESSING_SECONDS.labels(worker, queue).observe(elapsed_seconds)


def message_delete_finished(worker: str, queue: str, result: str) -> None:
    WORKER_DELETE.labels(worker, queue, result).inc()
