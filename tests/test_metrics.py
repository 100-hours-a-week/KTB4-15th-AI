import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.clients.sqs import FittingQueue, SqsOperation, SqsOutcome, SqsResult
from app.metrics import setup_metrics
from app.virtual_fitting.async_request import RequestBatch
from app.virtual_fitting.request_worker import RequestOutcome, RequestWorker
from tests.test_virtual_fitting_request_worker import batch, valid_message


def metric_value(name: str, labels: dict[str, str]) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return 0 if value is None else value


def test_metrics_endpoint_exposes_http_metrics_and_is_hidden_from_schema():
    app = FastAPI()
    setup_metrics(app)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        first = client.get("/metrics")
        second = client.get("/metrics")
        schema = client.get("/openapi.json").json()

    assert first.status_code == 200
    assert first.headers["content-type"].startswith("text/plain")
    assert 'http_requests_total{handler="/health",method="GET",status="200"}' in first.text
    assert "http_request_duration_seconds_bucket" in first.text
    assert "/metrics" not in schema["paths"]
    assert 'handler="/metrics"' not in second.text


def test_worker_metrics_record_processing_delete_and_receive_failure():
    worker_labels = {"worker": "request", "queue": "be-fitting-request"}
    processed_labels = {**worker_labels, "action": "test_done"}
    delete_succeeded_labels = {**worker_labels, "result": "succeeded"}
    delete_failed_labels = {**worker_labels, "result": "failed"}
    received_before = metric_value("ai_worker_messages_received_total", worker_labels)
    processed_before = metric_value("ai_worker_messages_processed_total", processed_labels)
    delete_succeeded_before = metric_value("ai_worker_delete_total", delete_succeeded_labels)
    delete_failed_before = metric_value("ai_worker_delete_total", delete_failed_labels)
    errors_before = metric_value("ai_worker_receive_errors_total", worker_labels)
    saw_in_progress = []
    calls = 0

    def receive():
        nonlocal calls
        calls += 1
        if calls == 1:
            return batch(
                valid_message(message_id="metrics-success"),
                valid_message(message_id="metrics-failure"),
            )
        if calls == 2:
            return RequestBatch(
                SqsResult(
                    SqsOperation.RECEIVE,
                    FittingQueue.BE_FITTING_REQUEST,
                    SqsOutcome.FAILED,
                    reason="access_denied",
                )
            )
        return batch()

    def process(item):
        saw_in_progress.append(metric_value("ai_worker_in_progress", worker_labels))
        return RequestOutcome(item.message_id, 1, "test_done", delete=True)

    def delete(receipt):
        outcome = SqsOutcome.FAILED if receipt.endswith("failure") else SqsOutcome.SUCCEEDED
        return SqsResult(SqsOperation.DELETE, FittingQueue.BE_FITTING_REQUEST, outcome)

    async def scenario():
        worker = RequestWorker(
            receive,
            process,
            delete,
            idle_seconds=0.01,
            backoff_initial_seconds=0.01,
            backoff_max_seconds=0.01,
        )
        worker.start()
        await asyncio.sleep(0.08)
        await worker.stop(timeout=1)

    asyncio.run(scenario())

    assert metric_value("ai_worker_messages_received_total", worker_labels) == received_before + 2
    assert metric_value("ai_worker_messages_processed_total", processed_labels) == processed_before + 2
    assert (
        metric_value("ai_worker_delete_total", delete_succeeded_labels)
        == delete_succeeded_before + 1
    )
    assert metric_value("ai_worker_delete_total", delete_failed_labels) == delete_failed_before + 1
    assert metric_value("ai_worker_receive_errors_total", worker_labels) >= errors_before + 1
    assert saw_in_progress == [1, 1]
    assert metric_value("ai_worker_in_progress", worker_labels) == 0
    assert metric_value("ai_worker_running", worker_labels) == 0
    assert metric_value("ai_worker_crashed", worker_labels) == 0


def test_worker_metrics_expose_unexpected_crash_and_release_in_progress_gauge():
    worker_labels = {"worker": "request", "queue": "be-fitting-request"}
    exception_labels = {**worker_labels, "action": "exception"}
    exceptions_before = metric_value("ai_worker_messages_processed_total", exception_labels)

    def crash(_item):
        raise RuntimeError("unexpected processor failure")

    async def scenario():
        worker = RequestWorker(lambda: batch(valid_message()), crash, lambda receipt: None)
        with pytest.raises(RuntimeError, match="unexpected processor failure"):
            await worker.start()
        await asyncio.sleep(0)

    asyncio.run(scenario())

    assert metric_value("ai_worker_messages_processed_total", exception_labels) == (
        exceptions_before + 1
    )
    assert metric_value("ai_worker_in_progress", worker_labels) == 0
    assert metric_value("ai_worker_running", worker_labels) == 0
    assert metric_value("ai_worker_crashed", worker_labels) == 1
