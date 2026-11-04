"""AI Server(remote) 실행 2·대기 2 정책. 기본값은 remote runtime 에만 걸리고 local 모드·Modal worker 는 그대로다."""

import importlib
import logging
import re
import time
from collections import Counter
from contextlib import contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.body_image_validation import router as validation_router
from app.body_image_validation.diagnostics import REQUEST_ID_HEADER
from app.body_image_validation.exceptions import SYSTEM_FAILURE_MESSAGE
from app.body_image_validation.runtime import BodyImageRuntimeManager, RemoteBodyImageRuntime
from app.config import body_image_validation as config
from app.config import settings
from tests.test_body_image_concurrent_api import Api, HeldWorker, WorkerSlot
from tests.test_body_image_modal_worker import PRODUCTION_GPU, load, load_production
from tests.test_body_image_queue_deadline import BUSY, wait_until
from tests.test_body_image_remote import SUCCESS, TIMEOUT, common
from tests.test_body_image_validation_service import service as local_service
from tests.test_body_image_warmup_api import remote, runtime  # noqa: F401 (pytest fixture)

LIMIT_ENV = ("MAX_CONCURRENT_BODY_VALIDATIONS", "MAX_WAITING_BODY_VALIDATIONS", "BODY_IMAGE_MAX_QUEUE_WAIT_SECONDS")
SUMMARY = re.compile(r"^body image request: request_id=(?P<request_id>\S+) outcome=(?P<outcome>\S+)")


# --- 설정: 같은 환경변수, 역할별 기본값 ---


@contextmanager
def config_from(monkeypatch, **env):
    """동시성 환경변수를 env 만 둔 상태로 config 를 다시 읽는다. 나오면 원래 환경으로 다시 읽는다."""
    try:
        with monkeypatch.context() as scoped:
            for name in LIMIT_ENV:
                scoped.delenv(name, raising=False)
            for name, value in env.items():
                scoped.setenv(name, value)
            importlib.reload(config)
            scoped.setattr(config, "REMOTE_URL", "https://example.modal.run")
            for name in ("BODY_IMAGE_REMOTE_PROXY_KEY", "BODY_IMAGE_REMOTE_PROXY_SECRET", "BODY_IMAGE_REMOTE_API_KEY"):
                scoped.setattr(settings, name, "test")
            yield
    finally:
        importlib.reload(config)


def remote_limits():
    limiter = RemoteBodyImageRuntime().service.limiter
    return limiter._max_running, limiter._max_waiting, limiter.max_wait_seconds


def local_limits():
    limiter = local_service().limiter
    return limiter._max_running, limiter._max_waiting, limiter.max_wait_seconds


def test_without_env_only_the_ai_server_runs_two_and_queues_two(monkeypatch):
    with config_from(monkeypatch):
        assert remote_limits() == (2, 2, 10.0)
        assert local_limits() == (1, 1, None)  # EC2 local·Modal worker 는 기존 그대로, 대기 시간 제한 없음
        assert RemoteBodyImageRuntime().service.total_deadline == 120.0


def test_explicit_env_wins_over_the_ai_server_defaults(monkeypatch):
    with config_from(monkeypatch, MAX_CONCURRENT_BODY_VALIDATIONS="1", MAX_WAITING_BODY_VALIDATIONS="1"):
        assert remote_limits() == (1, 1, 10.0)


def test_explicit_env_still_applies_to_local_mode(monkeypatch):
    with config_from(monkeypatch, MAX_CONCURRENT_BODY_VALIDATIONS="3", MAX_WAITING_BODY_VALIDATIONS="0",
                     BODY_IMAGE_MAX_QUEUE_WAIT_SECONDS="5"):
        assert local_limits() == (3, 0, None)
        assert remote_limits() == (3, 0, 5.0)


@pytest.mark.parametrize(
    "deploy_env",
    [
        {"BODY_IMAGE_WORKER_STAGE": "production"},
        {"BODY_IMAGE_WORKER_STAGE": "production", "BODY_IMAGE_WORKER_MAX_INPUTS": "2"},
        {**PRODUCTION_GPU, "BODY_IMAGE_WORKER_MAX_INPUTS": "2"},
        {"BODY_IMAGE_WORKER_STAGE": "staging", "BODY_IMAGE_WORKER_MAX_INPUTS": "2"},
    ],
    ids=["production-default", "production-2-cpu-snapshot", "production-2-gpu-snapshot", "staging-2"],
)
def test_worker_deploy_never_passes_a_queue_size(deploy_env):
    worker = load(**deploy_env)

    assert "MAX_WAITING_BODY_VALIDATIONS" not in worker.STAGE_ENV.get("env", {})


@pytest.mark.parametrize(("max_inputs", "expected"), [("1", (1, 1, None)), ("2", (2, 1, None))])
def test_worker_container_limiter_follows_its_inputs_and_keeps_one_waiting(monkeypatch, max_inputs, expected):
    worker = load_production(BODY_IMAGE_WORKER_MAX_INPUTS=max_inputs)
    container_env = worker.STAGE_ENV.get("env", {})

    with config_from(monkeypatch, **container_env):
        assert worker.MAX_INPUTS == int(max_inputs)
        assert local_limits() == expected  # AI Server 의 기본 대기 2 는 worker 에 오지 않는다


# --- 요청 처리: 실행 2 + 대기 2 ---


@pytest.fixture
def two_by_two(remote, monkeypatch):  # noqa: F811
    built = []

    def start(queue_wait=10.0, deadline=120.0):
        monkeypatch.setattr(config, "REMOTE_MAX_CONCURRENT_BODY_VALIDATIONS", 2)
        monkeypatch.setattr(config, "REMOTE_MAX_WAITING_BODY_VALIDATIONS", 2)
        monkeypatch.setattr(config, "MAX_QUEUE_WAIT_SECONDS", queue_wait)
        monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", deadline)
        monkeypatch.setattr(config, "REMOTE_READ_TIMEOUT_SECONDS", deadline)
        slot = WorkerSlot()
        manager = BodyImageRuntimeManager(remote_transport=slot)
        monkeypatch.setattr(main, "runtime_manager", manager)
        monkeypatch.setattr(validation_router, "runtime_manager", manager)
        client = TestClient(main.app, raise_server_exceptions=False)
        client.__enter__()
        built.append(Api(client, manager, slot))
        return built[-1]

    yield start
    for api in built:
        for worker in api.held:
            worker.released.set()
        api.client.__exit__(None, None, None)


def fill(api, worker):
    """실행 2 를 Modal 까지 보내고 대기 2 를 채운다."""
    running = [api.start(), api.start()]
    wait_until(lambda: worker.arrivals == 2)
    waiting = [api.start(), api.start()]
    wait_until(lambda: api.limiter.waiting == 2)
    return running, waiting


def summaries(caplog):
    return [m.groupdict() for r in caplog.records if (m := SUMMARY.match(r.getMessage()))]


def test_five_requests_run_two_queue_two_and_reject_the_fifth_at_once(two_by_two, caplog):
    api = two_by_two()  # 운영값: 대기 10초, 전체 120초
    worker = api.install(HeldWorker())

    with caplog.at_level(logging.INFO, logger="app.body_image_validation"):
        running, waiting = fill(api, worker)
        started = time.monotonic()
        fifth = api.post()
        rejected_in = time.monotonic() - started

        assert (fifth.status_code, fifth.json()) == (429, BUSY)
        assert rejected_in < 1.0  # 10초를 기다리지 않고 바로 거절했다
        assert (worker.arrivals, api.limiter.running, api.limiter.waiting) == (2, 2, 2)

        worker.released.set()
        responses = [call.result() for call in running + waiting]

    assert [r.status_code for r in responses] == [200] * 4
    assert worker.arrivals == 4  # 대기하던 두 요청은 슬롯을 얻은 뒤 Modal 을 불렀다
    wait_until(lambda: (api.limiter.running, api.limiter.waiting) == (0, 0))
    lines = summaries(caplog)
    assert Counter(line["outcome"] for line in lines) == {"response": 4, "queue_full": 1}
    # 요청 ID 는 요청마다 따로 만들어져 그 요청의 Modal 호출과 요약 로그에만 쓰였다.
    sent = [call.headers[REQUEST_ID_HEADER] for call in worker.calls]
    assert len(set(sent)) == 4
    assert sorted(sent) == sorted(line["request_id"] for line in lines if line["outcome"] == "response")


def test_waiters_give_up_after_the_queue_wait_and_the_slots_come_back(two_by_two, caplog):
    queue_wait = 0.5
    api = two_by_two(queue_wait=queue_wait, deadline=3.0)
    worker = api.install(HeldWorker())

    with caplog.at_level(logging.INFO, logger="app.body_image_validation"):
        running = [api.start(), api.start()]
        wait_until(lambda: worker.arrivals == 2)
        started = time.monotonic()
        waiting = [api.start(), api.start()]
        gave_up = [call.result() for call in waiting]
        waited = time.monotonic() - started

    assert [(r.status_code, r.json()) for r in gave_up] == [(429, BUSY)] * 2
    assert queue_wait * 0.9 <= waited < 3.0
    assert worker.arrivals == 2
    assert Counter(line["outcome"] for line in summaries(caplog)) == {"queue_timeout": 2}
    worker.released.set()
    assert [call.result().status_code for call in running] == [200, 200]
    wait_until(lambda: (api.limiter.running, api.limiter.waiting) == (0, 0))
    assert api.post().status_code == 200


def test_failed_and_timed_out_calls_free_their_slots_for_the_waiters(two_by_two):
    api = two_by_two()
    failed = common("BACKGROUND_REMOVAL_FAILED", SYSTEM_FAILURE_MESSAGE)
    worker = api.install(HeldWorker(steps=[(0, 500, {}, failed), httpx.ReadTimeout("timed out"), (0, 200, {}, SUCCESS)]))
    running, waiting = fill(api, worker)

    worker.released.set()

    assert sorted(call.result().status_code for call in running) == [500, 504]
    assert [call.result().status_code for call in waiting] == [200, 200]
    wait_until(lambda: (api.limiter.running, api.limiter.waiting) == (0, 0))


def test_queue_wait_counts_against_the_total_deadline(two_by_two):
    api = two_by_two(queue_wait=10.0, deadline=1.0)
    worker = api.install(HeldWorker())
    running, waiting = fill(api, worker)
    started = time.monotonic()

    timed_out = [call.result() for call in waiting]

    # 대기 제한(10초)보다 전체 deadline(1초)이 먼저 끝나 504 다.
    assert [(r.status_code, r.json()) for r in timed_out] == [(504, TIMEOUT)] * 2
    assert time.monotonic() - started < 2.0
    assert [call.result().status_code for call in running] == [504, 504]
    wait_until(lambda: (api.limiter.running, api.limiter.waiting) == (0, 0))
