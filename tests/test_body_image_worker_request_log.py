"""Modal worker 의 요청별 요약 로그. 진단이 꺼져 있어도 AI Server 의 request_id 로 요청을 이을 수 있어야 한다."""

import contextlib
import json
import logging
import re
import threading

import pytest

from app.body_image_validation.concurrency import BodyValidationLimiter
from app.body_image_validation.diagnostics import REQUEST_ID_HEADER
from app.body_image_validation.exceptions import BodyImageSystemError
from app.body_image_validation.service import BodyImageValidationService
from tests.test_body_image_deadline import started_worker, worker_client
from tests.test_body_image_validation_service import (
    FakeBackgroundRemover,
    FakePersonDetector,
    FakePoseDetector,
    FakeStorage,
    image_bytes,
)
from tests.test_body_image_worker import (
    AUTH,
    URL,
    jpeg,
    worker_env,  # noqa: F401 (pytest fixture)
)

SERVICE_LOGGER = "app.body_image_validation.service"
LINE = re.compile(
    r"^body image validation request: request_id=(?P<request_id>[0-9a-f]{16}) "
    r"ai_request_id=(?P<ai_request_id>\S+) outcome=(?P<outcome>\S+) total=(?P<total>\d+\.\d{3})s$"
)
AI_ID, OTHER_AI_ID = "85c390e8344842ad", "0123456789abcdef"


def request_lines(caplog):
    return [LINE.match(r.getMessage()).groupdict() for r in caplog.records
            if r.getMessage().startswith("body image validation request:")]


def diagnostic_lines(caplog):
    prefix = "body image validation diagnostics: "
    return [json.loads(r.getMessage()[len(prefix):]) for r in caplog.records if r.getMessage().startswith(prefix)]


@pytest.fixture
def keep_caplog(monkeypatch):
    """worker.create_app 의 dictConfig 가 테스트 중에 caplog handler 를 root 에서 떼지 않게 한다."""
    from app.config import logging_config

    monkeypatch.setattr(logging_config, "configure_logging", lambda: None)


# --- 진단이 꺼진 worker(production 기본) ---


def post(client, body, request_id=None):
    headers = dict(AUTH)
    if request_id is not None:
        headers[REQUEST_ID_HEADER] = request_id
    return client.post(URL, files={"image": ("body.jpg", body, "image/jpeg")}, headers=headers)


def test_without_diagnostics_each_request_leaves_one_line_with_both_ids(worker_env, keep_caplog, caplog):  # noqa: F811
    started_worker(worker_env)

    with caplog.at_level(logging.INFO, logger="app.body_image_validation"):
        response = post(worker_client(), jpeg(), AI_ID)

    assert response.status_code == 200
    [line] = request_lines(caplog)
    assert (line["ai_request_id"], line["outcome"]) == (AI_ID, "success")
    assert float(line["total"]) >= 0
    assert diagnostic_lines(caplog) == []  # 상세 진단(RSS·VRAM·Lock)은 켜지지 않는다
    s3_key = response.json()["data"]["s3_key"]
    for private in (s3_key, AUTH["Authorization"].split()[-1]):
        assert private not in caplog.text


@pytest.mark.parametrize("header", [None, "not-an-id", "0" * 15, AI_ID.upper()])
def test_missing_or_malformed_ai_id_is_logged_as_a_dash(worker_env, keep_caplog, caplog, header):  # noqa: F811
    started_worker(worker_env)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        assert post(worker_client(), jpeg(), header).status_code == 200

    [line] = request_lines(caplog)
    assert line["ai_request_id"] == "-"


def test_failures_log_the_code_without_the_message(worker_env, keep_caplog, caplog):  # noqa: F811
    started_worker(worker_env)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        response = post(worker_client(), b"not an image", AI_ID)

    [line] = request_lines(caplog)
    assert (line["ai_request_id"], line["outcome"]) == (AI_ID, response.json()["code"])
    assert response.json()["message"] not in caplog.text


def test_each_request_gets_its_own_worker_id(worker_env, keep_caplog, caplog):  # noqa: F811
    started_worker(worker_env)
    client = worker_client()

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        post(client, jpeg(), AI_ID)
        post(client, jpeg(), AI_ID)

    first, second = request_lines(caplog)
    assert first["request_id"] != second["request_id"]


# --- 동시 요청: 한 요청의 ID 가 다른 요청의 줄에 섞이지 않는다 ---


class OverlappingRemover(FakeBackgroundRemover):
    """두 요청이 모두 배경 제거에 들어온 뒤에야 진행한다. failing thread 의 요청만 실패한다."""

    def __init__(self, failing_thread):
        super().__init__()
        self.both = threading.Barrier(2, timeout=10)
        self.failing_thread = failing_thread

    def remove(self, image):
        self.both.wait()
        if threading.current_thread().name == self.failing_thread:
            raise RuntimeError("remover failed")
        return super().remove(image)


@pytest.mark.parametrize("diagnostics", [False, True], ids=["plain", "diagnostics"])
def test_overlapping_requests_keep_their_own_ids_and_outcomes(caplog, diagnostics):
    remover = OverlappingRemover(failing_thread=OTHER_AI_ID)
    service = BodyImageValidationService(
        FakePersonDetector(), FakePoseDetector(), remover, FakeStorage(),
        limiter=BodyValidationLimiter(2, 0), diagnostics=diagnostics,
    )

    def call(ai_request_id):
        with contextlib.suppress(BodyImageSystemError):  # 실패한 요청은 로그만 본다
            service.validate(image_bytes(), ai_request_id)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        threads = [threading.Thread(target=call, args=(ai,), name=ai) for ai in (AI_ID, OTHER_AI_ID)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

    lines = {line["ai_request_id"]: line for line in request_lines(caplog)}
    assert {ai: line["outcome"] for ai, line in lines.items()} == {
        AI_ID: "success", OTHER_AI_ID: "BACKGROUND_REMOVAL_FAILED"}
    assert lines[AI_ID]["request_id"] != lines[OTHER_AI_ID]["request_id"]
    if diagnostics:
        # 상세 진단 줄도 같은 worker request_id 로 이어진다.
        detailed = {line["ai_request_id"]: line for line in diagnostic_lines(caplog)}
        for ai in (AI_ID, OTHER_AI_ID):
            assert detailed[ai]["request_id"] == lines[ai]["request_id"]
            assert detailed[ai]["outcome"] == lines[ai]["outcome"]
            assert detailed[ai]["execution"]["active_max"] == 2  # 실제로 겹쳐 실행됐다
