"""AI Server deadline 을 worker 로 전하는 X-Body-Image-Deadline 과, worker 가 deadline 이 지난 요청을 멈추는 지점.

세 상태를 따로 본다.
- HTTP 연결 취소: AI Server 가 deadline 에 연결을 끊고 504 를 돌려준다(remote.py, asyncio.timeout).
- Modal 작업 취소 요청: 보내지 않는다(HTTP endpoint 로는 공식 방법이 없다). 로그에 remote_cancel=not_requested.
- 원격 작업 실제 종료: AI Server 는 알 수 없다(remote_stopped=unknown). worker 가 header 를 보고 다음 단계 전에
  스스로 멈추는 것만 worker 쪽 테스트로 확인한다.
"""

import logging
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.body_image_validation import deadline, router, worker
from app.body_image_validation.concurrency import BodyValidationLimiter
from app.body_image_validation.service import BodyImageValidationService
from app.config import settings
from tests.test_body_image_remote import (
    POLL,
    SCALE,
    SUCCESS,
    TIMEOUT,
    FakeWorker,
    jpeg,
    redirect,
    use,
)
from tests.test_body_image_remote import client as remote_client  # noqa: F401 (pytest fixture)
from tests.test_body_image_remote import post as remote_post
from tests.test_body_image_worker import (  # noqa: F401 (pytest fixture)
    AUTH,
    FakePersonDetector,
    FakePoseDetector,
    FakeRemover,
    worker_env,
)

URL = "/api/v1/validation/body-image"


def deadline_service(fake, deadline_seconds, wall_start=1_000_000.0):
    from app.body_image_validation.remote import RemoteBodyImageValidationService

    return RemoteBodyImageValidationService(
        "https://example--lookddak-body-image-staging.modal.run",
        proxy_key="wk-test",
        proxy_secret="ws-test",
        api_key="worker-key",
        timeout=deadline_seconds,
        total_deadline=deadline_seconds,
        limiter=BodyValidationLimiter(1, 0),
        transport=fake.transport,
        wall_clock=lambda: wall_start,
    )


# --- AI Server: deadline 을 보내고, 넘기면 연결을 끊고 504 ---


def test_deadline_header_is_the_absolute_time_on_post_and_redirect(remote_client, monkeypatch):  # noqa: F811
    fake = FakeWorker(steps=[redirect(POLL), (0, 200, {}, SUCCESS)])
    use(monkeypatch, deadline_service(fake, 50.0, wall_start=1_000_000.0))

    assert remote_post(remote_client, jpeg()).status_code == 200

    post_request, get_request = fake.calls
    # 요청 시작 시각(1_000_000) + 전체 deadline(50). 대기·처리에 쓴 몇 ms 만큼 앞당겨질 수는 있어도 늘어나지 않는다.
    sent = float(post_request.headers[deadline.HEADER])
    assert 1_000_049.9 < sent <= 1_000_050.0
    assert get_request.headers[deadline.HEADER] == post_request.headers[deadline.HEADER]  # redirect 로 늘지 않음


def test_within_50s_is_200_and_nothing_is_cancelled(remote_client, monkeypatch, caplog):  # noqa: F811
    fake = FakeWorker(steps=[(40 * SCALE, 200, {}, SUCCESS)])
    use(monkeypatch, deadline_service(fake, 50 * SCALE))

    with caplog.at_level(logging.INFO, logger="app.body_image_validation.remote"):
        response = remote_post(remote_client, jpeg())

    assert response.status_code == 200
    assert "http_cancelled=False" in caplog.text and "remote_stopped=n/a" in caplog.text


def test_over_50s_cancels_the_http_call_and_answers_504_once(remote_client, monkeypatch, caplog):  # noqa: F811
    fake = FakeWorker(steps=[(60 * SCALE, 200, {}, SUCCESS)])
    use(monkeypatch, deadline_service(fake, 50 * SCALE))

    started = time.monotonic()
    with caplog.at_level(logging.INFO, logger="app.body_image_validation.remote"):
        response = remote_post(remote_client, jpeg())
    elapsed = time.monotonic() - started

    assert response.status_code == 504
    assert response.json() == TIMEOUT
    assert elapsed < 60 * SCALE  # worker 응답(60초)까지 기다리지 않았다
    assert len(fake.calls) == 1  # 재시도나 새 GPU 작업 요청이 없다
    summary = next(r.getMessage() for r in caplog.records if "body image worker call" in r.getMessage())
    assert "outcome=deadline_exceeded" in summary
    assert "http_cancelled=True" in summary  # 이 쪽 연결은 끊었다
    assert "remote_cancel=not_requested" in summary  # Modal 취소 API 는 부르지 않았다
    assert "remote_stopped=unknown" in summary  # 원격 종료는 확인하지 못했다
    for secret in ("wk-test", "ws-test", "worker-key"):
        assert secret not in caplog.text


# --- worker: deadline 이 지난 요청은 다음 단계로 가지 않는다 ---


def worker_client():
    return TestClient(worker.create_app(), raise_server_exceptions=False)


def jpeg_body():
    return jpeg()


def worker_post(client, header_value=None):
    headers = dict(AUTH)
    if header_value is not None:
        headers[deadline.HEADER] = header_value
    return client.post(URL, files={"image": ("body.jpg", jpeg_body(), "image/jpeg")}, headers=headers)


def counts(worker_env, person, remover):  # noqa: F811
    return person.calls, remover.calls, len(worker_env["s3"].puts)


def started_worker(worker_env):  # noqa: F811
    person, pose = FakePersonDetector(), FakePoseDetector()
    worker.start_gpu(person, pose)  # warm-up 이 person/remover 를 한 번씩 부른다(S3 는 안 부름)
    remover = worker_env["manager"].runtime.background_remover
    return person, remover


def test_past_deadline_stops_before_any_gpu_work_or_upload(worker_env):  # noqa: F811
    person, remover = started_worker(worker_env)
    before = counts(worker_env, person, remover)

    response = worker_post(worker_client(), f"{time.time() - 5:.3f}")

    assert response.status_code == 504
    assert response.json() == TIMEOUT
    assert counts(worker_env, person, remover) == before  # 검출·배경 제거·S3 모두 안 함


def test_deadline_too_close_for_upload_skips_only_the_upload(worker_env):  # noqa: F811
    person, remover = started_worker(worker_env)
    before = counts(worker_env, person, remover)

    # 남은 시간(1초)이 업로드 여유(2초)보다 적다 → 검사·배경 제거는 했지만 S3 에 올리지 않는다
    response = worker_post(worker_client(), f"{time.time() + 1.0:.3f}")

    assert response.status_code == 504
    after = counts(worker_env, person, remover)
    assert after[0] == before[0] + 1 and after[1] == before[1] + 1
    assert after[2] == before[2] == 0


def test_enough_time_left_uploads_normally(worker_env):  # noqa: F811
    started_worker(worker_env)

    response = worker_post(worker_client(), f"{time.time() + 60:.3f}")

    assert response.status_code == 200
    assert len(worker_env["s3"].puts) == 1


@pytest.mark.parametrize("value", [None, "", "abc", "nan", "inf", "1e12", "-1"])
def test_missing_or_bogus_deadline_is_ignored(worker_env, value):  # noqa: F811
    started_worker(worker_env)

    assert worker_post(worker_client(), value).status_code == 200


def test_deadline_does_not_leak_into_the_next_request(worker_env):  # noqa: F811
    started_worker(worker_env)
    client = worker_client()

    assert worker_post(client, f"{time.time() - 5:.3f}").status_code == 504
    assert worker_post(client).status_code == 200


def test_ec2_local_app_ignores_the_header(monkeypatch):
    # EC2 app.main 에는 deadline middleware 가 없다. local 모드 동작은 그대로다.
    from app.main import app
    from tests.test_body_image_validation_service import (
        FakeBackgroundRemover,
        FakeStorage,
    )
    from tests.test_body_image_validation_service import FakePersonDetector as LocalPerson
    from tests.test_body_image_validation_service import FakePoseDetector as LocalPose

    storage = FakeStorage()
    service = BodyImageValidationService(LocalPerson(), LocalPose(), FakeBackgroundRemover(), storage)
    monkeypatch.setattr(router, "get_service", lambda: service)
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "worker-test-key")
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)

    response = TestClient(app, raise_server_exceptions=False).post(
        URL,
        files={"image": ("body.jpg", jpeg_body(), "image/jpeg")},
        headers={**AUTH, deadline.HEADER: f"{time.time() - 5:.3f}"},
    )

    assert response.status_code == 200
    assert len(storage.uploads) == 1


def test_parse_header_bounds():
    now = 2_000_000_000.0
    assert deadline.parse_header("2000000050.5", now=now) == 2000000050.5
    assert deadline.parse_header(str(now + 7200), now=now) is None
    assert deadline.parse_header(str(now - 7200), now=now) is None


def test_no_deadline_means_no_check():
    deadline.check("anything", margin=1000)  # deadline 이 없으면(EC2·직접 호출) 아무 일도 없다


def test_network_error_keeps_existing_mapping(remote_client, monkeypatch):  # noqa: F811
    fake = FakeWorker(error=httpx.ConnectError("refused"))
    use(monkeypatch, deadline_service(fake, 50 * SCALE))

    assert remote_post(remote_client, jpeg()).status_code == 503
