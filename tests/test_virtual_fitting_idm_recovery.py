"""IDM-VTON fallback 중단 복구: Watchdog 은 고정 Key(idm-vton.webp)만 확인·기록하고 Modal 을 부르지 않는다.

실제 Watchdog + AsyncJobStateRepository(격리 PostgreSQL) + IdmVtonFallbackHandler. S3·Modal 은 가짜다.
"""

import uuid

from app.virtual_fitting.idm_fallback_handler import META_ATTEMPT, META_JOB, RECOVERY_MARGIN_SECONDS
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_idm_fallback_handler import (
    KEY,
    UPLOAD_EXPIRES,
    FakeModal,
    FakeStorage,
    env,
    idm_job,
    model_failed,
    run,
    webp_bytes,
)
from tests.test_virtual_fitting_pruna_submit import JOB, db, job  # noqa: F401
from tests.test_virtual_fitting_watchdog import age, reasons, watchdog, wdb  # noqa: F401


def started_without_image(db, ago):  # noqa: F811
    """Modal 호출 기회를 선점(시작 기록)했지만 이미지 기록이 없고 lease 도 없는 작업."""
    idm_job(db)
    attempt = str(uuid.uuid4())
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET fallback_task_uuid = %s, fallback_attempt_count = 1, "
        "fallback_started_at = now() - make_interval(secs => %s), worker_lease_token = NULL, "
        "worker_lease_until = NULL WHERE fitting_job_id = %s",
        (attempt, ago, JOB),
    )
    age(db, "updated_at", 700)
    return attempt


def test_watchdog_records_a_matching_result_and_the_handler_resumes_without_a_call(wdb):  # noqa: F811
    attempt = started_without_image(wdb, 600)
    storage = FakeStorage()
    storage.put(KEY, webp_bytes(), metadata={META_JOB: str(JOB), META_ATTEMPT: attempt})
    w = watchdog(wdb, storage=storage)
    reports = w.run_once()
    assert "fallback_result_found" in reports[0].actions
    assert job(wdb).fallback_image_key == KEY
    assert [p["action"] for _, p in w.queue.sent] == ["RUN_FALLBACK"]
    modal = FakeModal(storage)
    run(wdb, env(wdb, storage=storage, modal=modal))
    assert modal.requests == [] and job(wdb).result_payload["result_image_key"] == KEY


def test_watchdog_waits_while_the_upload_url_is_still_valid(wdb):  # noqa: F811
    started_without_image(wdb, 600)
    w = watchdog(wdb, storage=FakeStorage())
    reports = w.run_once()
    assert "fallback_result_pending" in reports[0].actions and w.queue.sent == []
    assert ("fallback_result_unknown", "fallback", JOB) in reasons(w.sink)
    current = job(wdb)
    assert current.result_payload is None and current.fallback_image_key is None


def test_watchdog_hands_a_missing_result_back_to_the_handler_for_failure(wdb):  # noqa: F811
    started_without_image(wdb, UPLOAD_EXPIRES + RECOVERY_MARGIN_SECONDS + 60)
    storage = FakeStorage()
    w = watchdog(wdb, storage=storage)
    reports = w.run_once()
    assert "fallback_result_absent" in reports[0].actions
    assert [p["action"] for _, p in w.queue.sent] == ["RUN_FALLBACK"]
    assert ("fallback_result_absent", "fallback", JOB) in reasons(w.sink)
    assert job(wdb).result_payload is None  # Watchdog 은 결과를 쓰지 않는다
    modal = FakeModal(storage)
    run(wdb, env(wdb, storage=storage, modal=modal))
    assert modal.requests == [] and job(wdb).result_payload == model_failed()
