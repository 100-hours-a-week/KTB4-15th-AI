import json
import logging
from decimal import Decimal

import psycopg
import pytest

from app.clients.sqs import FittingQueue
from app.virtual_fitting import diagnostics as diag
from app.virtual_fitting.exceptions import FittingDatabaseError
from app.virtual_fitting.providers.runware_status import (
    TaskStatus,
    TaskStatusKind,
    image_ref,
    parse_status,
)
from app.virtual_fitting.repositories.async_job_repository import (
    AsyncJobRepository,
    AsyncJobStatus,
    NextAction,
)
from app.virtual_fitting.repositories.async_job_state import AsyncJobStateRepository
from app.virtual_fitting.runware_webhook import (
    EventKind,
    EventOutcome,
    WebhookEvent,
    WebhookFollowUpNotReadyError,
    parse_webhook,
)
from tests.test_virtual_fitting_async_job_repository import (
    AI_RESULT_MIGRATION,
    AUTH_HOLD_MIGRATION,
    FALLBACK_MIGRATION,
    IMAGE_REF_MIGRATION,
    MIGRATION,
    PRUNA_RETRY_MIGRATION,
    UnusablePool,
)
from tests.test_virtual_fitting_async_job_repository import _new_job as _repo_new_job
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_pruna_submit import JOB, db, job, limiter  # noqa: F401
from tests.test_virtual_fitting_watchdog import (  # noqa: F401 (pytest fixture)
    FakeStatus,
    clear_lease,
    waiting,
    watchdog,
    wdb,
)
from tests.test_virtual_fitting_webhook import (  # noqa: F401 (pytest fixture)
    IMAGE_URL,
    IMAGE_UUID,
    FakeQueue,
    api,
    in_flight,
    post,
    queue_settings,
    service,
    submitted,
    success_body,
    task_costs,
)

OTHER_URL = "https://im.runware.ai/image/os/a14d18/ws/2/ii/OTHER-IMAGE.webp"
OTHER_UUID = "7d444840-9dc0-11d1-b245-5ffdce74fad2"
PRESIGNED = (
    "https://bucket.s3.ap-northeast-2.amazonaws.com/virtual-fitting/results/3/pruna-async.jpg"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAXXXX&X-Amz-Security-Token=tok"
    "&X-Amz-Signature=abc"
)


def refs(db, job_id=JOB):  # noqa: F811
    current = job(db, job_id)
    return current.runware_image_url, current.runware_image_uuid


def body_with(task, **item):
    base = {"taskType": "imageInference", "taskUUID": task, "cost": 0.023}
    return json.dumps({**base, **item}).encode()


def processed_events(caplog):
    """진단 줄(vf_diag event=runware_result_processed key=value ...)을 dict 로."""
    found = []
    for record in caplog.records:
        message = record.getMessage()
        if record.name == diag.logger.name and "event=runware_result_processed " in message:
            found.append(dict(part.split("=", 1) for part in message.split()[1:]))
    return found


# --- 해석: Webhook·getResponse 가 같은 기준으로 꺼낸다 (DB 없음) ---


def test_webhook_and_status_parsers_extract_the_same_reference():
    task = "a770f077-f413-47de-9dac-be0b26a35da6"
    (event,) = parse_webhook(success_body(task))
    status = parse_status(
        json.dumps(
            {
                "data": [
                    {
                        "taskUUID": task,
                        "status": "success",
                        "imageURL": IMAGE_URL,
                        "imageUUID": IMAGE_UUID.upper(),
                    }
                ]
            }
        ).encode(),
        task,
    )
    assert (event.image_url, event.image_uuid, event.image_ref_invalid) == (
        IMAGE_URL,
        IMAGE_UUID,
        False,
    )
    assert (status.kind, status.image_url, status.image_uuid) == (
        TaskStatusKind.SUCCEEDED,
        IMAGE_URL,
        IMAGE_UUID,  # 대소문자 등 표기를 정규화해 두 경로가 같은 값이 된다
    )
    # repr 로 URL 이 새지 않는다(로그에 객체를 찍어도 안전).
    assert "SECRET-IMAGE" not in repr(event) and "SECRET-IMAGE" not in repr(status)


@pytest.mark.parametrize(
    "url",
    [
        "http://im.runware.ai/image/x.jpg",  # https 아님
        "HTTPS://im.runware.ai/image/x.jpg",  # DB CHECK(소문자 https:// 접두사)와 맞춘다
        PRESIGNED,  # uploadEndpoint 로 넘긴 Presigned PUT(S3 버킷 호스트)
        "https://cdn.example.com/x.jpg?token=abc",  # token 이 있어도 허용 호스트가 아니면 제외
        "https://im.runware.ai.evil.example/x.jpg",  # 허용 호스트로 시작하는 다른 호스트
        "https://evilim.runware.ai/x.jpg",  # 허용 호스트로 끝나는 다른 호스트
        "https://cdn.im.runware.ai/x.jpg",  # 하위 도메인(와일드카드 허용 안 함)
        "https://evil.example/im.runware.ai/x.jpg",  # 경로에만 있음
        "https://evil.example/x.jpg?u=https://im.runware.ai/x.jpg",  # 쿼리에만 있음
        "https://evil.example#@im.runware.ai/x.jpg",
        "https://im.runware.ai@evil.example/x.jpg",  # 사용자 정보 위치
        "https://user:pw@im.runware.ai/x.jpg",
        "https://im.runware.ai:8443/x.jpg",
        "https://im.runware.ai./x.jpg",
        "https://im%2Erunware.ai/x.jpg",
        "https://im.runware.ai/a b.jpg",
        "https:///no-host.jpg",
        "https://im.runware.ai/" + "a" * 2048,
        123,
    ],
)
def test_unstorable_url_is_dropped_but_success_is_kept(url):
    task = "a770f077-f413-47de-9dac-be0b26a35da6"
    (event,) = parse_webhook(body_with(task, imageURL=url, imageUUID=IMAGE_UUID))
    assert event.kind is EventKind.SUCCEEDED  # 기존 성공 판정은 그대로
    assert (event.image_url, event.image_uuid, event.image_ref_invalid) == (None, IMAGE_UUID, True)


def test_unstorable_uuid_is_dropped_but_success_is_kept():
    task = "a770f077-f413-47de-9dac-be0b26a35da6"
    (event,) = parse_webhook(body_with(task, imageURL=IMAGE_URL, imageUUID="not-a-uuid"))
    assert event.kind is EventKind.SUCCEEDED
    assert (event.image_url, event.image_uuid, event.image_ref_invalid) == (IMAGE_URL, None, True)


def test_missing_reference_is_not_invalid():
    assert image_ref({}) == (None, None, False)
    assert image_ref({"imageURL": "  ", "imageUUID": ""}) == (None, None, False)
    status = parse_status(
        json.dumps({"data": [{"taskUUID": IMAGE_UUID, "status": "success"}]}).encode(), IMAGE_UUID
    )
    # URL 없는 getResponse 성공도 성공이다(기존과 같음).
    assert status.kind is TaskStatusKind.SUCCEEDED
    assert (status.image_url, status.image_uuid, status.image_ref_invalid) == (None, None, False)


@pytest.mark.parametrize(
    "kwargs",
    [{"image_url": "http://im.runware.ai/x.jpg"}, {"image_url": PRESIGNED}, {"image_uuid": "x"}],
)
def test_repository_rejects_unstorable_reference_before_db(kwargs):
    with pytest.raises(ValueError):
        AsyncJobStateRepository(UnusablePool()).record_pruna_succeeded(IMAGE_UUID, **kwargs)


# --- 1. Webhook 성공 결과 저장 + 8. 기존 흐름 유지 ---


def test_webhook_success_stores_reference_before_publishing(db, queue_settings):  # noqa: F811
    task = submitted(db)
    seen_at_publish = []

    class OrderQueue(FakeQueue):
        def send_json(self, queue, payload):
            seen_at_publish.append(refs(db))  # 발행 시점에 이미 DB 에 저장돼 있어야 한다
            return super().send_json(queue, payload)

    queue = OrderQueue()
    (result,) = service(db, queue).handle(parse_webhook(success_body(task)))
    current = job(db)
    assert result.outcome is EventOutcome.APPLIED
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert refs(db) == (IMAGE_URL, IMAGE_UUID)
    assert seen_at_publish == [(IMAGE_URL, IMAGE_UUID)]
    # 후처리 메시지·이미지 위치·결과는 그대로(새 필드 없음, async_image_key 사용, 결과 미저장).
    assert queue.sent == [
        (
            FittingQueue.AI_FITTING_RESPONSE,
            {"fitting_job_id": JOB, "task_uuid": task, "action": "POSTPROCESS"},
        )
    ]
    assert current.async_image_key == f"virtual-fitting/results/{JOB}/pruna-async.jpg"
    assert current.result_payload is None and current.result_image_key is None
    assert task_costs(db)[0][1] == Decimal("0.023") and in_flight(db) == []


# --- 2. 상태 조회(Watchdog getResponse) 성공 결과 저장 ---


def test_watchdog_status_success_stores_the_same_reference(wdb):  # noqa: F811
    task = waiting(wdb, 200)
    status = TaskStatus(
        TaskStatusKind.SUCCEEDED,
        task,
        Decimal("0.015"),
        image_url=IMAGE_URL,
        image_uuid=IMAGE_UUID,
    )
    w = watchdog(wdb, FakeStatus(status))
    w.run_once()
    current = job(wdb)
    assert (current.status, current.next_action) == (
        AsyncJobStatus.POSTPROCESSING,
        NextAction.POSTPROCESS,
    )
    assert refs(wdb) == (IMAGE_URL, IMAGE_UUID)
    assert [p for _, p in w.queue.sent] == [
        {"fitting_job_id": JOB, "task_uuid": task, "action": "POSTPROCESS"}
    ]


def test_watchdog_then_webhook_with_the_same_reference_is_idempotent(wdb, queue_settings):  # noqa: F811
    task = waiting(wdb, 200)
    status = TaskStatus(
        TaskStatusKind.SUCCEEDED, task, Decimal("0.023"), image_url=IMAGE_URL, image_uuid=IMAGE_UUID
    )
    watchdog(wdb, FakeStatus(status)).run_once()
    (result,) = service(wdb).handle(parse_webhook(success_body(task)))
    assert result.outcome is EventOutcome.DUPLICATE
    assert refs(wdb) == (IMAGE_URL, IMAGE_UUID)


# --- 3. URL 없는 기존 응답 호환 ---


def test_webhook_success_without_url_keeps_the_existing_flow(db, queue_settings):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    (result,) = service(db, queue).handle(parse_webhook(body_with(task, imageUUID=IMAGE_UUID)))
    assert result.outcome is EventOutcome.APPLIED and result.published is NextAction.POSTPROCESS
    assert refs(db) == (None, IMAGE_UUID)
    assert [p["action"] for _, p in queue.sent] == ["POSTPROCESS"]


def test_status_success_without_reference_keeps_the_existing_flow(wdb):  # noqa: F811
    task = waiting(wdb, 200)
    w = watchdog(wdb, FakeStatus(TaskStatus(TaskStatusKind.SUCCEEDED, task, Decimal("0.015"))))
    w.run_once()
    assert job(wdb).status is AsyncJobStatus.POSTPROCESSING
    assert refs(wdb) == (None, None)
    assert [p["action"] for _, p in w.queue.sent] == ["POSTPROCESS"]


def test_invalid_url_is_not_stored_and_the_flow_continues(db, queue_settings, caplog):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    with caplog.at_level(logging.DEBUG):
        service(db, queue).handle(
            parse_webhook(body_with(task, imageURL=PRESIGNED, imageUUID=IMAGE_UUID))
        )
    assert refs(db) == (None, IMAGE_UUID)
    assert job(db).status is AsyncJobStatus.POSTPROCESSING and len(queue.sent) == 1
    text = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app"))
    assert "image ref ignored (invalid)" in text
    for secret in ("X-Amz", "amazonaws.com", "ASIAXXXX", IMAGE_UUID):
        assert secret not in text


# --- 4. 중복 알림 멱등 ---


def test_duplicate_webhook_changes_nothing(db, queue_settings, caplog):  # noqa: F811
    task = submitted(db)
    service(db).handle(parse_webhook(success_body(task)))
    before = db["sql"](
        "SELECT updated_at, runware_image_url, runware_image_uuid::text "
        "FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s",
        (JOB,),
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=diag.logger.name):
        (result,) = service(db).handle(parse_webhook(success_body(task)))
    after = db["sql"](
        "SELECT updated_at, runware_image_url, runware_image_uuid::text "
        "FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s",
        (JOB,),
    )
    assert result.outcome is EventOutcome.DUPLICATE
    assert before == after  # 행을 다시 쓰지도 않았다
    assert [e["image_ref"] for e in processed_events(caplog)] == ["same"]


# --- 5. 다른 task·충돌 결과 ---


def test_other_task_result_never_touches_this_job(db, queue_settings):  # noqa: F811
    first, second = submitted(db, 1), submitted(db, 2)
    service(db).handle(parse_webhook(success_body(first)))
    service(db).handle(parse_webhook(body_with(second, imageURL=OTHER_URL, imageUUID=OTHER_UUID)))
    assert refs(db, 1) == (IMAGE_URL, IMAGE_UUID)
    assert refs(db, 2) == (OTHER_URL, OTHER_UUID)


def test_conflicting_reference_for_the_same_task_keeps_the_first_and_reports(
    db,  # noqa: F811
    queue_settings,  # noqa: F811
    caplog,
):
    task = submitted(db)
    service(db).handle(parse_webhook(success_body(task)))
    conflicting = body_with(task, imageURL=OTHER_URL, imageUUID=OTHER_UUID)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        (result,) = service(db).handle(parse_webhook(conflicting))
    assert result.outcome is EventOutcome.DUPLICATE  # 상태 판단은 기존 그대로
    assert refs(db) == (IMAGE_URL, IMAGE_UUID)  # 덮어쓰지 않는다
    assert [e["image_ref"] for e in processed_events(caplog)] == ["conflict"]
    text = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app"))
    assert "image ref conflict kept existing" in text
    for secret in ("SECRET-IMAGE", "OTHER-IMAGE", "im.runware.ai", IMAGE_UUID, OTHER_UUID):
        assert secret not in text


def test_conflicting_results_in_one_batch_are_not_picked_arbitrarily(db, queue_settings):  # noqa: F811
    """같은 task 의 서로 다른 결과가 한 본문에 여럿이면(요청은 1개) 첫 항목을 고르지 않는다(3단계). 성공 처리는 그대로."""
    task = submitted(db)
    body = json.dumps(
        {
            "data": [
                json.loads(success_body(task)),
                json.loads(body_with(task, imageURL=OTHER_URL, imageUUID=OTHER_UUID)),
            ]
        }
    ).encode()
    events = parse_webhook(body)
    assert all(e.image_ref_invalid and e.image_url is None and e.image_uuid is None for e in events)
    results = service(db).handle(events)
    assert [r.outcome for r in results] == [EventOutcome.APPLIED, EventOutcome.DUPLICATE]
    assert refs(db) == (None, None) and job(db).status is AsyncJobStatus.POSTPROCESSING


# --- 6. 늦은 알림의 상태 역행 방지 ---


def test_late_success_after_completion_changes_nothing(db, queue_settings, caplog):  # noqa: F811
    task = submitted(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET status = 'COMPLETED', next_action = NULL "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    queue = FakeQueue()
    with caplog.at_level(logging.INFO, logger=diag.logger.name):
        (result,) = service(db, queue).handle(parse_webhook(success_body(task)))
    current = job(db)
    assert result.outcome is EventOutcome.IGNORED
    assert (current.status, current.next_action) == (AsyncJobStatus.COMPLETED, None)
    assert refs(db) == (None, None) and queue.sent == []
    assert [e["image_ref"] for e in processed_events(caplog)] == ["not_stored"]


def test_late_success_after_fallback_or_failure_result_changes_nothing(
    db,  # noqa: F811
    queue_settings,  # noqa: F811
    monkeypatch,
):
    from tests.runware_codes import register_generation_failed_code
    from tests.test_virtual_fitting_webhook import error_body

    fallback_task, failed_task = submitted(db, 1), submitted(db, 2)
    service(db).handle(parse_webhook(error_body(fallback_task)))  # timeoutProvider → fallback
    # 확정 생성 실패 결과 저장(결과 발행 경로 미준비라 503, 기존 동작).
    code = register_generation_failed_code(monkeypatch)
    with pytest.raises(WebhookFollowUpNotReadyError):
        service(db).handle(parse_webhook(error_body(failed_task, code=code)))
    for job_id, task in ((1, fallback_task), (2, failed_task)):
        before = job(db, job_id)
        try:
            service(db).handle(parse_webhook(success_body(task)))
        except WebhookFollowUpNotReadyError:  # PUBLISH_RESULT 작업은 늦은 알림에도 기존처럼 503
            assert job_id == 2
        after = job(db, job_id)
        assert (after.status, after.next_action, after.result_payload) == (
            before.status,
            before.next_action,
            before.result_payload,
        )
        assert refs(db, job_id) == (None, None)


def test_reference_survives_later_state_changes(db, queue_settings):  # noqa: F811
    task = submitted(db)
    service(db).handle(parse_webhook(success_body(task)))
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET status = 'FALLBACK_RUNNING', "
        "next_action = 'RUN_FALLBACK' WHERE fitting_job_id = %s",
        (JOB,),
    )
    service(db).handle(parse_webhook(body_with(task, imageURL=OTHER_URL, imageUUID=OTHER_UUID)))
    assert job(db).status is AsyncJobStatus.FALLBACK_RUNNING
    assert refs(db) == (IMAGE_URL, IMAGE_UUID)  # 복구용 기록은 남고 늦은 다른 값으로 바뀌지 않는다


# --- 7. DB 저장 실패 시 부분 저장 방지 ---


def test_db_failure_saves_neither_state_nor_reference_and_publishes_nothing(
    db,  # noqa: F811
    queue_settings,  # noqa: F811
):
    task = submitted(db)
    db["sql"](
        """
        CREATE FUNCTION fail_image_ref() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.runware_image_url IS NOT NULL THEN
                RAISE EXCEPTION 'injected failure';
            END IF;
            RETURN NEW;
        END $$;
        CREATE TRIGGER fail_image_ref BEFORE UPDATE ON virtual_fitting_async_jobs
            FOR EACH ROW EXECUTE FUNCTION fail_image_ref();
        """
    )
    queue = FakeQueue()
    with pytest.raises(FittingDatabaseError):
        service(db, queue).handle(parse_webhook(success_body(task)))
    current = job(db)
    assert (current.status, current.next_action) == (AsyncJobStatus.WAITING_PRUNA, None)
    assert refs(db) == (None, None)
    assert queue.sent == [] and task_costs(db) == [] and in_flight(db) == [(JOB, task)]

    db["sql"]("DROP TRIGGER fail_image_ref ON virtual_fitting_async_jobs")
    (result,) = service(db, queue).handle(parse_webhook(success_body(task)))  # Runware 재전송
    assert result.outcome is EventOutcome.APPLIED
    assert refs(db) == (IMAGE_URL, IMAGE_UUID)
    assert [p["action"] for _, p in queue.sent] == ["POSTPROCESS"]


def test_db_constraint_rejects_a_non_https_url(db):  # noqa: F811
    submitted(db)
    with pytest.raises(psycopg.errors.CheckViolation):
        db["sql"](
            "UPDATE virtual_fitting_async_jobs SET runware_image_url = 'http://x/y.jpg' "
            "WHERE fitting_job_id = %s",
            (JOB,),
        )


# --- 마이그레이션: 기존 행은 NULL, 재실행 안전 ---


def test_migration_leaves_existing_rows_null_and_can_run_again(pg):  # noqa: F811
    pg["sql"]("DROP TABLE virtual_fitting_async_jobs")
    for migration in (MIGRATION, FALLBACK_MIGRATION, AUTH_HOLD_MIGRATION):
        pg["sql"](migration.read_text(encoding="utf-8"))
    pg["sql"](
        "INSERT INTO virtual_fitting_async_jobs (fitting_job_id, request_fingerprint, requested_at) "
        "VALUES (3, 'fp', now())"
    )
    pg["sql"](IMAGE_REF_MIGRATION.read_text(encoding="utf-8"))
    pg["sql"](IMAGE_REF_MIGRATION.read_text(encoding="utf-8"))
    pg["sql"](AI_RESULT_MIGRATION.read_text(encoding="utf-8"))  # 3단계 컬럼(새 코드가 SELECT)
    pg["sql"](PRUNA_RETRY_MIGRATION.read_text(encoding="utf-8"))  # 보완 2-2 컬럼(새 코드가 SELECT)
    existing = AsyncJobRepository(pg["pool"]).find_by_fitting_job_id(3)
    assert (existing.runware_image_url, existing.runware_image_uuid) == (None, None)
    created = AsyncJobRepository(pg["pool"]).create(_repo_new_job(4)).job
    assert (created.runware_image_url, created.runware_image_uuid) == (None, None)


def test_event_defaults_keep_old_constructors_working():
    event = WebhookEvent("a770f077-f413-47de-9dac-be0b26a35da6", EventKind.SUCCEEDED)
    assert (event.image_url, event.image_uuid, event.image_ref_invalid) == (None, None, False)


# === 정상 Runware 결과 URL 전체 보존 (token·서명·만료 쿼리 포함) ===

TOKENED_URL = (
    "https://im.runware.ai/image/ws/2/ii/RESULT.webp"
    "?b=2&a=%2Fx%20y&token=tok-SECRET-123%3D%3D&X-Amz-Signature=sig-SECRET-456&Expires=1700000000"
)
SECRETS = ("tok-SECRET-123", "sig-SECRET-456", "RESULT.webp", "Expires=1700000000")


@pytest.mark.parametrize(
    "url",
    [
        "https://im.runware.ai/image/ws/2/ii/plain.webp",  # token 없음
        TOKENED_URL,  # token·서명·만료 + 순서·퍼센트 인코딩
        "https://im.runware.ai:443/image/x.webp?token=a&token=b",  # 명시 포트 443·중복 키
        "https://IM.RUNWARE.AI/image/x.webp",  # 호스트 대소문자(DNS 기준 같은 호스트)
    ],
)
def test_allowed_host_url_is_kept_verbatim_by_both_parsers(url):
    task = "a770f077-f413-47de-9dac-be0b26a35da6"
    (event,) = parse_webhook(body_with(task, imageURL=url, imageUUID=IMAGE_UUID))
    status = parse_status(
        json.dumps({"data": [{"taskUUID": task, "status": "success", "imageURL": url}]}).encode(),
        task,
    )
    assert (event.image_url, event.image_ref_invalid) == (url, False)  # 삭제·재정렬·재인코딩 없음
    assert (status.image_url, status.image_ref_invalid) == (url, False)


def test_tokened_url_is_stored_verbatim_and_never_exposed(db, queue_settings, caplog):  # noqa: F811
    task = submitted(db)
    queue = FakeQueue()
    with caplog.at_level(logging.DEBUG):
        (event,) = parse_webhook(body_with(task, imageURL=TOKENED_URL, imageUUID=IMAGE_UUID))
        service(db, queue).handle([event])
    assert refs(db) == (TOKENED_URL, IMAGE_UUID)
    stored = db["sql"](
        "SELECT runware_image_url FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s",
        (JOB,),
    )
    assert stored == [(TOKENED_URL,)]  # DB 에도 바이트 그대로
    exposed = [r.getMessage() for r in caplog.records] + [repr(event), repr(job(db))]
    exposed += [json.dumps(payload) for _, payload in queue.sent]
    with pytest.raises(ValueError) as rejected:
        AsyncJobStateRepository(UnusablePool()).record_pruna_succeeded(
            task, image_url=TOKENED_URL.replace("im.runware.ai", "evil.example")
        )
    exposed.append(str(rejected.value))
    text = "\n".join(exposed)
    for secret in SECRETS:
        assert secret not in text


def test_wrong_webhook_token_stores_no_reference(api):  # noqa: F811
    """실제 app·라우터 인증(api fixture). 토큰이 틀리면 상태·참조를 바꾸지 않고, 맞으면 같은 본문이 저장된다."""
    task = submitted(api["db"])
    body = body_with(task, imageURL=TOKENED_URL, imageUUID=IMAGE_UUID)
    for token in ("bad", None):
        assert post(api, body, token=token).status_code == 401
    assert refs(api["db"]) == (None, None)
    assert job(api["db"]).status is AsyncJobStatus.WAITING_PRUNA
    assert post(api, body).status_code == 200
    assert refs(api["db"]) == (TOKENED_URL, IMAGE_UUID)


# === 늦게 온 결과 참조 보완 ===


def uuid_only_success(db, job_id=JOB, image_uuid=IMAGE_UUID):  # noqa: F811
    """최초 성공 알림이 imageUUID 만 준 작업(POSTPROCESSING / POSTPROCESS, URL NULL)."""
    task = submitted(db, job_id)
    service(db).handle(parse_webhook(body_with(task, imageUUID=image_uuid)))
    assert refs(db, job_id) == (None, image_uuid)
    return task


SNAPSHOT_SQL = (
    "SELECT status, next_action, updated_at, runware_accepted_at, pruna_submit_started_at, "
    "worker_lease_token, worker_lease_until, result_payload, s3_check_attempt_count, "
    "llm_generation_attempt_count, fallback_attempt_count, result_message_attempt_count "
    "FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s"
)


def snapshot(db, job_id=JOB):  # noqa: F811
    return db["sql"](SNAPSHOT_SQL, (job_id,))


def test_late_url_from_watchdog_status_fills_uuid_only_reference(wdb, queue_settings):  # noqa: F811
    """Watchdog 이 대기 작업을 조회하는 사이 UUID 만 담긴 Webhook 이 먼저 성공 처리한 경합(실제 run_once 경로)."""
    waiting(wdb, 200)

    class RacingStatus(FakeStatus):
        def query(self, task_uuid):
            service(wdb).handle(parse_webhook(body_with(task_uuid, imageUUID=IMAGE_UUID)))
            return TaskStatus(
                TaskStatusKind.SUCCEEDED,
                task_uuid,
                Decimal("0.023"),
                image_url=TOKENED_URL,
                image_uuid=IMAGE_UUID,
            )

    w = watchdog(wdb, RacingStatus())
    w.run_once()
    assert refs(wdb) == (TOKENED_URL, IMAGE_UUID)
    assert job(wdb).status is AsyncJobStatus.POSTPROCESSING
    assert len(task_costs(wdb)) == 1 and in_flight(wdb) == []


def test_late_url_from_duplicate_webhook_fills_without_side_effects(db, queue_settings, caplog):  # noqa: F811
    task = uuid_only_success(db)
    clear_lease(db)  # 제출 단계 lease 정리 뒤 후처리 Worker 가 실행 중인 상황
    token = db["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60).token
    assert token is not None
    before, costs_before = snapshot(db), task_costs(db)
    queue = FakeQueue()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=diag.logger.name):
        (result,) = service(db, queue).handle(parse_webhook(success_body(task)))
    assert refs(db) == (IMAGE_URL, IMAGE_UUID)
    assert result.outcome is EventOutcome.DUPLICATE
    assert snapshot(db) == before  # 상태·next_action·updated_at·시각·lease·횟수 그대로
    assert task_costs(db) == costs_before  # 비용 중복 반영 없음
    assert db["state"].holds_lease(JOB, token)
    # 발행은 보완과 무관한 기존 중복 성공 처리(POSTPROCESS 재발행 1건) 그대로다. 보완이 더한 발행은 없다.
    assert [p for _, p in queue.sent] == [
        {"fitting_job_id": JOB, "task_uuid": task, "action": "POSTPROCESS"}
    ]
    assert [e["image_ref"] for e in processed_events(caplog)] == ["filled"]


def test_duplicate_without_anything_to_fill_publishes_the_same(db, queue_settings):  # noqa: F811
    """대조군: 보완할 것이 없는 같은 중복 알림도 같은 발행을 한다(발행 수는 보완과 무관)."""
    task = uuid_only_success(db)
    queue = FakeQueue()
    service(db, queue).handle(parse_webhook(body_with(task, imageUUID=IMAGE_UUID)))
    assert [p["action"] for _, p in queue.sent] == ["POSTPROCESS"]


def test_late_uuid_fills_url_only_reference(db, queue_settings):  # noqa: F811
    task = submitted(db)
    service(db).handle(parse_webhook(body_with(task, imageURL=IMAGE_URL)))
    assert refs(db) == (IMAGE_URL, None)
    service(db).handle(parse_webhook(success_body(task)))
    assert refs(db) == (IMAGE_URL, IMAGE_UUID)


def test_same_reference_again_is_idempotent_after_fill(db, queue_settings, caplog):  # noqa: F811
    task = uuid_only_success(db)
    service(db).handle(parse_webhook(success_body(task)))
    before = snapshot(db)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=diag.logger.name):
        service(db).handle(parse_webhook(success_body(task)))
    assert refs(db) == (IMAGE_URL, IMAGE_UUID) and snapshot(db) == before
    assert [e["image_ref"] for e in processed_events(caplog)] == ["same"]
    direct = db["state"].fill_pruna_image_ref(task, image_url=IMAGE_URL, image_uuid=IMAGE_UUID)
    assert not direct.applied


@pytest.mark.parametrize(
    ("first", "late", "expected", "code"),
    [
        # UUID 가 다르면 새 URL 을 붙이지 않는다(다른 결과 참조를 섞지 않음).
        (
            {"imageUUID": IMAGE_UUID},
            {"imageURL": OTHER_URL, "imageUUID": OTHER_UUID},
            (None, IMAGE_UUID),
            "conflict",
        ),
        # URL 이 다르면 새 UUID 를 붙이지 않는다.
        (
            {"imageURL": IMAGE_URL},
            {"imageURL": OTHER_URL, "imageUUID": OTHER_UUID},
            (IMAGE_URL, None),
            "conflict",
        ),
        # 같은 결과라는 근거(일치하는 참조)가 없으면 연결하지 않는다.
        ({"imageUUID": IMAGE_UUID}, {"imageURL": IMAGE_URL}, (None, IMAGE_UUID), "unconfirmed"),
        ({"imageURL": IMAGE_URL}, {"imageUUID": IMAGE_UUID}, (IMAGE_URL, None), "unconfirmed"),
    ],
)
def test_conflicting_or_unconfirmed_late_reference_is_not_mixed_in(
    db,  # noqa: F811
    queue_settings,  # noqa: F811
    caplog,
    first,
    late,
    expected,
    code,
):
    task = submitted(db)
    service(db).handle(parse_webhook(body_with(task, **first)))
    caplog.clear()
    with caplog.at_level(logging.INFO):
        service(db).handle(parse_webhook(body_with(task, **late)))
    assert refs(db) == expected
    assert [e["image_ref"] for e in processed_events(caplog)] == [code]
    text = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app"))
    assert f"image ref {code} kept existing" in text
    for secret in ("SECRET-IMAGE", "OTHER-IMAGE", IMAGE_UUID, OTHER_UUID):
        assert secret not in text
    # 서비스의 사전 판단 없이 repository 를 바로 불러도 UPDATE 조건이 같은 결과를 지킨다.
    direct = db["state"].fill_pruna_image_ref(
        task, image_url=late.get("imageURL"), image_uuid=late.get("imageUUID")
    )
    assert not direct.applied and refs(db) == expected


def test_missing_or_null_values_never_clear_existing_reference(db, queue_settings):  # noqa: F811
    task = submitted(db)
    service(db).handle(parse_webhook(success_body(task)))
    service(db).handle(parse_webhook(body_with(task, imageUUID=IMAGE_UUID)))
    service(db).handle(parse_webhook(body_with(task, imageURL=IMAGE_URL, imageUUID=None)))
    assert refs(db) == (IMAGE_URL, IMAGE_UUID)
    with pytest.raises(ValueError):
        db["state"].fill_pruna_image_ref(task)  # 보완할 값이 없으면 DB 에 가지 않는다
    assert refs(db) == (IMAGE_URL, IMAGE_UUID)


@pytest.mark.parametrize(
    "arrange_sql",
    [
        "status = 'COMPLETED', next_action = NULL",
        "status = 'FAILED', next_action = NULL",
        "status = 'FALLBACK_RUNNING', next_action = 'RUN_FALLBACK'",
        # 후처리가 결과를 저장해 발행 단계로 넘어간 작업(상태 이름은 아직 POSTPROCESSING)
        (
            "next_action = 'PUBLISH_RESULT', result_payload = "
            "jsonb_build_object('fitting_job_id', fitting_job_id, 'status', 'FAILED', "
            "'error_code', 'X', 'error_message', 'x')"
        ),
    ],
)
def test_completed_failed_fallback_or_published_jobs_are_not_filled(
    db,  # noqa: F811
    queue_settings,  # noqa: F811
    arrange_sql,
):
    task = uuid_only_success(db)
    db["sql"](
        f"UPDATE virtual_fitting_async_jobs SET {arrange_sql} WHERE fitting_job_id = %s", (JOB,)
    )
    before = snapshot(db)
    try:
        service(db).handle(parse_webhook(success_body(task)))
    except WebhookFollowUpNotReadyError:
        pass  # PUBLISH_RESULT 작업은 기존처럼 503(발행 경로 미준비)
    change = db["state"].fill_pruna_image_ref(task, image_url=IMAGE_URL, image_uuid=IMAGE_UUID)
    assert not change.applied
    assert refs(db) == (None, IMAGE_UUID) and snapshot(db) == before


def test_other_task_never_fills_this_job(db, queue_settings):  # noqa: F811
    uuid_only_success(db, 1)
    other = uuid_only_success(db, 2)
    # 2번 작업의 알림이 1번과 같은 imageUUID 를 주장해도 1번에는 쓰지 않는다(task_uuid 로만 찾는다).
    service(db).handle(parse_webhook(body_with(other, imageURL=IMAGE_URL, imageUUID=IMAGE_UUID)))
    assert refs(db, 1) == (None, IMAGE_UUID)
    assert refs(db, 2) == (IMAGE_URL, IMAGE_UUID)


def test_fill_failure_keeps_existing_record_and_the_existing_flow(db, queue_settings, caplog):  # noqa: F811
    task = uuid_only_success(db)
    db["sql"](
        """
        CREATE FUNCTION fail_fill() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF OLD.runware_image_url IS NULL AND NEW.runware_image_url IS NOT NULL THEN
                RAISE EXCEPTION 'injected failure';
            END IF;
            RETURN NEW;
        END $$;
        CREATE TRIGGER fail_fill BEFORE UPDATE ON virtual_fitting_async_jobs
            FOR EACH ROW EXECUTE FUNCTION fail_fill();
        """
    )
    before = snapshot(db)
    queue = FakeQueue()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        (result,) = service(db, queue).handle(parse_webhook(success_body(task)))
    assert result.outcome is EventOutcome.DUPLICATE  # 보완 실패가 기존 처리를 막지 않는다
    assert refs(db) == (None, IMAGE_UUID) and snapshot(db) == before
    assert [p["action"] for _, p in queue.sent] == ["POSTPROCESS"]
    assert [e["image_ref"] for e in processed_events(caplog)] == ["fill_failed"]
    text = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app"))
    assert "image ref fill_failed kept existing" in text and "SECRET-IMAGE" not in text

    db["sql"]("DROP TRIGGER fail_fill ON virtual_fitting_async_jobs")
    service(db).handle(parse_webhook(success_body(task)))  # 다음 알림에서 다시 보완
    assert refs(db) == (IMAGE_URL, IMAGE_UUID)


def test_fill_waits_for_a_concurrent_state_change_and_respects_it(db, queue_settings):  # noqa: F811
    """다른 트랜잭션이 행을 잠근 채 fallback 으로 바꾸는 동안 보완이 기다렸다가, 바뀐 상태로 판단해 쓰지 않는다."""
    import threading
    import time

    from tests.test_virtual_fitting_async_job_state import lock_waiters

    task = uuid_only_success(db)
    holder = db["connect"]()
    holder.execute(
        "SELECT 1 FROM virtual_fitting_async_jobs WHERE fitting_job_id = %s FOR UPDATE", (JOB,)
    )
    results = []
    thread = threading.Thread(
        target=lambda: results.append(
            db["state"].fill_pruna_image_ref(task, image_url=IMAGE_URL, image_uuid=IMAGE_UUID)
        )
    )
    thread.start()
    deadline = time.monotonic() + 3
    while lock_waiters(db) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert lock_waiters(db) >= 1  # 실제로 잠금을 기다리는 중
    holder.execute(
        "UPDATE virtual_fitting_async_jobs SET status = 'FALLBACK_RUNNING', "
        "next_action = 'RUN_FALLBACK' WHERE fitting_job_id = %s",
        (JOB,),
    )
    holder.commit()
    holder.close()
    thread.join(5)
    (change,) = results
    assert not change.applied and change.job.status is AsyncJobStatus.FALLBACK_RUNNING
    assert refs(db) == (None, IMAGE_UUID)


def test_service_uses_update_time_conditions_not_its_earlier_read(db, queue_settings):  # noqa: F811
    """서비스가 후처리 중이라고 읽은 뒤 보완 직전에 완료로 바뀌어도 쓰지 않는다."""
    task = uuid_only_success(db)
    real = db["state"]

    class CompletesFirst:
        def __getattr__(self, name):
            return getattr(real, name)

        def fill_pruna_image_ref(self, *args, **kwargs):
            db["sql"](
                "UPDATE virtual_fitting_async_jobs SET status = 'COMPLETED', next_action = NULL "
                "WHERE fitting_job_id = %s",
                (JOB,),
            )
            return real.fill_pruna_image_ref(*args, **kwargs)

    service(db, state=CompletesFirst()).handle(parse_webhook(success_body(task)))
    assert refs(db) == (None, IMAGE_UUID) and job(db).status is AsyncJobStatus.COMPLETED
