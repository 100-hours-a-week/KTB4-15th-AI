"""11단계: POSTPROCESS 핸들러 (S3 확인 → 제목·코멘트 → V2 결과 저장).

실제 S3·Runware LLM 은 부르지 않는다. 10단계 FollowUpProcessor(실행 권한·keepalive·삭제 판단) + 이 핸들러 + 실제
AsyncJobStateRepository(격리 PostgreSQL)로 돌리고, S3·LLM·상품 조회만 가짜로 바꾼다. 시간은 가짜 시계다.
"""

import threading
from types import SimpleNamespace

import pytest

from app.clients.s3 import ImageStorageError, S3ObjectCheck, S3ObjectState
from app.virtual_fitting.exceptions import FittingDatabaseError, FittingPostprocessError
from app.virtual_fitting.models import FittingProduct
from app.virtual_fitting.postprocess_handler import PostprocessHandler
from app.virtual_fitting.postprocess_worker import (
    HandlerContext,
    HandlerResult,
    HandlerStatus,
    PostprocessHandlers,
)
from app.virtual_fitting.providers.runware_comment import RunwareCommentProvider
from app.virtual_fitting.pruna_submitter import KeepaliveState, NextStep
from app.virtual_fitting.repositories.async_job_repository import AsyncJobStatus, NextAction
from app.virtual_fitting.service import FALLBACK_COMMENT, FALLBACK_TITLE
from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_postprocess_worker import (
    ReadyPaths,
    counts,
    item,
    postprocessing,
    processor,
)
from tests.test_virtual_fitting_pruna_submit import JOB, FakeClock, db, job  # noqa: F401
from tests.test_virtual_fitting_request_worker import FakeSqs

KEY = f"virtual-fitting/results/{JOB}/pruna-async.jpg"
EXISTS, NOT_FOUND = S3ObjectState.EXISTS, S3ObjectState.NOT_FOUND
DENIED, TRANSIENT, ERROR = (
    S3ObjectState.ACCESS_DENIED,
    S3ObjectState.TRANSIENT_ERROR,
    S3ObjectState.ERROR,
)


class FakeStorage:
    """HEAD 결과를 차례로 돌려주고, 부른 시각(가짜 시계)과 발급한 GET URL 을 기록한다."""

    def __init__(self, clock, states=(), presign_error=False):
        self.clock, self.states, self.presign_error = clock, list(states), presign_error
        self.heads, self.urls = [], []

    def head_object(self, key):
        self.heads.append((key, self.clock()))
        state = self.states.pop(0) if self.states else NOT_FOUND
        return S3ObjectCheck(state, key, error_code="X" if state is not EXISTS else None)

    def presigned_get_url(self, key, *, expires_in):
        if self.presign_error:
            raise ImageStorageError("signing failed")
        self.urls.append((key, expires_in))
        return f"https://bucket.s3.ap-northeast-2.amazonaws.com/{key}?X-Amz-Signature=SIG{len(self.urls)}"


class FakeComments:
    """comment/title 응답을 차례로. Exception 이면 던진다."""

    def __init__(self, comments=("좋은 코디입니다.\n잘 어울립니다.",), titles=("차분한 데일리룩",)):
        self.comments, self.titles = list(comments), list(titles)
        self.comment_calls, self.title_calls = [], []

    def generate_comment(self, url, summaries):
        self.comment_calls.append((url, tuple(summaries)))
        value = self.comments.pop(0) if self.comments else FittingPostprocessError("no more")
        if isinstance(value, Exception):
            raise value
        return value

    def generate_title(self, comment):
        self.title_calls.append(comment)
        value = self.titles.pop(0) if self.titles else FittingPostprocessError("no more")
        if isinstance(value, Exception):
            raise value
        return value


class FakeProducts:
    def __init__(self, summary="화이트 셔츠"):
        self.summary, self.calls = summary, []

    def find_by_codes(self, codes):
        self.calls.append(tuple(codes))
        return [
            FittingProduct(str(int(c)), "https://img", "상의", "셔츠", self.summary) for c in codes
        ]


def env(db, states=(), comments=None, **kwargs):  # noqa: F811
    clock = FakeClock()
    storage = FakeStorage(clock, states, kwargs.pop("presign_error", False))
    llm = comments or FakeComments()
    handler = PostprocessHandler(
        jobs=db["jobs"],
        state=db["state"],
        products=kwargs.pop("products", FakeProducts()),
        storage=storage,
        comments=llm,
        clock=clock,
        sleep=kwargs.pop("sleep", clock.sleep),
        result_url_expires_seconds=600,
    )
    return SimpleNamespace(clock=clock, storage=storage, llm=llm, handler=handler)


def run(db, e, *, follow_up=None, should_stop=lambda: False, sqs=None):  # noqa: F811
    task = job(db).task_uuid
    return processor(
        db,
        PostprocessHandlers(postprocess=e.handler),  # 테스트에서만 POSTPROCESS 단독 주입
        sqs=sqs,
        follow_up=follow_up,
        should_stop=should_stop,
    ).process(item(JOB, task))


def success_payload(title="차분한 데일리룩", comment="좋은 코디입니다.\n잘 어울립니다."):
    return {
        "fitting_job_id": JOB,
        "status": "COMPLETED",
        "result_image_key": KEY,
        "llm_title": title,
        "llm_comment": comment,
    }


def verified(db):  # noqa: F811
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET verified_image_key = %s, image_verified_at = now(), "
        "s3_check_attempt_count = 1 WHERE fitting_job_id = %s",
        (KEY, JOB),
    )


# --- 정상 성공 ---


def test_success_saves_verified_image_result_and_publish_result(db):  # noqa: F811
    postprocessing(db)
    e = env(db, [EXISTS])
    outcome = run(db, e)
    current = job(db)
    assert (outcome.action, outcome.delete) == (
        "handler_done",
        False,
    )  # 결과 발행 경로(14단계) 없음
    assert current.verified_image_key == KEY and current.image_verified_at is not None
    assert current.result_payload == success_payload() and current.result_image_key == KEY
    assert current.next_action is NextAction.PUBLISH_RESULT
    assert current.status is AsyncJobStatus.POSTPROCESSING  # 최종 상태 확정은 14단계
    assert current.result_message_status.value == "PENDING"
    assert counts(db) == (1, 2, 0, 0)  # LLM 호출 2번(코멘트 1·제목 1)
    assert len(e.llm.comment_calls) == 1 and len(e.llm.title_calls) == 1
    assert e.storage.heads == [(KEY, 0.0)]  # async_image_key 로 HEAD 한 번
    # LLM 에는 확인한 결과 이미지의 새 Presigned GET URL 과 상품 설명 요약이 들어간다.
    ((url, summaries),) = e.llm.comment_calls
    assert url.startswith(
        f"https://bucket.s3.ap-northeast-2.amazonaws.com/{KEY}?"
    ) and summaries == ("화이트 셔츠", "화이트 셔츠")
    assert e.storage.urls == [(KEY, 600)]
    assert e.llm.title_calls == ["좋은 코디입니다.\n잘 어울립니다."]
    assert "SIG" not in str(current.result_payload)  # URL 은 저장하지 않는다


def test_result_saved_with_a_ready_publish_path_deletes_the_message(db):  # noqa: F811
    postprocessing(db)
    outcome = run(db, env(db, [EXISTS]), follow_up=ReadyPaths(NextStep.PUBLISH_RESULT))
    assert (outcome.action, outcome.delete) == ("handler_done", True)


# --- S3 ---


def test_s3_bundle_checks_at_0_2_7_seconds_then_second_bundle(db):  # noqa: F811
    postprocessing(db)
    e = env(db, [NOT_FOUND, NOT_FOUND, NOT_FOUND, NOT_FOUND, EXISTS])
    run(db, e)
    times = [t for _, t in e.storage.heads]
    assert times[:3] == [0.0, 2.0, 7.0]  # 첫 묶음
    assert [round(t - times[3], 6) for t in times[3:]] == [0.0, 2.0]  # 두 번째 묶음, 성공 뒤 중단
    assert counts(db)[0] == 2 and job(db).verified_image_key == KEY
    assert all(s <= 0.5 for s in e.clock.sleeps)  # 대기를 나눠 그 사이 종료·연장을 확인


def test_second_bundle_confirmed_not_found_moves_to_fallback_only_db(db):  # noqa: F811
    postprocessing(db)
    e = env(db, [NOT_FOUND] * 6)
    outcome = run(db, e)
    current = job(db)
    assert len(e.storage.heads) == 6
    assert (current.status, current.next_action) == (
        AsyncJobStatus.FALLBACK_RUNNING,
        NextAction.RUN_FALLBACK,
    )
    assert counts(db) == (2, 0, 0, 0)  # fallback 횟수는 13단계가 늘린다
    assert e.llm.comment_calls == []
    assert outcome.delete is False  # RUN_FALLBACK 경로(13단계)가 준비됐다고 보지 않는다


def test_first_bundle_not_found_alone_is_not_fallback(db):  # noqa: F811
    postprocessing(db)
    e = env(db, [NOT_FOUND, NOT_FOUND, NOT_FOUND, NOT_FOUND, NOT_FOUND, TRANSIENT])
    outcome = run(db, e)
    # 두 번째 묶음의 마지막이 일시 오류면 객체가 없다고 단정하지 않는다.
    assert outcome.action == "handler_deferred_kept"
    assert job(db).status is AsyncJobStatus.POSTPROCESSING and counts(db)[0] == 2


@pytest.mark.parametrize("state", [DENIED, ERROR])
def test_access_or_config_error_is_held_not_missing(db, state):  # noqa: F811
    postprocessing(db)
    e = env(db, [state])
    outcome = run(db, e)
    assert outcome.action == "handler_deferred_kept"
    assert len(e.storage.heads) == 1  # 권한·설정 문제는 묶음을 멈춘다
    current = job(db)
    assert current.status is AsyncJobStatus.POSTPROCESSING and current.result_payload is None
    assert counts(db) == (1, 0, 0, 0)


def test_transient_errors_continue_within_the_bundle(db):  # noqa: F811
    postprocessing(db)
    e = env(db, [TRANSIENT, TRANSIENT, EXISTS])
    run(db, e)
    assert len(e.storage.heads) == 3 and job(db).verified_image_key == KEY and counts(db)[0] == 1


def test_verified_image_is_reused_without_head(db):  # noqa: F811
    postprocessing(db)
    verified(db)
    e = env(db)
    run(db, e)
    assert e.storage.heads == [] and job(db).result_payload == success_payload()
    assert counts(db) == (1, 2, 0, 0)


def test_exhausted_s3_checks_with_a_verified_image_still_run_llm(db):  # noqa: F811
    postprocessing(db)
    verified(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET s3_check_attempt_count = 2 WHERE fitting_job_id = %s",
        (JOB,),
    )
    e = env(db)
    run(db, e)
    assert e.storage.heads == [] and len(e.llm.comment_calls) == 1


def test_interrupted_second_bundle_is_not_guessed_as_failed(db):  # noqa: F811
    postprocessing(db)
    # 이전 Worker 가 두 번째 묶음을 확보한 뒤 중단됐다: 횟수 2, 확인 이미지 없음.
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET s3_check_attempt_count = 2 WHERE fitting_job_id = %s",
        (JOB,),
    )
    e = env(db)
    outcome = run(db, e)
    assert outcome.action == "handler_deferred_kept"
    current = job(db)
    assert current.status is AsyncJobStatus.POSTPROCESSING  # fallback 추측 없음
    assert counts(db) == (2, 0, 0, 0)  # 횟수 초기화·추가 없음
    assert e.storage.heads == [] and e.llm.comment_calls == []


def test_s3_counter_is_kept_across_redelivery(db):  # noqa: F811
    postprocessing(db)
    stop = {"now": False}
    clock_holder = {}

    def stopping_sleep(seconds):
        clock_holder["clock"].sleep(seconds)
        stop["now"] = True  # 첫 묶음 대기 중 종료

    e = env(db, [NOT_FOUND], sleep=stopping_sleep)
    clock_holder["clock"] = e.clock
    first = run(db, e, should_stop=lambda: stop["now"])
    assert first.action == "handler_deferred_kept" and counts(db)[0] == 1
    assert len(e.storage.heads) == 1  # 종료 뒤 새 HEAD 없음
    again = env(db, [EXISTS])  # 재수신, 다른 Worker
    run(db, again)
    assert counts(db)[0] == 2 and job(db).verified_image_key == KEY


def test_concurrent_claims_never_exceed_two_bundles(db):  # noqa: F811
    postprocessing(db)
    token = db["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60).token
    results, barrier = [], threading.Barrier(6)

    def claim():
        barrier.wait()
        results.append(db["state"].claim_s3_check_attempt(JOB, token).applied)

    threads = [threading.Thread(target=claim) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert sum(results) == 2 and counts(db)[0] == 2


# --- LLM ---


def test_comment_and_title_each_retry_once_with_new_urls_without_s3_again(db):  # noqa: F811
    postprocessing(db)
    comments = FakeComments(
        comments=(FittingPostprocessError("x"), "코멘트 두 번째", "쓰이지 않음"),
        titles=(FittingPostprocessError("y"), "제목 두 번째", "쓰이지 않음"),
    )
    e = env(db, [EXISTS], comments=comments)
    run(db, e)
    assert len(e.storage.heads) == 1  # S3 확인은 처음 한 번만
    assert [key for key, _ in e.storage.urls] == [KEY, KEY]  # 같은 Key, 코멘트 시도마다 새 URL
    urls = [url for url, _ in comments.comment_calls]
    assert len(urls) == len(set(urls)) == 2  # 코멘트 2회
    assert comments.title_calls == [
        "코멘트 두 번째",
        "코멘트 두 번째",
    ]  # 제목 2회, 같은 성공 코멘트로
    assert counts(db)[1] == 4
    assert job(db).result_payload == success_payload("제목 두 번째", "코멘트 두 번째")


def test_comment_failing_twice_uses_defaults_without_a_title_call(db):  # noqa: F811
    postprocessing(db)
    comments = FakeComments(comments=[FittingPostprocessError("x")] * 5)
    e = env(db, [EXISTS], comments=comments)
    run(db, e)
    assert len(comments.comment_calls) == 2 and counts(db)[1] == 2  # 3번째 호출 없음
    assert comments.title_calls == []  # 코멘트가 없으면 제목을 부르지 않는다
    current = job(db)
    assert current.result_payload == success_payload(FALLBACK_TITLE, FALLBACK_COMMENT)
    assert (
        current.status is AsyncJobStatus.POSTPROCESSING
        and current.next_action is NextAction.PUBLISH_RESULT
    )
    assert counts(db)[2] == 0  # fallback 없음


def test_already_exhausted_llm_attempts_use_default_text_without_calls(db):  # noqa: F811
    postprocessing(db)
    verified(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET llm_generation_attempt_count = 4 WHERE fitting_job_id = %s",
        (JOB,),
    )
    e = env(db)
    run(db, e)
    assert e.llm.comment_calls == [] and e.storage.heads == [] and e.storage.urls == []
    assert job(db).result_payload == success_payload(FALLBACK_TITLE, FALLBACK_COMMENT)
    assert counts(db)[1] == 4


def test_redelivery_uses_only_the_remaining_llm_attempts(db):  # noqa: F811
    postprocessing(db)
    verified(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET llm_generation_attempt_count = 3 WHERE fitting_job_id = %s",
        (JOB,),
    )
    comments = FakeComments(comments=["남은 한 번으로 만든 코멘트"])
    e = env(db, comments=comments)
    run(db, e)
    # 호출 한도(4)가 1번만 남았다: 코멘트 1번으로 다 쓰고, 제목은 부르지 않고 기본 제목(초기화 없음).
    assert len(comments.comment_calls) == 1 and comments.title_calls == [] and counts(db)[1] == 4
    assert len(e.storage.urls) == 1
    assert job(db).result_payload == success_payload(FALLBACK_TITLE, "남은 한 번으로 만든 코멘트")


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_llm_text_is_not_a_success(db, bad):  # noqa: F811
    postprocessing(db)
    comments = FakeComments(comments=[bad, "정상 코멘트"], titles=["정상 제목"])
    e = env(db, [EXISTS], comments=comments)
    run(db, e)
    assert counts(db)[1] == 3 and comments.title_calls == ["정상 코멘트"]
    assert job(db).result_payload == success_payload("정상 제목", "정상 코멘트")


def test_real_provider_makes_one_http_request_per_attempt(db):  # noqa: F811
    """V1 Provider 를 그대로 쓴다. urllib 한 번씩이라 내부 재시도가 없다(코멘트 2회 실패 → 요청 2번, 제목 없음)."""
    postprocessing(db)
    requests = []

    def opener(request, timeout):
        requests.append((request.full_url, timeout))
        raise TimeoutError()

    provider = RunwareCommentProvider("k", opener=opener)
    e = env(db, [EXISTS], comments=provider)
    run(db, e)
    assert len(requests) == 2  # comment 가 두 번 모두 실패해 title 은 부르지 않았다
    assert all(timeout == 30.0 for _, timeout in requests)
    assert job(db).result_payload["llm_comment"] == FALLBACK_COMMENT


def test_presign_failure_is_an_s3_error_not_an_llm_attempt(db):  # noqa: F811
    postprocessing(db)
    verified(db)
    e = env(db, presign_error=True)
    outcome = run(db, e)
    assert outcome.action == "handler_deferred_kept"
    assert counts(db)[1] == 0 and e.llm.comment_calls == [] and e.llm.title_calls == []
    assert job(db).result_payload is None  # LLM 시도를 쓰지 않았고 결과도 저장하지 않았다


def test_no_description_summary_uses_default_text(db):  # noqa: F811
    postprocessing(db)
    e = env(db, [EXISTS], products=FakeProducts(summary=None))
    run(db, e)
    assert e.llm.comment_calls == []
    assert job(db).result_payload["llm_comment"] == FALLBACK_COMMENT


# --- 중복·권한·저장 ---


def test_saved_result_is_not_reprocessed(db):  # noqa: F811
    postprocessing(db)
    run(db, env(db, [EXISTS]))
    saved = job(db)
    e = env(db, [EXISTS])
    outcome = run(db, e)  # 재수신: 이제 PUBLISH_RESULT 라 핸들러를 부르지 않는다
    assert outcome.action == "state_mismatch"
    assert e.storage.heads == [] and e.llm.comment_calls == [] and job(db) == saved
    # 핸들러가 직접 불려도 결과가 있으면 외부 호출 없이 끝낸다.
    context = HandlerContext(SimpleNamespace(beat=lambda: True), KeepaliveState(), lambda: False)
    direct = e.handler(saved, "token", context)
    assert direct == HandlerResult(HandlerStatus.DONE, "result_already_saved")
    assert e.storage.heads == []


def test_stop_and_extension_failure_stop_new_external_calls(db):  # noqa: F811
    postprocessing(db)
    verified(db)
    stop = {"now": False}
    comments = FakeComments(comments=[FittingPostprocessError("x")] * 5)
    original = comments.generate_comment

    def comment_then_stop(url, summaries):
        stop["now"] = True  # 첫 LLM 시도 중 종료 요청
        return original(url, summaries)

    comments.generate_comment = comment_then_stop
    e = env(db, comments=comments)
    outcome = run(db, e, should_stop=lambda: stop["now"])
    assert outcome.action == "handler_deferred_kept"
    assert len(comments.comment_calls) == 1 and counts(db)[1] == 1  # 다음 시도를 시작하지 않았다
    assert job(db).result_payload is None


def test_visibility_failure_stops_before_the_next_head(db):  # noqa: F811
    postprocessing(db)
    sqs = FakeSqs()
    clock_holder = {}

    def failing_sleep(seconds):
        clock_holder["clock"].sleep(seconds)
        sqs.visibility_ok = False  # 대기 중 Visibility 연장 실패

    e = env(db, [NOT_FOUND, NOT_FOUND], sleep=failing_sleep)
    clock_holder["clock"] = e.clock
    outcome = run(db, e, sqs=sqs)
    assert outcome.action == "keepalive_failed_kept" and outcome.delete is False
    assert len(e.storage.heads) == 1


def test_expired_lease_rejects_the_late_result_save(db):  # noqa: F811
    """연장 확인을 통과했다고 해도(이전 Worker 의 오래된 판단) 만료된 토큰의 저장은 DB 가 거절한다."""
    postprocessing(db)
    verified(db)
    token = db["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60).token
    comments = FakeComments()
    original = comments.generate_title

    def title_after_expiry(comment):
        db["sql"](
            "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second' "
            "WHERE fitting_job_id = %s",
            (JOB,),
        )
        return original(comment)

    comments.generate_title = title_after_expiry
    e = env(db, comments=comments)
    context = HandlerContext(SimpleNamespace(beat=lambda: True), KeepaliveState(), lambda: False)
    result = e.handler(job(db), token, context)
    assert result == HandlerResult(HandlerStatus.DEFERRED, "save_lease_invalid")
    assert job(db).result_payload is None


def test_expired_lease_during_the_worker_stops_before_saving(db):  # noqa: F811
    postprocessing(db)
    verified(db)
    comments = FakeComments()
    original = comments.generate_title

    def title_after_expiry(comment):
        db["sql"](
            "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second' "
            "WHERE fitting_job_id = %s",
            (JOB,),
        )
        return original(comment)

    comments.generate_title = title_after_expiry
    outcome = run(db, env(db, comments=comments))
    assert (outcome.action, outcome.delete) == ("keepalive_failed_kept", False)
    assert job(db).result_payload is None


def test_db_error_does_not_reset_or_add_attempts(db, monkeypatch):  # noqa: F811
    postprocessing(db)
    verified(db)
    real = db["state"].save_final_result

    def broken(*args, **kwargs):
        raise FittingDatabaseError("down")

    monkeypatch.setattr(db["state"], "save_final_result", broken)
    first = run(db, env(db))
    assert first.action == "handler_error_kept" and counts(db)[1] == 2  # 코멘트 1·제목 1 호출
    monkeypatch.setattr(db["state"], "save_final_result", real)
    comments = FakeComments(comments=[FittingPostprocessError("x")] * 5)
    run(db, env(db, comments=comments))
    assert counts(db)[1] == 4 and len(comments.comment_calls) == 2  # 남은 2번만, 한도 그대로
    assert job(db).result_payload == success_payload(FALLBACK_TITLE, FALLBACK_COMMENT)


def test_postprocess_alone_is_still_not_ready():
    from app.virtual_fitting import postprocess_worker as pw

    handlers = pw.connected_postprocess_handlers()
    assert isinstance(handlers.postprocess, PostprocessHandler)
    alone = PostprocessHandlers(
        postprocess=handlers.postprocess
    )  # 한쪽만으로 실제 큐를 소비하지 않는다
    assert alone.missing() == ["run_fallback"]
    with pytest.raises(ValueError):
        pw.create_postprocess_worker(alone)


# --- 코멘트는 성공, 제목만 실패 ---


def test_title_failing_twice_keeps_the_generated_comment_with_the_default_title(db):  # noqa: F811
    postprocessing(db)
    comments = FakeComments(
        comments=("코멘트 1", "코멘트 2", "코멘트 3"),
        titles=[FittingPostprocessError("title failed")] * 4,
    )
    e = env(db, [EXISTS], comments=comments)
    outcome = run(db, e)

    # 코멘트는 첫 시도에 성공했으므로 다시 만들지 않는다. 제목만 같은 코멘트로 최대 2회.
    assert len(comments.comment_calls) == 1
    assert comments.title_calls == ["코멘트 1", "코멘트 1"]
    assert counts(db)[1] == 3
    # 이미지 URL 은 코멘트 시도에만(제목에는 이미지가 필요 없다).
    assert [key for key, _ in e.storage.urls] == [KEY]
    # 제목 2회 모두 실패 → 생성 코멘트 유지 + 기본 제목, 이미지 성공 유지
    current = job(db)
    assert current.result_payload == success_payload(FALLBACK_TITLE, "코멘트 1")
    assert current.result_payload["status"] == "COMPLETED"
    assert current.result_image_key == KEY and current.verified_image_key == KEY
    assert current.next_action is NextAction.PUBLISH_RESULT
    assert counts(db) == (1, 3, 0, 0)  # S3 1묶음, fallback 없음
    assert len(e.storage.heads) == 1 and outcome.action == "handler_done"


def test_title_only_failures_through_the_real_v1_provider_http_calls(db):  # noqa: F811
    """V1 RunwareCommentProvider 를 그대로 쓰고 HTTP 계층(opener)만 가짜로 둔다."""
    import io
    import json
    from urllib.error import HTTPError

    postprocessing(db)
    requests = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def opener(request, timeout):
        payload = json.loads(request.data)
        user = payload["messages"][1]["content"]
        kind = "comment" if isinstance(user, list) else "title"
        requests.append((kind, user if kind == "title" else None))
        if kind == "comment":
            n = sum(1 for k, _ in requests if k == "comment")
            body = {"choices": [{"message": {"content": f"코멘트 {n}"}, "finish_reason": "stop"}]}
            return Response(json.dumps(body).encode())
        raise HTTPError(request.full_url, 500, "server error", {}, io.BytesIO(b"{}"))

    provider = RunwareCommentProvider("k", opener=opener)
    e = env(db, [EXISTS], comments=provider)
    run(db, e)

    assert [kind for kind, _ in requests] == [
        "comment",
        "title",
        "title",
    ]  # 실제 HTTP 3번, 4번째 없음
    # 제목 재시도에도 같은 성공 코멘트(V1 build_title_request 형식), 코멘트는 다시 만들지 않는다.
    assert [text for kind, text in requests if kind == "title"] == ["코디 설명: 코멘트 1"] * 2
    assert counts(db)[1] == 3
    assert job(db).result_payload == success_payload(FALLBACK_TITLE, "코멘트 1")


@pytest.mark.parametrize("title_failure", ["length", "http_400"])
def test_title_token_limit_failures_use_two_title_attempts_then_the_default_title(
    db,  # noqa: F811
    title_failure,
):
    """제목이 출력 한도로 실패하는 두 형태(HTTP 200 finish_reason=length, HTTP 400)를 실제 V1 Provider 로."""
    import io
    import json
    from urllib.error import HTTPError

    postprocessing(db)
    sent = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def opener(request, timeout):
        payload = json.loads(request.data)
        user = payload["messages"][1]["content"]
        kind = "comment" if isinstance(user, list) else "title"
        sent.append((kind, payload["max_completion_tokens"], user if kind == "title" else None))
        if kind == "comment":
            n = sum(1 for k, *_ in sent if k == "comment")
            body = {"choices": [{"message": {"content": f"코멘트 {n}"}, "finish_reason": "stop"}]}
            return Response(json.dumps(body).encode())
        if title_failure == "length":
            body = {"choices": [{"message": {"content": "잘린"}, "finish_reason": "length"}]}
            return Response(json.dumps(body).encode())
        raise HTTPError(
            request.full_url, 400, "bad request", {}, io.BytesIO(b'{"error": {"message": "limit"}}')
        )

    e = env(db, [EXISTS], comments=RunwareCommentProvider("k", opener=opener))
    run(db, e)

    # 재시도에도 같은 토큰 한도(코멘트 512·제목 1024), 제목 재시도는 같은 성공 코멘트로
    assert [(kind, limit) for kind, limit, _ in sent] == [
        ("comment", 512),
        ("title", 1024),
        ("title", 1024),
    ]
    assert [text for kind, _, text in sent if kind == "title"] == ["코디 설명: 코멘트 1"] * 2
    current = job(db)
    assert current.result_payload == success_payload(FALLBACK_TITLE, "코멘트 1")
    assert current.verified_image_key == KEY and current.next_action is NextAction.PUBLISH_RESULT
    assert counts(db) == (1, 3, 0, 0)  # S3 확인 1회 유지, LLM 호출 3번, fallback·결과 전송 0
    assert len(e.storage.heads) == 1  # 이미지 재확인·재생성 없음
    run(db, e)  # 재수신: 결과가 있어 추가 호출 없음
    assert len(sent) == 3 and counts(db) == (1, 3, 0, 0)


# --- 항목별 재시도 정책(코멘트·제목 각각 최초 1회 + 재시도 1회) ---


def test_title_retry_succeeds_with_the_same_first_comment(db):  # noqa: F811
    postprocessing(db)
    comments = FakeComments(
        comments=["첫 코멘트", "쓰이지 않음"], titles=[FittingPostprocessError("x"), "두 번째 제목"]
    )
    e = env(db, [EXISTS], comments=comments)
    run(db, e)
    assert len(comments.comment_calls) == 1  # 제목 실패로 코멘트를 다시 만들지 않는다
    assert comments.title_calls == ["첫 코멘트", "첫 코멘트"]
    assert job(db).result_payload == success_payload("두 번째 제목", "첫 코멘트")
    assert counts(db) == (1, 3, 0, 0)


FAILURE_KINDS = ["http_500", "timeout", "length", "content_filter", "empty"]


def _scripted_provider(failures):
    """실제 RunwareCommentProvider 에 HTTP 계층만 가짜. failures: {"comment": [...], "title": [...]} 차례로 쓰는 응답 종류."""
    import io
    import json
    from urllib.error import HTTPError

    sent = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def reply(content, finish_reason="stop"):
        return Response(
            json.dumps(
                {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
            ).encode()
        )

    def opener(request, timeout):
        payload = json.loads(request.data)
        user = payload["messages"][1]["content"]
        kind = "comment" if isinstance(user, list) else "title"
        sent.append((kind, payload["max_completion_tokens"], user if kind == "title" else None))
        mode = failures[kind].pop(0) if failures[kind] else "ok"
        if mode == "http_500":
            raise HTTPError(request.full_url, 500, "x", {}, io.BytesIO(b"{}"))
        if mode == "timeout":
            raise TimeoutError()
        if mode == "length":
            return reply("잘린", finish_reason="length")
        if mode == "content_filter":
            return reply("차단", finish_reason="content_filter")
        if mode == "empty":
            return reply("   ")
        if isinstance(mode, str) and mode.startswith("text:"):
            return reply(mode[5:])
        return reply("생성 코멘트" if kind == "comment" else "생성 제목")

    return RunwareCommentProvider("k", opener=opener), sent


@pytest.mark.parametrize("failure", FAILURE_KINDS)
def test_each_failure_kind_gets_one_retry_per_item(db, failure):  # noqa: F811
    postprocessing(db)
    provider, sent = _scripted_provider({"comment": [failure], "title": [failure]})
    e = env(db, [EXISTS], comments=provider)
    run(db, e)
    assert [(kind, limit) for kind, limit, _ in sent] == [
        ("comment", 512),
        ("comment", 512),
        ("title", 1024),
        ("title", 1024),
    ]  # 항목별 2번, 재시도도 같은 토큰 한도, Provider 내부 재시도 없음
    assert [text for kind, _, text in sent if kind == "title"] == ["코디 설명: 생성 코멘트"] * 2
    assert job(db).result_payload == success_payload("생성 제목", "생성 코멘트")


@pytest.mark.parametrize("failure", FAILURE_KINDS)
def test_title_failing_twice_with_each_kind_keeps_the_comment(db, failure):  # noqa: F811
    postprocessing(db)
    provider, sent = _scripted_provider({"comment": [], "title": [failure, failure]})
    run(db, env(db, [EXISTS], comments=provider))
    assert [kind for kind, _, _ in sent] == ["comment", "title", "title"]
    assert job(db).result_payload == success_payload(FALLBACK_TITLE, "생성 코멘트")


@pytest.mark.parametrize("failure", FAILURE_KINDS)
def test_comment_failing_twice_with_each_kind_skips_the_title(db, failure):  # noqa: F811
    postprocessing(db)
    provider, sent = _scripted_provider({"comment": [failure, failure], "title": []})
    run(db, env(db, [EXISTS], comments=provider))
    assert [kind for kind, _, _ in sent] == ["comment", "comment"]  # 제목 호출 없음
    assert job(db).result_payload == success_payload(FALLBACK_TITLE, FALLBACK_COMMENT)


def test_title_length_boundary_100_passes_and_101_fails(db):  # noqa: F811
    postprocessing(db)
    exactly_100 = "가" * 100
    provider, sent = _scripted_provider(
        {"comment": [], "title": ["text:" + "가" * 101, f"text:  {exactly_100}  "]}
    )
    run(db, env(db, [EXISTS], comments=provider))
    assert [kind for kind, _, _ in sent] == ["comment", "title", "title"]  # 101자는 실패 → 재시도
    assert job(db).result_payload == success_payload(
        exactly_100, "생성 코멘트"
    )  # 공백 제거 후 100자 통과


def test_title_too_long_twice_uses_the_default_title_without_truncating(db):  # noqa: F811
    postprocessing(db)
    long_title = "나" * 150
    provider, sent = _scripted_provider({"comment": [], "title": ["text:" + long_title] * 2})
    e = env(db, [EXISTS], comments=provider)
    run(db, e)
    payload = job(db).result_payload
    assert [kind for kind, _, _ in sent] == ["comment", "title", "title"]
    assert payload == success_payload(FALLBACK_TITLE, "생성 코멘트")
    assert long_title[:100] not in payload["llm_title"]  # 잘라서 저장하지 않는다
    # 텍스트 실패로 이미지 재확인·fallback 없음, 정상 이미지로 완료
    assert counts(db) == (1, 3, 0, 0) and len(e.storage.heads) == 1
    assert job(db).next_action is NextAction.PUBLISH_RESULT and job(db).verified_image_key == KEY


def test_short_titles_outside_the_prompt_target_are_accepted(db):  # noqa: F811
    postprocessing(db)
    provider, _ = _scripted_provider({"comment": [], "title": ["text:룩", "text:" + "다" * 60]})
    run(db, env(db, [EXISTS], comments=provider))
    assert job(db).result_payload == success_payload(
        "룩", "생성 코멘트"
    )  # 10~20자 목표 밖이어도 통과


def test_stop_between_comment_and_title_defers_without_default_text(db):  # noqa: F811
    postprocessing(db)
    comments = FakeComments(comments=["코멘트"], titles=["제목"])
    e = env(db, [EXISTS], comments=comments)
    stop = {"now": False}
    original = comments.generate_comment

    def comment_then_stop(url, summaries):
        value = original(url, summaries)
        stop["now"] = True  # 코멘트 직후 종료 요청
        return value

    comments.generate_comment = comment_then_stop
    outcome = run(db, e, should_stop=lambda: stop["now"])
    assert outcome.delete is False and comments.title_calls == []
    assert job(db).result_payload is None  # 기본 문구로 성공 처리하지 않는다
    assert counts(db)[1] == 1


def test_completed_job_redelivery_makes_no_calls_or_saves(db):  # noqa: F811
    postprocessing(db)
    comments = FakeComments(comments=[FittingPostprocessError("x"), "코멘트"], titles=["제목"])
    e = env(db, [EXISTS], comments=comments)
    run(db, e)
    saved = job(db)
    assert saved.result_payload == success_payload("제목", "코멘트")
    calls = (len(comments.comment_calls), len(comments.title_calls), len(e.storage.urls))
    run(db, e)  # 같은 메시지 재전달
    assert (len(comments.comment_calls), len(comments.title_calls), len(e.storage.urls)) == calls
    assert job(db) == saved  # 결과·횟수·updated_at 그대로


# --- URL 생성과 횟수 확보의 순서 ---


def _context():
    return HandlerContext(SimpleNamespace(beat=lambda: True), KeepaliveState(), lambda: False)


def test_claim_rejected_after_url_by_the_limit_makes_no_llm_call(db):  # noqa: F811
    """앞의 조회(횟수 3)와 확보 사이에 다른 Worker 가 마지막 호출을 가져갔다 → 확보가 거절돼 호출 없음."""
    postprocessing(db)
    verified(db)
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET llm_generation_attempt_count = 3 WHERE fitting_job_id = %s",
        (JOB,),
    )
    e = env(db)
    original = e.storage.presigned_get_url

    def presign_then_race(key, *, expires_in):
        url = original(key, expires_in=expires_in)
        db["sql"](
            "UPDATE virtual_fitting_async_jobs SET llm_generation_attempt_count = 4 "
            "WHERE fitting_job_id = %s",
            (JOB,),
        )
        return url

    e.storage.presigned_get_url = presign_then_race
    run(db, e)
    assert len(e.storage.urls) == 1 and e.llm.comment_calls == [] and e.llm.title_calls == []
    assert counts(db)[1] == 4  # 조건부 확보가 한도를 지켰다(5가 되지 않음)
    assert job(db).result_payload == success_payload(FALLBACK_TITLE, FALLBACK_COMMENT)


def test_claim_rejected_after_url_by_a_lost_lease_makes_no_llm_call(db):  # noqa: F811
    postprocessing(db)
    verified(db)
    token = db["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60).token
    e = env(db)
    original = e.storage.presigned_get_url

    def presign_then_expire(key, *, expires_in):
        url = original(key, expires_in=expires_in)
        db["sql"](
            "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second' "
            "WHERE fitting_job_id = %s",
            (JOB,),
        )
        return url

    e.storage.presigned_get_url = presign_then_expire
    result = e.handler(job(db), token, _context())
    assert result == HandlerResult(HandlerStatus.DEFERRED, "llm_claim_lease_invalid")
    assert len(e.storage.urls) == 1 and e.llm.comment_calls == []
    assert counts(db)[1] == 0 and job(db).result_payload is None


def test_lost_lease_before_the_loop_makes_no_url_or_call(db):  # noqa: F811
    postprocessing(db)
    verified(db)
    token = db["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60).token
    db["sql"](
        "UPDATE virtual_fitting_async_jobs SET worker_lease_until = now() - interval '1 second' "
        "WHERE fitting_job_id = %s",
        (JOB,),
    )
    e = env(db)
    result = e.handler(job(db), token, _context())
    assert result == HandlerResult(HandlerStatus.DEFERRED, "llm_lease_lost")
    assert e.storage.urls == [] and e.llm.comment_calls == [] and counts(db)[1] == 0
    assert job(db).result_payload is None


def test_result_saved_meanwhile_is_not_overwritten(db):  # noqa: F811
    """핸들러가 받은 기록에는 결과가 없지만, 반복 시작 때 DB 에는 이미 결과가 있다."""
    postprocessing(db)
    verified(db)
    token = db["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60).token
    stale = job(db)
    other = success_payload("다른 Worker 제목", "다른 Worker 코멘트")
    assert db["state"].save_final_result(JOB, token, other, KEY).applied
    e = env(db)
    result = e.handler(stale, token, _context())
    assert result == HandlerResult(HandlerStatus.DONE, "result_already_saved")
    assert e.storage.urls == [] and e.llm.comment_calls == []
    assert job(db).result_payload == other and counts(db)[1] == 0


def test_result_saved_between_attempts_stops_the_loop_without_overwrite(db):  # noqa: F811
    postprocessing(db)
    verified(db)
    token = db["state"].acquire_lease(JOB, NextAction.POSTPROCESS, 60).token
    other = success_payload("다른 제목", "다른 코멘트")
    comments = FakeComments(comments=["코멘트 1"], titles=[FittingPostprocessError("x")])
    original = comments.generate_title

    def title_fails_after_someone_saved(comment):
        assert db["state"].save_final_result(JOB, token, other, KEY).applied
        return original(comment)

    comments.generate_title = title_fails_after_someone_saved
    e = env(db, comments=comments)
    result = e.handler(job(db), token, _context())
    assert result == HandlerResult(HandlerStatus.DONE, "result_already_saved")
    assert len(e.storage.urls) == 1 and counts(db)[1] == 2  # 제목 재시도 호출 없음
    assert job(db).result_payload == other
