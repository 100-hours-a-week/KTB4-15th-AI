"""8단계 보완 회귀 테스트: (1) Runware 대기 안내를 줄이거나 놓치지 않는지, (2) 사용량 보정과 작업별 비용 기록이
순서·겹침과 관계없이 같은 비용을 두 번 세지 않는지.

처음에는 xfail(strict=True) 로 문제를 재현했던 테스트들이다. 조건은 그대로 두고 표시만 지워 일반 테스트로 바꿨다.
실제 Runware·AWS 는 부르지 않고, 시간은 가짜 시계나 DB 의 next_send_at 을 옮겨 흉내 낸다(실제 600초 대기 없음).
"""

import io
import threading
import time
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError

import pytest

from app.virtual_fitting.budget import FittingBudget
from app.virtual_fitting.exceptions import FittingDailyBudgetExceededError
from app.virtual_fitting.providers.base import FittingInput
from app.virtual_fitting.providers.runware_async import (
    RetryAfter,
    RunwarePrunaAsyncProvider,
    SubmissionOutcome,
    classify_response,
    parse_retry_after,
)
from app.virtual_fitting.repositories.pruna_limiter import (
    SubmissionGrant,
    SubmissionReason,
    TaskCostRepository,
)
from app.virtual_fitting.repositories.usage_repository import UsageRepository
from tests.test_virtual_fitting_async_job_state import (  # noqa: F401 (pytest fixture)
    lock_waiters,
    pg,
    run_contended,
)
from tests.test_virtual_fitting_pruna_submit import (  # noqa: F401 (pytest fixture)
    LIMITS_SQL,
    USAGE_SQL,
    FakeClock,
    FakeProvider,
    ScriptedLimiter,
    build,
    counts,
    db,
    job,
    limiter,
    ready_job,
)
from tests.test_virtual_fitting_request_worker import FakeSqs, valid_message

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SQL = ROOT / "scripts" / "db" / "usage_amount_split_totals.sql"
TASK = "a770f077-f413-47de-9dac-be0b26a35da6"


def seconds_until_next_send(db):  # noqa: F811
    return db["sql"](
        "SELECT EXTRACT(EPOCH FROM next_send_at - statement_timestamp()) FROM virtual_fitting_pruna_limiter"
    )[0][0]


def elapse(db, seconds):  # noqa: F811
    """DB 시계를 앞당기는 대신 공유 대기 시각을 그만큼 당겨, seconds 가 지난 상황을 만든다."""
    db["sql"](
        "UPDATE virtual_fitting_pruna_limiter SET next_send_at = next_send_at - make_interval(secs => %s)",
        (seconds,),
    )


def submit_with_retry_after(db, retry_after):  # noqa: F811
    """Runware 가 429(UNKNOWN rate_limited)와 retryAfter 를 돌려준 제출 한 번. 같은 작업을 다시 보내지 않는다."""
    provider = FakeProvider(
        SubmissionOutcome.UNKNOWN, reason="rate_limited", http_status=429, retry_after=retry_after
    )
    processor, _ = build(db, provider)
    processor.process(valid_message())
    assert len(provider.calls) == 1
    return provider


class HttpErrorOpener:
    def __init__(self, error):
        self.error, self.calls = error, 0

    def __call__(self, request, timeout):
        self.calls += 1
        raise self.error


def http_429(
    body=b'{"errors": [{"code": "serviceOverloaded", "message": "queue full"}]}', **headers
):
    return HTTPError("u", 429, "Too Many Requests", headers, io.BytesIO(body))


def submit_real_provider(error):
    return RunwarePrunaAsyncProvider("k", opener=HttpErrorOpener(error)).submit(
        FittingInput("https://p", ["https://g"], "x"),
        task_uuid=TASK,
        webhook_url="https://hook",
        upload_endpoint="https://put",
    )


# --- 1. 대기 안내: DB 공유 대기 시각 ---


def test_600s_guidance_is_stored_as_600s_in_the_shared_wait(db):  # noqa: F811
    submit_with_retry_after(db, 600)
    assert seconds_until_next_send(db) > 590


def test_after_300s_of_a_600s_guidance_no_worker_may_send(db):  # noqa: F811
    submit_with_retry_after(db, 600)
    elapse(db, 300)
    other_token, other_task = ready_job(db, 7)
    # 다른 Worker(다른 PrunaLimiter 인스턴스, 같은 DB)도 같은 대기 시각을 본다.
    grant = limiter(db, interval=0).begin_submission(7, other_token, other_task)
    assert (grant.granted, grant.reason) == (False, SubmissionReason.INTERVAL_WAIT)
    assert grant.wait_seconds > 290
    assert job(db, 7).pruna_submit_started_at is None


def test_after_the_full_600s_other_limits_decide(db):  # noqa: F811
    submit_with_retry_after(db, 600)
    elapse(db, 601)
    other_token, other_task = ready_job(db, 7)
    grant = limiter(db, interval=0).begin_submission(7, other_token, other_task)
    # 대기 시각은 지났다. 진행 중 자리(UNKNOWN 이 1개 쥠, 한도 3)·전송 자리가 남아 있어 허용된다.
    assert grant.granted
    assert counts(db) == (2, 1)


def test_after_the_full_600s_a_full_in_flight_limit_still_blocks(db):  # noqa: F811
    submit_with_retry_after(db, 600)
    elapse(db, 601)
    other_token, other_task = ready_job(db, 7)
    # 대기 안내가 끝나도 진행 중 작업 한도(여기서는 1, UNKNOWN 작업이 쥠)로 판단한다.
    grant = limiter(db, in_flight=1, interval=0).begin_submission(7, other_token, other_task)
    assert grant.reason is SubmissionReason.IN_FLIGHT_FULL


def test_after_the_full_600s_the_minimum_interval_still_applies(db):  # noqa: F811
    submit_with_retry_after(db, 600)
    elapse(db, 601)
    shared = limiter(db, interval=10)
    first_token, first_task = ready_job(db, 7)
    second_token, second_task = ready_job(db, 8)
    assert shared.begin_submission(7, first_token, first_task).granted
    wait = shared.begin_submission(8, second_token, second_task)
    assert wait.reason is SubmissionReason.INTERVAL_WAIT and 9 < wait.wait_seconds <= 10


def test_shared_wait_applies_to_other_workers(db):  # noqa: F811
    limiter(db, interval=0).defer_sends(600)
    other_token, other_task = ready_job(db, 7)
    grant = limiter(db, interval=0).begin_submission(7, other_token, other_task)
    assert grant.reason is SubmissionReason.INTERVAL_WAIT and grant.wait_seconds > 590


def test_shorter_guidance_never_brings_the_wait_forward(db):  # noqa: F811
    shared = limiter(db, interval=0)
    shared.defer_sends(600)
    shared.defer_sends(30)
    assert seconds_until_next_send(db) > 590


def test_shorter_guidance_through_the_submit_path_does_not_shorten_an_existing_wait(db):  # noqa: F811
    limiter(db, interval=0).defer_sends(600)
    elapse(db, 600)  # 대기가 막 끝났다고 하고 제출이 진행된 뒤
    submit_with_retry_after(db, 30)
    first = seconds_until_next_send(db)
    limiter(db, interval=0).defer_sends(600)
    submit_seconds = seconds_until_next_send(db)
    limiter(db, interval=0).defer_sends(5)
    assert 20 < first <= 31 and submit_seconds > 590 and seconds_until_next_send(db) > 590


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5, 0, True, "600", None])
def test_defer_sends_ignores_values_that_are_not_a_positive_finite_number(db, bad):  # noqa: F811
    before = seconds_until_next_send(db)
    limiter(db, interval=0).defer_sends(bad)
    assert seconds_until_next_send(db) == pytest.approx(before, abs=1)


def test_no_automatic_resend_of_the_same_request_after_the_wait(db):  # noqa: F811
    provider = submit_with_retry_after(db, 600)
    elapse(db, 601)
    # 같은 메시지가 다시 와도(Visibility 만료 등) 호출 시작 기록이 있으므로 같은 생성 요청을 다시 보내지 않는다.
    processor, _ = build(db, provider)
    processor.process(valid_message())
    assert len(provider.calls) == 1


# --- 1. 대기 안내: Worker 가 기다리는 방식(호출 허용 시각과 별개) ---


def test_worker_waits_in_short_slices_and_handles_stop_during_a_long_wait(db):  # noqa: F811
    clock = FakeClock()
    stop = {"now": False}
    scripted = ScriptedLimiter(limiter(db, interval=0), [SubmissionReason.INTERVAL_WAIT] * 100)
    scripted_wait = SubmissionGrant(False, SubmissionReason.INTERVAL_WAIT, wait_seconds=600)
    scripted.begin_submission = lambda *args: scripted_wait

    def sleep(seconds):
        clock.sleep(seconds)
        if len(clock.sleeps) == 3:
            stop["now"] = True  # 기다리는 중 종료 신호

    provider = FakeProvider()
    processor, _ = build(
        db,
        provider,
        limiter_=scripted,
        clock=clock,
        sleep=sleep,
        poll_seconds=2,
        max_wait_seconds=900,
        should_stop=lambda: stop["now"],
    )
    processor.process(valid_message())
    # 한 번에 600초를 자지 않고 2초씩 나눠 기다렸고, 종료 신호 뒤 바로 멈췄다(시작 기록·전송 없음).
    assert clock.sleeps == [2, 2, 2]
    assert provider.calls == [] and job(db).pruna_submit_started_at is None


def test_worker_extends_visibility_while_waiting_and_stops_when_it_fails(db):  # noqa: F811
    limiter(db, interval=0).defer_sends(600)  # 실제 DB 공유 대기(600초)
    clock = FakeClock()
    sqs = FakeSqs()

    def sleep(seconds):
        clock.sleep(seconds)
        if len(clock.sleeps) == 2:
            sqs.visibility_ok = False  # 세 번째 확인 전에 Visibility 연장이 실패한다

    provider = FakeProvider()
    processor, _ = build(
        db, provider, sqs=sqs, clock=clock, sleep=sleep, poll_seconds=2, max_wait_seconds=900
    )
    processor.process(valid_message())
    assert clock.sleeps == [2, 2]
    assert len(sqs.visibility_calls) >= 3  # 기다리는 동안 계속 연장했다
    assert provider.calls == [] and job(db).pruna_submit_started_at is None
    assert sqs.deletes == []


def test_worker_gives_up_after_its_own_wait_budget_without_changing_the_shared_wait(db):  # noqa: F811
    limiter(db, interval=0).defer_sends(600)
    clock = FakeClock()
    provider = FakeProvider()
    processor, _ = build(
        db, provider, clock=clock, sleep=clock.sleep, poll_seconds=2, max_wait_seconds=6
    )
    processor.process(valid_message())
    # Worker 는 자기 대기 한도(6초)만큼만 기다리고 보류한다. 공유 대기 시각(호출 허용 시각)은 그대로 600초 뒤다.
    assert clock.sleeps == [2, 2, 2] and provider.calls == []
    assert seconds_until_next_send(db) > 590


# --- 1. 대기 안내: Provider 의 본문·헤더 해석 ---


def test_http_retry_after_header_is_used_when_the_body_has_none():
    result = submit_real_provider(http_429(**{"Retry-After": "120"}))
    assert result.outcome is SubmissionOutcome.UNKNOWN
    assert result.retry_after == 120 and result.retryable


def test_header_only_guidance_without_an_error_body_is_used():
    result = submit_real_provider(http_429(b"", **{"Retry-After": "45"}))
    assert result.reason == "rate_limited" and result.retry_after == 45


def test_body_retry_after_is_read_as_given():
    body = b'{"errors": [{"code": "serviceOverloaded", "message": "x", "retryAfter": 600}]}'
    result = submit_real_provider(http_429(body))
    assert result.retry_after == 600  # 줄이지 않는다


@pytest.mark.parametrize(
    ("body_value", "header", "expected"),
    [
        (600, "120", 600),  # 본문이 더 길다
        (30, "120", 120),  # 헤더가 더 길다
        (12.5, None, 12.5),
        (None, "7", 7),
    ],
)
def test_longer_of_body_and_header_is_used(body_value, header, expected):
    retry = "" if body_value is None else f', "retryAfter": {body_value}'
    body = f'{{"errors": [{{"code": "rateLimitExceeded", "message": "x"{retry}}}]}}'.encode()
    headers = {} if header is None else {"Retry-After": header}
    assert submit_real_provider(http_429(body, **headers)).retry_after == expected


@pytest.mark.parametrize(
    "bad_body", ['"600"', "-5", "NaN", "Infinity", "-Infinity", "true", "{}", "[1]", '"9e999"']
)
@pytest.mark.parametrize(
    "bad_header",
    [None, "abc", "-5", "nan", "inf", "1e3", "", " ", "Wed, 21 Oct 2026 07:28:00 GMT", "1.2.3"],
)
def test_invalid_values_are_not_used_as_guidance(bad_body, bad_header):
    body = (
        f'{{"errors": [{{"code": "rateLimitExceeded", "message": "x", "retryAfter": {bad_body}}}]}}'
    )
    headers = {} if bad_header is None else {"Retry-After": bad_header}
    result = classify_response(
        429, body.encode(), TASK, retry_after_header=headers.get("Retry-After")
    )
    assert result.retry_after is None and not result.retry_after_unrepresentable
    # 429 자체는 여전히 다시 보내도 될 수 있는 응답이다(자동 재시도는 하지 않음)
    assert result.retryable


def test_invalid_body_value_does_not_hide_a_valid_header():
    body = b'{"errors": [{"code": "x", "retryAfter": "soon"}]}'
    assert classify_response(429, body, TASK, retry_after_header="90").retry_after == 90
    assert parse_retry_after([-1, float("nan")], "15") == RetryAfter(seconds=15)
    assert parse_retry_after(["abc"], "Wed, 21 Oct 2026 07:28:00 GMT") == RetryAfter()


def test_header_only_guidance_reaches_the_shared_wait_through_the_submit_path(db):  # noqa: F811
    opener = HttpErrorOpener(http_429(b"", **{"Retry-After": "600"}))
    processor, _ = build(db, RunwarePrunaAsyncProvider("k", opener=opener))
    processor.process(valid_message())
    assert opener.calls == 1  # 같은 생성 요청을 다시 보내지 않았다
    assert seconds_until_next_send(db) > 590


def test_invalid_guidance_through_the_submit_path_keeps_only_the_normal_interval(db):  # noqa: F811
    opener = HttpErrorOpener(http_429(b"", **{"Retry-After": "nan"}))
    processor, _ = build(db, RunwarePrunaAsyncProvider("k", opener=opener), limiter_=limiter(db))
    processor.process(valid_message())
    assert 0 < seconds_until_next_send(db) <= 10  # 최소 간격(10초)만 적용됐다


def test_guidance_log_has_no_body_secret_or_signed_url(db, caplog):  # noqa: F811
    body = (
        b'{"errors": [{"code": "rateLimitExceeded", "message": "see https://x.example/a?sig=SECRET",'
        b' "retryAfter": 600}]}'
    )
    opener = HttpErrorOpener(http_429(body, **{"Retry-After": "120"}))
    with caplog.at_level("DEBUG"):
        processor, _ = build(db, RunwarePrunaAsyncProvider("APIKEYSECRET", opener=opener))
        processor.process(valid_message())
    text = caplog.text
    assert "retry_after=600" in text
    for secret in ("APIKEYSECRET", "SECRET", "X-Amz-Signature", "WEBHOOKTOKEN", "queue full"):
        assert secret not in text


# --- 1. 하루를 넘는 대기 안내와 표현할 수 없는 큰 값 ---

DAY_SECONDS = 86400
LONG = 90000


def held(db):  # noqa: F811
    return db["sql"](
        "SELECT send_hold_reason, send_hold_at IS NOT NULL FROM virtual_fitting_pruna_limiter"
    )[0]


def test_body_90000s_is_used_as_is():
    body = b'{"errors": [{"code": "rateLimitExceeded", "message": "x", "retryAfter": 90000}]}'
    result = submit_real_provider(http_429(body))
    assert result.retry_after == LONG and not result.retry_after_unrepresentable


def test_header_90000s_is_used_as_is():
    result = submit_real_provider(http_429(b"", **{"Retry-After": "90000"}))
    assert result.retry_after == LONG


def test_longer_of_body_and_header_beyond_a_day():
    body = b'{"errors": [{"code": "x", "retryAfter": 90000}]}'
    assert submit_real_provider(http_429(body, **{"Retry-After": "120"})).retry_after == LONG
    body = b'{"errors": [{"code": "x", "retryAfter": 120}]}'
    assert submit_real_provider(http_429(body, **{"Retry-After": "172800"})).retry_after == 172800


def test_90000s_guidance_still_blocks_every_worker_after_a_day(db):  # noqa: F811
    opener = HttpErrorOpener(http_429(b"", **{"Retry-After": "90000"}))
    processor, _ = build(db, RunwarePrunaAsyncProvider("k", opener=opener))
    processor.process(valid_message())
    assert opener.calls == 1
    assert seconds_until_next_send(db) > LONG - 10
    elapse(db, DAY_SECONDS)  # 86400초가 지났다
    other_token, other_task = ready_job(db, 7)
    grant = limiter(db, interval=0).begin_submission(7, other_token, other_task)
    assert grant.reason is SubmissionReason.INTERVAL_WAIT and grant.wait_seconds > 3000
    assert job(db, 7).pruna_submit_started_at is None


def test_after_90000s_other_limits_decide(db):  # noqa: F811
    submit_with_retry_after(db, LONG)
    elapse(db, LONG + 1)
    other_token, other_task = ready_job(db, 7)
    assert limiter(db, interval=0).begin_submission(7, other_token, other_task).granted
    third_token, third_task = ready_job(db, 8)
    grant = limiter(db, in_flight=2, interval=0).begin_submission(8, third_token, third_task)
    assert grant.reason is SubmissionReason.IN_FLIGHT_FULL


def test_shorter_guidance_does_not_shorten_a_90000s_wait(db):  # noqa: F811
    submit_with_retry_after(db, LONG)
    limiter(db, interval=0).defer_sends(600)
    limiter(db, interval=0).defer_sends(DAY_SECONDS)
    assert seconds_until_next_send(db) > LONG - 10


def test_very_long_but_storable_guidance_is_kept_without_reading_it_as_a_datetime(db):  # noqa: F811
    # 1e12초 ≈ 3만 년 뒤. Python datetime(9999년까지)으로 읽을 수 없지만 DB 에는 저장되고 그대로 막는다.
    assert limiter(db, interval=0).defer_sends(1e12) is True
    other_token, other_task = ready_job(db, 7)
    grant = limiter(db, interval=0).begin_submission(7, other_token, other_task)
    assert grant.reason is SubmissionReason.INTERVAL_WAIT and grant.wait_seconds > 9.9e11
    assert held(db) == (None, False)


TOO_LARGE_BODY = b'{"errors": [{"code": "x", "retryAfter": 1e400}]}'
TOO_LARGE_HEADER = "9" * 400


@pytest.mark.parametrize(
    ("body", "header"),
    [
        (TOO_LARGE_BODY, None),
        (b'{"errors": [{"code": "x", "retryAfter": ' + b"9" * 5000 + b"}]}", None),
        (b"", TOO_LARGE_HEADER),
        # 한쪽만 표현할 수 없게 크다: 다른 쪽의 짧은 값을 채택하지 않는다.
        (TOO_LARGE_BODY, "30"),
        (b'{"errors": [{"code": "x", "retryAfter": 30}]}', TOO_LARGE_HEADER),
    ],
)
def test_unrepresentable_guidance_is_reported_not_dropped(body, header):
    headers = {} if header is None else {"Retry-After": header}
    result = submit_real_provider(http_429(body, **headers))
    assert result.retry_after is None and result.retry_after_unrepresentable
    assert result.retryable and result.outcome is SubmissionOutcome.UNKNOWN


def test_infinity_literal_is_invalid_not_unrepresentable():
    result = classify_response(429, b'{"errors": [{"code": "x", "retryAfter": Infinity}]}', TASK)
    assert result.retry_after is None and not result.retry_after_unrepresentable
    assert parse_retry_after([], "inf") == RetryAfter()


@pytest.mark.parametrize(("body", "header"), [(TOO_LARGE_BODY, "30"), (b"", TOO_LARGE_HEADER)])
def test_unrepresentable_guidance_holds_new_sends_for_every_worker(db, body, header):  # noqa: F811
    opener = HttpErrorOpener(http_429(body, **{"Retry-After": header}))
    processor, _ = build(db, RunwarePrunaAsyncProvider("k", opener=opener), limiter_=limiter(db))
    processor.process(valid_message())
    assert opener.calls == 1  # 같은 생성 요청을 다시 보내지 않았다
    assert held(db) == ("retry_after_unrepresentable", True)
    # 기본 간격(10초)이 지나도, 다른 Worker 도 새 생성 요청 권한을 받지 못한다.
    elapse(db, 3600)
    assert seconds_until_next_send(db) < 0
    other_token, other_task = ready_job(db, 7)
    grant = limiter(db, interval=0).begin_submission(7, other_token, other_task)
    assert (grant.granted, grant.reason) == (False, SubmissionReason.SEND_HOLD)
    assert job(db, 7).pruna_submit_started_at is None


def test_value_the_database_cannot_store_holds_new_sends(db, caplog):  # noqa: F811
    # float 로는 표현되지만 DB 시각·간격 범위(약 29만 년)를 넘는다.
    with caplog.at_level("ERROR"):
        assert limiter(db, interval=0).defer_sends(1e300) is True
    assert held(db) == ("retry_after_out_of_range", True)
    assert "holding new sends" in caplog.text
    other_token, other_task = ready_job(db, 7)
    grant = limiter(db, interval=0).begin_submission(7, other_token, other_task)
    assert grant.reason is SubmissionReason.SEND_HOLD


def test_storage_overflow_through_the_submit_path_holds(db):  # noqa: F811
    submit_with_retry_after(db, 1e300)
    assert held(db) == ("retry_after_out_of_range", True)


def test_hold_keeps_the_first_reason_and_shorter_guidance_does_not_release_it(db):  # noqa: F811
    shared = limiter(db, interval=0)
    shared.hold_sends("retry_after_unrepresentable")
    query = "SELECT send_hold_reason, send_hold_at FROM virtual_fitting_pruna_limiter"
    first = db["sql"](query)[0]
    shared.defer_sends(1e300)
    shared.defer_sends(5)
    assert db["sql"](query)[0] == first and first[0] == "retry_after_unrepresentable"


def test_held_worker_does_not_send_and_reports_the_reason(db, caplog):  # noqa: F811
    limiter(db, interval=0).hold_sends("retry_after_unrepresentable")
    clock, sqs = FakeClock(), FakeSqs()
    provider = FakeProvider()
    processor, _ = build(db, provider, sqs=sqs, clock=clock, sleep=clock.sleep, poll_seconds=2)
    with caplog.at_level("ERROR"):
        processor.process(valid_message())
    # 기다려도 풀리지 않는 보류이므로 바로 보류하고(조용히 버리지 않음) 메시지를 지우지 않는다.
    assert provider.calls == [] and job(db).pruna_submit_started_at is None
    assert clock.sleeps == [] and sqs.deletes == []
    assert "pruna submit held" in caplog.text and "send_hold" in caplog.text


def test_stop_signal_during_a_hold_sends_nothing(db):  # noqa: F811
    limiter(db, interval=0).hold_sends("retry_after_unrepresentable")
    provider = FakeProvider()
    processor, _ = build(db, provider, should_stop=lambda: True)
    processor.process(valid_message())
    assert provider.calls == [] and job(db).pruna_submit_started_at is None


def test_worker_waits_in_slices_during_a_90000s_wait_and_stops(db):  # noqa: F811
    limiter(db, interval=0).defer_sends(LONG)
    clock = FakeClock()
    stop = {"now": False}

    def sleep(seconds):
        clock.sleep(seconds)
        stop["now"] = len(clock.sleeps) == 2

    provider = FakeProvider()
    processor, _ = build(
        db,
        provider,
        clock=clock,
        sleep=sleep,
        poll_seconds=2,
        max_wait_seconds=900,
        should_stop=lambda: stop["now"],
    )
    processor.process(valid_message())
    assert clock.sleeps == [2, 2] and provider.calls == []
    assert seconds_until_next_send(db) > LONG - 10  # Worker 대기가 끝나도 호출 가능 시각은 그대로


# --- 2. 사용량 보정과 작업별 비용 기록 ---

TASK_A = "a770f077-f413-47de-9dac-00000000000a"
TASK_B = "a770f077-f413-47de-9dac-00000000000b"
DAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)


class RunwareTotal:
    """Runware getUsageActivity 의 하루 합계(Mock). 값이 작업 A 비용을 포함하는지는 각 테스트가 정한다."""

    def __init__(self, total):
        self.total, self.days = Decimal(total), []

    def spend_on(self, day):
        self.days.append(day)
        return self.total


def budget(db, runware_total, now=NOW):  # noqa: F811
    # 보정 시각이 없거나 600초가 지났으면 ensure_available 이 보정한다(실제 FittingBudget·UsageRepository).
    return FittingBudget(
        UsageRepository(db["pool"]),
        RunwareTotal(runware_total),
        daily_limit=Decimal(5),
        reconcile_interval=timedelta(seconds=600),
        now=lambda: now,
    )


def amount(db, day=DAY):  # noqa: F811
    return UsageRepository(db["pool"]).get(day).amount


def totals(db, day=DAY):  # noqa: F811
    """(legacy, recorded, reconciled, amount) 저장값."""
    rows = db["sql"](
        "SELECT legacy_amount, recorded_amount, reconciled_amount, amount FROM usage_amount "
        "WHERE usage_date = %s",
        (day,),
    )
    return tuple(Decimal(v) for v in rows[0]) if rows else None


def baseline(db, value="2.00"):  # noqa: F811
    UsageRepository(db["pool"]).add(DAY, Decimal(value))  # reconciled_at 없음 → 다음 확인 때 보정


def record_a(db, task=TASK_A, cost="0.02", day=DAY):  # noqa: F811
    return TaskCostRepository(db["pool"]).record(task, 123, Decimal(cost), day)


def test_case_a_cost_then_reconcile_with_a_total_that_includes_it(db):  # noqa: F811
    baseline(db)
    assert record_a(db) is True
    assert amount(db) == Decimal("2.02")
    budget(db, "2.02").ensure_available()  # Runware 합계 2.02 는 작업 A(0.02)를 포함한다
    assert amount(db) == Decimal("2.02")


def test_case_b_reconcile_including_a_then_late_cost_record(db):  # noqa: F811
    baseline(db)
    budget(db, "2.02").ensure_available()  # Runware 합계 2.02 는 이미 작업 A(0.02)를 포함한다
    assert amount(db) == Decimal("2.02")
    assert record_a(db) is True  # 같은 task 의 첫 기록이라 UNIQUE 는 막지 않는다
    assert amount(db) == Decimal("2.02")  # 수정 전: 2.04 (이중 합산)


def test_case_b_keeps_the_two_totals_apart(db):  # noqa: F811
    # 수정 전에는 2.04 를 관찰했다. 이제 두 합계가 따로 남고 판단 금액은 큰 쪽이다.
    baseline(db)
    budget(db, "2.02").ensure_available()
    record_a(db)
    legacy, recorded, reconciled, stored = totals(db)
    assert (legacy, recorded, reconciled) == (Decimal(0), Decimal("2.02"), Decimal("2.02"))
    assert stored == amount(db) == Decimal("2.02")


def test_case_d_stale_total_never_lowers_the_amount(db):  # noqa: F811
    baseline(db)
    record_a(db)
    budget(db, "2.00").ensure_available()  # 작업 A 가 아직 들어가지 않은 오래된 Runware 합계
    assert amount(db) == Decimal("2.02")


def test_stale_reconcile_never_lowers_an_earlier_reconciled_value(db):  # noqa: F811
    usage = UsageRepository(db["pool"])
    assert usage.reconcile(DAY, Decimal("3.00")) == Decimal("3.00")
    assert usage.reconcile(DAY, Decimal("2.50")) == Decimal("3.00")  # 늦게 도착한 오래된 합계
    assert totals(db)[2] == Decimal("3.00")


def test_cost_not_yet_in_the_reconciled_total_is_still_counted(db):  # noqa: F811
    baseline(db)
    budget(db, "2.00").ensure_available()  # 보정 합계에는 작업 A 가 아직 없다
    record_a(db)
    assert amount(db) == Decimal("2.02")


def test_known_limit_cost_only_in_the_reconciled_total_plus_a_new_cost(db):  # noqa: F811
    # 한계(문서화): 우리가 기록하지 못한 비용 0.50 이 보정값에만 있고(2.50), 그 뒤 보정에 없는 새 비용 0.02 가
    # 기록되면 판단 금액은 max(2.02, 2.50) = 2.50 이다. 실제 2.52 와의 차이는 다음 보정까지 남는다.
    baseline(db)
    budget(db, "2.50").ensure_available()
    record_a(db)
    assert amount(db) == Decimal("2.50")
    assert UsageRepository(db["pool"]).reconcile(DAY, Decimal("2.52")) == Decimal("2.52")


def test_same_task_cost_twice_is_added_once(db):  # noqa: F811
    baseline(db)
    assert record_a(db) is True and record_a(db) is False
    assert amount(db) == Decimal("2.02")
    assert totals(db)[1] == Decimal("2.02")


@contextmanager
def usage_row_locked(db):  # noqa: F811
    connection = db["connect"]()
    try:
        connection.execute("SELECT 1 FROM usage_amount WHERE usage_date = %s FOR UPDATE", (DAY,))
        yield
    finally:
        connection.rollback()
        connection.close()


def overlap(db, first, second):  # noqa: F811
    """usage_amount 행을 다른 연결이 잠근 동안 first → second 순서로 각자 다른 연결에서 보내 둘 다 잠금을 기다리게
    한 뒤 푼다. PostgreSQL 은 행 잠금 대기를 도착 순서대로 넘겨준다(실제 겹친 트랜잭션)."""
    errors = []

    def run(call):
        try:
            call()
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    with usage_row_locked(db):
        threads = []
        for expected, call in enumerate((first, second), start=1):
            thread = threading.Thread(target=run, args=(call,))
            thread.start()
            threads.append(thread)
            deadline = time.monotonic() + 10
            while lock_waiters(db) < expected:
                assert time.monotonic() < deadline, "요청이 usage_amount 잠금을 기다리지 않았다"
                time.sleep(0.02)
    for thread in threads:
        thread.join(10)
    assert not errors, errors


def test_case_c_overlap_cost_first_then_reconcile(db):  # noqa: F811
    baseline(db)
    overlap(
        db,
        lambda: record_a(db),
        lambda: UsageRepository(db["pool"]).reconcile(DAY, Decimal("2.02")),
    )
    assert amount(db) == Decimal("2.02")  # 누락·덮어쓰기 없음
    assert db["sql"]("SELECT count(*) FROM virtual_fitting_task_costs")[0][0] == 1


def test_case_c_overlap_reconcile_first_then_cost(db):  # noqa: F811
    baseline(db)
    overlap(
        db,
        lambda: UsageRepository(db["pool"]).reconcile(DAY, Decimal("2.02")),
        lambda: record_a(db),
    )
    assert amount(db) == Decimal("2.02")  # 수정 전: 2.04


def test_case_c_overlap_never_loses_or_overwrites_a_cost(db):  # noqa: F811
    # 어느 순서로 겹쳐도 개별 비용 0.02 는 개별 합계에 남고, 보정값도 사라지지 않는다(원자적 UPSERT).
    baseline(db)
    overlap(
        db,
        lambda: UsageRepository(db["pool"]).reconcile(DAY, Decimal("2.02")),
        lambda: record_a(db),
    )
    assert totals(db)[1:] == (Decimal("2.02"), Decimal("2.02"), Decimal("2.02"))
    assert db["sql"]("SELECT count(*) FROM virtual_fitting_task_costs")[0][0] == 1


def test_many_records_and_reconciles_overlapping_lose_nothing(db):  # noqa: F811
    baseline(db, "1.00")
    tasks = [f"a770f077-f413-47de-9dac-{i:012d}" for i in range(1, 9)]
    calls = [lambda t=t: record_a(db, task=t, cost="0.01") for t in tasks]
    calls += [lambda t=t: record_a(db, task=t, cost="0.01") for t in tasks[:3]]  # 같은 task 중복
    calls += [
        lambda v=v: UsageRepository(db["pool"]).reconcile(DAY, Decimal(v)) for v in ("1.03", "1.05")
    ]

    def hold():
        return usage_row_locked(db)

    run_contended(db, calls, hold)
    _, recorded, reconciled, stored = totals(db)
    assert recorded == Decimal("1.08")  # 1.00 + 0.01 × 8 (중복 3건은 더하지 않음)
    assert reconciled == Decimal("1.05")
    assert stored == amount(db) == Decimal("1.08")
    assert db["sql"]("SELECT count(*) FROM virtual_fitting_task_costs")[0][0] == 8


def test_days_are_kept_apart_and_use_the_utc_date(db):  # noqa: F811
    next_day = DAY + timedelta(days=1)
    record_a(db, day=DAY)
    record_a(db, task=TASK_B, cost="0.05", day=next_day)
    late = budget(db, "3.00", now=datetime(2026, 10, 17, 23, 59, tzinfo=UTC))
    late.ensure_available()
    assert late._usage_source.days == [DAY]
    # 보정은 그날(UTC) 줄에만 반영되고 다음 날 줄은 그대로다(기존 기준: UTC 날짜, KST 09:00 에 바뀜).
    assert (amount(db, DAY), amount(db, next_day)) == (Decimal("3.00"), Decimal("0.05"))
    assert totals(db, next_day)[2] == Decimal(0)


# --- 2. 일일 예산 판단 (실제 DB) ---


def test_budget_passes_at_4_99_and_blocks_at_5_00(db):  # noqa: F811
    baseline(db, "4.99")
    budget(db, "4.99").ensure_available()
    record_a(db, cost="0.01")
    with pytest.raises(FittingDailyBudgetExceededError) as error:
        budget(db, "4.99").ensure_available()
    assert error.value.code == "FITTING_DAILY_BUDGET_EXCEEDED"


def test_budget_blocks_when_only_the_reconciled_total_reaches_the_limit(db):  # noqa: F811
    baseline(db, "1.00")
    with pytest.raises(FittingDailyBudgetExceededError):
        budget(db, "5.00").ensure_available()


def test_budget_does_not_add_the_two_totals(db):  # noqa: F811
    # 개별 2.50 + 보정 2.50 를 더하면 5.00(차단)이지만, 같은 비용일 수 있으므로 큰 쪽 2.50 으로 판단한다.
    baseline(db, "2.50")
    budget(db, "2.50").ensure_available()
    assert amount(db) == Decimal("2.50")


# --- 2. V1·V2 공용 경로 ---


def test_v1_record_and_v2_task_cost_follow_the_same_rule(db):  # noqa: F811
    v1 = budget(db, "2.02")
    v1.ensure_available()  # 보정 합계 2.02(작업 A 포함), 개별 기록 없음
    v1.record(Decimal("0.02"))  # V1 try_on 뒤 기록(같은 ADD_SQL)
    assert amount(db) == Decimal("2.02")
    other = DAY + timedelta(days=2)
    UsageRepository(db["pool"]).reconcile(other, Decimal("2.02"))
    record_a(db, day=other)  # V2 작업별 기록
    assert (
        totals(db)[1:]
        == totals(db, other)[1:]
        == (
            Decimal("0.02"),
            Decimal("2.02"),
            Decimal("2.02"),
        )
    )


def test_v1_and_v2_records_share_the_recorded_total(db):  # noqa: F811
    budget(db, "0").record(Decimal("0.01"))
    record_a(db)
    assert totals(db)[1] == Decimal("0.03")


# --- 2. 변경 SQL 과 기존 데이터 전환 ---


@pytest.fixture
def old_usage(pg):  # noqa: F811
    """변경 SQL 적용 전 구조(schema.sql 그대로)."""
    pg["sql"](LIMITS_SQL.read_text(encoding="utf-8"))
    pg["sql"](USAGE_SQL)
    return pg


def columns(pg):  # noqa: F811
    return {
        name: (nullable, default)
        for name, nullable, default in pg["sql"](
            "SELECT column_name, is_nullable, column_default FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'usage_amount'"
        )
    }


def apply_split(pg):  # noqa: F811
    pg["sql"](SPLIT_SQL.read_text(encoding="utf-8"))


def test_migration_on_an_empty_table(old_usage):
    apply_split(old_usage)
    cols = columns(old_usage)
    for name in ("legacy_amount", "recorded_amount", "reconciled_amount"):
        nullable, default = cols[name]
        assert nullable == "NO" and default == "0"
    assert old_usage["sql"]("SELECT count(*) FROM usage_amount")[0][0] == 0
    UsageRepository(old_usage["pool"]).add(DAY, Decimal("0.02"))
    assert amount(old_usage) == Decimal("0.02")


def test_migration_keeps_existing_rows_and_their_amount(old_usage):
    sql = old_usage["sql"]
    sql(
        "INSERT INTO usage_amount (usage_date, amount, reconciled_at) VALUES "
        "('2026-10-16', 3.10, '2026-10-16 23:50+00'), ('2026-10-17', 2.02, '2026-10-17 11:55+00')"
    )
    apply_split(old_usage)
    rows = sql(
        "SELECT usage_date, amount, legacy_amount, recorded_amount, reconciled_amount, reconciled_at "
        "FROM usage_amount ORDER BY usage_date"
    )
    # 기존 금액을 출처 미구분 금액으로 옮길 뿐, 다시 더하거나 0 으로 만들지 않는다. 보정 시각도 그대로다.
    assert [(r[1], r[2], r[3], r[4]) for r in rows] == [
        (Decimal("3.10"), Decimal("3.10"), Decimal(0), Decimal(0)),
        (Decimal("2.02"), Decimal("2.02"), Decimal(0), Decimal(0)),
    ]
    assert rows[1][5] is not None
    usage = UsageRepository(old_usage["pool"])
    assert usage.get(date(2026, 10, 16)).amount == Decimal("3.10")  # 판단 금액은 전환 전과 같다

    # 재실행해도 아무것도 바뀌지 않는다(같은 금액을 다시 옮기거나 더하지 않음).
    apply_split(old_usage)
    assert sql("SELECT amount, legacy_amount FROM usage_amount ORDER BY usage_date") == [
        (Decimal("3.10"), Decimal("3.10")),
        (Decimal("2.02"), Decimal("2.02")),
    ]


def test_migrated_day_continues_counting_new_costs_on_top_of_the_legacy_amount(old_usage):
    old_usage["sql"]("INSERT INTO usage_amount (usage_date, amount) VALUES ('2026-10-17', 2.00)")
    apply_split(old_usage)
    TaskCostRepository(old_usage["pool"]).record(TASK_B, 1, Decimal("0.02"), DAY)
    assert amount(old_usage) == Decimal(
        "2.02"
    )  # legacy 2.00 + 새 기록 0.02 (새 비용이 빠지지 않음)
    UsageRepository(old_usage["pool"]).reconcile(DAY, Decimal("2.02"))  # 같은 비용을 포함한 보정
    assert amount(old_usage) == Decimal("2.02")
    # 새 날짜는 legacy 0 에서 시작한다.
    TaskCostRepository(old_usage["pool"]).record(
        TASK_A, 1, Decimal("0.03"), DAY + timedelta(days=1)
    )
    assert totals(old_usage, DAY + timedelta(days=1))[:2] == (Decimal(0), Decimal("0.03"))


def test_known_limit_on_the_migrated_day_only(old_usage):
    # 한계(문서화): 전환 전 보정으로 이미 작업 A 를 포함한 2.02 가 기존 amount 에 있으면, 전환 뒤 늦게 온 작업 A 기록은
    # legacy + recorded 로 다시 센다(그날 하루, 크게). 출처를 알 수 없어 자동으로 빼지 않는다. 다음 날짜는 해당 없음.
    old_usage["sql"](
        "INSERT INTO usage_amount (usage_date, amount, reconciled_at) VALUES ('2026-10-17', 2.02, now())"
    )
    apply_split(old_usage)
    record_a(old_usage)
    assert amount(old_usage) == Decimal("2.04")
    next_day = DAY + timedelta(days=1)
    UsageRepository(old_usage["pool"]).reconcile(next_day, Decimal("2.02"))
    record_a(old_usage, task=TASK_B, day=next_day)
    assert amount(old_usage, next_day) == Decimal("2.02")


def test_new_code_needs_the_migration(old_usage):
    from app.virtual_fitting.exceptions import FittingDatabaseError

    with pytest.raises(FittingDatabaseError):
        UsageRepository(old_usage["pool"]).get(DAY)
