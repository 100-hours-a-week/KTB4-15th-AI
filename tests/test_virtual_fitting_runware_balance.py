"""Runware 잔액 조회(accountManagement / getDetails) 응답 해석.

지원 형식: 공식 문서·공식 TypeScript SDK 의 객체 {amount, freeBalance?, currency} 와, 우리 계정에서 관찰한 숫자 balance
(같은 계정 대시보드 달러 잔액과 표시 정밀도 기준 일치 확인 → USD). 실제 계정 정보 없이 가짜 응답만 쓴다.
"""

import json
from decimal import Decimal

import pytest

from app.virtual_fitting.exceptions import FittingBalanceExhaustedError, FittingBalanceUnknownError
from app.virtual_fitting.providers.runware_balance import (
    MinimumBalanceCheck,
    RunwareBalanceClient,
    parse_balance,
)
from tests.test_virtual_fitting_pruna_submit import FakeOpener

# 관찰된 형태(값은 가짜): data[0].balance 가 숫자, 그 밖 조직·팀 정보는 생략
OBSERVED_NUMERIC = (
    b'{"data": [{"taskType": "accountManagement", "operation": "getDetails", "balance": 12.34}]}'
)


def body(balance, **extra):
    return json.dumps(
        {"data": [{"taskType": "accountManagement", "balance": balance}], **extra}
    ).encode()


def raw(balance_json: str) -> bytes:
    return ('{"data": [{"balance": ' + balance_json + "}]}").encode()


def test_observed_numeric_balance_is_read_as_exact_usd():
    reading = parse_balance(OBSERVED_NUMERIC)
    assert (reading.amount, reading.currency, reading.format) == (Decimal("12.34"), "USD", "number")
    assert reading.checked_at.tzinfo is not None
    assert parse_balance(raw("0.1000000000000000055511")).amount == Decimal(
        "0.1000000000000000055511"
    )
    assert parse_balance(raw("7")).amount == Decimal(7)
    assert parse_balance(raw("0")).amount == Decimal(0)  # 유효한 값, 최소 기준에서 막힌다


def test_numeric_balance_with_a_currency_is_accepted_only_for_usd():
    usd = b'{"data": [{"balance": 3.5, "currency": "USD"}]}'
    assert parse_balance(usd).amount == Decimal("3.5")
    for currency in ('"KRW"', '"usd"', "null", "1"):
        with pytest.raises(FittingBalanceUnknownError):
            parse_balance(('{"data": [{"balance": 3.5, "currency": ' + currency + "}]}").encode())


@pytest.mark.parametrize(
    ("amount", "blocked"),
    [("0", True), ("1.00", True), ("1", True), ("1.0001", False), ("1.01", False)],
)
def test_minimum_policy_with_the_numeric_form(amount, blocked):
    opener = FakeOpener(raw(amount))
    check = MinimumBalanceCheck(RunwareBalanceClient("k", opener=opener), Decimal("1.00"))
    if blocked:
        with pytest.raises(FittingBalanceExhaustedError):
            check.ensure_available()
    else:
        check.ensure_available()
    assert len(opener.requests) == 1


def test_documented_object_form_with_exact_decimal():
    reading = parse_balance(raw('{"amount": 2450.75, "freeBalance": 120.00, "currency": "USD"}'))
    assert reading.amount == Decimal("2450.75") and reading.currency == "USD"
    assert parse_balance(
        raw('{"amount": 0.1000000000000000055511, "currency": "USD"}')
    ).amount == Decimal("0.1000000000000000055511")
    assert parse_balance(raw('{"amount": 7, "currency": "USD"}')).amount == Decimal(7)


def test_zero_is_a_valid_reading_that_the_minimum_blocks():
    reading = parse_balance(raw('{"amount": 0, "currency": "USD"}'))
    assert reading.amount == Decimal(0)


@pytest.mark.parametrize(
    ("amount", "blocked"),
    [("0", True), ("1.00", True), ("1", True), ("1.0001", False), ("1.01", False)],
)
def test_minimum_policy_with_the_real_parser(amount, blocked):
    opener = FakeOpener(raw('{"amount": ' + amount + ', "currency": "USD"}'))
    check = MinimumBalanceCheck(RunwareBalanceClient("k", opener=opener), Decimal("1.00"))
    if blocked:
        with pytest.raises(FittingBalanceExhaustedError):
            check.ensure_available()
    else:
        check.ensure_available()
    assert len(opener.requests) == 1


@pytest.mark.parametrize(
    "amount_json",
    ["-0.01", "-5", "true", "false", '"12.5"', "null", "NaN", "Infinity", "-Infinity", "[]", "{}"],
)
def test_invalid_amounts_are_unknown_not_zero(amount_json):
    with pytest.raises(FittingBalanceUnknownError):
        parse_balance(raw('{"amount": ' + amount_json + ', "currency": "USD"}'))


@pytest.mark.parametrize(
    "value",
    ["true", "false", '"12.5"', "null", "NaN", "Infinity", "-Infinity", "-1", "-0.01", "[]"],
)
def test_invalid_top_level_balance_values(value):
    with pytest.raises(FittingBalanceUnknownError):
        parse_balance(raw(value))


@pytest.mark.parametrize(
    "balance_json",
    [
        '{"amount": 10, "currency": "KRW"}',
        '{"amount": 10, "currency": "usd"}',
        '{"amount": 10}',
        '{"currency": "USD"}',
        '{"amount": 10, "currency": null}',
        '{"amount": 10, "currency": 1}',
    ],
)
def test_currency_mismatch_and_missing_fields(balance_json):
    with pytest.raises(FittingBalanceUnknownError):
        parse_balance(raw(balance_json))


@pytest.mark.parametrize(
    "payload",
    [
        b"[]",
        b'{"data": {}}',
        b'{"data": [1]}',
        b'{"data": [{}]}',
        b'{"data": null}',
        b"\xff\xfe",
    ],
)
def test_wrong_structures(payload):
    with pytest.raises(FittingBalanceUnknownError):
        parse_balance(payload)


def test_http_200_with_errors_is_never_a_balance():
    ok_balance = {"amount": 10, "currency": "USD"}
    for errors in ([{"code": "invalidApiKey"}], [{}]):
        with pytest.raises(FittingBalanceUnknownError):
            parse_balance(body(ok_balance, errors=errors))
    with pytest.raises(FittingBalanceUnknownError):
        parse_balance(b'{"errors": [{"code": "x"}]}')


UNREADABLE = b'{"data": [{"balance": "12.34"}]}'  # 숫자 문자열: 정상 잔액이 아니다


def test_client_reads_once_without_retry_on_an_unreadable_balance():
    opener = FakeOpener(UNREADABLE)
    with pytest.raises(FittingBalanceUnknownError):
        RunwareBalanceClient("k", opener=opener).read()
    assert len(opener.requests) == 1


# --- 잔액 해석 실패 시 유료 생성·fallback 없음(실제 파서 + 실제 Submitter/fallback 핸들러) ---

from tests.test_virtual_fitting_async_job_state import pg  # noqa: F401 (pytest fixture)
from tests.test_virtual_fitting_pruna_submit import (  # noqa: F401 (pytest fixture)
    FakeProvider,
    build,
    counts,
    db,
    job,
)
from tests.test_virtual_fitting_request_worker import valid_message


def unreadable_balance_check():
    return MinimumBalanceCheck(
        RunwareBalanceClient("k", opener=FakeOpener(UNREADABLE)), Decimal("1.00")
    )


def test_unreadable_balance_postpones_the_async_submission(db):  # noqa: F811
    # 실제 조립(create_request_worker)처럼 잔액 확인은 RequestPrecheck 에서 한다(Submitter 는 긴 대기 뒤 재확인만).
    from app.virtual_fitting.request_worker import RequestPrecheck, RequestProcessor
    from tests.test_virtual_fitting_pruna_submit import budget_with
    from tests.test_virtual_fitting_request_worker import BOTTOM, TOP, FakeHead, FakeProducts
    from tests.test_virtual_fitting_request_worker import FakeSqs as RequestSqs

    provider = FakeProvider()
    built, _ = build(db, provider, balance=unreadable_balance_check())
    precheck = RequestPrecheck(
        FakeProducts((TOP, BOTTOM)), FakeHead(), budget_with("0"), unreadable_balance_check()
    )
    processor = RequestProcessor(
        db["jobs"],
        db["state"],
        precheck,
        RequestSqs(),
        lease_seconds=60,
        visibility_seconds=120,
        heartbeat_interval_seconds=0,
        submit=built._submit,
    )
    outcome = processor.process(valid_message())
    assert outcome.action == "retry_FITTING_BALANCE_UNKNOWN"
    current = job(db)
    assert outcome.delete is False and provider.calls == []  # 유료 생성 요청 없음, 메시지 보존
    assert current.pruna_submit_started_at is None and current.result_payload is None
    assert (
        current.next_action is not None and current.next_action.value == "SUBMIT_PRUNA"
    )  # fallback 아님
    assert counts(db) == (0, 0)  # 공유 제한 자리·전송 자리도 잡지 않음


def test_unreadable_balance_holds_the_fallback_without_generation(db):  # noqa: F811
    from tests.test_virtual_fitting_fallback_handler import env, fallback_job, run

    fallback_job(db)
    e = env(db, balance=unreadable_balance_check())
    run(db, e)
    current = job(db)
    assert e.opener.requests == []  # V1 동기 생성 호출 없음
    assert current.result_payload is None and current.fallback_attempt_count == 0
