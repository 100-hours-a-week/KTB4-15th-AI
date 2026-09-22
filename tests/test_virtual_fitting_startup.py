"""Runware 키(VTON / LLM)는 요청 시점이 아니라 앱이 뜰 때 확인한다."""

import logging

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.virtual_fitting.providers.runware import RunwareConfigError

VTON = "RUNWARE_VTON_API_KEY"
LLM = "RUNWARE_LLM_API_KEY"
# 두 값은 서로 달라야 한다. 같은 값이면 키가 섞여도 테스트가 알아채지 못한다.
VTON_SECRET = "rw-vton-secret-1234567890"
LLM_SECRET = "rw-llm-secret-0987654321"


@pytest.fixture(autouse=True)
def valid_keys(monkeypatch):
    monkeypatch.setattr(settings, "CHECKPOINT_DSN", "")
    monkeypatch.setenv(VTON, VTON_SECRET)
    monkeypatch.setenv(LLM, LLM_SECRET)


def test_startup_succeeds_when_both_keys_are_set():
    with TestClient(app) as client:
        assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize("name", [VTON, LLM])
def test_startup_fails_when_a_key_is_missing(monkeypatch, name):
    monkeypatch.delenv(name)

    with pytest.raises(RunwareConfigError) as exc_info, TestClient(app):
        pass

    assert name in str(exc_info.value)


@pytest.mark.parametrize("value", ["", "   "], ids=["empty", "whitespace"])
@pytest.mark.parametrize("name", [VTON, LLM])
def test_startup_fails_when_a_key_is_blank(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(RunwareConfigError) as exc_info, TestClient(app):
        pass

    assert str(exc_info.value) == f"{name} 환경변수가 설정되지 않았습니다."


def test_the_legacy_generic_key_does_not_satisfy_startup(monkeypatch):
    monkeypatch.delenv(VTON)
    monkeypatch.delenv(LLM)
    monkeypatch.setenv("RUNWARE_API_KEY", "legacy-key-value")

    with pytest.raises(RunwareConfigError), TestClient(app):
        pass


def test_startup_failure_happens_before_any_request(monkeypatch):
    monkeypatch.delenv(VTON)
    entered = []

    with pytest.raises(RunwareConfigError), TestClient(app):
        entered.append(True)

    assert entered == []


def test_key_values_are_not_leaked_on_success(caplog, capsys):
    with caplog.at_level(logging.DEBUG), TestClient(app):
        pass

    output = capsys.readouterr()
    everything = caplog.text + output.out + output.err
    assert VTON_SECRET not in everything
    assert LLM_SECRET not in everything


@pytest.mark.parametrize("name", [VTON, LLM])
def test_key_values_are_not_leaked_on_failure(monkeypatch, caplog, capsys, name):
    monkeypatch.setenv(name, "")

    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(RunwareConfigError) as exc_info,
        TestClient(app),
    ):
        pass

    output = capsys.readouterr()
    everything = caplog.text + output.out + output.err + str(exc_info.value) + repr(exc_info.value)
    assert VTON_SECRET not in everything
    assert LLM_SECRET not in everything
