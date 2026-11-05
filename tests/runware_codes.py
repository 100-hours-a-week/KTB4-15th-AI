"""테스트 전용 Runware 오류 코드.

확정 생성 실패(fallback 대상 아님) 코드는 공식 근거가 아직 없어 운영 목록(GENERATION_FAILED_CODES)이 비어 있다. 최종
FITTING_MODEL_FAILED 저장·발행 경로를 검증하는 테스트는 근거가 확인된 코드가 생긴 경우를 가정해 테스트 전용 코드를 등록해
쓴다. 운영 목록과 알 수 없는 코드의 분류(불명확)는 바꾸지 않는다.
"""

import pytest

from app.virtual_fitting.providers import runware_errors

GENERATION_FAILED_TEST_CODE = "testConfirmedGenerationFailure"


def register_generation_failed_code(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(
        runware_errors, "GENERATION_FAILED_CODES", frozenset({GENERATION_FAILED_TEST_CODE})
    )
    return GENERATION_FAILED_TEST_CODE
