from enum import Enum

# 이 task 가 처리되지 않았다(재요청 가능). 자동 재요청 연결은 다음 단계다(지금은 보류).
NOT_ACCEPTED_CODES = frozenset({"serviceOverloaded", "inferenceRateLimitExceeded"})
# 그 task 의 생성 단계 종료 실패 중 V1 fallback 으로 다시 만들어 볼 수 있다고 확인한 코드.
FALLBACK_ERROR_CODES = frozenset({"timeoutProvider"})
# 그 task 의 생성 단계 종료 실패이고 fallback 대상이 아니라고 근거로 확인한 코드(최종 FITTING_MODEL_FAILED).
# 공식 문서에 그런 코드가 아직 없다. 근거 없이 채우지 않는다.
GENERATION_FAILED_CODES: frozenset[str] = frozenset()
# 우리 API 키 거절. 사용자용 생성 실패로 저장하지 않고 운영 확인으로 보류한다.
AUTH_ERROR_CODES = frozenset({"invalidApiKey"})


class TaskErrorKind(str, Enum):
    NOT_ACCEPTED = "not_accepted"  # 처리되지 않음(재요청 가능)
    GENERATION_FALLBACK = "generation_fallback"  # 생성 종료 실패, fallback 조건 평가
    GENERATION_FAILED = "generation_failed"  # 생성 종료 실패, fallback 대상 아님(최종 실패)
    AUTH = "auth"  # 인증 오류(운영 확인)
    UNCLEAR = "unclear"  # 근거 부족(상태 유지, 조회·운영 확인)


def classify_task_error(code: str | None) -> TaskErrorKind:
    """현재 task 에 연결된 생성 task 결과(Webhook·getResponse 의 errors)의 오류 코드를 분류한다."""
    if code in NOT_ACCEPTED_CODES:
        return TaskErrorKind.NOT_ACCEPTED
    if code in FALLBACK_ERROR_CODES:
        return TaskErrorKind.GENERATION_FALLBACK
    if code in GENERATION_FAILED_CODES:
        return TaskErrorKind.GENERATION_FAILED
    if code in AUTH_ERROR_CODES:
        return TaskErrorKind.AUTH
    return TaskErrorKind.UNCLEAR
