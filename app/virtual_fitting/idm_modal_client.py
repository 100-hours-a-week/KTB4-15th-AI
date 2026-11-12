"""배포된 IDM-VTON Modal 함수(deploy/modal/idm_vton/service.py: lookddak-idm-vton-fallback / generate) 호출.

이 모듈은 함수 조회·원격 호출(spawn 1회)·결과 대기·오류 분류·응답 검증만 한다. DB·S3·재시도 정책은 호출하는 쪽
(idm_fallback_handler)이 정한다. 같은 요청을 다시 보내는 메서드는 없다(spawn 은 호출부가 작업당 1회만 부른다).

결과 분류(IdmCallKind):
- SUCCEEDED: 함수가 끝났고 성공 요약이 형식에 맞는다(업로드 PUT 2xx 를 받았다는 함수의 보고. S3 확인은 호출부가 한다).
- FAILED: 함수가 끝났다는 근거가 있다 — 정해진 실패 요약(error_kind), 실행 시간 초과(FunctionTimeoutError),
  RemoteError·UserCodeException·InputCancellation, 응답 형식 오류. 더 이상 결과가 올라오지 않는다.
- UNKNOWN: 끝났는지 모른다 — spawn 요청 자체의 오류(접수됐을 수 있음), 연결·gRPC 오류, 결과 보관 만료, 그 밖의 예외.
  호출부는 UNKNOWN 을 실패로 확정하지 않는다(업로드 URL 만료 뒤 S3 확인으로만 확정).
wait() 가 None 을 돌려주면 아직 실행 중이다(이번 대기 시간 안에 끝나지 않음).

Modal 인증은 MODAL_TOKEN_ID·MODAL_TOKEN_SECRET 환경변수를 SDK 가 읽는다. 요청의 URL 은 로그에 남기지 않는다.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# service.py ERROR_KINDS 와 같다.
SERVICE_ERROR_KINDS = frozenset(
    {
        "invalid_request",
        "deadline_exceeded",
        "person_download_failed",
        "garment_download_failed",
        "person_image_invalid",
        "garment_image_invalid",
        "preprocess_rejected",
        "inference_failed",
        "postprocess_failed",
        "upload_failed",
        "internal_error",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_RESULT_BYTES = 20 * 1024 * 1024


class IdmCallKind(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IdmCallResult:
    kind: IdmCallKind
    # 로그용 고정 사유(error_kind·예외 클래스 이름). URL 없음.
    reason: str
    output: dict[str, Any] | None = None


class IdmModalUnavailable(RuntimeError):
    """배포된 함수를 찾지 못했다(호출 전). 생성 기회를 쓰지 않고 보류한다."""


def validate_output(output: Any, fitting_job_id: int, attempt_id: str) -> IdmCallResult:
    """함수 반환값을 확인한다. 끝난 함수의 반환값이므로 형식이 틀려도 UNKNOWN 이 아니라 FAILED 다."""
    if not isinstance(output, dict):
        return IdmCallResult(IdmCallKind.FAILED, "invalid_response_type")
    status = output.get("status")
    if status == "failed":
        kind = output.get("error_kind")
        reason = kind if kind in SERVICE_ERROR_KINDS else "unknown_error_kind"
        return IdmCallResult(IdmCallKind.FAILED, reason, output)
    if status != "succeeded":
        return IdmCallResult(IdmCallKind.FAILED, "invalid_response_status")
    size = output.get("bytes")
    checks = (
        output.get("fitting_job_id") == fitting_job_id,
        output.get("attempt_id") == attempt_id,
        output.get("content_type") == "image/webp",
        isinstance(size, int) and not isinstance(size, bool) and 0 < size <= MAX_RESULT_BYTES,
        isinstance(output.get("sha256"), str) and bool(_SHA256.match(output["sha256"])),
        all(isinstance(output.get(name), int) and output[name] > 0 for name in ("width", "height")),
    )
    if not all(checks):
        return IdmCallResult(IdmCallKind.FAILED, "invalid_response_fields", output)
    return IdmCallResult(IdmCallKind.SUCCEEDED, "succeeded", output)


class IdmVtonModalClient:
    """동기 API(Worker thread 에서 부른다). 함수 핸들은 처음 resolve() 때 찾아 둔다."""

    def __init__(
        self,
        app_name: str | None = None,
        function_name: str | None = None,
        *,
        environment: str | None = None,
        function: Any = None,
    ) -> None:
        if app_name is None or function_name is None:
            from app.config import settings

            app_name = app_name or settings.IDM_VTON_MODAL_APP_NAME
            function_name = function_name or settings.IDM_VTON_MODAL_FUNCTION_NAME
            environment = environment or settings.IDM_VTON_MODAL_ENVIRONMENT
        self._app_name, self._function_name, self._environment = (
            app_name,
            function_name,
            environment,
        )
        self._function = function

    def resolve(self) -> Any:
        """배포된 함수를 찾는다(호출 전 확인). 없거나 인증·연결 실패면 IdmModalUnavailable."""
        if self._function is not None:
            return self._function
        try:
            import modal

            function = modal.Function.from_name(
                self._app_name, self._function_name, environment_name=self._environment
            )
            function.hydrate()
        except Exception as error:  # 호출 전: 어떤 실패도 '부르지 않았음'이다
            raise IdmModalUnavailable(
                f"IDM-VTON Modal 함수를 찾지 못했습니다: {type(error).__name__}"
            ) from error
        self._function = function
        return function

    def spawn(self, request: dict[str, Any]) -> tuple[Any, IdmCallResult | None]:
        """생성 함수를 한 번 시작한다. (호출 핸들, None) 또는 (None, UNKNOWN 결과). 다시 보내지 않는다."""
        function = self.resolve()
        try:
            return function.spawn(request), None
        except Exception as error:  # noqa: BLE001 (접수됐을 수 있다 — 결과 불명확)
            logger.warning("idm modal spawn failed error=%s", type(error).__name__)
            return None, IdmCallResult(IdmCallKind.UNKNOWN, f"spawn_{type(error).__name__}")

    def wait(
        self, call: Any, timeout_seconds: float, fitting_job_id: int, attempt_id: str
    ) -> IdmCallResult | None:
        """최대 timeout_seconds 동안 결과를 기다린다. 아직 실행 중이면 None. 실행 상태 조회일 뿐 다시 부르지 않는다."""
        from modal import exception as modal_exception

        try:
            output = call.get(timeout=timeout_seconds)
        except modal_exception.FunctionTimeoutError:
            return IdmCallResult(IdmCallKind.FAILED, "function_timeout")
        except modal_exception.OutputExpiredError:
            return IdmCallResult(IdmCallKind.UNKNOWN, "output_expired")
        except (modal_exception.TimeoutError, TimeoutError):
            return None
        except (
            modal_exception.RemoteError,
            modal_exception.UserCodeException,
            modal_exception.InputCancellation,
        ) as error:
            # 함수(또는 그 입력)가 예외·취소로 끝났다는 Modal 의 응답.
            return IdmCallResult(IdmCallKind.FAILED, f"remote_{type(error).__name__}")
        except Exception as error:  # noqa: BLE001 (연결·gRPC·역직렬화 오류 등: 끝났는지 모른다)
            return IdmCallResult(IdmCallKind.UNKNOWN, f"wait_{type(error).__name__}")
        return validate_output(output, fitting_job_id, attempt_id)
