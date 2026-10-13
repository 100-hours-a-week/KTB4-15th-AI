"""AI Server 의 전체 deadline 을 Modal worker 로 전해, 이미 응답을 포기한 요청을 worker 가 더 처리하지 않게 한다.

AI Server(remote 모드)는 deadline 이 지나면 HTTP 연결을 끊고 504 를 돌려주지만, 그것만으로 Modal 의 작업은
취소되지 않는다(스케줄링 대기 중이던 요청도 나중에 실행됐다). 그래서 AI Server 가 요청마다 deadline 시각을
header 로 보내고, worker 는 처리 단계 사이에서 그 시각을 넘겼는지 보고 스스로 멈춘다.

- 시각은 Unix epoch 초(wall clock)다. 서로 다른 machine 이라 monotonic 시계를 쓸 수 없다. NTP 오차는 여유로 덮는다.
- header 는 worker 앱(Modal)에서만 읽는다. EC2 local 모드에는 이 값이 없어 아무 영향이 없다.
- 이미 시작한 GPU 추론·S3 업로드를 중간에 끊지는 않는다. 다음 단계로 넘어가기 전에만 멈춘다.
"""

import contextvars
import logging
import math
import time

from app.body_image_validation.exceptions import BodyImageValidationError

logger = logging.getLogger(__name__)

HEADER = "X-Body-Image-Deadline"
# S3 에 올린 결과를 AI Server 가 deadline 안에 받아야 의미가 있다. 남은 시간이 이보다 적으면 올리지 않는다.
UPLOAD_MARGIN_SECONDS = 2.0
# 잘못되거나 조작된 값은 무시한다(지금으로부터 이 범위 밖).
_MAX_DISTANCE_SECONDS = 3600.0

_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("body_image_deadline", default=None)


class RequestDeadlineExceeded(BodyImageValidationError):
    """AI Server 의 deadline 이 지나 처리를 멈췄다. AI Server 가 아직 기다리고 있으면 504 로 전해진다."""

    status_code = 504

    def __init__(self, stage: str) -> None:
        super().__init__("BODY_IMAGE_WORKER_TIMEOUT")
        self.stage = stage


def parse_header(value: str | None, now: float | None = None) -> float | None:
    if not value:
        return None
    try:
        deadline = float(value)
    except ValueError:
        return None
    now = time.time() if now is None else now
    if not math.isfinite(deadline) or abs(deadline - now) > _MAX_DISTANCE_SECONDS:  # NaN·inf·터무니없는 값
        return None
    return deadline


def set_deadline(deadline: float | None) -> contextvars.Token:
    return _deadline.set(deadline)


def reset(token: contextvars.Token) -> None:
    _deadline.reset(token)


def check(stage: str, margin: float = 0.0) -> None:
    """deadline 까지 margin 초도 남지 않았으면 RequestDeadlineExceeded. deadline 이 없으면 아무것도 하지 않는다."""
    deadline = _deadline.get()
    if deadline is None:
        return
    remaining = deadline - time.time()
    if remaining <= margin:
        logger.warning(
            "body image request stopped at %s: AI Server deadline passed (remaining=%.1fs margin=%.1fs)",
            stage, remaining, margin,
        )
        raise RequestDeadlineExceeded(stage)
