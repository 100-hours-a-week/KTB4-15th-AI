import asyncio
import contextvars
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.body_image_validation.exceptions import BodyImageValidationError

logger = logging.getLogger(__name__)

HEADER = "X-Body-Image-Deadline"
# S3 에 올린 결과를 AI Server 가 deadline 안에 받아야 의미가 있다. 남은 시간이 이보다 적으면 올리지 않는다.
UPLOAD_MARGIN_SECONDS = 2.0
# 잘못되거나 조작된 값은 무시한다(지금으로부터 이 범위 밖).
_MAX_DISTANCE_SECONDS = 3600.0

_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("body_image_deadline", default=None)
# AI Server 의 ASGI scope 에서 RequestDeadline 을 찾는 key.
SCOPE_KEY = "body_image_request_deadline"


class RequestDeadlineExceeded(BodyImageValidationError):
    """AI Server 의 deadline 이 지나 처리를 멈췄다. AI Server 가 아직 기다리고 있으면 504 로 전해진다."""

    status_code = 504
    deadline_stop = True

    def __init__(self, stage: str) -> None:
        super().__init__("BODY_IMAGE_WORKER_TIMEOUT")
        self.stage = stage


@dataclass
class RequestDeadline:
    """AI Server(remote 모드)에서 요청 하나가 쓰는 전체 제한시간. 시각은 모두 monotonic 이다.

    ASGI middleware 가 요청마다 하나 만들어 scope 에 둔다. hand_over() 전(본문 수신·파싱·인증·service 조회·파일
    읽기)은 middleware 의 guard 가, 그 뒤(입력 검사·대기열·Modal 호출)는 service 가 같은 deadline_at 으로 지킨다.
    """

    total: float
    clock: Callable[[], float] = time.monotonic
    source: str = "asgi"  # total 을 잰 기준점. 요약 로그의 total_from 이다.
    stage: str = "request_parse"
    guard: asyncio.Future[None] | None = None  # service 가 이어받으면 끝난다
    spent: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.started = self._entered = self.clock()
        self.deadline_at = self.started + self.total

    def enter(self, stage: str) -> None:
        self.finish_stage()
        self.stage, self._entered = stage, self.clock()

    def finish_stage(self) -> None:
        self.spent[self.stage] = self.clock() - self._entered

    def hand_over(self) -> None:
        """여기부터 service 가 deadline_at 을 지키고 요약 로그를 남긴다."""
        self.finish_stage()
        if self.guard is not None:
            self.guard.set_result(None)
            self.guard = None


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
