"""로그 포맷 한 곳 모음. 앱 로그와 uvicorn 로그가 같은 timestamp 포맷을 쓴다.

uvicorn 기본 LOGGING_CONFIG(uvicorn.config)와 같은 handler/logger 구성을 유지하고
포맷에 `%(asctime)s` 와 logger 이름만 더했다. 출력은 기존처럼 stdout/stderr 로만 보낸다.

시각은 컨테이너 시간대(UTC)와 관계없이 formatter 에서 KST 로 바꿔 찍는다. Backend 로그와
같은 ISO 8601 형식(2026-10-01T11:20:18.472+09:00)이라 CloudWatch 에서 나란히 비교할 수 있다.
"""

import logging
import logging.config
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from uvicorn.logging import AccessFormatter, DefaultFormatter

LOG_TIMEZONE = ZoneInfo("Asia/Seoul")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


class _KstTimeMixin:
    """`%(asctime)s` 를 KST 기준 ISO 8601(밀리초, +09:00)으로 만든다."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, LOG_TIMEZONE).isoformat(
            timespec="milliseconds"
        )


class KstDefaultFormatter(_KstTimeMixin, DefaultFormatter):
    pass


class KstAccessFormatter(_KstTimeMixin, AccessFormatter):
    pass


LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        # uvicorn 이 use_colors 옵션을 이 두 키에 써 넣으므로 이름을 바꾸지 않는다.
        "default": {
            "()": KstDefaultFormatter,
            "fmt": LOG_FORMAT,
            "use_colors": False,
        },
        # AccessFormatter 가 status_code 에 "200 OK" 처럼 reason phrase 를 붙여 준다.
        "access": {
            "()": KstAccessFormatter,
            "fmt": '%(asctime)s %(levelname)s %(name)s - %(client_addr)s - "%(request_line)s" '
            "%(status_code)s",
            "use_colors": False,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        # 앱 모듈은 logging.getLogger(__name__) 로 `app.*` 아래에 붙는다.
        "app": {"level": "INFO"},
    },
    # 서드파티 라이브러리는 이전과 같이 WARNING 이상만 남긴다.
    "root": {"handlers": ["default"], "level": "WARNING"},
}


def configure_logging() -> None:
    logging.config.dictConfig(LOGGING_CONFIG)
