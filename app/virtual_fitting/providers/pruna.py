"""Pruna 공식 API 를 직접 호출하는 p-image-try-on 동기 가상피팅 Provider.

POST https://api.pruna.ai/v1/predictions
  헤더: apikey / Model: p-image-try-on / Try-Sync: true
  본문: {"input": {"person_image", "garment_images", "prompt"}}
  성공: {"status": "succeeded", "generation_url": "..."}
"""

import json
import os
from collections.abc import Callable, Mapping
from typing import Any
from urllib.request import Request, urlopen

from app.virtual_fitting.exceptions import FittingModelError
from app.virtual_fitting.providers.base import FittingInput, FittingResult
from app.virtual_fitting.providers.http import send_request

PRUNA_ENDPOINT = "https://api.pruna.ai/v1/predictions"
PRUNA_MODEL = "p-image-try-on"
API_KEY_ENV = "PRUNA_API_KEY"
# Try-Sync 는 서버에서 60초까지 기다리므로 그보다 조금 길게 잡는다.
DEFAULT_TIMEOUT = 70.0


class PrunaConfigError(RuntimeError):
    """PRUNA_API_KEY가 설정되지 않았을 때 발생한다."""


def get_pruna_api_key() -> str:
    key = os.getenv(API_KEY_ENV)
    if not key or not key.strip():
        raise PrunaConfigError(f"{API_KEY_ENV} 환경변수가 설정되지 않았습니다.")
    return key.strip()


def build_payload(fitting_input: FittingInput) -> dict:
    return {
        "input": {
            "person_image": fitting_input.person_image_url,
            "garment_images": list(fitting_input.garment_image_urls),
            "prompt": fitting_input.prompt,
        }
    }


def parse_result(body: bytes) -> FittingResult:
    """Pruna 원본 응답에서 결과 이미지 URL만 꺼내 내부 타입으로 변환한다."""
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError) as error:
        raise FittingModelError("Pruna 응답이 올바른 JSON이 아닙니다.") from error
    if not isinstance(data, dict):
        raise FittingModelError("Pruna 응답이 JSON 객체가 아닙니다.")

    status = data.get("status")
    if status is not None and status != "succeeded":
        raise FittingModelError(f"Pruna 가상피팅이 실패했습니다: status={status!r}")

    url = data.get("generation_url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise FittingModelError("Pruna 응답에 결과 이미지 URL(generation_url)이 없습니다.")
    return FittingResult(result_image_url=url)


class PrunaDirectProvider:
    """FittingProvider 구현체. Pruna API 를 직접 호출한다."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = PRUNA_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else get_pruna_api_key()
        self.endpoint = endpoint
        self.timeout = timeout
        self._opener = opener or urlopen

    def try_on(self, fitting_input: FittingInput) -> FittingResult:
        request = Request(
            self.endpoint,
            data=json.dumps(build_payload(fitting_input)).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        body = send_request(
            request, opener=self._opener, timeout=self.timeout, service="Pruna"
        )
        return parse_result(body)

    def _headers(self) -> Mapping[str, str]:
        return {
            "apikey": self._api_key,
            "Model": PRUNA_MODEL,
            "Try-Sync": "true",
            "Content-Type": "application/json",
        }
