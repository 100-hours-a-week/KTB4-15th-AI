"""Runware 플랫폼 경유 Pruna p-image-try-on 동기 가상피팅 Provider.

POST https://api.runware.ai/v1
  헤더: Authorization: Bearer <RUNWARE_VTON_API_KEY>
  본문: [{"taskType": "imageInference", "model": "prunaai:p-image@try-on",
          "inputs": {"referenceImages": [{"image", "role": "person" | "garment"}]},
          "positivePrompt", "deliveryMethod": "sync", ...}]
  성공: {"data": [{"imageURL": "..."}]}
"""

import json
import os
import uuid
from collections.abc import Callable, Mapping
from typing import Any
from urllib.request import Request, urlopen

from app.virtual_fitting.exceptions import FittingModelError
from app.virtual_fitting.providers.base import FittingInput, FittingResult
from app.virtual_fitting.providers.http import send_request

RUNWARE_ENDPOINT = "https://api.runware.ai/v1"
RUNWARE_MODEL = "prunaai:p-image@try-on"
# Runware 키는 용도별로 나눈다. VTON Provider 는 VTON 키만 쓴다.
VTON_API_KEY_ENV = "RUNWARE_VTON_API_KEY"
LLM_API_KEY_ENV = "RUNWARE_LLM_API_KEY"
DEFAULT_TIMEOUT = 60.0
_ERROR_DETAIL_LIMIT = 200


class RunwareConfigError(RuntimeError):
    """RUNWARE_VTON_API_KEY / RUNWARE_LLM_API_KEY가 설정되지 않았을 때 발생한다."""


def _read_api_key(env_name: str) -> str:
    key = os.getenv(env_name)
    if not key or not key.strip():
        raise RunwareConfigError(f"{env_name} 환경변수가 설정되지 않았습니다.")
    return key.strip()


def get_runware_vton_api_key() -> str:
    return _read_api_key(VTON_API_KEY_ENV)


def get_runware_llm_api_key() -> str:
    """Runware LLM comment/title Provider 가 쓸 키. 지금은 startup 검증에만 쓴다."""
    return _read_api_key(LLM_API_KEY_ENV)


def build_payload(fitting_input: FittingInput, task_uuid: str) -> list:
    """Runware 는 task 객체의 배열을 받는다. garment 는 입력 순서를 유지한다."""
    reference_images = [{"image": fitting_input.person_image_url, "role": "person"}]
    reference_images.extend(
        {"image": url, "role": "garment"} for url in fitting_input.garment_image_urls
    )
    return [
        {
            "taskType": "imageInference",
            "taskUUID": task_uuid,
            "model": RUNWARE_MODEL,
            "inputs": {"referenceImages": reference_images},
            "positivePrompt": fitting_input.prompt,
            "deliveryMethod": "sync",
            "outputType": "URL",
        }
    ]


def parse_result(body: bytes) -> FittingResult:
    """Runware 원본 응답에서 결과 이미지 URL만 꺼내 내부 타입으로 변환한다."""
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError) as error:
        raise FittingModelError("Runware 응답이 올바른 JSON이 아닙니다.") from error
    if not isinstance(data, dict):
        raise FittingModelError("Runware 응답이 JSON 객체가 아닙니다.")

    errors = data.get("errors")
    if errors:
        raise FittingModelError(
            f"Runware 가상피팅이 실패했습니다: {str(errors)[:_ERROR_DETAIL_LIMIT]}"
        )

    items = data.get("data")
    item = items[0] if isinstance(items, list) and items else None
    url = item.get("imageURL") if isinstance(item, dict) else None
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise FittingModelError("Runware 응답에 결과 이미지 URL(imageURL)이 없습니다.")
    # imageURL 은 Runware CDN URL 이라 장기 보관용이 아니다. VirtualFittingService 가 이 URL 의
    # 이미지를 내려받아 S3 에 저장하고, Backend 에는 URL 이 아니라 result_image_key 만 돌려준다.
    return FittingResult(result_image_url=url)


class RunwarePrunaProvider:
    """FittingProvider 구현체. Runware 플랫폼의 Pruna try-on 모델을 호출한다."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = RUNWARE_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else get_runware_vton_api_key()
        self.endpoint = endpoint
        self.timeout = timeout
        self._opener = opener or urlopen

    def try_on(self, fitting_input: FittingInput) -> FittingResult:
        request = Request(
            self.endpoint,
            data=json.dumps(build_payload(fitting_input, str(uuid.uuid4()))).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        body = send_request(
            request, opener=self._opener, timeout=self.timeout, service="Runware"
        )
        return parse_result(body)

    def _headers(self) -> Mapping[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
