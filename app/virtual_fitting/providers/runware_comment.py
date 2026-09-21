"""Runware LLM 으로 llm_comment / llm_title 을 만드는 Provider.

Runware 의 OpenAI 호환 endpoint(/chat/completions)는 이미지 입력을 지원하지 않는다.
그래서 native API 의 textInference task 를 쓴다.

POST https://api.runware.ai/v1
  헤더: Authorization: Bearer <RUNWARE_LLM_API_KEY>
  본문: [{"taskType": "textInference", "model": "google-gemini-3-5-flash",
          "inputs": {"images": [<결과 이미지 URL>]},
          "messages": [{"role": "user", "content": ...}],
          "settings": {"systemPrompt", "temperature", "maxTokens", "thinkingLevel"},
          "deliveryMethod": "sync"}]
  성공: {"data": [{"text": "...", "finishReason": "stop"}]}

comment 는 결과 이미지 + 상품 설명 요약으로, title 은 comment 만으로 만든다(호출 2회).
결과 이미지는 접근 가능한 URL 이면 된다. Runware 원본인지 S3 인지 구분하지 않는다.
"""

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.request import Request, urlopen

from app.virtual_fitting.exceptions import (
    FittingModelError,
    FittingPostprocessError,
    FittingTimeoutError,
)
from app.virtual_fitting.providers.http import send_request
from app.virtual_fitting.providers.runware import RUNWARE_ENDPOINT, get_runware_llm_api_key

# 이미지 입력을 받는 멀티모달 모델. 바꿀 때는 여기 한 곳만 고친다.
LLM_MODEL = "google-gemini-3-5-flash"
DEFAULT_TIMEOUT = 30.0
_ERROR_DETAIL_LIMIT = 200
_TRUNCATED_FINISH_REASONS = ("length", "content_filter")

COMMENT_SYSTEM_PROMPT = (
    "당신은 패션 쇼핑 앱에서 가상 피팅 결과를 설명하는 작성자입니다. "
    "첨부된 가상 피팅 결과 이미지를 보고, 사용자에게 직접 보여줄 코디 설명을 작성하세요.\n"
    "규칙:\n"
    "- 결과 이미지에 실제로 보이는 코디를 중심으로 쓰고, 상품 설명 요약은 보조 근거로만 쓰세요.\n"
    "- 이미지와 상품 설명에서 확인되지 않는 소재, 핏, 색상, 디테일은 쓰지 마세요.\n"
    "- 과장된 표현을 쓰지 마세요.\n"
    "- 자연스러운 한국어 1~2문장으로 쓰세요.\n"
    "- 설명 문장만 출력하세요. 제목, 목록, 따옴표, 머리말은 쓰지 마세요."
)
TITLE_SYSTEM_PROMPT = (
    "당신은 코디 설명을 읽고 그 코디의 이름을 짓는 작성자입니다.\n"
    "규칙:\n"
    "- 10~20자 내외의 짧은 한국어 제목 형태로 쓰세요. 문장으로 쓰지 마세요.\n"
    "- 코디 설명에 없는 스타일 정보를 새로 추가하지 마세요.\n"
    "- 제목만 출력하세요. 따옴표, 마침표, 설명은 쓰지 마세요."
)


def build_comment_task(
    result_image_url: str, description_summaries: Sequence[str], task_uuid: str
) -> dict:
    summaries = "\n".join(
        f"{number}. {summary}" for number, summary in enumerate(description_summaries, start=1)
    )
    return {
        "taskType": "textInference",
        "taskUUID": task_uuid,
        "model": LLM_MODEL,
        "inputs": {"images": [result_image_url]},
        "messages": [{"role": "user", "content": f"상품 설명 요약:\n{summaries}"}],
        "settings": {
            "systemPrompt": COMMENT_SYSTEM_PROMPT,
            "temperature": 0.5,
            "maxTokens": 512,
            "thinkingLevel": "off",
        },
        "deliveryMethod": "sync",
    }


def build_title_task(comment: str, task_uuid: str) -> dict:
    """title 은 comment 만 입력으로 받는다. 이미지(inputs)와 상품 정보는 넣지 않는다."""
    return {
        "taskType": "textInference",
        "taskUUID": task_uuid,
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": f"코디 설명: {comment}"}],
        "settings": {
            "systemPrompt": TITLE_SYSTEM_PROMPT,
            "temperature": 0.7,
            "maxTokens": 128,
            "thinkingLevel": "off",
        },
        "deliveryMethod": "sync",
    }


def parse_text(body: bytes) -> str:
    """Runware 원본 응답에서 생성된 문장만 꺼낸다."""
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError) as error:
        raise FittingPostprocessError("Runware LLM 응답이 올바른 JSON이 아닙니다.") from error
    if not isinstance(data, dict):
        raise FittingPostprocessError("Runware LLM 응답이 JSON 객체가 아닙니다.")

    errors = data.get("errors") or data.get("error")
    if errors:
        raise FittingPostprocessError(
            f"Runware LLM 이 실패했습니다: {str(errors)[:_ERROR_DETAIL_LIMIT]}"
        )

    items = data.get("data")
    item = items[0] if isinstance(items, list) and items else None
    if not isinstance(item, dict):
        raise FittingPostprocessError("Runware LLM 응답에 data 가 없습니다.")

    finish_reason = item.get("finishReason")
    if finish_reason in _TRUNCATED_FINISH_REASONS:
        raise FittingPostprocessError(
            f"Runware LLM 응답이 완성되지 않았습니다: finishReason={finish_reason!r}"
        )

    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        raise FittingPostprocessError("Runware LLM 응답에 text 가 없습니다.")
    return text.strip()


class RunwareCommentProvider:
    """Runware LLM 기반 comment / title Provider.

    generate_comment 는 결과 이미지 URL 과 상품 설명 요약을 직접 받는다. 그래서 아직
    CommentProvider(comment.py)를 만족하지 않고, production 에도 연결하지 않았다.
    description_summary 가 DB 에서 오면 그때 Service 와 함께 맞춘다.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = RUNWARE_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else get_runware_llm_api_key()
        self.endpoint = endpoint
        self.timeout = timeout
        self._opener = opener or urlopen

    def generate_comment(self, result_image_url: str, description_summaries: Sequence[str]) -> str:
        if not result_image_url.startswith(("http://", "https://")):
            raise ValueError("result_image_url 은 http(s) URL 이어야 합니다.")
        if not description_summaries or any(
            not summary.strip() for summary in description_summaries
        ):
            raise ValueError("description_summaries 는 비어 있지 않은 문자열이어야 합니다.")

        return self._run(build_comment_task(result_image_url, description_summaries, _task_uuid()))

    def generate_title(self, comment: str) -> str:
        if not comment.strip():
            raise ValueError("comment 는 비어 있지 않아야 합니다.")

        return self._run(build_title_task(comment, _task_uuid()))

    def _run(self, task: dict) -> str:
        request = Request(
            self.endpoint,
            data=json.dumps([task]).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            body = send_request(
                request, opener=self._opener, timeout=self.timeout, service="Runware LLM"
            )
        except (FittingModelError, FittingTimeoutError) as error:
            # 이미지는 이미 만들어졌으므로 VTON 실패와 구분되는 후처리 실패로 낸다.
            raise FittingPostprocessError(str(error)) from error
        return parse_text(body)

    def _headers(self) -> Mapping[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }


def _task_uuid() -> str:
    return str(uuid.uuid4())
