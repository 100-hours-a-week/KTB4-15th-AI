"""Runware LLM 으로 llm_comment / llm_title 을 만드는 Provider.

Runware 의 OpenAI 호환 endpoint 를 non-streaming 으로 쓴다. OpenAI API 를 직접 부르지 않으며
Runware 키로 Runware 계정에 과금된다.

POST https://api.runware.ai/v1/chat/completions
  헤더: Authorization: Bearer <RUNWARE_LLM_API_KEY>
  본문: {"model": "openai:gpt@5.6-luna",
         "messages": [{"role": "system", ...}, {"role": "user", "content": ...}],
         "max_completion_tokens": ..., "stream": false}
  성공: {"choices": [{"message": {"content": "..."}, "finish_reason": "stop"}]}

comment 는 결과 이미지 + 상품 설명 요약으로, title 은 comment 만으로 만든다(호출 2회).
이미지는 OpenAI 표준 image_url content part 로 보낸다. Runware 문서는 이 endpoint 의 요청
형식이 OpenAI Chat Completions 와 같다고만 하고 이미지 입력을 따로 명시하지 않는다.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.request import Request, urlopen

from app.virtual_fitting.exceptions import (
    FittingModelError,
    FittingPostprocessError,
    FittingTimeoutError,
)
from app.virtual_fitting.providers.http import send_request
from app.virtual_fitting.providers.runware import get_runware_llm_api_key

LLM_ENDPOINT = "https://api.runware.ai/v1/chat/completions"
# 모델은 여기 한 곳에서만 바꾼다. Runware AIR 형식이다.
LLM_MODEL = "openai:gpt@5.6-luna"
DEFAULT_TIMEOUT = 30.0
_ERROR_DETAIL_LIMIT = 200
_TRUNCATED_FINISH_REASONS = ("length", "content_filter")

COMMENT_SYSTEM_PROMPT = (
    "당신은 패션 쇼핑 앱에서 가상 피팅 결과를 설명하는 작성자입니다. "
    "첨부된 가상 피팅 결과 이미지를 보고, 사용자에게 직접 보여줄 짧고 읽기 쉬운 코디 설명을 작성하세요.\n"
    "규칙:\n"
    "- 결과 이미지에 실제로 보이는 코디를 중심으로 쓰고, 상품 설명 요약은 보조 근거로만 쓰세요.\n"
    "- 이미지와 상품 설명에서 확인되지 않는 소재, 핏, 색상, 디테일은 쓰지 마세요.\n"
    "- 과장된 표현을 쓰지 마세요.\n"
    "- 정확히 2개의 짧은 문장으로 작성하세요.\n"
    "- 각 문장은 한 줄에 하나씩 쓰고, 문장 사이에는 반드시 줄바꿈을 넣으세요.\n"
    "- 첫 문장에서는 가장 눈에 띄는 아이템이나 디테일을 설명하세요.\n"
    "- 두 번째 문장에서는 색상이나 아이템 간의 조화와 전체적인 분위기를 설명하세요.\n"
    "- 한 문장에 너무 많은 정보를 넣지 마세요.\n"
    "- 같은 색상명이나 표현을 반복해서 사용하지 마세요.\n"
    "- 제목, 목록 기호, 번호, 따옴표, 머리말은 쓰지 마세요.\n"
    "- 코디 설명만 출력하세요."
)
TITLE_SYSTEM_PROMPT = (
    "당신은 코디 설명을 읽고 그 코디의 이름을 짓는 작성자입니다.\n"
    "규칙:\n"
    "- 10~20자 내외의 짧은 한국어 제목 형태로 쓰세요. 문장으로 쓰지 마세요.\n"
    "- 코디 설명에 없는 스타일 정보를 새로 추가하지 마세요.\n"
    "- 제목만 출력하세요. 따옴표, 마침표, 설명은 쓰지 마세요."
)


def build_comment_request(result_image_url: str, description_summaries: Sequence[str]) -> dict:
    summaries = "\n".join(
        f"{number}. {summary}" for number, summary in enumerate(description_summaries, start=1)
    )
    return {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": COMMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"상품 설명 요약:\n{summaries}"},
                    {"type": "image_url", "image_url": {"url": result_image_url}},
                ],
            },
        ],
        "max_completion_tokens": 512,
        "stream": False,
    }


def build_title_request(comment: str) -> dict:
    """title 은 comment 만 입력으로 받는다. 이미지와 상품 정보는 넣지 않는다."""
    return {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": TITLE_SYSTEM_PROMPT},
            {"role": "user", "content": f"코디 설명: {comment}"},
        ],
        "max_completion_tokens": 128,
        "stream": False,
    }


def parse_text(body: bytes) -> str:
    """OpenAI Chat Completions 응답에서 choices[0].message.content 만 꺼낸다."""
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError) as error:
        raise FittingPostprocessError("Runware LLM 응답이 올바른 JSON이 아닙니다.") from error
    if not isinstance(data, dict):
        raise FittingPostprocessError("Runware LLM 응답이 JSON 객체가 아닙니다.")

    errors = data.get("error") or data.get("errors")
    if errors:
        raise FittingPostprocessError(
            f"Runware LLM 이 실패했습니다: {str(errors)[:_ERROR_DETAIL_LIMIT]}"
        )

    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    if not isinstance(choice, dict):
        raise FittingPostprocessError("Runware LLM 응답에 choices 가 없습니다.")

    finish_reason = choice.get("finish_reason")
    if finish_reason in _TRUNCATED_FINISH_REASONS:
        raise FittingPostprocessError(
            f"Runware LLM 응답이 완성되지 않았습니다: finish_reason={finish_reason!r}"
        )

    message = choice.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise FittingPostprocessError("Runware LLM 응답에 message.content 가 없습니다.")
    return text.strip()


class RunwareCommentProvider:
    """Runware LLM 기반 comment / title Provider. production(router)에서 쓴다."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = LLM_ENDPOINT,
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

        return self._run(build_comment_request(result_image_url, description_summaries))

    def generate_title(self, comment: str) -> str:
        if not comment.strip():
            raise ValueError("comment 는 비어 있지 않아야 합니다.")

        return self._run(build_title_request(comment))

    def _run(self, payload: dict) -> str:
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
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
