"""LLM Client — OpenAI 요청 형식을 아는 유일한 자리 (단계6 §4).

모델이나 요청 스키마가 바뀌면 여기만 고친다. v2에서 analyze를 로컬 모델로 옮길 때도
바꾸는 것은 이 파일이지 노드가 아니다 (단계2 §7.1).
"""

import json
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings


class LLMError(RuntimeError):
    """LLM 호출 실패. 스트림 안에서는 llm_generation_failed로 나간다."""


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.MODEL
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """첫 호출 때 만든다.

        생성자에서 만들면 OPENAI_API_KEY 가 없는 환경에서 import 만으로 예외가 난다.
        CI 의 test stage 에는 키가 없으므로 그 자리에서 전부 깨진다.
        """
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

    async def complete_json(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int,
        reasoning_effort: str = "none",
    ) -> dict:
        """구조화된 출력을 한 번에 받는다. 스트리밍하지 않는다."""
        try:
            response = await self.client.responses.create(
                model=self._model,
                input=messages,
                reasoning={"effort": reasoning_effort},
                max_output_tokens=max_output_tokens,
                text={"format": {"type": "json_object"}},
            )
            return json.loads(response.output_text)
        except Exception as exc:  # 상위에서 SSE error 이벤트로 변환한다
            raise LLMError(str(exc)) from exc

    async def stream_text(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int,
    ) -> AsyncIterator[str]:
        """생성되는 대로 조각을 내보낸다. 사용자에게 보이는 말풍선은 전부 이 경로를 쓴다."""
        try:
            stream = await self.client.responses.create(
                model=self._model,
                input=messages,
                max_output_tokens=max_output_tokens,
                stream=True,
            )
            async for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
        except Exception as exc:  # 상위에서 SSE error 이벤트로 변환한다
            raise LLMError(str(exc)) from exc


llm = LLMClient()
