"""LLM Client — OpenAI 요청 형식을 아는 유일한 자리 (단계6 §4).

모델이나 요청 스키마가 바뀌면 여기만 고친다. v2에서 analyze를 로컬 모델로 옮길 때도
바꾸는 것은 이 파일이지 노드가 아니다 (단계2 §7.1).

2026-10-07 — 채팅(complete_json·stream_text)은 Runware, 임베딩(embed)은 OpenAI 로 간다.
Runware 는 /embeddings 를 제공하지 않고, 상품 벡터가 OpenAI 임베딩 모델로 만들어져 있다.
"""

import json
import os
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings


class LLMError(RuntimeError):
    """LLM 호출 실패. 스트림 안에서는 LLM_GENERATION_FAILED로 나간다."""


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.MODEL
        self._chat_client: AsyncOpenAI | None = None
        self._embed_client: AsyncOpenAI | None = None

    # 두 클라이언트 모두 첫 호출 때 만든다. 생성자에서 만들면 키가 없는 환경에서 import 만으로
    # 예외가 난다. CI 의 test stage 에는 키가 없으므로 그 자리에서 전부 깨진다.

    @property
    def chat_client(self) -> AsyncOpenAI:
        """Runware 로 가는 클라이언트.

        api_key 를 None 으로 넘기면 SDK 가 OPENAI_API_KEY 를 대신 읽어 Runware 로 보낸다.
        그래서 os.getenv 가 아니라 os.environ[...] 로 읽어, 키가 없으면 여기서 멈춘다.
        """
        if self._chat_client is None:
            self._chat_client = AsyncOpenAI(
                base_url=settings.LLM_BASE_URL,
                api_key=os.environ["RUNWARE_LLM_API_KEY"],
            )
        return self._chat_client

    @property
    def embed_client(self) -> AsyncOpenAI:
        """OpenAI 로 가는 클라이언트. 키와 주소는 SDK 기본값(OPENAI_API_KEY)을 쓴다."""
        if self._embed_client is None:
            self._embed_client = AsyncOpenAI()
        return self._embed_client

    async def complete_json(
        self,
        messages: list[dict],
        *,
        max_output_tokens: int,
        reasoning_effort: str = "none",
    ) -> dict:
        """구조화된 출력을 한 번에 받는다. 스트리밍하지 않는다."""
        try:
            response = await self.chat_client.responses.create(
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
            stream = await self.chat_client.responses.create(
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

    async def embed(self, text: str) -> list[float]:
        """검색 질의 한 줄을 벡터로 바꾼다. 모델은 상품 쪽 임베딩과 같은 settings.EMBEDDING_MODEL."""
        try:
            response = await self.embed_client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=[text],
            )
            return response.data[0].embedding
        except Exception as exc:  # 상위에서 RECOMMENDATION_SEARCH_FAILED 로 변환한다
            raise LLMError(str(exc)) from exc


llm = LLMClient()
