"""대화 API 의 요청·응답 스키마."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/v1/chat/stream 요청 (단계1 §7)."""

    chat_id: int
    user_id: int
    # 길이 상한은 Backend 의 ChatMessageCreateRequest(@Size(max = 500)) 와 맞춘다.
    message: str = Field(
        max_length=500, description="칩 클릭이면 Backend 가 칩 라벨을 넣어 보낸다"
    )
    # Backend 의 ChatSourceType enum 값을 그대로 받는다.
    source_type: Literal["GENERAL", "WISHLIST"] = "GENERAL"
    product_ids: list[str] = Field(
        default_factory=list, description="source_type 이 WISHLIST 일 때 최근 찜 상품 10개"
    )


class ChatDeleteResponse(BaseModel):
    """DELETE /api/v1/chat/{chat_id} 응답."""

    code: int = 200
    message: str = "chat_deleted"
    data: dict | None = None
