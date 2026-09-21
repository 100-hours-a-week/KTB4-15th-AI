"""대화 및 상품 추천 스트리밍 (단계1 §7).

사용자의 모든 채팅 입력이 이 라우터로 온다. 스트림을 열기 전에 확인할 수 있는 오류는
HTTP 로, 연 뒤의 오류는 error 이벤트로 낸다.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.chat import controller
from app.chat.schemas import ChatDeleteResponse, ChatRequest
from app.config.checkpointer import get_checkpointer, get_graph
from app.errors import ErrorResponse, error_response
from app.security import verify_internal_key

router = APIRouter(
    tags=["chat"],
    dependencies=[Depends(verify_internal_key)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)

# Annotated 로 감싸 두면 기본값 자리에서 Depends 를 호출하지 않아도 된다.
Graph = Annotated[Any, Depends(get_graph)]
Checkpointer = Annotated[Any, Depends(get_checkpointer)]


@router.post("/api/v1/chat/stream", response_class=StreamingResponse)
async def chat_stream(request: ChatRequest, graph: Graph):
    # sabu: 스트림을 여는 시점 — 404 chat_not_found 와 503 은 스트림 전에 내야 하는데,
    #       추천 분기인지는 analyze 가 끝나야 안다. 이 두 오류는 언제 판단하지?
    found = await controller.find_pre_stream_error(graph, request)
    if found is not None:
        return error_response(*found)

    return StreamingResponse(
        controller.stream_chat(graph, request),
        media_type="text/event-stream",
    )


@router.delete("/api/v1/chat/{chat_id}", response_model=ChatDeleteResponse)
async def delete_chat(chat_id: int, checkpointer: Checkpointer):
    await controller.delete_chat_state(checkpointer, chat_id)
    return ChatDeleteResponse()
