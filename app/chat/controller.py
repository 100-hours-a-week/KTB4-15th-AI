"""대화 API 의 처리 로직.

라우터는 HTTP 를 다루고, 그래프를 돌리고 SSE 를 만드는 일은 여기서 한다.
"""

from collections.abc import AsyncIterator

from app.chat import sse
from app.chat.graph.state import initial_state
from app.chat.schemas import ChatRequest
from app.clients.llm import LLMError
from app.recommendation import RecommendationError, WishlistCommentError

# 단계1 §7 — Backend 가 최근 찜 상품 10개를 선별해 전달하는 계약
WISHLIST_PRODUCT_COUNT = 10


def thread_config(request: ChatRequest) -> dict:
    return {"configurable": {"thread_id": str(request.chat_id), "user_id": request.user_id}}


async def find_pre_stream_error(graph, request: ChatRequest) -> tuple[int, str, dict] | None:
    """스트림을 열기 전에 확인할 수 있는 오류만 여기서 찾는다.

    찾으면 (status, message, data) 를 돌려주고, 없으면 None 을 돌려준다.
    """
    if request.source_type != "WISHLIST":
        return None

    ids = request.product_ids
    if len(ids) != WISHLIST_PRODUCT_COUNT or len(set(ids)) != len(ids):
        return 400, "invalid_wishlist_products", {"chat_id": request.chat_id}

    # 찜 추천 칩은 대화를 시작할 때만 누를 수 있다. 진행 중인 대화에 오면 거절한다.
    snapshot = await graph.aget_state(thread_config(request))
    if snapshot.values.get("messages"):
        # 이 message 코드는 Backend 와 합의가 필요하다.
        return 400, "wishlist_not_at_chat_start", {"chat_id": request.chat_id}

    return None


async def stream_chat(graph, request: ChatRequest) -> AsyncIterator[str]:
    """그래프를 돌리며 SSE 이벤트를 만든다."""
    chat_id = request.chat_id
    state = initial_state(
        request.message,
        source_type=request.source_type,
        product_ids=request.product_ids,
    )

    # done 에 실어 보낼 전체 문장. 체크포인트에서 다시 읽지 않고 실제로 내보낸 조각을
    # 모은다. 오류로 중간에 끊긴 턴에서는 그때까지 보낸 만큼만 담긴다.
    spoken: list[str] = []

    try:
        async for chunk in graph.astream(state, thread_config(request), stream_mode="custom"):
            if chunk["event"] == sse.TOKEN:
                spoken.append(chunk["content"])
                yield sse.token(chat_id, chunk["content"])
            elif chunk["event"] == sse.PRODUCTS:
                yield sse.products(chat_id, chunk["products"])
            elif chunk["event"] == sse.STATUS:
                yield sse.status(chat_id, chunk["label"])
    except LLMError:
        yield sse.error(chat_id, "llm_generation_failed", "응답 생성 중 오류가 발생했습니다.")
    except WishlistCommentError:
        yield sse.error(
            chat_id,
            "wishlist_comment_generation_failed",
            "추천 이유를 만드는 중 오류가 발생했습니다.",
        )
    except RecommendationError:
        yield sse.error(
            chat_id, "recommendation_search_failed", "상품 추천 처리 중 오류가 발생했습니다."
        )

    yield sse.done(chat_id, "".join(spoken))


# sabu: SSE 가 중간에 끊겨도 그래프는 끝까지 돌고 checkpoint 에 봇 응답이 저장된다.
#       사용자가 못 본 요약에 다음 턴 "응"이 오면 어떻게 되지?


async def delete_chat_state(checkpointer, chat_id: int) -> None:
    """채팅방 삭제·회원 탈퇴 시 Backend 가 요청한다 (2026-09-17 결정).

    AI 쪽 대화 상태는 checkpointer 에 남으므로 Backend 기록만 지우면 지워지지 않는다.
    없는 thread 를 지우는 요청도 성공으로 처리한다(멱등).
    """
    # 메서드 이름은 langgraph-checkpoint 버전에 따라 다를 수 있다. 설치 후 확인할 것.
    await checkpointer.adelete_thread(str(chat_id))
