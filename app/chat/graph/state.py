"""Chat 그래프의 State.

checkpointer가 thread_id(= chat_id)별로 이 값을 보관하고, 매 턴 START에서 복원한다.
"""

from typing import Annotated, Literal, TypedDict

# 플래그가 꺼져 있을 때 analyze가 낼 수 있는 값
OPEN_INTENTS = ("chat", "recommend")
# 플래그가 켜져 있을 때(= 직전 턴이 요약 확인이었을 때) 낼 수 있는 값
CONFIRM_INTENTS = ("confirm", "reject_only", "reject_with_conditions", "chat")

Intent = Literal["chat", "recommend", "confirm", "reject_only", "reject_with_conditions"]

# Backend 의 ChatSourceType enum 값을 그대로 쓴다. 칩 클릭이면 WISHLIST.
SourceType = Literal["GENERAL", "WISHLIST"]


def append_messages(left: list[dict], right: list[dict]) -> list[dict]:
    """대화 기록 reducer. 노드가 돌려준 메시지를 기존 기록 뒤에 붙인다."""
    return left + right


class Conditions(TypedDict, total=False):
    category: str | None
    color: str | None
    max_price: int | None
    dislikes: list[dict]  # [{"field": "color", "value": "레드"}]


class ChatState(TypedDict):
    messages: Annotated[list[dict], append_messages]
    conditions: Conditions
    semantic_query: str
    awaiting_confirm: bool  # 수명은 다음 사용자 턴 하나
    intent: Intent  # 이번 턴에만 쓰는 값
    source_type: SourceType  # 이번 턴에만 쓰는 값. 매 턴 입력으로 덮어쓴다
    product_ids: list[str]  # source_type이 WISHLIST일 때만 채워진다


def initial_state(
    message: str,
    *,
    source_type: str = "GENERAL",
    product_ids: list[str] | None = None,
) -> dict:
    """새 턴의 입력. 기존 값은 checkpointer가 복원하므로 여기서 덮어쓰지 않는다.

    source_type과 product_ids는 턴마다 새로 오는 값이라 매번 채워 넣는다.
    """
    return {
        "messages": [{"role": "user", "content": message}],
        "source_type": source_type,
        "product_ids": product_ids or [],
    }
