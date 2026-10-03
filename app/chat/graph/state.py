"""Chat 그래프의 State.

checkpointer가 thread_id(= chat_id)별로 이 값을 보관하고, 매 턴 START에서 복원한다.
"""

from typing import Annotated, Literal, TypedDict

# 직전 턴에 봇이 사용자에게 물어 둔 것. 없으면 None.
#   confirm_summary  summarize 가 조건을 요약하고 이대로 추천할지 물었다
#   ask_change       ask_change 가 무엇을 바꾸고 싶은지 물었다
PendingQuestion = Literal["confirm_summary", "ask_change"]

# 봇의 질문에 대한 사용자의 답. analyze(LLM)가 낸다. 봇이 묻고 있는 것이 없으면 none.
ANSWERS = ("yes", "no", "none")
Answer = Literal["yes", "no", "none"]

# Backend 의 ChatSourceType enum 값을 그대로 쓴다. 칩 클릭이면 WISHLIST.
SourceType = Literal["GENERAL", "WISHLIST"]


def append_messages(left: list[dict], right: list[dict]) -> list[dict]:
    """대화 기록 reducer. 노드가 돌려준 메시지를 기존 기록 뒤에 붙인다."""
    return left + right


class Conditions(TypedDict, total=False):
    category: str | None
    color: str | None
    min_price: int | None
    max_price: int | None
    dislikes: list[dict]  # [{"field": "color", "value": "레드"}]


class ChatState(TypedDict):
    messages: Annotated[list[dict], append_messages]
    conditions: Conditions
    # 검색 질의 재료 두 칸 (2026-10-03, QA 10번). 사용자가 실제로 말한 표현만 담는다.
    detail_category: str  # 세부 종류·핏·소재("자켓", "와이드", "데님"). 새로 말하면 덮어쓰고, category 가 바뀌면 지운다
    mood: str  # 상황·분위기("데이트", "출근"). 새로 말하면 덮어쓰고, category 가 바뀌어도 유지한다
    # 수명은 다음 사용자 턴 하나. 매 턴 끝나는 노드가 새로 쓴다
    pending_question: PendingQuestion | None
    answer: Answer  # 이번 턴에만 쓰는 값
    said_conditions: bool  # 이번 턴에만 쓰는 값. 이번 턴 발화에서 조건이 하나라도 뽑혔는가
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
