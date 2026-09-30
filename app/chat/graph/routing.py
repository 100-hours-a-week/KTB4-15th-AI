"""분기 함수. State를 읽고 다음 노드 이름만 돌려준다. State에는 쓰지 않는다."""

# START 직후. source_type은 Backend가 준 값이므로 LLM 판단이 필요 없다.
# 값은 Backend 의 ChatSourceType enum 을 그대로 쓴다.
ENTRY_ROUTES = {
    "WISHLIST": "wishlist",
    "GENERAL": "analyze",
}

# analyze 이후. 갈 곳은 LLM 이 이름표로 고르지 않고 route 가 코드로 정한다 (2026-09-30 결정).
ROUTES = {
    "chat": "chat",
    "summarize": "summarize",
    "ask_change": "ask_change",
    "search": "search",
}

# 봇이 물어 둔 질문별로, 조건 없이 온 답이 갈 곳. 여기 없는 답은 chat 으로 보낸다.
#
# ask_change 뒤의 "no"(바꿀 것을 말하지 못함)는 같은 질문을 되풀이하지 않고 chat 으로 보낸다.
# chat 은 카탈로그 종류를 예로 들며 종류·색·예산 중 하나를 묻는다.
_ANSWER_ROUTES = {
    "confirm_summary": {"yes": "search", "no": "ask_change", "none": "chat"},
    "ask_change": {"yes": "search", "no": "chat", "none": "chat"},
}


def entry_route(state: dict) -> str:
    """돌려주는 값은 노드 이름이 아니라 ENTRY_ROUTES 의 **키**다.

    LangGraph 는 이 반환값을 path map 에서 다시 찾아 노드를 고른다. 노드 이름을
    그대로 돌려주면 map 에 그 키가 없어 KeyError 가 난다.
    """
    return state.get("source_type", "GENERAL")


def route(state: dict) -> str:
    """위에서부터 먼저 맞는 갈래로 간다.

    1. 이번 턴에 조건을 말했으면 summarize. 예/아니오와 상관없이 기존 조건의 수정으로 보고 다시 요약한다.
    2. 봇이 물어 둔 질문이 있으면 그 질문에 대한 답(yes / no / none)으로 정한다.
    3. 봇이 묻고 있는 것이 없으면 chat. 조건 없는 추천 요청은 chat 이 조건을 되묻는다 (QA 3번 결정).

    answer 는 LLM 이 낸 값이라 목록 밖 값·null 이 올 수 있다. 그때도 KeyError 없이 chat 으로 간다.
    chat 은 무슨 말이 와도 되묻기·가벼운 대화·거절 중 하나로 답하므로 화면이 비지 않는다.
    """
    if state.get("said_conditions"):
        return "summarize"

    answers = _ANSWER_ROUTES.get(state.get("pending_question"), {})
    return answers.get(state.get("answer"), "chat")
