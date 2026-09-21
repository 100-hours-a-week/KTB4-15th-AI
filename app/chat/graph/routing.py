"""분기 함수. State를 읽고 다음 노드 이름만 돌려준다. State에는 쓰지 않는다."""

# START 직후. source_type은 Backend가 준 값이므로 LLM 판단이 필요 없다.
# 값은 Backend 의 ChatSourceType enum 을 그대로 쓴다.
ENTRY_ROUTES = {
    "WISHLIST": "wishlist",
    "GENERAL": "analyze",
}

# analyze 이후
ROUTES = {
    "chat": "chat",
    "recommend": "summarize",
    "reject_with_conditions": "summarize",  # 조건은 analyze가 이미 병합했다
    "reject_only": "ask_change",
    "confirm": "search",
}


def entry_route(state: dict) -> str:
    """돌려주는 값은 노드 이름이 아니라 ENTRY_ROUTES 의 **키**다.

    LangGraph 는 이 반환값을 path map 에서 다시 찾아 노드를 고른다. 노드 이름을
    그대로 돌려주면 map 에 그 키가 없어 KeyError 가 난다.
    """
    return state.get("source_type", "GENERAL")


def route(state: dict) -> str:
    # sabu: 목록 밖 값 — analyze가 다섯 갈래에 없는 intent를 내면 이 함수는 어떻게 되지?
    return state["intent"]
