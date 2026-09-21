"""프롬프트. 허용값 목록은 vocab 한 곳에서 가져온다.

analyze는 자유 텍스트를 받고 나중에 동의어로 맞추지 않는다. 허용값을 먼저 주고 그
안에서만 고르게 한다 (단계2 §5-1에서 채택한 방식).
"""

from app.chat import vocab
from app.chat.graph.state import CONFIRM_INTENTS, OPEN_INTENTS
from app.config import settings

# 추천 목록을 담은 assistant 메시지의 표시. 프롬프트로 나갈 때는 떼어낸다.
RECOMMENDATION_KIND = "recommendation"


def history_for_prompt(messages: list[dict]) -> list[dict]:
    """LLM에 넘길 대화 기록.

    추천 목록은 최근 RECOMMENDATION_MEMORY_TURNS 번만 남긴다. 기록 자체는 지우지
    않고 프롬프트에서만 덜어낸다. messages는 대화의 기록이기도 하기 때문이다.
    """
    kept: list[dict] = []
    seen = 0
    for message in reversed(messages):
        if message.get("kind") == RECOMMENDATION_KIND:
            seen += 1
            if seen > settings.RECOMMENDATION_MEMORY_TURNS:
                continue
        kept.append({"role": message["role"], "content": message["content"]})
    return list(reversed(kept))

_ANALYZE_RULES = """너는 패션 쇼핑 대화의 한 턴을 해석한다.

1. intent 를 아래 목록에서 정확히 하나 고른다: {intents}
2. 사용자가 이번 턴에 말한 조건만 뽑는다. 말하지 않은 조건은 만들어내지 않는다.
3. color 는 다음 목록의 값만 쓴다: {colors}
   목록에 없는 색이면 null 로 둔다.
4. category 는 다음 목록의 값만 쓴다: {categories}
   목록이 비어 있거나 해당하는 값이 없으면 null 로 둔다.
5. 싫다고 말한 것은 dislikes 에 넣는다. field 는 color, category, max_price, style 중 하나다.
6. semantic_query 는 색상·가격·카테고리를 빼고, 상황·분위기·핏 같은 말만 담은 한 문장이다.

JSON만 출력한다:
{{"intent": "...", "metadata": {{"color": null, "category": null, "max_price": null}},
  "dislikes": [{{"field": "color", "value": "레드"}}], "semantic_query": "..."}}"""

_INTENT_MEANING = """intent 의 뜻:
- chat: 추천과 상관없는 일반 대화
- recommend: 상품을 추천받고 싶다는 뜻
- confirm: 직전에 봇이 요약한 조건으로 진행하겠다는 동의
- reject_only: 거절만 했고 바꿀 조건은 말하지 않음
- reject_with_conditions: 거절하면서 바꿀 조건도 함께 말함"""


def analyze_prompt(state: dict, message: str) -> list[dict]:
    # sabu: 재검색 — analyze 는 누적 조건과 현재 메시지만 보고 추천 기록은 보지 않는다.
    #       "다른 것도 보여줘"가 왔을 때 이미 보여준 상품을 빼고 다시 찾으려면
    #       무엇이 더 필요하지? 그 값은 어느 State 필드에 있어야 하지?
    intents = CONFIRM_INTENTS if state.get("awaiting_confirm") else OPEN_INTENTS
    rules = _ANALYZE_RULES.format(
        intents=", ".join(intents),
        colors=", ".join(vocab.ALLOWED_COLORS),
        categories=", ".join(vocab.ALLOWED_CATEGORIES) or "(아직 정해지지 않음)",
    )
    context = {
        "누적 조건": state.get("conditions", {}),
        "직전 턴에 봇이 조건을 요약하고 확인을 물었는가": bool(state.get("awaiting_confirm")),
    }
    return [
        {"role": "system", "content": f"{rules}\n\n{_INTENT_MEANING}"},
        {"role": "user", "content": f"[대화 상태]\n{context}\n\n[이번 사용자 입력]\n{message}"},
    ]


CHAT_SYSTEM = """너는 패션 쇼핑을 돕는 상담원이다.
사용자의 패션 관련 요청을 이해하고, 이전 대화의 조건을 고려하여 자연스럽게 답한다.
추천 상품 목록을 네가 지어내지 않는다."""


def chat_prompt(state: dict) -> list[dict]:
    # sabu: 문맥 길이 — 추천 목록은 덜어냈지만 대화 자체는 턴마다 자란다.
    #       어디서부터 TTFT 가 나빠지고, 무엇을 기준으로 자를까?
    return [
        {"role": "system", "content": CHAT_SYSTEM},
        *history_for_prompt(state["messages"]),
    ]


def summarize_prompt(state: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "지금까지 사용자가 말한 조건을 한두 문장으로 요약하고, "
                "이 조건으로 추천해도 될지 묻는 말로 끝낸다. "
                "사용자가 말하지 않은 조건을 덧붙이지 않는다. 다른 질문은 하지 않는다."
            ),
        },
        {
            "role": "user",
            "content": f"조건: {state.get('conditions', {})}\n분위기: {state.get('semantic_query', '')}",
        },
    ]


def ask_change_prompt(state: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "사용자가 추천을 거절했지만 바꾸고 싶은 조건은 말하지 않았다. "
                "무엇을 바꾸고 싶은지 한 문장으로 짧게 묻는다."
            ),
        },
        *history_for_prompt(state["messages"])[-2:],
    ]
