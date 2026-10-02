"""프롬프트. 허용값 목록은 vocab 한 곳에서 가져온다.

analyze는 자유 텍스트를 받고 나중에 동의어로 맞추지 않는다. 허용값을 먼저 주고 그
안에서만 고르게 한다 (단계2 §5-1에서 채택한 방식).
"""

from app.chat import vocab
from app.chat.graph.state import ANSWERS
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

1. answer 는 봇이 물어 둔 질문에 대한 사용자의 답이다. 아래 목록에서 정확히 하나 고른다: {answers}
2. 사용자가 이번 턴에 말한 조건만 뽑는다. 말하지 않은 조건은 만들어내지 않는다.
3. color 는 다음 목록의 값만 쓴다: {colors}
   목록에 없는 색이면 null 로 둔다.
4. category 는 다음 목록의 값만 쓴다: {categories}
   목록이 비어 있거나 해당하는 값이 없으면 null 로 둔다.
5. 싫다고 말한 것은 dislikes 에 넣는다. field 는 {dislike_fields} 중 하나다.
6. semantic_query 는 색상·가격·카테고리를 빼고, 상황·분위기·핏 같은 말만 담은 한 문장이다.
   사용자가 이번 턴에 상황·분위기·핏을 말하지 않았으면 빈 문자열로 둔다. 요청을 요약하거나 지어내지 않는다.

JSON만 출력한다:
{{"answer": "...", "metadata": {{"color": null, "category": null}},
  "dislikes": [{{"field": "color", "value": "레드"}}], "semantic_query": "..."}}"""

_ANSWER_MEANING = """answer 의 뜻:
- yes: 봇의 질문에 동의한다 (요약한 조건으로 진행, 이전 조건 그대로 진행)
- no: 거절한다. 또는 무엇을 바꿀지 묻는 질문에 바꿀 것을 말하지 못한다
- none: 봇의 질문과 상관없는 말이다. 봇이 묻고 있는 것이 없으면 항상 none 이다
조건을 말했는지는 answer 로 나타내지 않는다. 조건은 metadata·dislikes·semantic_query 에만 담는다."""

# pending_question 별로 analyze 에게 알려줄 직전 상황
_PENDING_CONTEXT = {
    "confirm_summary": "봇이 지금까지의 조건을 요약하고, 이 조건으로 추천해도 될지 물었다.",
    "ask_change": "봇이 추천 조건 중 무엇을 바꾸고 싶은지 물었다.",
}
_NO_PENDING_CONTEXT = "봇이 묻고 있는 것이 없다."


# sabu: 배포 전 대화 — 운영 checkpointer 에는 pending_question 없이 awaiting_confirm=True 로 저장된
#       대화가 남아 있다. 배포 직후 그 대화에 "응"이 오면 여기서는 무엇으로 읽히고, 어느 노드로 가지?
def analyze_prompt(state: dict, message: str) -> list[dict]:
    # sabu: 재검색 — analyze 는 누적 조건과 현재 메시지만 보고 추천 기록은 보지 않는다.
    #       "다른 것도 보여줘"가 왔을 때 이미 보여준 상품을 빼고 다시 찾으려면
    #       무엇이 더 필요하지? 그 값은 어느 State 필드에 있어야 하지?
    rules = _ANALYZE_RULES.format(
        answers=", ".join(ANSWERS),
        colors=", ".join(vocab.ALLOWED_COLORS),
        categories=", ".join(vocab.ALLOWED_CATEGORIES) or "(아직 정해지지 않음)",
        dislike_fields=", ".join(vocab.DISLIKE_FIELDS),
    )
    context = {
        "누적 조건": state.get("conditions", {}),
        "봇이 물어 둔 질문": _PENDING_CONTEXT.get(state.get("pending_question"), _NO_PENDING_CONTEXT),
    }
    return [
        {"role": "system", "content": f"{rules}\n\n{_ANSWER_MEANING}"},
        {"role": "user", "content": f"[대화 상태]\n{context}\n\n[이번 사용자 입력]\n{message}"},
    ]


# chat 노드는 검색 수단이 없고 다음 엣지도 END 다. 추천을 원하는 턴이 여기 오면
# 다음 턴에 조건을 말해 summarize 로 가도록 말로 안내만 한다 (2026-09-28 QA "옷 미추천").
# 종류 예시는 vocab 의 카테고리 안에서만 들게 한다. 카탈로그에 없는 신발·원피스를 예로 들면
# 사용자가 그대로 말했을 때 검색이 0건이 된다.
CHAT_SYSTEM = f"""너는 남성 패션 쇼핑을 돕는 상담원이다.
사용자의 패션 관련 요청을 이해하고, 이전 대화의 조건을 고려하여 자연스럽게 답한다.
패션과 상관없는 요청(음식, 맛집, 날씨, 일반 상식 등)에는 답하지 않는다. 옷 쇼핑만 도울 수 있다고
짧게 말하고, 찾는 옷이 있는지 묻는다. 인사나 감사처럼 가벼운 말에는 짧게 받아 준다.

- 상품은 실제 판매 중인 상품을 검색해서만 보여준다. 옷이나 코디 조합을 글로 지어내 추천하지 않는다.
- 이 답변에서는 검색하지 않는다. "추천해드릴게요"처럼 추천을 약속하지 않는다.
- 사용자가 옷을 찾는 것 같으면, 종류·색·예산 중 하나라도 말해 주면 실제 상품에서 찾아 준다고 안내한다.
- 옷 종류를 예로 들 때는 다음 목록 안에서만 든다: {", ".join(vocab.ALLOWED_CATEGORIES)}"""


def chat_prompt(state: dict) -> list[dict]:
    # sabu: 문맥 길이 — 추천 목록은 덜어냈지만 대화 자체는 턴마다 자란다.
    #       어디서부터 TTFT 가 나빠지고, 무엇을 기준으로 자를까?
    return [
        {"role": "system", "content": CHAT_SYSTEM},
        *history_for_prompt(state["messages"]),
    ]


def summarize_prompt(state: dict) -> list[dict]:
    # 출구: 운영에 저장된 옛 대화에 남은 목록 밖 제외 조건(가격 등)은 요약에 넘기지 않는다
    conditions = dict(state.get("conditions", {}))
    if "dislikes" in conditions:
        conditions["dislikes"] = vocab.known_dislikes(conditions["dislikes"])
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
            "content": f"조건: {conditions}\n분위기: {state.get('semantic_query', '')}",
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
