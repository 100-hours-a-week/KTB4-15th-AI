"""노드 여섯 개.

analyze 만 LLM 결과를 통째로 받고, 말풍선을 만드는 세 개는 생성되는 대로 흘린다.
search 와 wishlist 는 LLM 을 직접 부르지 않고 recommendation 함수를 호출한다.
"""

from langgraph.config import get_stream_writer

from app import recommendation
from app.chat import price, sse, vocab
from app.chat.graph import prompts
from app.clients.llm import llm
from app.config import settings


def _merge_conditions(current: dict, metadata: dict | None, dislikes: list | None) -> dict:
    """이번 턴에 말한 값만 덮어쓴다. 말하지 않은 조건은 그대로 둔다."""
    merged = dict(current or {})
    for key, value in (metadata or {}).items():
        if value is not None:
            merged[key] = value

    if dislikes:
        existing = list(merged.get("dislikes", []))
        for item in dislikes:
            if item not in existing:
                existing.append(item)
            # 앞 턴에 저장된 포함 조건과 같은 값을 이번 턴에 빼 달라고 하면 포함 조건을 지운다.
            # 남겨 두면 "데님이면서 데님이 아닌 것"을 찾아 0건이 된다 (2026-10-02)
            if merged.get(item.get("field")) == item.get("value"):
                del merged[item["field"]]
        merged["dislikes"] = existing

    # sabu: 조건 병합 — 사용자가 "아 빨간색도 괜찮아"라고 번복하면 dislikes 에서 어떻게 빠지지?
    return merged


def _recommendation_message(products: list[dict]) -> dict:
    """추천 목록을 대화 기록에 남긴다.

    product_code와 상품명만 남기고 가격·이미지는 남기지 않는다. 가격은 배치로 갱신되므로
    대화 기록에 박아두면 낡은 값이 프롬프트로 들어간다.

    "두 번째 거"를 풀 수 있도록 번호를 붙인다.
    """
    if products:
        body = "\n".join(
            f"{index}. {product['product_name']} ({product['product_code']})"
            for index, product in enumerate(products, start=1)
        )
    else:
        body = "조건에 맞는 상품을 찾지 못했다."

    return {
        "role": "assistant",
        "content": f"[추천한 상품]\n{body}",
        "kind": prompts.RECOMMENDATION_KIND,
    }


def postprocess_analysis(raw: dict, message: str) -> dict:
    """LLM 이 낸 analyze 결과를 코드 규칙으로 다듬는다. 조건 병합·판정 전에 반드시 거친다.

    기타/bench_analyze_route.py 도 이 함수를 불러 서비스와 같은 조건 판정을 잰다.
    """
    raw = _drop_unknown_category(_apply_price_rule(raw, message))
    # 목록 밖 필드의 제외 조건(가격 등)은 상태에 넣지 않고, 조건을 말한 턴으로도 세지 않는다
    raw = {**raw, "dislikes": vocab.known_dislikes(raw.get("dislikes"))}
    return _drop_included_dislikes(raw)


def _drop_included_dislikes(raw: dict) -> dict:
    """같은 턴에 같은 값이 포함과 제외에 둘 다 있으면 제외를 믿고 포함 쪽을 null 로 바꾼다.

    "데님은 빼고" 를 LLM 이 category=데님 + 제외 데님 으로 함께 내는 경우(2026-10-02 운영, 재현 2/4).
    제외로 나왔다는 것은 부정 표현이 있었다는 뜻이다. null 은 병합에서 무시되므로 앞 턴 값이 남는다.
    """
    metadata = dict(raw.get("metadata") or {})
    for item in raw.get("dislikes") or []:
        field = item.get("field")
        if field in ("color", "category") and metadata.get(field) == item.get("value"):
            metadata[field] = None
    return {**raw, "metadata": metadata}


def _drop_unknown_category(raw: dict) -> dict:
    """목록 밖 category 는 null 로 바꾼다. json_object 모드는 값이 허용 목록 안인지 보장하지 않는다.

    null 은 병합에서 무시되므로 앞 턴의 category 가 그대로 남고, 조건을 말한 턴으로도 세지 않는다.
    """
    metadata = dict(raw.get("metadata") or {})
    if metadata.get("category") not in (None, *vocab.ALLOWED_CATEGORIES):
        metadata["category"] = None
    return {**raw, "metadata": metadata}


def _apply_price_rule(raw: dict, message: str) -> dict:
    """가격은 LLM 이 아니라 규칙으로 뽑는다. 모델이 가격을 내더라도 쓰지 않는다."""
    metadata = {
        key: value
        for key, value in (raw.get("metadata") or {}).items()
        if key not in ("min_price", "max_price")
    }
    price_range = price.parse_price(message)
    if price_range:
        metadata["min_price"], metadata["max_price"] = price_range
    return {**raw, "metadata": metadata}


def _said_conditions(raw: dict) -> bool:
    """이번 턴 발화에서 조건이 하나라도 뽑혔는가. 갈 곳은 이 값과 answer 로 코드가 정한다."""
    metadata = raw.get("metadata") or {}
    return (
        any(value is not None for value in metadata.values())
        or bool(raw.get("dislikes"))
        or bool((raw.get("detail_category") or "").strip())
        or bool((raw.get("mood") or "").strip())
    )


# sabu: 조건의 기준 — 규칙 7 은 LLM 이 detail_category·mood 를 지어낼 확률만 낮춘다. 지어낸 값을 코드가 걸러내려면
#       무엇과 무엇을 대조하면 되지? "MZ" 처럼 짧은 분위기 말은 그 대조를 어떻게 통과하지?
async def analyze(state: dict) -> dict:
    """봇이 물어 둔 질문에 대한 답(answer)과 이번 턴에 말한 조건을 뽑는다. LLM 1회.

    갈 곳은 정하지 않는다. routing.route 가 answer 와 said_conditions 로 정한다.
    """
    message = state["messages"][-1]["content"]
    raw = await llm.complete_json(
        prompts.analyze_prompt(state, message),
        max_output_tokens=settings.ANALYZE_MAX_OUTPUT_TOKENS,
        reasoning_effort=settings.ANALYZE_REASONING_EFFORT,
    )

    raw = postprocess_analysis(raw, message)
    conditions = _merge_conditions(
        state.get("conditions", {}), raw.get("metadata"), raw.get("dislikes")
    )
    detail_category, mood = _merge_query_words(state, raw, conditions)

    return {
        "answer": raw.get("answer"),
        "said_conditions": _said_conditions(raw),
        "conditions": conditions,
        "detail_category": detail_category,
        "mood": mood,
    }


def _merge_query_words(state: dict, raw: dict, conditions: dict) -> tuple[str, str]:
    """검색 질의 재료 두 칸을 이번 턴 값으로 갱신한다 (2026-10-03 결정).

    - detail_category(세부 종류·핏·소재): 새로 말하면 덮어쓴다. category 가 바뀌면 지운다.
      "와이드 팬츠" 를 보다가 "셔츠도" 라고 하면 와이드는 셔츠로 이어지지 않는다.
    - mood(상황·분위기): 새로 말하면 덮어쓴다. category 가 바뀌어도 유지한다.
    """
    new_detail = (raw.get("detail_category") or "").strip()
    new_mood = (raw.get("mood") or "").strip()
    category_changed = conditions.get("category") != state.get("conditions", {}).get("category")

    if new_detail:
        detail_category = new_detail
    elif category_changed:
        detail_category = ""
    else:
        detail_category = state.get("detail_category", "")

    mood = new_mood or state.get("mood", "")
    return detail_category, mood


async def _stream_reply(messages: list[dict], max_output_tokens: int) -> str:
    """토큰을 SSE 로 흘리면서 전체 문장을 모아 돌려준다."""
    writer = get_stream_writer()
    parts: list[str] = []
    async for delta in llm.stream_text(messages, max_output_tokens=max_output_tokens):
        writer({"event": sse.TOKEN, "content": delta})
        parts.append(delta)
    return "".join(parts)


async def chat(state: dict) -> dict:
    text = await _stream_reply(prompts.chat_prompt(state), settings.CHAT_MAX_OUTPUT_TOKENS)
    return {
        "messages": [{"role": "assistant", "content": text}],
        "pending_question": None,
    }


async def summarize(state: dict) -> dict:
    """조건을 요약하고 추천해도 될지 묻는다."""
    text = await _stream_reply(
        prompts.summarize_prompt(state), settings.SUMMARY_MAX_OUTPUT_TOKENS
    )
    return {
        "messages": [{"role": "assistant", "content": text}],
        "pending_question": "confirm_summary",
    }


async def ask_change(state: dict) -> dict:
    """거절만 한 경우에만 무엇을 바꾸고 싶은지 묻는다.

    물은 것을 pending_question 에 남긴다. 남기지 않으면 다음 턴의 "가을 말고 여름"을 analyze 가
    질문에 대한 답으로 읽지 못해 일반 대화로 빠진다 (2026-09-28 QA).
    """
    text = await _stream_reply(
        prompts.ask_change_prompt(state), settings.SUMMARY_MAX_OUTPUT_TOKENS
    )
    return {
        "messages": [{"role": "assistant", "content": text}],
        "pending_question": "ask_change",
    }


def _search_query(state: dict) -> str:
    """검색에 넘길 의미 질의. 색 + 세부 카테고리 + 분위기를 이어 붙인다 (2026-10-03 측정으로 결정).

    카테고리 문구("팬츠", "아우터")는 붙이지 않는다. 붙이면 세부 단어의 신호가 묻힌다
    ("슬랙스 출근" 8/10 → "블랙 슬랙스 팬츠 출근" 0/10). 카테고리는 이미 필터가 거른다.
    색은 붙인다. 빼면 "데님 데이트" 가 코디 문장 쪽으로 끌려간다(1/10 → 색을 붙이면 8/10).
    세부 카테고리도 분위기도 없으면 색 + 카테고리 문구로 대신한다("청바지 하나 보여줘").
    아무것도 없으면 빈 문자열이다(search.py "빈 질의" 마커).

    사용자 표기는 상품 설명 표기로 바꾼다("자켓" → "재킷"). State 에는 채워 넣지 않는다.
    State 와 요약은 사용자가 실제로 말한 것만 담는다.
    """
    conditions = state.get("conditions", {})
    detail = (state.get("detail_category") or "").strip()
    mood = (state.get("mood") or "").strip()
    if detail or mood:
        words = [conditions.get("color"), detail, mood]
    else:
        words = [conditions.get("color"), vocab.CATEGORY_QUERY_PHRASE.get(conditions.get("category"))]
    query = " ".join(word.strip() for word in words if word and word.strip())
    return vocab.to_catalog_spelling(query)


async def search(state: dict) -> dict:
    """Recommendation 모듈에 검색을 맡기고 결과만 products 로 내보낸다. LLM 없음."""
    writer = get_stream_writer()
    conditions = state.get("conditions", {})

    # dislikes 중 필터로 거를 수 있는 항목만 넘긴다 ((ㄱ) 결정).
    # sabu: 버려지는 제외 조건 — "오버핏은 싫어"는 여기서 조용히 사라진다.
    #       사용자에게는 반영된 것처럼 보이는데, 그대로 둬도 괜찮은가?
    dislikes = [
        item
        for item in conditions.get("dislikes", [])
        if item.get("field") in vocab.FILTERABLE_FIELDS
    ]

    products = await recommendation.search_products(
        color=conditions.get("color"),
        category=conditions.get("category"),
        min_price=conditions.get("min_price"),
        max_price=conditions.get("max_price"),
        dislikes=dislikes,
        semantic_query=_search_query(state),
        top_k=settings.TOP_K,
    )

    # 0건이면 빈 배열 그대로 내보낸다. 문구는 프론트가 만든다.
    # done 의 content 는 말풍선이 없는 이 턴을 위해 합의한 고정 문구로 채운다.
    writer({"event": sse.PRODUCTS, "products": products, "done_content": sse.SEARCH_DONE_CONTENT})

    return {
        "messages": [_recommendation_message(products)],
        "pending_question": None,
    }


async def wishlist(state: dict) -> dict:
    """찜 기반 추천. 대화 첫 턴에 칩을 눌렀을 때만 들어온다. 이 노드도 LLM을 직접 부르지 않는다.

    추천이 끝나면 이 턴은 종료되고, 다음 턴부터는 평소대로 analyze를 거친다.
    """
    writer = get_stream_writer()

    def on_progress(label: str) -> None:
        writer({"event": sse.STATUS, "label": label})

    products = await recommendation.recommend_from_wishlist(
        product_ids=state.get("product_ids", []),
        on_progress=on_progress,
    )

    writer({"event": sse.PRODUCTS, "products": products})
    return {
        "messages": [_recommendation_message(products)],
        "pending_question": None,
    }
