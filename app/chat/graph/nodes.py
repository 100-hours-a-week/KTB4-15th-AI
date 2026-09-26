"""노드 여섯 개.

analyze 만 LLM 결과를 통째로 받고, 말풍선을 만드는 세 개는 생성되는 대로 흘린다.
search 와 wishlist 는 LLM 을 직접 부르지 않고 recommendation 함수를 호출한다.
"""

from langgraph.config import get_stream_writer

from app import recommendation
from app.chat import sse, vocab
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
        merged["dislikes"] = existing

    # sabu: 조건 병합 — 사용자가 "아 빨간색도 괜찮아"라고 번복하면 dislikes 에서 어떻게 빠지지?
    return merged


def _recommendation_message(products: list[dict]) -> dict:
    """추천 목록을 대화 기록에 남긴다.

    product_id와 상품명만 남기고 가격·이미지는 남기지 않는다. 가격은 배치로 갱신되므로
    대화 기록에 박아두면 낡은 값이 프롬프트로 들어간다.

    "두 번째 거"를 풀 수 있도록 번호를 붙인다.
    """
    if products:
        body = "\n".join(
            f"{index}. {product['product_name']} ({product['product_id']})"
            for index, product in enumerate(products, start=1)
        )
    else:
        body = "조건에 맞는 상품을 찾지 못했다."

    return {
        "role": "assistant",
        "content": f"[추천한 상품]\n{body}",
        "kind": prompts.RECOMMENDATION_KIND,
    }


async def analyze(state: dict) -> dict:
    """이번 턴의 의도를 정하고, 말한 조건이 있으면 같이 뽑는다. LLM 1회."""
    message = state["messages"][-1]["content"]
    raw = await llm.complete_json(
        prompts.analyze_prompt(state, message),
        max_output_tokens=settings.ANALYZE_MAX_OUTPUT_TOKENS,
        reasoning_effort=settings.ANALYZE_REASONING_EFFORT,
    )

    conditions = _merge_conditions(
        state.get("conditions", {}), raw.get("metadata"), raw.get("dislikes")
    )
    semantic_query = raw.get("semantic_query") or state.get("semantic_query", "")

    return {
        "intent": raw.get("intent"),
        "conditions": conditions,
        "semantic_query": semantic_query,
    }


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
        "awaiting_confirm": False,
    }


async def summarize(state: dict) -> dict:
    """조건을 요약하고 추천해도 될지 묻는다. 여기서만 플래그를 켠다."""
    text = await _stream_reply(
        prompts.summarize_prompt(state), settings.SUMMARY_MAX_OUTPUT_TOKENS
    )
    return {
        "messages": [{"role": "assistant", "content": text}],
        "awaiting_confirm": True,
    }


async def ask_change(state: dict) -> dict:
    """거절만 한 경우에만 무엇을 바꾸고 싶은지 묻는다."""
    text = await _stream_reply(
        prompts.ask_change_prompt(state), settings.SUMMARY_MAX_OUTPUT_TOKENS
    )
    return {
        "messages": [{"role": "assistant", "content": text}],
        "awaiting_confirm": False,
    }


def _search_query(state: dict) -> str:
    """검색에 넘길 의미 질의.

    사용자가 분위기를 말하지 않으면 semantic_query 가 빈다("청바지 하나 보여줘").
    빈 문자열을 임베딩하면 의미 없는 벡터가 나와 유사도 순위가 아무 근거 없이 정해지므로,
    말한 조건에서 최소한의 문장을 만들어 넘긴다.

    State 에는 채워 넣지 않는다. State 는 사용자가 실제로 말한 것만 담는다.
    """
    written = (state.get("semantic_query") or "").strip()
    if written:
        return written

    conditions = state.get("conditions", {})
    words = [conditions.get("color"), vocab.CATEGORY_QUERY_PHRASE.get(conditions.get("category"))]
    return " ".join(word for word in words if word)


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
        "awaiting_confirm": False,
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
        "awaiting_confirm": False,
    }
