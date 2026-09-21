"""프롬프트에 넣을 대화 기록. 추천 목록은 최근 2번만 남긴다."""

from app.chat.graph.nodes import _recommendation_message
from app.chat.graph.prompts import RECOMMENDATION_KIND, history_for_prompt
from app.config import settings


def _products(name, product_id):
    return [{"product_name": name, "product_id": product_id}]


def test_recommendation_message_is_numbered():
    message = _recommendation_message(_products("오버핏 코튼 셔츠", "0000001"))
    assert "1. 오버핏 코튼 셔츠 (0000001)" in message["content"]
    assert message["kind"] == RECOMMENDATION_KIND


def test_empty_result_is_also_recorded():
    assert "찾지 못했다" in _recommendation_message([])["content"]


def test_extra_key_is_stripped_before_the_llm_sees_it():
    messages = [_recommendation_message(_products("셔츠", "1"))]
    assert all(set(m) == {"role", "content"} for m in history_for_prompt(messages))


def test_only_recent_recommendations_are_kept():
    messages = []
    for index in range(settings.RECOMMENDATION_MEMORY_TURNS + 2):
        messages.append({"role": "user", "content": "다른 것도 보여줘"})
        messages.append(_recommendation_message(_products(f"셔츠{index}", str(index))))

    kept = history_for_prompt(messages)
    recommendations = [m for m in kept if m["content"].startswith("[추천한 상품]")]
    assert len(recommendations) == settings.RECOMMENDATION_MEMORY_TURNS
    assert "셔츠3" in recommendations[-1]["content"]


def test_plain_conversation_is_never_dropped():
    messages = [{"role": "user", "content": f"{i}번째 말"} for i in range(20)]
    assert len(history_for_prompt(messages)) == 20
