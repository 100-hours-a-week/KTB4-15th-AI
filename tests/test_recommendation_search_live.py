"""검색 기반 추천 — 실제 DB 와 임베딩 API 로 끝까지 돌린다 (스모크).

test_recommendation_search.py 는 SQL 문자열만 본다. 여기서는 그 SQL 이 Postgres 에서
실제로 실행되는지, 질의 벡터가 컬럼과 맞는지까지 본다. 결과가 조건에 맞는지는 아직 보지 않는다.

실행하려면 두 환경변수가 필요하다. pytest 는 .env 를 읽지 않는다.
    TEST_DATABASE_URL=postgresql://localhost/team_project_exp
    OPENAI_API_KEY=...
아래 중 하나라도 해당하면 건너뛴다 (CI 와 DB 가 없는 로컬 포함).
    TEST_DATABASE_URL 없음 / DB 연결 실패 / 임베딩이 든 행 0개 / OPENAI_API_KEY 없음
"""

import asyncio
import os

import psycopg
import pytest

from app.clients.llm import LLMClient
from app.config import database, settings
from app.recommendation import search as search_module
from app.recommendation.search import search_products

PRODUCT_KEYS = {
    "product_code", "product_name", "image_url", "detail_url", "color", "price", "item_type",
    "llm_comment",
}


@pytest.fixture
def live_search(monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL 이 없다")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY 가 없다")
    try:
        with psycopg.connect(url, connect_timeout=3) as connection:
            embedded = connection.execute(
                "SELECT count(*) FROM products WHERE embedding IS NOT NULL"
            ).fetchone()[0]
    except psycopg.Error as error:
        pytest.skip(f"테스트 DB 에 연결할 수 없다: {error}")
    if embedded == 0:
        pytest.skip("임베딩이 든 행이 없다")

    monkeypatch.setenv("DATABASE_URL", url)
    # 테스트마다 asyncio.run 으로 루프가 새로 뜨므로, 이전 루프에 묶인 클라이언트를 쓰지 않게 한다
    monkeypatch.setattr(search_module, "llm", LLMClient())


def _run(**kwargs) -> list[dict]:
    # 운영에서는 lifespan 이 pool 을 연다. async pool 은 이벤트 루프에 묶이므로 루프 안에서 열고 닫는다.
    async def search() -> list[dict]:
        await database.open_async_pool()
        try:
            return await search_products(top_k=settings.TOP_K, **kwargs)
        finally:
            await database.close_async_pool()

    return asyncio.run(search())


def test_search_runs_without_conditions(live_search):
    products = _run(
        color=None,
        category=None,
        max_price=None,
        dislikes=[],
        semantic_query="블랙 와이드 팬츠",
    )

    assert len(products) <= settings.TOP_K
    for product in products:
        assert set(product) == PRODUCT_KEYS


# sabu: 결과 검증 — 지금은 "실행되는가"만 본다. 결과가 조건에 맞는지 보려면:
#       서브 일치 상품은 color 가 조건과 다른데, 메인/서브 각각 무엇이 성립해야 하나?
#       10행에 조건을 다 걸면 0건이 나오기 쉽다. 0건일 때 "모든 상품이 조건을 만족한다"는 무엇을 증명하나?
def test_search_runs_with_every_filter_clause(live_search):
    products = _run(
        color="블랙",
        category="팬츠",
        max_price=100000,
        dislikes=[
            {"field": "color", "value": "레드"},
            {"field": "category", "value": "셔츠"},
        ],
        semantic_query="블랙 팬츠",
    )

    assert len(products) <= settings.TOP_K
    for product in products:
        assert set(product) == PRODUCT_KEYS
