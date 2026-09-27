"""로컬에서 대화를 직접 해보기 위한 개발용 서버.

찜 추천은 아직 비어 있으므로(NotImplementedError) 여기서만 가짜 결과를 꽂는다.
검색 추천은 기본이 가짜이고, DEV_REAL_SEARCH=1 이면 실제 DB(.env 의 DATABASE_URL)와
임베딩 API 로 검색한다. `app/` 안에는 가짜 데이터를 두지 않는다. Dockerfile 도 이 디렉터리를 COPY 하지 않는다.

    python scripts/dev_server.py
    DEV_REAL_SEARCH=1 python scripts/dev_server.py     # 검색만 실제 DB

인증은 기본으로 꺼진다. 켜서 확인하려면 AUTH_DISABLED=0 INTERNAL_API_KEY=... 로 띄운다.
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# .env 는 개발 편의를 위한 것이다. app/ 은 환경변수만 읽으며 .env 를 알지 못한다.
# config 가 import 시점에 값을 읽으므로 app 을 가져오기 전에 먼저 불러와야 한다.
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

os.environ.setdefault("AUTH_DISABLED", "1")
os.environ.setdefault("RELOAD", "0")

# DB 가 없는 팀원도 대화를 해볼 수 있도록 실제 검색은 켤 때만 쓴다.
REAL_SEARCH = os.environ.get("DEV_REAL_SEARCH") == "1"

from app import recommendation
from app.config import settings

SAMPLE = [
    {
        "product_code": "0000001",
        "product_name": "오버핏 코튼 셔츠",
        "image_url": "https://example.com/products/0000001.jpg",
        "detail_url": "https://shop.example.com/products/0000001",
        "color": "네이비",
        "price": 39000,
        "item_type": "TOP",
        "llm_comment": "차분한 네이비 컬러와 여유로운 실루엣이 특징인 셔츠입니다.",
    },
    {
        "product_code": "0000002",
        "product_name": "미니멀 드롭 숄더 셔츠",
        "image_url": "https://example.com/products/0000002.jpg",
        "detail_url": "https://shop.example.com/products/0000002",
        "color": "블랙",
        "price": 42000,
        "item_type": "TOP",
        "llm_comment": "깔끔한 디자인과 오버핏이 특징인 셔츠입니다.",
    },
    {
        "product_code": "0000003",
        "product_name": "데일리 레귤러 셔츠",
        "image_url": "https://example.com/products/0000003.jpg",
        "detail_url": "https://shop.example.com/products/0000003",
        "color": "화이트",
        "price": 35000,
        "item_type": "TOP",
        "llm_comment": "단정한 실루엣이 특징인 데일리 셔츠입니다.",
    },
]


async def fake_search_products(**kwargs):
    print(f"[dev] 검색 조건: {kwargs}", flush=True)
    return SAMPLE[: kwargs.get("top_k", 3)]


async def fake_recommend_from_wishlist(*, product_ids, on_progress=None):
    print(f"[dev] 찜 상품 {len(product_ids)}개로 추천", flush=True)
    for label in ("찜한 상품을 살펴보고 있어요", "비슷한 상품을 찾고 있어요"):
        if on_progress is not None:
            on_progress(label)
        await asyncio.sleep(0.6)  # 진행 표시가 보이도록 일부러 늦춘다
    return SAMPLE


if not REAL_SEARCH:
    recommendation.search_products = fake_search_products
recommendation.recommend_from_wishlist = fake_recommend_from_wishlist

if __name__ == "__main__":
    import uvicorn

    from app.main import app

    search_mode = "실제 DB" if REAL_SEARCH else "가짜 데이터"
    print(f"[dev] 검색 추천: {search_mode} / 찜 추천: 가짜 데이터. "
          f"인증 {'꺼짐' if settings.AUTH_DISABLED else '켜짐'}")
    if REAL_SEARCH and not os.environ.get("DATABASE_URL"):
        print("[dev] DATABASE_URL 이 없습니다. 검색 추천이 실패합니다.")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[dev] OPENAI_API_KEY 가 없습니다. 찜 추천 경로만 끝까지 동작합니다.")

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
