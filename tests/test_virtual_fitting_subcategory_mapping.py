"""sub_category → 영어 garment 명칭 매핑 검증."""

import json
from pathlib import Path

import pytest

from app.virtual_fitting.exceptions import UnsupportedSubCategoryError
from app.virtual_fitting.subcategory_mapping import (
    SUBCATEGORY_ENGLISH_NAMES,
    get_english_sub_category,
)

# crawling/data/products.json 의 고유 sub_category 스냅샷(52개).
PRODUCT_SUB_CATEGORIES = [
    "경량패딩",
    "기타 니트",
    "기타 아우터",
    "기타 팬츠",
    "긴소매 셔츠",
    "긴소매 티셔츠",
    "나일론/코치 재킷",
    "데님 팬츠",
    "레깅스",
    "레더 재킷",
    "롱코트",
    "롱패딩",
    "무스탕",
    "바람막이",
    "바시티",
    "반소매 셔츠",
    "반소매 티셔츠",
    "배스트",
    "베스트",
    "부츠컷",
    "브이넥",
    "블레이저",
    "블루종",
    "쇼트",
    "숏코트",
    "숏패딩",
    "스웨트셔츠",
    "스트레이트 팬츠",
    "슬랙스",
    "슬리브리스",
    "슬림 팬츠",
    "아노락",
    "야상",
    "와이드 팬츠",
    "점퍼",
    "집업",
    "카디건",
    "코튼 팬츠",
    "크루넥",
    "터틀넥",
    "테님 재킷",
    "트레이닝 재킷",
    "트레이닝 팬츠",
    "트렌치/맥코트",
    "퍼 재킷",
    "폴로셔츠",
    "플리스",
    "피케/카라 티셔츠",
    "하프코트",
    "후드",
    "후드 집업",
    "후디",
]
PRODUCTS_JSON = Path(__file__).resolve().parents[2] / "crawling" / "data" / "products.json"


def test_mapping_covers_exactly_the_known_sub_categories():
    assert len(PRODUCT_SUB_CATEGORIES) == 52
    assert set(SUBCATEGORY_ENGLISH_NAMES) == set(PRODUCT_SUB_CATEGORIES)


@pytest.mark.skipif(not PRODUCTS_JSON.exists(), reason="crawling/data/products.json 이 없는 환경")
def test_mapping_covers_every_sub_category_in_products_json():
    products = json.loads(PRODUCTS_JSON.read_text(encoding="utf-8"))
    actual = {product["sub_category"] for product in products}
    assert actual == set(PRODUCT_SUB_CATEGORIES)
    assert actual <= set(SUBCATEGORY_ENGLISH_NAMES)


@pytest.mark.parametrize(
    ("korean", "english"), [("스웨트셔츠", "sweatshirt"), ("슬림 팬츠", "slim pants")]
)
def test_known_mappings(korean, english):
    assert get_english_sub_category(korean) == english


@pytest.mark.parametrize("sub_category", PRODUCT_SUB_CATEGORIES)
def test_english_names_are_non_empty_ascii(sub_category):
    english = get_english_sub_category(sub_category)
    assert english.strip() == english and english
    assert english.isascii(), f"{sub_category!r} -> {english!r}"


@pytest.mark.parametrize("sub_category", ["없는 카테고리", "", " 스웨트셔츠", "sweatshirt"])
def test_unknown_sub_category_raises_instead_of_falling_back(sub_category):
    with pytest.raises(UnsupportedSubCategoryError) as exc_info:
        get_english_sub_category(sub_category)
    assert exc_info.value.sub_category == sub_category
    assert exc_info.value.message == "unsupported_sub_category"
    assert exc_info.value.status_code == 500
