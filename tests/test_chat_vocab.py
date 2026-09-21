"""허용값 표.

수집 파이프라인과 analyze 가 같은 표를 보기 때문에, 여기가 어긋나면 필터가 0건을 낸다.
"""

import json
from pathlib import Path

import pytest

from app.chat import vocab

# 단계2 §5-1 에서 쓴 29CM 카테고리 51종. 이 저장소에는 없으므로 개수만 고정해 둔다.
SOURCE_CATEGORY_COUNT = 51


def test_every_source_category_maps_to_one_filter_value():
    sources = [s for values in vocab.CATEGORY_GROUPS.values() for s in values]
    assert len(sources) == SOURCE_CATEGORY_COUNT
    assert len(set(sources)) == len(sources)  # 두 곳에 걸친 값이 없다


def test_filter_values_are_ten():
    assert len(vocab.CATEGORY_GROUPS) == 10
    # '기타'는 상품 정리용이라 사용자 발화에서 고르게 하지 않는다
    assert "기타" not in vocab.ALLOWED_CATEGORIES
    assert len(vocab.ALLOWED_CATEGORIES) == 9


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("긴소매 셔츠", "셔츠"),
        ("반소매 티셔츠", "티셔츠"),
        ("데님 팬츠", "데님"),
        ("와이드 팬츠", "팬츠"),
        ("롱패딩", "아우터"),
        ("터틀넥", "니트"),
        ("존재하지 않는 카테고리", "기타"),
    ],
)
def test_normalize_category(source, expected):
    assert vocab.normalize_category(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [("카키", "그린"), ("아이보리", "화이트"), ("다크네이비", "네이비"), ("형광색", "기타")],
)
def test_normalize_color(source, expected):
    assert vocab.normalize_color(source) == expected


def test_every_filter_value_has_a_fitting_category():
    """가상피팅은 TOP / BOTTOM 만 받는다 (단계1 §6).

    v1 은 아우터와 니트웨어까지 상의로 취급한다.
    """
    assert set(vocab.FITTING_MAIN_CATEGORY) == set(vocab.ALLOWED_CATEGORIES)
    assert set(vocab.FITTING_MAIN_CATEGORY.values()) == {"TOP", "BOTTOM"}
    assert vocab.FITTING_MAIN_CATEGORY["아우터"] == "TOP"
    assert vocab.FITTING_MAIN_CATEGORY["니트"] == "TOP"


def test_source_values_match_the_bench_vocabulary():
    """bench_data/allowed_values.json 과 대조한다.

    수집·측정 저장소가 따로 있으므로 파일이 없으면 건너뛴다.
    """
    path = Path("../bench_data/allowed_values.json")
    if not path.exists():
        pytest.skip("bench_data 를 찾을 수 없다")

    bench = json.loads(path.read_text())
    ours = {s for values in vocab.CATEGORY_GROUPS.values() for s in values}
    assert ours == set(bench["카테고리"])
