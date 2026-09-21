"""DB sub_category(한글) → Runware Pruna prompt 용 영어 garment 명칭.

한글 명칭으로는 garment 선택이 실패하는 것을 확인했다. 값은 크롤러가 수집하는 고정된
sub_category 목록이며, 여기에 없는 값은 원문으로 대체하지 않고 예외를 낸다.
"""

from app.virtual_fitting.exceptions import UnsupportedSubCategoryError

SUBCATEGORY_ENGLISH_NAMES = {
    # 상의
    "반소매 셔츠": "short-sleeve shirt",
    "반소매 티셔츠": "short-sleeve t-shirt",
    "긴소매 셔츠": "long-sleeve shirt",
    "긴소매 티셔츠": "long-sleeve t-shirt",
    "피케/카라 티셔츠": "collared t-shirt",
    "폴로셔츠": "polo shirt",
    "슬리브리스": "sleeveless top",
    "스웨트셔츠": "sweatshirt",
    "후디": "hoodie",
    "후드": "hooded sweater",
    "후드 집업": "zip-up hoodie",
    "집업": "zip-up jacket",
    # 니트
    "기타 니트": "knit sweater",
    "크루넥": "crew neck sweater",
    "브이넥": "v-neck sweater",
    "터틀넥": "turtleneck sweater",
    "카디건": "cardigan",
    "베스트": "vest",
    # 아우터
    "배스트": "vest",
    "블레이저": "blazer",
    "블루종": "bomber jacket",
    "바시티": "varsity jacket",
    "야상": "field jacket",
    "점퍼": "jacket",
    "기타 아우터": "outerwear jacket",
    "바람막이": "windbreaker",
    "아노락": "anorak",
    "플리스": "fleece jacket",
    "무스탕": "shearling jacket",
    "퍼 재킷": "fur jacket",
    "레더 재킷": "leather jacket",
    "테님 재킷": "denim jacket",
    "트레이닝 재킷": "track jacket",
    "나일론/코치 재킷": "nylon coach jacket",
    "숏코트": "short coat",
    "하프코트": "mid-length coat",
    "롱코트": "long coat",
    "트렌치/맥코트": "trench coat",
    "경량패딩": "lightweight puffer jacket",
    "숏패딩": "short puffer jacket",
    "롱패딩": "long puffer coat",
    # 하의
    "슬림 팬츠": "slim pants",
    "스트레이트 팬츠": "straight pants",
    "와이드 팬츠": "wide-leg pants",
    "부츠컷": "bootcut pants",
    "데님 팬츠": "jeans",
    "코튼 팬츠": "cotton pants",
    "슬랙스": "slacks",
    "트레이닝 팬츠": "track pants",
    "기타 팬츠": "pants",
    "레깅스": "leggings",
    "쇼트": "shorts",
}


def get_english_sub_category(sub_category: str) -> str:
    try:
        return SUBCATEGORY_ENGLISH_NAMES[sub_category]
    except KeyError:
        raise UnsupportedSubCategoryError(sub_category) from None
