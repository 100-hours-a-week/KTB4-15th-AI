"""사용자 발화에서 가격 범위를 규칙으로 뽑는다.

LLM 에 맡기면 같은 말에도 결과가 흔들리고, 예시를 넣으면 analyze 가 느려질 수 있다
(2026-09-30 측정). 가격 표현은 형태가 정해져 있어 코드가 뽑는다. analyze 프롬프트는 가격을 다루지 않는다.

숫자는 "5만원", "5만 원", "50000원", "5,000원", "5천원", "오만원", "삼만오천원"을 받는다.
"""

import re

# "이상"은 상한이 없지만 min/max 를 항상 둘 다 채우기로 했다 (2026-09-30 결정).
# sabu: 상한 없음 — "천만원 이상"이면 (min, max) 는 무엇이 되고, 검색은 몇 건을 돌려주지?
NO_UPPER_BOUND = 9_999_999

_DIGITS = {"영": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_SMALL_UNITS = {"십": 10, "백": 100, "천": 1000}

_NUM = r"[0-9,일이삼사오육칠팔구십백천만]+"
_WON = r"\s?원"
_RANGE = re.compile(
    rf"(?P<low>{_NUM})(?:{_WON})?\s*(?:~|-|〜|에서|부터)\s*(?P<high>{_NUM}){_WON}"
)
_SINGLE = re.compile(
    rf"(?P<num>{_NUM}){_WON}\s*"
    r"(?P<q>대|이하|이내|까지|아래|미만|이상|초과|이?\s*넘|정도|쯤|내외|전후)?"
)


def _to_int(text: str) -> int | None:
    """"3만5천" → 35000, "십오만" → 150000, "만" → 10000. 읽을 수 없으면 None."""
    text = text.replace(",", "")
    total = 0
    section = 0  # 만 아래 자리
    num: int | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if char.isdigit():
            end = index
            while end < len(text) and text[end].isdigit():
                end += 1
            num = int(text[index:end])
            index = end
            continue
        if char in _DIGITS:
            num = _DIGITS[char]
        elif char in _SMALL_UNITS:
            section += (1 if num is None else num) * _SMALL_UNITS[char]
            num = None
        elif char == "만":
            section += 0 if num is None else num
            total += (section or 1) * 10_000
            section = 0
            num = None
        index += 1
    total += section + (0 if num is None else num)
    return total or None


def _band_width(value: int) -> int:
    """"N만원대"의 폭. 끝자리가 0이면 한 자리 위로 올라간다.

    5만 → 1만(50000~59999), 10만 → 10만(100000~199999), 15만 → 1만, 20만 → 10만.
    """
    width = 1
    while value % (width * 10) == 0:
        width *= 10
    return width


def _range_low(low: str, high: str) -> int | None:
    """"5~7만원"처럼 앞 숫자에 단위가 없으면 뒤 숫자의 마지막 단위를 붙인다."""
    if not any(unit in low for unit in "십백천만") and high[-1] in "십백천만":
        low += high[-1]
    return _to_int(low)


# sabu: 가격 판정 — 가격만 말한 "5만원짜리"(표의 ? 칸)는 지금 무엇을 돌려주지?
#       한 발화에 가격이 둘("5만원 이하, 아니 7만원 이하")이거나 "5만원 넘는 건 싫어"처럼 부정이면 무엇이 잡히지?
def parse_price(message: str) -> tuple[int, int] | None:
    """발화에서 (min_price, max_price) 를 뽑는다. 가격 조건이 없으면 None."""
    matched = _RANGE.search(message)
    if matched:
        low = _range_low(matched["low"], matched["high"])
        high = _to_int(matched["high"])
        if low is not None and high is not None:
            return low, high

    for matched in _SINGLE.finditer(message):
        value = _to_int(matched["num"])
        qualifier = (matched["q"] or "").replace(" ", "")
        if value is None or not qualifier:
            continue
        if qualifier == "대":
            return value, value + _band_width(value) - 1
        if qualifier in ("이하", "이내", "까지", "아래"):
            return 0, value
        if qualifier == "미만":
            return 0, value - 1
        if qualifier in ("이상", "초과", "넘", "이넘"):
            return value, NO_UPPER_BOUND
        return round(value * 0.9), round(value * 1.1)
    return None
