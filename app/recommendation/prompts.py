"""찜 기반 추천의 프롬프트 두 개 (단계5 §4 "찜 기반 추천 - LLM 입력 / 출력 포맷").

공통 특징 추출: 한 갈래로 고른 찜 상품들의 image_description → 검색 질의 한 문장
추천 이유:     그 상품을 추천하게 된 계기 + 추천 상품의 image_description → 이유 한두 문장

둘 다 complete_json 으로 받는다. 키 이름은 아래 상수와 프롬프트 문구가 같아야 한다.
"""

FEATURES_KEY = "features"
REASON_KEY = "reason"

# 갈래 이름 → 계기 문장에 쓸 말
_BRANCH_LABELS = {"color": "색상", "category": "종류"}

_FEATURES_SYSTEM = f"""너는 패션 쇼핑몰의 상품 검색 질의를 만든다.

사용자가 최근 찜한 옷 몇 벌의 설명을 받는다. 설명들에 반복해서 나타나는 특징을 찾아,
비슷한 옷을 찾는 검색 질의 한 문장으로 쓴다.

1. 여러 설명에 함께 나타나는 핏·실루엣·소재·디테일·분위기만 쓴다. 한 벌에만 있는 특징은 쓰지 않는다.
2. 색상 이름과 옷 종류 이름(니트, 팬츠 등)은 쓰지 않는다. 그 조건은 검색 필터가 따로 건다.
3. 설명에 없는 특징을 지어내지 않는다.
4. 명사로 끝나는 짧은 한 문장으로 쓴다. 예: 여유로운 실루엣과 단정한 디테일의 미니멀한 디자인

JSON 으로만 답한다: {{"{FEATURES_KEY}": "검색 질의"}}"""

_REASON_SYSTEM = f"""너는 패션 쇼핑몰에서 상품을 추천하는 이유를 쓴다.

사용자가 최근 찜한 상품을 분석한 결과(추천 계기)와 추천할 상품의 설명을 받는다.
이 상품을 왜 추천하는지 사용자에게 말하듯 쓴다.

1. 추천 계기와 추천 상품 설명이 겹치는 점을 연결한다. 예: 자주 찜하신 블랙 색상, 찜하신 옷들처럼 여유로운 실루엣
2. 추천 상품 설명에 없는 특징을 지어내지 않는다.
3. 가격·브랜드·할인은 말하지 않는다.
4. 한두 문장, 80자 이내로 쓴다. "~입니다" 체로 끝낸다.

JSON 으로만 답한다: {{"{REASON_KEY}": "추천 이유"}}"""


def features_prompt(descriptions: list[str]) -> list[dict]:
    listed = "\n".join(f"- {description}" for description in descriptions)
    return [
        {"role": "system", "content": _FEATURES_SYSTEM},
        {"role": "user", "content": f"[찜한 옷의 설명]\n{listed}"},
    ]


def reason_prompt(triggers: list[dict], description: str) -> list[dict]:
    """triggers 는 이 상품을 찾아낸 갈래들이다. 두 갈래에 다 나온 상품은 두 개를 받는다.

    갈래 하나는 {"kind": "color" | "category", "value": "블랙", "features": "..."} 이다.
    """
    lines = [
        f"- 최근 찜한 상품 중 {_BRANCH_LABELS[t['kind']]} '{t['value']}' 이(가) 가장 많았다. "
        f"그 상품들의 공통 특징: {t['features']}"
        for t in triggers
    ]
    return [
        {"role": "system", "content": _REASON_SYSTEM},
        {
            "role": "user",
            "content": "[추천 계기]\n" + "\n".join(lines) + f"\n\n[추천 상품 설명]\n{description}",
        },
    ]
