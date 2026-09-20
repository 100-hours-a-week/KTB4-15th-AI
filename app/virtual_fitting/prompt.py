from typing import Sequence

from app.virtual_fitting.models import FittingProduct

_INSTRUCTION = "Dress the person with the provided garments."


def build_fitting_prompt(products: Sequence[FittingProduct]) -> str:
    """products 의 순서 그대로 Garment N 을 붙인다. 정렬은 호출 전에 끝나 있어야 한다."""
    lines = [
        f"Garment {index}: {product.sub_category}."
        for index, product in enumerate(products, start=1)
    ]
    lines.append(_INSTRUCTION)
    return "\n".join(lines)
