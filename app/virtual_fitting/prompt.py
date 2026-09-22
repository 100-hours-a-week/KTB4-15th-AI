from collections.abc import Sequence

from app.virtual_fitting.models import FittingProduct
from app.virtual_fitting.subcategory_mapping import get_english_sub_category

_ORDINALS = (
    "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "eleventh",
)
_PRESERVE = (
    "Preserve the person's face, hair, body shape, pose, hands, "
    "camera angle, and background."
)


def build_fitting_prompt(products: Sequence[FittingProduct]) -> str:
    """products 의 순서 그대로 N 번째 garment image 를 영어 sub_category 명칭과 연결한다.

    정렬은 호출 전에 끝나 있어야 하며, garment_image_urls 도 같은 순서여야 한다.
    """
    parts = [
        f"the {get_english_sub_category(product.sub_category)} "
        f"from the {_ORDINALS[index]} garment image"
        for index, product in enumerate(products)
    ]
    if len(parts) == 1:
        dress = f"Dress the person in {parts[0]}."
        realism = (
            "Keep the garment realistic with natural folds, accurate fit, "
            "and believable shadows."
        )
    else:
        dress = f"Dress the person in {', '.join(parts[:-1])} and {parts[-1]}."
        which = "both garments" if len(parts) == 2 else "all garments"
        realism = (
            f"Keep {which} realistic with natural folds, accurate fit, "
            "clean layering at the waist, and believable shadows."
        )
    return f"{dress}\n{_PRESERVE}\n{realism}"
