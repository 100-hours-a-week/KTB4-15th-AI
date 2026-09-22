import pytest

from app.virtual_fitting.exceptions import UnsupportedSubCategoryError
from app.virtual_fitting.product_selection import validate_and_sort_products
from app.virtual_fitting.prompt import build_fitting_prompt
from tests.virtual_fitting_fixtures import make_bottom, make_top


def test_top_and_bottom():
    prompt = build_fitting_prompt(
        [make_top(sub_category="스웨트셔츠"), make_bottom(sub_category="슬림 팬츠")]
    )
    assert prompt == (
        "Dress the person in the sweatshirt from the first garment image "
        "and the slim pants from the second garment image.\n"
        "Preserve the person's face, hair, body shape, pose, hands, camera angle, and background.\n"
        "Keep both garments realistic with natural folds, accurate fit, "
        "clean layering at the waist, and believable shadows."
    )


def test_single_product_references_only_first_garment_image():
    prompt = build_fitting_prompt([make_top(sub_category="스웨트셔츠")])
    assert prompt == (
        "Dress the person in the sweatshirt from the first garment image.\n"
        "Preserve the person's face, hair, body shape, pose, hands, camera angle, and background.\n"
        "Keep the garment realistic with natural folds, accurate fit, and believable shadows."
    )
    assert "second" not in prompt
    assert "both" not in prompt


def test_single_bottom_uses_english_name():
    prompt = build_fitting_prompt([make_bottom(sub_category="슬림 팬츠")])
    assert prompt.startswith("Dress the person in the slim pants from the first garment image.")


def test_prompt_contains_no_korean():
    prompt = build_fitting_prompt(
        [make_top(sub_category="후드 집업"), make_bottom(sub_category="트레이닝 팬츠")]
    )
    assert prompt.isascii()


def test_prompt_order_matches_sorted_garment_images():
    top = make_top(sub_category="스웨트셔츠", image_url="https://img/top.jpg")
    bottom = make_bottom(sub_category="슬림 팬츠", image_url="https://img/bottom.jpg")

    ordered = validate_and_sort_products([bottom, top])
    prompt = build_fitting_prompt(ordered)
    garment_images = [product.image_url for product in ordered]

    assert garment_images == ["https://img/top.jpg", "https://img/bottom.jpg"]
    assert "the sweatshirt from the first garment image" in prompt
    assert "the slim pants from the second garment image" in prompt
    assert prompt.index("sweatshirt") < prompt.index("slim pants")


def test_unmapped_sub_category_raises_and_is_not_used_verbatim():
    with pytest.raises(UnsupportedSubCategoryError):
        build_fitting_prompt([make_top(sub_category="없는 카테고리")])
