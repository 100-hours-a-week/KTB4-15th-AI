from app.virtual_fitting.product_selection import validate_and_sort_products
from app.virtual_fitting.prompt import build_fitting_prompt
from tests.virtual_fitting_fixtures import make_bottom, make_top


def test_top_and_bottom():
    prompt = build_fitting_prompt(
        [make_top(sub_category="스웨트셔츠"), make_bottom(sub_category="데님 팬츠")]
    )
    assert prompt == (
        "Garment 1: 스웨트셔츠.\n"
        "Garment 2: 데님 팬츠.\n"
        "Dress the person with the provided garments."
    )


def test_single_product_has_only_garment_1():
    prompt = build_fitting_prompt([make_bottom(sub_category="데님 팬츠")])
    assert prompt == "Garment 1: 데님 팬츠.\nDress the person with the provided garments."
    assert "Garment 2" not in prompt


def test_prompt_order_matches_sorted_garment_images():
    top = make_top(sub_category="후디", image_url="https://img/top.jpg")
    bottom = make_bottom(sub_category="와이드 팬츠", image_url="https://img/bottom.jpg")

    ordered = validate_and_sort_products([bottom, top])
    prompt = build_fitting_prompt(ordered)
    garment_images = [product.image_url for product in ordered]

    assert garment_images == ["https://img/top.jpg", "https://img/bottom.jpg"]
    assert prompt.index("Garment 1: 후디.") < prompt.index("Garment 2: 와이드 팬츠.")
