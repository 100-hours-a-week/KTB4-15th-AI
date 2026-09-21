from app.virtual_fitting.models import BOTTOM_CATEGORY, TOP_CATEGORY, FittingProduct


def make_product(
    code: str,
    main_category: str = TOP_CATEGORY,
    sub_category: str = "스웨트셔츠",
    image_url: str | None = "https://img.29cm.co.kr/item/example.jpg",
) -> FittingProduct:
    return FittingProduct(
        product_code=code,
        image_url=image_url,
        main_category=main_category,
        sub_category=sub_category,
    )


def make_top(code: str = "1", **kwargs) -> FittingProduct:
    return make_product(code, TOP_CATEGORY, **kwargs)


def make_bottom(code: str = "2", **kwargs) -> FittingProduct:
    kwargs.setdefault("sub_category", "데님 팬츠")
    return make_product(code, BOTTOM_CATEGORY, **kwargs)
