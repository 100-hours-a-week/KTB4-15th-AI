from app.virtual_fitting.models import BOTTOM_CATEGORY, TOP_CATEGORY, FittingProduct


def make_product(
    code: str,
    main_category: str = TOP_CATEGORY,
    sub_category: str = "스웨트셔츠",
    image_url: str | None = "https://img.29cm.co.kr/item/example.jpg",
    description_summary: str | None = "부드러운 소재의 기본 스웨트셔츠",
) -> FittingProduct:
    return FittingProduct(
        product_code=code,
        image_url=image_url,
        main_category=main_category,
        sub_category=sub_category,
        description_summary=description_summary,
    )


def make_top(code: str = "1", **kwargs) -> FittingProduct:
    return make_product(code, TOP_CATEGORY, **kwargs)


def make_bottom(code: str = "2", **kwargs) -> FittingProduct:
    kwargs.setdefault("sub_category", "데님 팬츠")
    return make_product(code, BOTTOM_CATEGORY, **kwargs)


class AllowAllBudget:
    """FittingBudget 대역. 막지 않고, 기록된 금액만 모은다."""

    def __init__(self):
        self.checks = 0
        self.recorded = []

    def ensure_available(self):
        self.checks += 1

    def record(self, cost):
        self.recorded.append(cost)
