"""가상피팅 결과의 llm_comment / llm_title 생성 Provider."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.virtual_fitting.models import FittingProduct

MOCK_COMMENT = "Mock 코멘트: 선택한 상품이 자연스럽게 어우러지는 코디입니다."
MOCK_TITLE = "Mock 제목"


@runtime_checkable
class CommentProvider(Protocol):
    def generate_comment(self, products: Sequence[FittingProduct]) -> str: ...

    def generate_title(self, comment: str) -> str: ...


class MockCommentProvider:
    """항상 고정 문자열을 반환한다."""

    def generate_comment(self, products: Sequence[FittingProduct]) -> str:
        return MOCK_COMMENT

    def generate_title(self, comment: str) -> str:
        return MOCK_TITLE
