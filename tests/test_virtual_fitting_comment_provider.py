from app.virtual_fitting.providers.comment import (
    MOCK_COMMENT,
    MOCK_TITLE,
    CommentProvider,
    MockCommentProvider,
)
from tests.virtual_fitting_fixtures import make_bottom, make_top


def test_mock_provider_satisfies_comment_provider_interface():
    assert isinstance(MockCommentProvider(), CommentProvider)


def test_mock_provider_returns_fixed_values_regardless_of_input():
    provider = MockCommentProvider()

    assert provider.generate_comment([make_top()]) == MOCK_COMMENT
    assert provider.generate_comment([make_top(), make_bottom()]) == MOCK_COMMENT
    assert provider.generate_title(MOCK_COMMENT) == MOCK_TITLE
    assert provider.generate_title("다른 코멘트") == MOCK_TITLE


def test_mock_values_are_non_empty_strings():
    assert MOCK_COMMENT.strip()
    assert MOCK_TITLE.strip()
