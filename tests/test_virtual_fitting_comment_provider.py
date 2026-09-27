from app.virtual_fitting.providers.comment import (
    MOCK_COMMENT,
    MOCK_TITLE,
    CommentProvider,
    MockCommentProvider,
)
from app.virtual_fitting.providers.runware_comment import RunwareCommentProvider

RESULT_URL = "https://im.runware.ai/image/result.jpg"


def test_mock_and_runware_providers_satisfy_comment_provider_interface():
    assert isinstance(MockCommentProvider(), CommentProvider)
    assert isinstance(RunwareCommentProvider(api_key="test-llm-key"), CommentProvider)


def test_mock_provider_returns_fixed_values_regardless_of_input():
    provider = MockCommentProvider()

    assert provider.generate_comment(RESULT_URL, ["상의 설명"]) == MOCK_COMMENT
    assert provider.generate_comment(RESULT_URL, ["상의 설명", "하의 설명"]) == MOCK_COMMENT
    assert provider.generate_title(MOCK_COMMENT) == MOCK_TITLE
    assert provider.generate_title("다른 코멘트") == MOCK_TITLE


def test_mock_values_are_non_empty_strings():
    assert MOCK_COMMENT.strip()
    assert MOCK_TITLE.strip()
