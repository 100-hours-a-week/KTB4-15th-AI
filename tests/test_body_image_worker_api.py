"""EC2 앱의 전신 이미지 API 계약 테스트를 Modal worker 앱에 그대로 다시 돌린다.

test_body_image_validation_api.py 의 test_* 를 가져와, 같은 이름의 client fixture 만 worker 앱으로 바꾼다.
endpoint·method·multipart 필드·응답 형식·오류 코드·HTTP status 가 두 앱에서 같아야 한다.
"""

import pytest
from fastapi.testclient import TestClient

from app.body_image_validation import worker
from app.config import settings
from tests import test_body_image_validation_api as ec2_api_tests

globals().update(
    {name: value for name, value in vars(ec2_api_tests).items() if name.startswith("test_")}
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", ec2_api_tests.KEY)
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)
    with TestClient(worker.create_app(), raise_server_exceptions=False) as test_client:
        yield test_client
