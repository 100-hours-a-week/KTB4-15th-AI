import pytest

from app.body_image_validation import runtime
from app.body_image_validation.runtime import BodyImageRuntimeManager
from app.clients.s3 import S3ConfigError
from app.config import settings


def test_missing_bucket_is_a_s3_config_error_before_loading_any_model(monkeypatch):
    def fail_if_built():
        raise AssertionError("S3_BUCKET 이 없으면 무거운 모델을 로드하기 전에 멈춰야 한다")

    monkeypatch.setattr(settings, "S3_BUCKET", "")
    monkeypatch.setattr(runtime, "BodyImageRuntime", fail_if_built)
    manager = BodyImageRuntimeManager()

    manager.start()
    with pytest.raises(S3ConfigError):
        manager.get_service()
