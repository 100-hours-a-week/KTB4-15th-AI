import pytest

from app.body_image_validation import runtime
from app.body_image_validation.runtime import BodyImageRuntimeManager
from app.clients.s3 import S3ConfigError
from app.config import body_image_validation as config
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


class _Fake:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def close(self):
        pass


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_passes_cpu_mem_arena_setting_to_rembg(monkeypatch, tmp_path, enabled):
    for name in ("PERSON_DETECTOR_MODEL", "POSE_LANDMARKER_MODEL", "REMBG_MODEL"):
        path = tmp_path / name
        path.touch()
        monkeypatch.setattr(config, name, path)
    monkeypatch.setattr(config, "REMBG_ENABLE_CPU_MEM_ARENA", enabled)
    for name in (
        "S3ImageStorage",
        "MediaPipePersonDetector",
        "MediaPipePoseDetector",
        "RembgBackgroundRemover",
    ):
        monkeypatch.setattr(runtime, name, _Fake)

    remover = runtime.BodyImageRuntime().background_remover

    assert remover.args == (str(tmp_path / "REMBG_MODEL"),)
    assert remover.kwargs == {"enable_cpu_mem_arena": enabled}
