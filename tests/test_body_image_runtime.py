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
    monkeypatch.setattr(config, "REMBG_PROVIDERS", ())  # worker 이미지의 provider 설정과 무관하게
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


def test_prebuilt_detectors_are_used_and_closed_by_the_runtime(monkeypatch, tmp_path):
    for name in ("PERSON_DETECTOR_MODEL", "POSE_LANDMARKER_MODEL", "REMBG_MODEL"):
        path = tmp_path / name
        path.touch()
        monkeypatch.setattr(config, name, path)

    def must_not_build(*args, **kwargs):
        raise AssertionError("미리 만든 detector 가 있으면 새로 만들지 않는다")

    for name in ("S3ImageStorage", "RembgBackgroundRemover"):
        monkeypatch.setattr(runtime, name, _Fake)
    monkeypatch.setattr(runtime, "MediaPipePersonDetector", must_not_build)
    monkeypatch.setattr(runtime, "MediaPipePoseDetector", must_not_build)
    closed = []

    class Detector:
        def close(self):
            closed.append(self)

    person, pose = Detector(), Detector()
    built = runtime.BodyImageRuntime(person_detector=person, pose_detector=pose)

    assert built.service.person_detector is person and built.service.pose_detector is pose
    built.close()
    assert closed == [person, pose]
