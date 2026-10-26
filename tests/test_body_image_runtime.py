import pytest

from app.body_image_validation import inference, runtime
from app.body_image_validation.inference import BodyImageModels, BodyImageRuntimeError
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


@pytest.fixture
def model_files(monkeypatch, tmp_path):
    for name in ("PERSON_DETECTOR_MODEL", "POSE_LANDMARKER_MODEL", "REMBG_MODEL"):
        path = tmp_path / name
        path.touch()
        monkeypatch.setattr(config, name, path)
    monkeypatch.setattr(config, "GPU_PREPROCESSING", False)
    monkeypatch.setattr(config, "REMBG_PROVIDERS", ())  # worker 이미지의 provider 설정과 무관하게
    return tmp_path


class Detector:
    def __init__(self, closed):
        self.closed_log = closed

    def close(self):
        self.closed_log.append(self)


def must_not_build(*args, **kwargs):
    raise AssertionError("만들면 안 되는 객체다")


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_passes_cpu_mem_arena_setting_to_rembg(monkeypatch, model_files, enabled):
    monkeypatch.setattr(config, "REMBG_ENABLE_CPU_MEM_ARENA", enabled)
    monkeypatch.setattr(runtime, "S3ImageStorage", _Fake)
    for name in ("MediaPipePersonDetector", "MediaPipePoseDetector", "RembgBackgroundRemover"):
        monkeypatch.setattr(inference, name, _Fake)

    remover = runtime.BodyImageRuntime.create().background_remover

    assert remover.args == (str(model_files / "REMBG_MODEL"),)
    assert remover.kwargs == {"enable_cpu_mem_arena": enabled}


def test_prebuilt_detectors_are_used_and_closed_by_the_runtime(monkeypatch, model_files):
    monkeypatch.setattr(runtime, "S3ImageStorage", _Fake)
    monkeypatch.setattr(inference, "RembgBackgroundRemover", _Fake)
    monkeypatch.setattr(inference, "MediaPipePersonDetector", must_not_build)
    monkeypatch.setattr(inference, "MediaPipePoseDetector", must_not_build)
    closed = []
    person, pose = Detector(closed), Detector(closed)

    models = BodyImageModels.load(person_detector=person, pose_detector=pose)
    built = runtime.BodyImageRuntime.create(models)

    assert built.service.person_detector is person and built.service.pose_detector is pose
    built.close()
    built.close()  # 두 번 닫아도 detector 는 한 번만 닫는다
    assert closed == [person, pose]


def test_models_are_built_without_s3_and_runtime_adds_the_storage_and_limiter(monkeypatch, model_files):
    for name in ("MediaPipePersonDetector", "MediaPipePoseDetector", "RembgBackgroundRemover"):
        monkeypatch.setattr(inference, name, _Fake)
    monkeypatch.setattr(runtime, "S3ImageStorage", must_not_build)

    models = BodyImageModels.load()  # S3 client 없이 만든다

    storage = _Fake()
    built = runtime.BodyImageRuntime(models, storage)
    assert built.models is models
    assert built.service.image_storage is storage
    assert built.service.limiter is not None
    # runtime 마다 새 service·limiter 다(모델은 공유해도 요청 상태는 공유하지 않는다).
    assert runtime.BodyImageRuntime(models, storage).service.limiter is not built.service.limiter


def test_rembg_failure_closes_the_detectors_already_built(monkeypatch, model_files):
    def broken_remover(*args, **kwargs):
        raise RuntimeError("CUDA init failed")

    monkeypatch.setattr(inference, "RembgBackgroundRemover", broken_remover)
    closed = []
    person, pose = Detector(closed), Detector(closed)

    with pytest.raises(BodyImageRuntimeError):
        BodyImageModels.load(person_detector=person, pose_detector=pose)
    assert closed == [person, pose]


def test_gpu_preprocessor_failure_closes_the_detectors(monkeypatch, model_files):
    monkeypatch.setattr(inference, "RembgBackgroundRemover", _Fake)
    monkeypatch.setattr(inference, "build_gpu_preprocessor", lambda: (_ for _ in ()).throw(TypeError("bug")))
    closed = []
    person, pose = Detector(closed), Detector(closed)

    with pytest.raises(TypeError):  # 코드 오류는 감싸지 않고 그대로 낸다
        BodyImageModels.load(person_detector=person, pose_detector=pose)
    assert closed == [person, pose]


def test_storage_failure_closes_the_given_models_once(monkeypatch, model_files):
    def broken_storage():
        raise RuntimeError("bad boto3 config")

    monkeypatch.setattr(runtime, "S3ImageStorage", broken_storage)
    monkeypatch.setattr(inference, "RembgBackgroundRemover", _Fake)
    closed = []
    models = BodyImageModels.load(person_detector=Detector(closed), pose_detector=Detector(closed))

    with pytest.raises(BodyImageRuntimeError):
        runtime.BodyImageRuntime.create(models)
    models.close()  # 호출자가 다시 닫아도 중복 종료하지 않는다
    assert len(closed) == 2


def test_local_runtime_builds_s3_before_loading_models(monkeypatch, model_files):
    def broken_storage():
        raise RuntimeError("bad boto3 config")

    monkeypatch.setattr(runtime, "S3ImageStorage", broken_storage)
    for name in ("MediaPipePersonDetector", "MediaPipePoseDetector", "RembgBackgroundRemover"):
        monkeypatch.setattr(inference, name, must_not_build)

    with pytest.raises(BodyImageRuntimeError):
        runtime.BodyImageRuntime.create()


# --- 모델 초기화 실패 시 정리와 중복 종료 ---


def test_missing_model_file_closes_the_given_detectors(monkeypatch, model_files):
    monkeypatch.setattr(config, "REMBG_MODEL", model_files / "missing.onnx")
    monkeypatch.setattr(inference, "RembgBackgroundRemover", must_not_build)
    closed = []
    person, pose = Detector(closed), Detector(closed)

    with pytest.raises(BodyImageRuntimeError, match="모델 파일"):
        BodyImageModels.load(person_detector=person, pose_detector=pose)
    assert closed == [person, pose]


def test_pose_failure_closes_the_person_detector_built_here(monkeypatch, model_files):
    closed = []
    built = []

    def person_factory(path):
        built.append(Detector(closed))
        return built[-1]

    def broken_pose(path):
        raise RuntimeError("bad pose model")

    monkeypatch.setattr(inference, "MediaPipePersonDetector", person_factory)
    monkeypatch.setattr(inference, "MediaPipePoseDetector", broken_pose)
    monkeypatch.setattr(inference, "RembgBackgroundRemover", must_not_build)

    with pytest.raises(BodyImageRuntimeError):
        BodyImageModels.load()
    assert closed == built and len(built) == 1


def test_only_the_missing_detector_is_built_and_both_are_closed_on_failure(monkeypatch, model_files):
    closed = []
    given_person = Detector(closed)
    built = []

    def pose_factory(path):
        built.append(Detector(closed))
        return built[-1]

    def broken_remover(*args, **kwargs):
        raise RuntimeError("CUDA init failed")

    monkeypatch.setattr(inference, "MediaPipePersonDetector", must_not_build)
    monkeypatch.setattr(inference, "MediaPipePoseDetector", pose_factory)
    monkeypatch.setattr(inference, "RembgBackgroundRemover", broken_remover)

    with pytest.raises(BodyImageRuntimeError):
        BodyImageModels.load(person_detector=given_person)
    assert closed == [given_person, *built]


def test_a_failing_close_does_not_hide_the_init_error_or_skip_the_other(monkeypatch, model_files):
    closed = []

    class BrokenClose:
        def close(self):
            raise RuntimeError("close failed")

    def broken_remover(*args, **kwargs):
        raise RuntimeError("CUDA init failed")

    monkeypatch.setattr(inference, "RembgBackgroundRemover", broken_remover)
    pose = Detector(closed)

    with pytest.raises(BodyImageRuntimeError) as exc_info:
        BodyImageModels.load(person_detector=BrokenClose(), pose_detector=pose)
    assert str(exc_info.value.__cause__) == "CUDA init failed"
    assert closed == [pose]


def test_models_close_is_idempotent_and_survives_a_failing_detector(monkeypatch, model_files):
    calls = []

    class Counting:
        def __init__(self, fail=False):
            self.fail = fail

        def close(self):
            calls.append(self)
            if self.fail:
                raise RuntimeError("close failed")

    monkeypatch.setattr(inference, "RembgBackgroundRemover", _Fake)
    person, pose = Counting(fail=True), Counting()
    models = BodyImageModels.load(person_detector=person, pose_detector=pose)

    models.close()
    models.close()
    assert calls == [person, pose]


@pytest.mark.parametrize("cls_name", ["MediaPipePersonDetector", "MediaPipePoseDetector"])
def test_mediapipe_detector_close_is_idempotent(cls_name):
    import threading

    from app.body_image_validation import detectors

    class Task:
        closes = 0

        def close(self):
            Task.closes += 1

    detector = getattr(detectors, cls_name).__new__(getattr(detectors, cls_name))
    detector._detector = Task()
    detector._lock = threading.Lock()
    detector._closed = False

    detector.close()
    detector.close()
    assert Task.closes == 1



def test_rembg_profiling_is_passed_only_when_configured(monkeypatch, model_files):
    monkeypatch.setattr(inference, "RembgBackgroundRemover", _Fake)

    monkeypatch.setattr(config, "REMBG_PROFILE_RUNS", 0)
    assert "profile_runs" not in inference.build_background_remover().kwargs

    monkeypatch.setattr(config, "REMBG_PROFILE_RUNS", 2)
    assert inference.build_background_remover().kwargs["profile_runs"] == 2


def test_runtime_passes_the_diagnostics_setting_to_the_service(monkeypatch, model_files):
    monkeypatch.setattr(runtime, "S3ImageStorage", _Fake)
    for name in ("MediaPipePersonDetector", "MediaPipePoseDetector", "RembgBackgroundRemover"):
        monkeypatch.setattr(inference, name, _Fake)

    monkeypatch.setattr(config, "DIAGNOSTICS", True)
    assert runtime.BodyImageRuntime.create().service._diagnostics is not None
    monkeypatch.setattr(config, "DIAGNOSTICS", False)
    assert runtime.BodyImageRuntime.create().service._diagnostics is None
