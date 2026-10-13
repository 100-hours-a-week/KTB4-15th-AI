"""Modal T4 worker 의 앱 구성과 초기화 순서. Modal·CUDA 없이 CPU 에서 돈다.

실제 MediaPipe·rembg·S3 대신 fake 를 넣고, runtime 을 만드는 순서·한 번만 만드는지·warm-up 이
S3 를 건드리지 않는지·S3 업로드 성공/실패가 기존 API 응답으로 나가는지를 확인한다.
"""

import json
import logging
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.body_image_validation import router, runtime, worker
from app.body_image_validation.models import BoundingBox, PersonDetection
from app.body_image_validation.runtime import BodyImageRuntimeManager
from app.body_image_validation.service import BodyImageValidationService
from app.clients.s3 import S3ImageStorage
from app.config import body_image_validation as config
from app.config import settings
from tests.test_body_image_validation_service import valid_landmarks

URL = "/api/v1/validation/body-image"
KEY = "worker-test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
ROOT = Path(__file__).resolve().parents[1]


class FakePersonDetector:
    instances = 0

    def __init__(self, *args):
        type(self).instances += 1
        self.calls = 0
        self.closed = False

    def detect(self, image):
        self.calls += 1
        height, width = image.array.shape[:2]
        return [PersonDetection(BoundingBox(10, 10, width - 20, height - 20), 0.9)]

    def close(self):
        self.closed = True


class FakePoseDetector(FakePersonDetector):
    def detect(self, image):
        self.calls += 1
        return valid_landmarks()


class FakeRemover:
    instances = 0

    def __init__(self, model_path, **options):
        type(self).instances += 1
        self.options = options
        self.calls = 0

    @property
    def providers(self):
        return list(self.options.get("providers", ["CPUExecutionProvider"]))

    def remove(self, image):
        self.calls += 1
        return image.convert("RGBA")


class FakeS3Client:
    def __init__(self, error=None):
        self.error = error
        self.puts = []

    def put_object(self, **kwargs):
        if self.error:
            raise self.error
        self.puts.append(kwargs)


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """모델 파일·S3·MediaPipe·rembg 를 fake 로 바꾸고 새 runtime manager 를 router 와 공유한다."""
    for name in ("PERSON_DETECTOR_MODEL", "POSE_LANDMARKER_MODEL", "REMBG_MODEL"):
        path = tmp_path / name
        path.touch()
        monkeypatch.setattr(config, name, path)
    monkeypatch.setattr(config, "GPU_PREPROCESSING", False)
    monkeypatch.setattr(config, "REMBG_PROVIDERS", ())
    monkeypatch.setattr(settings, "S3_BUCKET", "test-bucket")
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", KEY)
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)
    s3 = FakeS3Client()
    monkeypatch.setattr(runtime, "S3ImageStorage", lambda: S3ImageStorage(client=s3, bucket="test-bucket"))
    monkeypatch.setattr(runtime, "MediaPipePersonDetector", FakePersonDetector)
    monkeypatch.setattr(runtime, "MediaPipePoseDetector", FakePoseDetector)
    monkeypatch.setattr(runtime, "RembgBackgroundRemover", FakeRemover)
    FakePersonDetector.instances = FakePoseDetector.instances = FakeRemover.instances = 0
    manager = BodyImageRuntimeManager()
    monkeypatch.setattr(runtime, "runtime_manager", manager)
    monkeypatch.setattr(router, "runtime_manager", manager)
    yield {"manager": manager, "s3": s3}
    manager.close()


def jpeg(size=(600, 800)):
    output = BytesIO()
    Image.new("RGB", size, (180, 170, 160)).save(output, format="JPEG")
    return output.getvalue()


def post(client, body, headers=AUTH):
    return client.post(URL, files={"image": ("body.jpg", body, "image/jpeg")}, headers=headers)


# --- 앱 구성 ---


def test_worker_app_serves_only_the_body_image_endpoint_and_health():
    paths = {path: sorted(methods) for path, methods in worker.create_app().openapi()["paths"].items()}

    assert paths == {URL: ["post"], "/health": ["get"]}


def test_snapshot_phase_modules_do_not_read_secret_settings():
    # prepare_cpu() 가 쓰는 app 모듈(worker·detectors)이 settings(INTERNAL_API_KEY·S3·AWS)를
    # import 하지 않아야 snapshot 에 secret 이 담기지 않는다. 다른 테스트가 이미 import 했을 수 있어
    # 새 프로세스에서 확인한다. 무거운 라이브러리 import(preload_modules)는 여기서 하지 않는다.
    code = (
        "import sys, json\n"
        "from app.body_image_validation import worker\n"
        "import app.body_image_validation.detectors\n"
        "print(json.dumps({'settings': 'app.config.settings' in sys.modules,"
        " 's3': 'app.clients.s3' in sys.modules, 'torch': 'torch' in sys.modules}))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "settings": False, "s3": False, "torch": False,
    }


def test_preload_imports_torch_only_for_gpu_preprocessing(monkeypatch):
    imported = []
    monkeypatch.setattr(worker.importlib, "import_module", imported.append)

    monkeypatch.setattr(config, "GPU_PREPROCESSING", False)
    worker.preload_modules()
    assert "rembg" in imported and "torch" not in imported

    imported.clear()
    monkeypatch.setattr(config, "GPU_PREPROCESSING", True)
    worker.preload_modules()
    assert {"rembg", "torch", "torchvision"} <= set(imported)


# --- 초기화 순서와 모델 재사용 ---


def test_start_gpu_builds_the_runtime_once_with_the_snapshot_detectors(worker_env):
    person, pose = FakePersonDetector(), FakePoseDetector()

    timings = worker.start_gpu(person, pose)
    worker.start_gpu(person, pose)  # 두 번째 호출은 runtime 을 다시 만들지 않는다

    service = worker_env["manager"].get_service()
    # worker 는 기존 local service 만 쓴다(Modal 을 다시 부르는 remote service 가 아니다).
    assert type(service) is BodyImageValidationService
    assert service.person_detector is person and service.pose_detector is pose
    # 위에서 만든 것뿐이다. runtime 이 detector 를 새로 만들지 않는다.
    assert (FakePersonDetector.instances, FakePoseDetector.instances, FakeRemover.instances) == (1, 1, 1)
    assert set(timings) == {"preprocess", "person_pose", "rembg", "png"}


def test_warm_up_runs_every_stage_without_uploading_to_s3(worker_env):
    person, pose = FakePersonDetector(), FakePoseDetector()

    worker.start_gpu(person, pose)

    remover = worker_env["manager"].runtime.background_remover
    assert (person.calls, pose.calls, remover.calls) == (1, 1, 1)
    assert worker_env["s3"].puts == []


def test_requests_reuse_the_models_built_at_startup(worker_env):
    worker.start_gpu(FakePersonDetector(), FakePoseDetector())
    client = TestClient(worker.create_app(), raise_server_exceptions=False)

    for _ in range(3):
        assert post(client, jpeg()).status_code == 200

    assert (FakePersonDetector.instances, FakePoseDetector.instances, FakeRemover.instances) == (1, 1, 1)


def test_rembg_providers_come_from_config(worker_env, monkeypatch):
    monkeypatch.setattr(config, "REMBG_PROVIDERS", ("CUDAExecutionProvider", "CPUExecutionProvider"))
    monkeypatch.setattr(worker, "_preload_cuda_libraries", lambda: None)

    worker.start_gpu(FakePersonDetector(), FakePoseDetector())

    remover = worker_env["manager"].runtime.background_remover
    assert remover.options["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert remover.options["enable_cpu_mem_arena"] is config.REMBG_ENABLE_CPU_MEM_ARENA


def test_cuda_requested_but_cpu_active_is_logged(worker_env, monkeypatch, caplog):
    monkeypatch.setattr(config, "REMBG_PROVIDERS", ("CUDAExecutionProvider", "CPUExecutionProvider"))
    monkeypatch.setattr(worker, "_preload_cuda_libraries", lambda: None)
    monkeypatch.setattr(FakeRemover, "providers", property(lambda self: ["CPUExecutionProvider"]))

    with caplog.at_level(logging.WARNING, logger="app.body_image_validation.worker"):
        worker.start_gpu(FakePersonDetector(), FakePoseDetector())

    assert "rembg CUDA provider is not active" in caplog.text


# --- 인증 설정 ---


def test_auth_disabled_refuses_to_start(worker_env, monkeypatch):
    monkeypatch.setattr(settings, "AUTH_DISABLED", True)

    with pytest.raises(worker.WorkerConfigError):
        worker.start_gpu(FakePersonDetector(), FakePoseDetector())
    assert worker_env["manager"].runtime is None


def test_missing_internal_key_rejects_every_request(worker_env, monkeypatch, caplog):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "")

    with caplog.at_level(logging.ERROR, logger="app.body_image_validation.worker"):
        worker.start_gpu(FakePersonDetector(), FakePoseDetector())
    client = TestClient(worker.create_app(), raise_server_exceptions=False)

    assert "INTERNAL_API_KEY is not set" in caplog.text
    for headers in ({}, {"Authorization": "Bearer "}, {"Authorization": "Bearer anything"}):
        response = post(client, jpeg(), headers)
        assert response.status_code == 401
        assert response.json() == {"code": "UNAUTHORIZED", "data": None, "message": "인증에 실패했습니다."}
    assert worker_env["s3"].puts == []


# --- S3 업로드 (mock client) ---


def test_pass_uploads_the_png_to_s3_before_answering(worker_env):
    worker.start_gpu(FakePersonDetector(), FakePoseDetector())
    client = TestClient(worker.create_app(), raise_server_exceptions=False)

    response = post(client, jpeg())

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "BODY_IMAGE_UPLOAD_SUCCESS"
    [put] = worker_env["s3"].puts
    assert put["Bucket"] == "test-bucket"
    assert put["Key"] == body["data"]["s3_key"]
    assert put["Key"].startswith("body-images/") and put["Key"].endswith(".png")
    assert put["ContentType"] == "image/png"
    assert Image.open(BytesIO(put["Body"])).mode == "RGBA"


def test_s3_failure_is_a_500_not_a_success(worker_env):
    worker.start_gpu(FakePersonDetector(), FakePoseDetector())
    worker_env["s3"].error = RuntimeError("AccessDenied")
    client = TestClient(worker.create_app(), raise_server_exceptions=False)

    response = post(client, jpeg())

    assert response.status_code == 500
    assert response.json() == {
        "code": "IMAGE_UPLOAD_FAILED",
        "data": None,
        "message": "전신 사진 처리 중 서버 오류가 발생했습니다.",
    }


def test_validation_failure_does_not_upload(worker_env, monkeypatch):
    worker.start_gpu(FakePersonDetector(), FakePoseDetector())
    service = worker_env["manager"].get_service()
    monkeypatch.setattr(service.person_detector, "detect", lambda image: [])
    client = TestClient(worker.create_app(), raise_server_exceptions=False)

    response = post(client, jpeg())

    assert response.status_code == 422
    assert response.json()["code"] == "PERSON_NOT_FOUND"
    assert worker_env["s3"].puts == []


def test_missing_bucket_starts_without_a_runtime_and_answers_s3_config_error(worker_env, monkeypatch):
    monkeypatch.setattr(settings, "S3_BUCKET", "")

    assert worker.start_gpu(FakePersonDetector(), FakePoseDetector()) is None
    client = TestClient(worker.create_app(), raise_server_exceptions=False)
    response = post(client, jpeg())

    assert response.status_code == 500
    assert response.json()["code"] == "S3_CONFIG_ERROR"
    assert worker_env["s3"].puts == []
