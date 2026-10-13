"""deploy/modal/body_image_worker.py 의 Modal 정의. Modal SDK 가 있을 때만 돈다(원격 호출 없음).

Modal 객체 정의는 import 시점에 검증되므로(asgi_app 은 인자 없는 sync method 여야 하는 등) 모듈을
불러오는 것만으로 구성 오류를 잡는다. 자원·환경변수·CUDA 버전은 모듈 상수로 확인한다.
"""

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("modal")
WORKER_FILE = Path(__file__).resolve().parents[1] / "deploy/modal/body_image_worker.py"
if not WORKER_FILE.is_file():
    pytest.skip("deploy/ 가 없는 환경(Docker test stage)", allow_module_level=True)


_NAME_VARS = (
    "BODY_IMAGE_WORKER_STAGE",
    "BODY_IMAGE_WORKER_APP_NAME",
    "BODY_IMAGE_WORKER_LABEL",
    "BODY_IMAGE_WORKER_SECRET_NAME",
)


def load(**env):
    """환경변수를 지정한 채 배포 파일을 새로 불러온다. Modal 에 아무것도 보내지 않는다."""
    saved = {name: os.environ.pop(name, None) for name in _NAME_VARS}
    os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location("body_image_worker", WORKER_FILE)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        for name in _NAME_VARS:
            os.environ.pop(name, None)
            if saved[name] is not None:
                os.environ[name] = saved[name]


@pytest.fixture(scope="module")
def module():
    return load()


def test_resources_match_the_t4_benchmark(module):
    assert module.WORKER_RESOURCES == {
        "gpu": "T4",
        "cpu": 4,
        "memory": 8192,
        "scaledown_window": 30,
        "min_containers": 0,
        "enable_memory_snapshot": True,
    }
    # GPU memory snapshot 은 experimental_options 로만 켜진다. 넘기지 않는다.
    assert "experimental_options" not in module.WORKER_RESOURCES


def test_cold_start_and_gpu_settings_are_in_the_image_env(module):
    env = module.WORKER_ENV
    assert env["NUMBA_CPU_NAME"] == "generic"
    assert env["NUMBA_CACHE_DIR"] == "/opt/numba_cache"
    assert env["BODY_IMAGE_GPU_PREPROCESSING"] == "true"
    assert env["BODY_IMAGE_REMBG_PROVIDERS"] == "CUDAExecutionProvider,CPUExecutionProvider"
    assert env["REMBG_HOME"] == env["BODY_IMAGE_MODEL_DIR"] == module.MODEL_DIR


def test_cuda_stack_is_the_benchmarked_one(module):
    assert module.ONNXRUNTIME_GPU == "onnxruntime-gpu[cuda,cudnn]==1.30.0"
    assert module.TORCH == ("torch==2.14.1", "torchvision==0.29.1")
    assert module.TORCH_INDEX_URL.endswith("/cu130")


def test_proxy_auth_is_on_by_default(module):
    assert module.REQUIRES_PROXY_AUTH is True


def test_worker_class_and_app_are_defined(module):
    import modal

    assert isinstance(module.app, modal.App)
    assert isinstance(module.BodyImageWorker, modal.Cls)
    assert module.app.name == module.APP_NAME


def test_snapshot_hooks_keep_cuda_out_of_the_snapshot(module):
    source = WORKER_FILE.read_text()
    snap = source.split("@modal.enter(snap=True)")[1].split("@modal.enter(snap=False)")[0]
    assert "prepare_cpu" in snap and "start_gpu" not in snap
    assert "@modal.concurrent" not in source  # 컨테이너당 동시 요청 1(기본값)


def web_endpoint(loaded):
    from modal._partial_function import _find_partial_methods_for_user_cls, _PartialFunctionFlags

    [partial] = _find_partial_methods_for_user_cls(
        loaded.BodyImageWorker._get_user_cls(), _PartialFunctionFlags.WEB_INTERFACE
    ).values()
    return partial.params.webhook_config


def test_production_defaults_have_no_container_cap(module):
    assert module.STAGE == "production"
    assert (module.APP_NAME, module.ENDPOINT_LABEL, module.SECRET_NAME) == (
        "lookddak-body-image-worker", "lookddak-body-image", "lookddak-body-image-worker",
    )
    assert module.STAGE_SCALING == {}
    assert web_endpoint(module).requested_suffix == "lookddak-body-image"


def test_staging_caps_containers_and_uses_staging_names():
    staging = load(BODY_IMAGE_WORKER_STAGE="staging")

    assert staging.STAGE_SCALING == {"max_containers": 1}
    assert staging.app.name == "lookddak-body-image-worker-staging"
    assert staging.SECRET_NAME == "lookddak-body-image-worker-staging"
    assert web_endpoint(staging).requested_suffix == "lookddak-body-image-staging"
    assert web_endpoint(staging).requires_proxy_auth is True
    assert staging.WORKER_RESOURCES == load().WORKER_RESOURCES  # 나머지 자원은 production 과 같다


def test_name_variables_override_the_defaults():
    custom = load(
        BODY_IMAGE_WORKER_STAGE="staging",
        BODY_IMAGE_WORKER_APP_NAME="app-x",
        BODY_IMAGE_WORKER_LABEL="label-x",
        BODY_IMAGE_WORKER_SECRET_NAME="secret-x",
    )

    assert (custom.app.name, custom.SECRET_NAME) == ("app-x", "secret-x")
    assert web_endpoint(custom).requested_suffix == "label-x"


def test_empty_name_variables_fall_back_to_the_defaults():
    loaded = load(BODY_IMAGE_WORKER_APP_NAME="", BODY_IMAGE_WORKER_LABEL="", BODY_IMAGE_WORKER_SECRET_NAME="")

    assert (loaded.app.name, loaded.SECRET_NAME) == ("lookddak-body-image-worker", "lookddak-body-image-worker")


@pytest.mark.parametrize(
    "env",
    [
        {"BODY_IMAGE_WORKER_STAGE": "staging", "BODY_IMAGE_WORKER_APP_NAME": "lookddak-body-image-worker"},
        {"BODY_IMAGE_WORKER_STAGE": "staging", "BODY_IMAGE_WORKER_LABEL": "lookddak-body-image"},
        {"BODY_IMAGE_WORKER_STAGE": "stagin"},
    ],
    ids=["staging-with-production-app", "staging-with-production-label", "unknown-stage"],
)
def test_unsafe_or_unknown_stage_settings_stop_before_deploy(env):
    with pytest.raises(ValueError):
        load(**env)


def load_as_in_container():
    """Modal 컨테이너처럼 이 파일만 /root/body_image_worker.py 로 놓였을 때의 import 를 재현한다."""
    namespace = {"__name__": "body_image_worker", "__file__": "/root/body_image_worker.py"}
    code = compile(WORKER_FILE.read_text(), "/root/body_image_worker.py", "exec")
    saved = {name: os.environ.pop(name, None) for name in _NAME_VARS}
    try:
        exec(code, namespace)  # noqa: S102 - 배포 파일 자체를 컨테이너 경로로 실행한다
    finally:
        for name in _NAME_VARS:
            if saved[name] is not None:
                os.environ[name] = saved[name]
    return namespace


def test_import_from_the_container_path_does_not_depend_on_the_local_layout():
    import modal

    loaded = load_as_in_container()

    assert loaded["LOCAL_PROJECT_ROOT"] is None
    assert isinstance(loaded["app"], modal.App) and isinstance(loaded["BodyImageWorker"], modal.Cls)
    # 컨테이너 쪽 경로는 이미지 안의 절대 경로만 쓴다.
    assert loaded["WORKER_ENV"]["PYTHONPATH"] == "/root"
    assert loaded["MODEL_DIR"] == "/app/models/body_image_validation"


def test_local_deploy_uses_the_repository_root(module):
    repo = Path(__file__).resolve().parents[1]

    assert module.LOCAL_PROJECT_ROOT == repo
    for path in ("requirements.txt", "scripts/download_body_image_models.py", "app/body_image_validation/worker.py"):
        assert (module.LOCAL_PROJECT_ROOT / path).is_file()


def registered_options(monkeypatch, **env):
    """배포 파일을 불러올 때 Modal 에 실제로 넘어가는 함수 옵션(gpu·컨테이너 수 등)을 가로챈다."""
    import modal._functions as functions

    seen = {}
    original = functions._Function._from_local

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(functions._Function, "_from_local", staticmethod(spy))
    load(**env)
    return seen


def test_production_keeps_a_single_t4(monkeypatch):
    options = registered_options(monkeypatch)

    assert options["gpu"] == "T4"
    assert options["max_containers"] is None
    assert (options["min_containers"], options["scaledown_window"]) == (0, 30)
    assert options["enable_memory_snapshot"] is True


def test_staging_prefers_t4_and_falls_back_to_l4(monkeypatch):
    options = registered_options(monkeypatch, BODY_IMAGE_WORKER_STAGE="staging")

    assert options["gpu"] == ["T4", "L4"]  # 순서가 우선순위다
    assert options["max_containers"] == 1
    assert (options["min_containers"], options["scaledown_window"]) == (0, 30)
    assert (options["cpu"], options["memory"]) == (4, 8192)
    assert options["enable_memory_snapshot"] is True
