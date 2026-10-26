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
    "BODY_IMAGE_WORKER_DIAGNOSTICS",
    "BODY_IMAGE_WORKER_IMAGE_STAGE",
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


def test_one_request_per_container(module):
    assert "@modal.concurrent" not in WORKER_FILE.read_text()  # 컨테이너당 동시 요청 1(기본값)


# --- lifecycle: production 은 CPU snapshot, staging 은 GPU snapshot ---


def lifecycle(loaded):
    """Modal 컨테이너가 쓰는 방식(_find_callables_for_obj)으로 사용자 클래스의 hook 을 찾는다."""
    from modal._partial_function import _find_callables_for_obj, _PartialFunctionFlags

    obj = loaded.BodyImageWorker._get_user_cls()()

    def hooks(flag):
        return list(_find_callables_for_obj(obj, flag).values())

    return (
        obj,
        hooks(_PartialFunctionFlags.ENTER_PRE_SNAPSHOT),
        hooks(_PartialFunctionFlags.ENTER_POST_SNAPSHOT),
        hooks(_PartialFunctionFlags.EXIT),
    )


@pytest.fixture
def fake_worker(monkeypatch):
    """hook 이 부르는 worker 함수를 기록용 fake 로 바꾼다(모델·CUDA·S3 없음)."""
    from app.body_image_validation import worker

    calls = []
    detectors, models, warmup = ("person", "pose"), object(), {"rembg": 1.0}

    def record(name, result=None):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs) if kwargs else (name, args))
            return result
        return fake

    monkeypatch.setattr(worker, "prepare_cpu", record("prepare_cpu", detectors))
    monkeypatch.setattr(worker, "prepare_gpu_snapshot", record("prepare_gpu_snapshot", (models, warmup)))
    monkeypatch.setattr(worker, "start_gpu", record("start_gpu"))
    monkeypatch.setattr(worker, "start_restored_runtime", record("start_restored_runtime"))
    monkeypatch.setattr(worker, "close", record("close"))
    return {"calls": calls, "detectors": detectors, "models": models, "warmup": warmup}


def run(hooks):
    for hook in hooks:
        hook()


def test_each_stage_has_one_hook_per_phase(module):
    for loaded in (module, load(BODY_IMAGE_WORKER_STAGE="staging")):
        _, pre, post, exit_hooks = lifecycle(loaded)
        assert (len(pre), len(post), len(exit_hooks)) == (1, 1, 1)


def test_production_initializes_only_cpu_in_the_snapshot_and_gpu_after_restore(module, fake_worker):
    calls = fake_worker["calls"]
    _, pre, post, exit_hooks = lifecycle(module)

    run(pre)
    assert calls == [("prepare_cpu", ())]  # CUDA·S3 를 건드리는 함수는 snapshot 단계에서 부르지 않는다
    run(post)
    assert calls[1:] == [("start_gpu", fake_worker["detectors"])]  # 기존 production 순서
    run(exit_hooks)
    assert calls[2:] == [("close", ())]


STAGE_VARS = ("BODY_IMAGE_WORKER_STAGE", "BODY_IMAGE_WORKER_IMAGE_STAGE")


def set_container_stages(monkeypatch, function_stage, image_stage):
    """컨테이너가 받는 두 값: 함수 env(@app.cls(env=...))와 이미지 env(Image.env). None 이면 없음."""
    for name, value in zip(STAGE_VARS, (function_stage, image_stage), strict=True):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def as_staging_container(monkeypatch, loaded):
    set_container_stages(monkeypatch, "staging", "staging")
    return loaded


def test_staging_initializes_gpu_in_the_snapshot_and_only_the_runtime_after_restore(fake_worker, monkeypatch):
    calls = fake_worker["calls"]
    staging = as_staging_container(monkeypatch, load(BODY_IMAGE_WORKER_STAGE="staging"))
    _, pre, post, exit_hooks = lifecycle(staging)

    run(pre)
    assert calls == [("prepare_gpu_snapshot", (), {"require_cuda": True})]
    run(post)
    # snapshot 에 담긴 모델로 runtime 만 시작한다. GPU 초기화(start_gpu)를 다시 하지 않는다.
    assert calls[1:] == [("start_restored_runtime", (fake_worker["models"], fake_worker["warmup"]))]
    run(exit_hooks)
    assert calls[2:] == [("close", ())]


def test_staging_snapshot_without_models_fails_fast(fake_worker, monkeypatch):
    from app.body_image_validation import worker

    monkeypatch.setattr(worker, "prepare_gpu_snapshot", lambda require_cuda: (None, None))
    staging = as_staging_container(monkeypatch, load(BODY_IMAGE_WORKER_STAGE="staging"))
    _, pre, _, _ = lifecycle(staging)

    with pytest.raises(staging.SnapshotStateError):  # runtime 없이 시작하지 않는다
        run(pre)


def test_staging_restore_without_models_fails_fast(fake_worker, monkeypatch):
    staging = as_staging_container(monkeypatch, load(BODY_IMAGE_WORKER_STAGE="staging"))
    obj, pre, post, _ = lifecycle(staging)
    run(pre)
    obj.models = None  # 복원된 상태에 모델이 없다

    with pytest.raises(staging.SnapshotStateError):
        run(post)
    assert [call[0] for call in fake_worker["calls"]] == ["prepare_gpu_snapshot"]


# --- stage 불일치는 멈춘다(production 경로로 조용히 바꾸지 않는다). Modal 내부 API 없이 두 공개 경로만 비교 ---


@pytest.mark.parametrize(
    ("function_stage", "image_stage", "expected"),
    [
        ("staging", "staging", True),  # staging 배포: 함수 env·이미지 env 모두 staging
        (None, None, False),  # production 배포: 둘 다 없음(기존 그대로)
        ("production", None, False),
        ("production", "production", False),
    ],
)
def test_matching_stages_pick_the_lifecycle(module, monkeypatch, function_stage, image_stage, expected):
    set_container_stages(monkeypatch, function_stage, image_stage)

    assert module.resolve_gpu_snapshot() is expected


@pytest.mark.parametrize(
    ("function_stage", "image_stage"),
    [
        (None, "staging"),  # staging 이미지인데 함수 env 가 전달되지 않음
        ("production", "staging"),
        ("staging", None),  # 함수 env 는 staging 인데 이미지가 staging 이 아님
        ("staging", "production"),
        ("stagin", "stagin"),  # 알 수 없는 값
    ],
    ids=["function-env-missing", "function-env-production", "image-env-missing", "image-env-production",
         "unknown-stage"],
)
def test_mismatched_or_unknown_stages_fail_fast(module, monkeypatch, function_stage, image_stage):
    set_container_stages(monkeypatch, function_stage, image_stage)

    with pytest.raises(module.StageConfigError):
        module.resolve_gpu_snapshot()


def test_snapshot_hook_stops_instead_of_falling_back_to_the_production_path(fake_worker, monkeypatch):
    staging = load(BODY_IMAGE_WORKER_STAGE="staging")
    set_container_stages(monkeypatch, None, "staging")  # 함수 env 가 컨테이너에 전달되지 않음
    _, pre, _, _ = lifecycle(staging)

    with pytest.raises(staging.StageConfigError):
        run(pre)
    assert fake_worker["calls"] == []  # prepare_cpu(production 경로)로 넘어가지 않았다


def test_restore_with_a_different_stage_fails_fast(fake_worker, monkeypatch):
    staging = as_staging_container(monkeypatch, load(BODY_IMAGE_WORKER_STAGE="staging"))
    _, pre, post, _ = lifecycle(staging)
    run(pre)
    monkeypatch.delenv("BODY_IMAGE_WORKER_STAGE")  # 복원된 컨테이너에서 stage 가 사라짐

    with pytest.raises(staging.StageConfigError):
        run(post)
    assert [call[0] for call in fake_worker["calls"]] == ["prepare_gpu_snapshot"]


def test_staging_bakes_its_stage_into_the_image_and_production_does_not(monkeypatch):
    import modal

    baked = []
    original = modal.Image.env

    def spy(self, vars):
        baked.append(dict(vars))
        return original(self, vars)

    monkeypatch.setattr(modal.Image, "env", spy)
    load()
    production_envs = list(baked)
    baked.clear()
    load(BODY_IMAGE_WORKER_STAGE="staging")

    assert {"BODY_IMAGE_WORKER_IMAGE_STAGE": "staging"} in baked
    assert all("BODY_IMAGE_WORKER_IMAGE_STAGE" not in env for env in production_envs)  # production 이미지는 그대로


@pytest.mark.parametrize("stage", ["production", "staging"])
def test_no_build_step_runs_after_the_local_code_layer(monkeypatch, stage):
    # modal deploy 는 copy 없이 add_local_* 한 뒤의 build step(env·run_commands 등)을 거절한다. 로컬 import 로는
    # 드러나지 않으므로 공개 Image 메서드의 호출 순서로 확인한다. 코드(add_local_dir)는 마지막이어야 한다.
    import modal

    calls = []
    for name in ("env", "run_commands", "pip_install", "apt_install", "add_local_dir"):
        original = getattr(modal.Image, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(modal.Image, name, spy)
    load(BODY_IMAGE_WORKER_STAGE=stage)

    assert calls[-1] == "add_local_dir" and calls.count("add_local_dir") == 1


def test_worker_does_not_use_modal_private_runtime_api():
    """배포 파일이 Modal 비공개 API(modal._*, _experimental_* 속성)에 런타임에 기대지 않는다."""
    import ast

    tree = ast.parse(WORKER_FILE.read_text())
    private_imports = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("modal._"))
        or (isinstance(node, ast.Import) and alias.name.startswith("modal._"))
    ]
    private_attributes = [
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr.startswith(("_experimental", "_singleton"))
    ]
    assert private_imports == [] and private_attributes == []


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


def load_as_in_container(**env):
    """Modal 컨테이너처럼 이 파일만 /root/body_image_worker.py 로 놓였을 때의 import 를 재현한다.

    env 는 컨테이너 환경변수(Modal 이 env= 로 넣는 값)다. 로컬 배포 환경변수는 컨테이너에 없다.
    """
    namespace = {"__name__": "body_image_worker", "__file__": "/root/body_image_worker.py"}
    code = compile(WORKER_FILE.read_text(), "/root/body_image_worker.py", "exec")
    saved = {name: os.environ.pop(name, None) for name in _NAME_VARS}
    os.environ.update(env)
    try:
        exec(code, namespace)  # noqa: S102 - 배포 파일 자체를 컨테이너 경로로 실행한다
    finally:
        for name in _NAME_VARS:
            os.environ.pop(name, None)
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
    assert loaded["GPU_SNAPSHOT"] is False  # stage 를 받지 않은 컨테이너는 production lifecycle


def test_staging_container_picks_the_gpu_snapshot_lifecycle_from_its_env(monkeypatch):
    # staging 배포는 env={"BODY_IMAGE_WORKER_STAGE": "staging"} 을 넘기므로 컨테이너도 같은 lifecycle 을 고른다.
    staging = load(BODY_IMAGE_WORKER_STAGE="staging")
    loaded = load_as_in_container(**staging.STAGE_ENV["env"], BODY_IMAGE_WORKER_IMAGE_STAGE="staging")

    assert loaded["GPU_SNAPSHOT"] is True


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


def test_staging_is_pinned_to_t4(monkeypatch):
    options = registered_options(monkeypatch, BODY_IMAGE_WORKER_STAGE="staging")

    assert options["gpu"] == "T4"  # GPU snapshot 검증은 production·benchmark 와 같은 T4 에서 한다(L4 fallback 없음)
    assert options["max_containers"] == 1
    assert (options["min_containers"], options["scaledown_window"]) == (0, 30)
    assert (options["cpu"], options["memory"]) == (4, 8192)
    assert options["enable_memory_snapshot"] is True


def test_production_keeps_gpu_snapshot_off(monkeypatch):
    options = registered_options(monkeypatch)

    assert not options["experimental_options"]  # CPU memory snapshot 만
    production = load()
    assert production.GPU_SNAPSHOT is False
    assert production.STAGE_SNAPSHOT == {} and production.STAGE_ENV == {}


def test_staging_enables_gpu_snapshot_and_passes_its_stage_to_the_container(monkeypatch):
    # registered_options 를 두 번 부르면 spy 가 겹치므로 production 값은 먼저 읽어 둔다.
    production_secrets = len(registered_options(monkeypatch)["secrets"])
    options = registered_options(monkeypatch, BODY_IMAGE_WORKER_STAGE="staging")

    # SDK 가 값을 문자열로 바꿔 등록한다.
    assert options["experimental_options"] == {"enable_gpu_snapshot": "True"}
    assert options["enable_memory_snapshot"] is True  # GPU snapshot 은 memory snapshot 위에 켠다
    # App.cls(env=...) 는 env 를 Secret 하나로 바꿔 넘긴다. production 보다 하나 많다.
    assert len(options["secrets"]) == production_secrets + 1
    assert load(BODY_IMAGE_WORKER_STAGE="staging").STAGE_ENV == {
        "env": {"BODY_IMAGE_WORKER_STAGE": "staging"}
    }


# --- 진단(staging 검증용) ---


def test_diagnostics_are_off_unless_requested():
    assert load(BODY_IMAGE_WORKER_STAGE="staging").STAGE_ENV == {"env": {"BODY_IMAGE_WORKER_STAGE": "staging"}}


def test_staging_diagnostics_pass_profiling_settings_to_the_container():
    staging = load(BODY_IMAGE_WORKER_STAGE="staging", BODY_IMAGE_WORKER_DIAGNOSTICS="true")

    assert staging.STAGE_ENV == {"env": {
        "BODY_IMAGE_WORKER_STAGE": "staging",
        "BODY_IMAGE_DIAGNOSTICS": "true",
        "BODY_IMAGE_REMBG_PROFILE_RUNS": "2",  # snapshot 전 warm-up 1번 + 복원 뒤 첫 요청 1번
    }}


def test_production_refuses_diagnostics():
    with pytest.raises(ValueError):
        load(BODY_IMAGE_WORKER_DIAGNOSTICS="true")
