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


PREFIX = "BODY_IMAGE_WORKER_"


def _with_env(env, action):
    """셸의 BODY_IMAGE_WORKER_* 를 모두 치우고 env 만 둔 채 action 을 실행한다."""
    saved = {name: os.environ.pop(name) for name in list(os.environ) if name.startswith(PREFIX)}
    os.environ.update(env)
    try:
        return action()
    finally:
        for name in [name for name in os.environ if name.startswith(PREFIX)]:
            del os.environ[name]
        os.environ.update(saved)


def load(**env):
    """배포 시점처럼(저장소 경로에서) 환경변수를 지정한 채 배포 파일을 새로 불러온다. Modal 에 아무것도 보내지 않는다."""
    def action():
        spec = importlib.util.spec_from_file_location("body_image_worker", WORKER_FILE)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    return _with_env(env, action)


def load_production(**env):
    return load(BODY_IMAGE_WORKER_STAGE="production", **env)


@pytest.fixture(scope="module")
def module():
    return load_production()


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


STAGE_VARS = (
    "BODY_IMAGE_WORKER_STAGE",  # 함수 env
    "BODY_IMAGE_WORKER_IMAGE_STAGE",  # 이미지 env
    "BODY_IMAGE_WORKER_GPU_SNAPSHOT",  # 함수 env
    "BODY_IMAGE_WORKER_IMAGE_GPU_SNAPSHOT",  # 이미지 env
)


def set_container_stages(monkeypatch, function_stage, image_stage, function_gpu=None, image_gpu=None):
    """컨테이너가 받는 네 값: 함수 env(@app.cls(env=...))·이미지 env(Image.env)의 stage 와 GPU snapshot. None 이면 없음."""
    for name, value in zip(STAGE_VARS, (function_stage, image_stage, function_gpu, image_gpu), strict=True):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def as_staging_container(monkeypatch, loaded):
    set_container_stages(monkeypatch, "staging", "staging", "true", "true")
    return loaded


def as_production_gpu_container(monkeypatch, loaded):
    set_container_stages(monkeypatch, "production", "production", "true", "true")
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
    ("function_stage", "image_stage", "function_gpu", "image_gpu", "expected"),
    [
        ("staging", "staging", "true", "true", True),  # staging 배포
        (None, None, None, None, False),  # production 기본(CPU snapshot): 네 값 모두 없음(기존 그대로)
        ("production", None, None, None, False),
        ("production", "production", None, None, False),
        ("production", "production", "true", "true", True),  # production GPU snapshot(명시적 활성화)
    ],
    ids=["staging", "production-cpu-default", "production-cpu-function-stage", "production-cpu-both-stages",
         "production-gpu"],
)
def test_matching_stages_pick_the_lifecycle(
    module, monkeypatch, function_stage, image_stage, function_gpu, image_gpu, expected
):
    set_container_stages(monkeypatch, function_stage, image_stage, function_gpu, image_gpu)

    assert module.resolve_gpu_snapshot() is expected


@pytest.mark.parametrize(
    ("function_stage", "image_stage", "function_gpu", "image_gpu"),
    [
        (None, "staging", None, "true"),  # staging 이미지인데 함수 env 가 전달되지 않음
        ("production", "staging", "true", "true"),
        ("staging", None, "true", None),  # 함수 env 는 staging 인데 이미지가 staging 이 아님
        ("staging", "production", "true", "true"),
        ("stagin", "stagin", "true", "true"),  # 알 수 없는 값
        ("staging", "staging", None, None),  # staging 인데 GPU snapshot 설정이 없음
        # production GPU snapshot: 한쪽 경로의 GPU snapshot 값이 빠짐 → CPU 경로로 조용히 가지 않는다
        ("production", "production", None, "true"),
        ("production", "production", "true", None),
        (None, None, "true", None),
    ],
    ids=["function-env-missing", "function-env-production", "image-env-missing", "image-env-production",
         "unknown-stage", "staging-without-gpu-flags", "production-gpu-function-flag-missing",
         "production-gpu-image-flag-missing", "production-gpu-only-function-flag"],
)
def test_mismatched_or_unknown_stages_fail_fast(module, monkeypatch, function_stage, image_stage, function_gpu,
                                                image_gpu):
    set_container_stages(monkeypatch, function_stage, image_stage, function_gpu, image_gpu)

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
    load_production()
    production_envs = list(baked)
    baked.clear()
    load(BODY_IMAGE_WORKER_STAGE="staging")

    assert {"BODY_IMAGE_WORKER_IMAGE_STAGE": "staging", "BODY_IMAGE_WORKER_IMAGE_GPU_SNAPSHOT": "true"} in baked
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
    assert staging.WORKER_RESOURCES == load_production().WORKER_RESOURCES  # 나머지 자원은 production 과 같다


@pytest.mark.parametrize("name", ["APP_NAME", "LABEL", "SECRET_NAME", "PROXY_AUTH"])
@pytest.mark.parametrize("stage", ["staging", "production"])
@pytest.mark.parametrize("value", ["lookddak-body-image-worker", "lookddak-body-image-worker-staging", "false", ""])
def test_name_and_proxy_overrides_are_rejected_before_deploy(name, stage, value):
    # 이름은 stage 로만 정해진다. 상대 stage 이름·임의 값·빈 값·proxy 인증 끄기 모두 배포 전에 거절한다.
    with pytest.raises(ValueError, match="지원하지 않는 배포 환경변수"):
        load(BODY_IMAGE_WORKER_STAGE=stage, **{PREFIX + name: value})


@pytest.mark.parametrize("name", ["BODY_IMAGE_WORKER_GPU_SNAPHOT", "BODY_IMAGE_WORKER_IMAGE_STAGE",
                                  "BODY_IMAGE_WORKER_IMAGE_GPU_SNAPSHOT"])
def test_typos_and_container_only_variables_are_rejected_before_deploy(name):
    with pytest.raises(ValueError, match="지원하지 않는 배포 환경변수"):
        load(BODY_IMAGE_WORKER_STAGE="staging", **{name: "true"})


@pytest.mark.parametrize("env", [{}, {"BODY_IMAGE_WORKER_STAGE": ""}, {"BODY_IMAGE_WORKER_STAGE": "  "}],
                         ids=["missing", "empty", "blank"])
def test_deploy_without_a_stage_stops_instead_of_choosing_production(env):
    with pytest.raises(ValueError, match="BODY_IMAGE_WORKER_STAGE"):
        load(**env)


@pytest.mark.parametrize("stage", ["stagin", "prod", "dev"])
def test_unknown_stages_stop_before_deploy(stage):
    with pytest.raises(ValueError, match="production 또는 staging"):
        load(BODY_IMAGE_WORKER_STAGE=stage)


def test_modal_profile_does_not_choose_the_stage(monkeypatch):
    # MODAL_PROFILE 은 Workspace 만 고른다. 팀 profile 이어도 stage 를 대신하지 않는다.
    monkeypatch.setenv("MODAL_PROFILE", "lookddak")
    with pytest.raises(ValueError, match="BODY_IMAGE_WORKER_STAGE"):
        load()
    assert load(BODY_IMAGE_WORKER_STAGE="staging").APP_NAME == "lookddak-body-image-worker-staging"


def test_staging_and_production_share_no_names(module):
    production, staging = module.DEPLOY_TARGETS["production"], module.DEPLOY_TARGETS["staging"]
    assert set(production.values()).isdisjoint(staging.values())
    for stage, target in module.DEPLOY_TARGETS.items():
        loaded = module if stage == "production" else load(BODY_IMAGE_WORKER_STAGE=stage)
        assert (loaded.APP_NAME, loaded.ENDPOINT_LABEL, loaded.SECRET_NAME) == (
            target["app_name"], target["label"], target["secret_name"])


@pytest.mark.parametrize(
    ("stage", "flag", "expected"),
    [("staging", None, True), ("staging", "true", True), ("production", None, False), ("production", "false", False),
     ("production", "true", True)],
)
def test_snapshot_mode_table(module, stage, flag, expected):
    env = {"BODY_IMAGE_WORKER_STAGE": stage, "BODY_IMAGE_WORKER_MAX_CONTAINERS": "2" if stage == "production" else ""}
    if flag is not None:
        env["BODY_IMAGE_WORKER_GPU_SNAPSHOT"] = flag
    config = module.resolve_deploy_config(env, in_container=False)

    assert config.gpu_snapshot is expected
    assert config.max_containers == (1 if stage == "staging" else 2)


def test_deploy_config_is_resolved_once_and_exposed(module):
    assert module.CONFIG == module.resolve_deploy_config({"BODY_IMAGE_WORKER_STAGE": "production"}, in_container=False)
    assert (module.STAGE, module.APP_NAME, module.SECRET_NAME, module.GPU_SNAPSHOT, module.MAX_CONTAINERS) == (
        module.CONFIG.stage, module.CONFIG.app_name, module.CONFIG.secret_name, module.CONFIG.gpu_snapshot,
        module.CONFIG.max_containers)


def test_invalid_diagnostics_value_stops_before_deploy():
    with pytest.raises(ValueError, match="true 또는 false"):
        load(BODY_IMAGE_WORKER_STAGE="staging", BODY_IMAGE_WORKER_DIAGNOSTICS="yes")


def test_container_config_does_not_require_deploy_only_settings(module):
    # 컨테이너: stage 가 없으면(CPU snapshot production 은 함수 env 를 등록하지 않는다) production. 상한·진단은 보지 않는다.
    assert module.resolve_deploy_config({}, in_container=True).stage == "production"
    config = module.resolve_deploy_config(
        {"BODY_IMAGE_WORKER_STAGE": "production", "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "true",
         "BODY_IMAGE_WORKER_IMAGE_STAGE": "production"}, in_container=True)
    assert config.gpu_snapshot is True and config.max_containers is None


def load_as_in_container(**env):
    """Modal 컨테이너처럼 이 파일만 /root/body_image_worker.py 로 놓였을 때의 import 를 재현한다.

    env 는 컨테이너 환경변수(Modal 이 env= 로 넣는 값)다. 로컬 배포 환경변수는 컨테이너에 없다.
    """
    namespace = {"__name__": "body_image_worker", "__file__": "/root/body_image_worker.py"}
    code = compile(WORKER_FILE.read_text(), "/root/body_image_worker.py", "exec")
    _with_env(env, lambda: exec(code, namespace))  # noqa: S102 - 배포 파일 자체를 컨테이너 경로로 실행한다
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
    loaded = load_as_in_container(**staging.STAGE_ENV["env"], BODY_IMAGE_WORKER_IMAGE_STAGE="staging",
                                  BODY_IMAGE_WORKER_IMAGE_GPU_SNAPSHOT="true")

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
    load(**{"BODY_IMAGE_WORKER_STAGE": "production", **env})  # STAGE 를 주지 않으면 production 으로 불러온다
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
    production = load_production()
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
        "env": {"BODY_IMAGE_WORKER_STAGE": "staging", "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "true"}
    }


# --- 진단(staging 검증용) ---


def test_diagnostics_are_off_unless_requested():
    assert load(BODY_IMAGE_WORKER_STAGE="staging").STAGE_ENV == {
        "env": {"BODY_IMAGE_WORKER_STAGE": "staging", "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "true"}
    }


def test_staging_diagnostics_pass_profiling_settings_to_the_container():
    staging = load(BODY_IMAGE_WORKER_STAGE="staging", BODY_IMAGE_WORKER_DIAGNOSTICS="true")

    assert staging.STAGE_ENV == {"env": {
        "BODY_IMAGE_WORKER_STAGE": "staging",
        "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "true",
        "BODY_IMAGE_DIAGNOSTICS": "true",
        "BODY_IMAGE_REMBG_PROFILE_RUNS": "2",  # snapshot 전 warm-up 1번 + 복원 뒤 첫 요청 1번
    }}


def test_production_refuses_diagnostics():
    with pytest.raises(ValueError, match="staging 배포에서만"):
        load_production(BODY_IMAGE_WORKER_DIAGNOSTICS="true")


# --- production GPU snapshot (배포 시점에 BODY_IMAGE_WORKER_GPU_SNAPSHOT=true 를 명시했을 때만) ---

# 테스트용 상한(2). 운영값은 production AI Server task 수 기준으로 정한다.
PRODUCTION_GPU = {"BODY_IMAGE_WORKER_STAGE": "production", "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "true",
                  "BODY_IMAGE_WORKER_MAX_CONTAINERS": "2"}


def test_production_gpu_snapshot_is_off_unless_explicitly_enabled():
    for flag in (None, "", "false"):
        env = {} if flag is None else {"BODY_IMAGE_WORKER_GPU_SNAPSHOT": flag}
        production = load_production(**env)
        assert production.GPU_SNAPSHOT is False
        assert production.STAGE_SNAPSHOT == {} and production.STAGE_ENV == {}


def test_production_gpu_snapshot_registers_the_staging_style_configuration(monkeypatch):
    import modal

    baked = []
    original = modal.Image.env
    monkeypatch.setattr(modal.Image, "env", lambda self, vars: baked.append(dict(vars)) or original(self, vars))
    options = registered_options(monkeypatch, **PRODUCTION_GPU)
    production = load(**PRODUCTION_GPU)

    assert options["experimental_options"] == {"enable_gpu_snapshot": "True"}
    assert options["gpu"] == "T4" and (options["cpu"], options["memory"]) == (4, 8192)
    assert options["max_containers"] == 2 and options["min_containers"] == 0  # 배포 때 준 상한
    assert production.APP_NAME == "lookddak-body-image-worker"
    assert production.SECRET_NAME == "lookddak-body-image-worker"
    assert production.STAGE_ENV == {"env": {"BODY_IMAGE_WORKER_STAGE": "production",
                                             "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "true"}}
    assert {"BODY_IMAGE_WORKER_IMAGE_STAGE": "production", "BODY_IMAGE_WORKER_IMAGE_GPU_SNAPSHOT": "true"} in baked


def test_production_gpu_snapshot_uses_the_staging_lifecycle(fake_worker, monkeypatch):
    calls = fake_worker["calls"]
    production = as_production_gpu_container(monkeypatch, load(**PRODUCTION_GPU))
    _, pre, post, exit_hooks = lifecycle(production)

    run(pre)
    assert calls == [("prepare_gpu_snapshot", (), {"require_cuda": True})]  # CUDA·rembg·nvJPEG·warm-up 까지
    run(post)
    # 복원 뒤에는 secret·S3·limiter 로 runtime 만 만든다(GPU 모델을 다시 만들지 않는다).
    assert calls[1:] == [("start_restored_runtime", (fake_worker["models"], fake_worker["warmup"]))]
    run(exit_hooks)
    assert calls[2:] == [("close", ())]


def test_production_gpu_snapshot_model_failure_fails_fast(fake_worker, monkeypatch):
    from app.body_image_validation import worker

    monkeypatch.setattr(worker, "prepare_gpu_snapshot", lambda require_cuda: (None, None))
    production = as_production_gpu_container(monkeypatch, load(**PRODUCTION_GPU))
    _, pre, _, _ = lifecycle(production)

    with pytest.raises(production.SnapshotStateError):
        run(pre)


def test_production_gpu_deploy_whose_container_lost_the_flag_fails_fast(fake_worker, monkeypatch):
    # GPU snapshot 으로 배포했는데 함수 env 가 빠지면 CPU 경로(prepare_cpu)로 가지 않고 멈춘다.
    production = load(**PRODUCTION_GPU)
    set_container_stages(monkeypatch, None, "production", None, "true")
    _, pre, _, _ = lifecycle(production)

    with pytest.raises(production.StageConfigError):
        run(pre)
    assert fake_worker["calls"] == []


def test_production_cpu_default_keeps_the_existing_lifecycle(module, fake_worker, monkeypatch):
    set_container_stages(monkeypatch, None, None, None, None)
    calls = fake_worker["calls"]
    _, pre, post, _ = lifecycle(module)

    run(pre)
    run(post)
    assert calls == [("prepare_cpu", ()), ("start_gpu", fake_worker["detectors"])]


@pytest.mark.parametrize(("env", "message"), [
    ({"BODY_IMAGE_WORKER_STAGE": "production", "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "yes"}, "true 또는 false"),
    ({"BODY_IMAGE_WORKER_STAGE": "staging", "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "false"}, "끌 수 없습니다"),
], ids=["invalid-flag", "staging-off"])
def test_invalid_gpu_snapshot_flags_stop_before_deploy(env, message):
    with pytest.raises(ValueError, match=message):
        load(**env)


# --- max_containers 안전장치 ---


def test_production_gpu_snapshot_without_a_container_cap_stops_before_deploy():
    with pytest.raises(ValueError, match="BODY_IMAGE_WORKER_MAX_CONTAINERS 가 필요"):
        load_production(BODY_IMAGE_WORKER_GPU_SNAPSHOT="true")


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5", "21"])
def test_invalid_container_caps_stop_before_deploy(value):
    with pytest.raises(ValueError, match="1~20 정수"):
        load_production(BODY_IMAGE_WORKER_GPU_SNAPSHOT="true", BODY_IMAGE_WORKER_MAX_CONTAINERS=value)


def test_production_cpu_keeps_no_cap_by_default_and_accepts_an_explicit_one():
    assert load_production().STAGE_SCALING == {}  # 기존 그대로(상한 없음)
    assert load_production(BODY_IMAGE_WORKER_MAX_CONTAINERS="3").STAGE_SCALING == {"max_containers": 3}


def test_staging_cap_is_fixed_to_one():
    assert load(BODY_IMAGE_WORKER_STAGE="staging").STAGE_SCALING == {"max_containers": 1}
    assert load(BODY_IMAGE_WORKER_STAGE="staging", BODY_IMAGE_WORKER_MAX_CONTAINERS="1").STAGE_SCALING == {
        "max_containers": 1
    }
    with pytest.raises(ValueError, match="1 로 고정"):
        load(BODY_IMAGE_WORKER_STAGE="staging", BODY_IMAGE_WORKER_MAX_CONTAINERS="2")


def test_production_gpu_container_import_does_not_require_the_cap():
    # 컨테이너에는 로컬 배포 환경변수(MAX_CONTAINERS)가 없다. 함수 env 만으로 import 가 되어야 한다(상한은 이미 등록됨).
    production = load(**PRODUCTION_GPU)
    loaded = load_as_in_container(**production.STAGE_ENV["env"])

    assert loaded["GPU_SNAPSHOT"] is True and loaded["LOCAL_PROJECT_ROOT"] is None
