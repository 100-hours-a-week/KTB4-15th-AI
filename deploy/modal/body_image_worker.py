import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import modal

# 경로가 두 가지다.
# - 로컬(modal deploy 를 실행하는 쪽): 이 파일은 <저장소>/deploy/modal/body_image_worker.py 다.
#   이미지에 올릴 requirements.txt·scripts/·app/ 을 저장소 루트 기준으로 찾는다.
# - Modal 컨테이너: 이 파일만 /root/body_image_worker.py 로 올라온다. app 패키지는 이미지의
#   /root/app(PYTHONPATH=/root), 모델은 /app/models/body_image_validation 에 있다. 이미지는 이미
#   빌드돼 있어 로컬 경로를 읽지 않으므로, 저장소 구조를 가정해 경로를 계산하면 안 된다.
# 배포 설정 검증도 이 구분을 쓴다(LOCAL_PROJECT_ROOT 가 None 이면 컨테이너, 아래 resolve_deploy_config).
_THIS_FILE = Path(__file__).resolve()
_IN_REPOSITORY = _THIS_FILE.parent.name == "modal" and _THIS_FILE.parent.parent.name == "deploy"
LOCAL_PROJECT_ROOT = _THIS_FILE.parents[2] if _IN_REPOSITORY else None
# 컨테이너에서는 아래 이미지 정의의 로컬 경로를 쓰지 않는다. 계산이 실패하지 않도록 자리값만 둔다.
ROOT = LOCAL_PROJECT_ROOT or _THIS_FILE.parent
MODEL_DIR = "/app/models/body_image_validation"
NUMBA_CACHE_DIR = "/opt/numba_cache"

# --- 배포 설정 ---------------------------------------------------------------------------------------
# 배포 대상별 이름. App·URL label·Modal Secret 은 stage 로만 정한다(이름을 바꾸는 환경변수는 두지 않는다).
DEPLOY_TARGETS = {
    "production": {
        "app_name": "lookddak-body-image-worker",
        "label": "lookddak-body-image",
        "secret_name": "lookddak-body-image-worker",
    },
    "staging": {
        "app_name": "lookddak-body-image-worker-staging",
        "label": "lookddak-body-image-staging",
        "secret_name": "lookddak-body-image-worker-staging",
    },
}
STAGE_VAR = "BODY_IMAGE_WORKER_STAGE"
GPU_SNAPSHOT_VAR = "BODY_IMAGE_WORKER_GPU_SNAPSHOT"
MAX_CONTAINERS_VAR = "BODY_IMAGE_WORKER_MAX_CONTAINERS"
DIAGNOSTICS_VAR = "BODY_IMAGE_WORKER_DIAGNOSTICS"
DEPLOY_VARS = (STAGE_VAR, GPU_SNAPSHOT_VAR, MAX_CONTAINERS_VAR, DIAGNOSTICS_VAR)
MAX_CONTAINERS_SANITY_LIMIT = 20
IMAGE_STAGE_VAR = "BODY_IMAGE_WORKER_IMAGE_STAGE"
IMAGE_GPU_SNAPSHOT_VAR = "BODY_IMAGE_WORKER_IMAGE_GPU_SNAPSHOT"


@dataclass(frozen=True)
class DeployConfig:
    """한 번 정한 배포 설정. 이 파일과 deploy_checks.py(preflight·staging 배포)가 같은 함수로 만든다."""

    stage: str
    app_name: str
    label: str
    secret_name: str
    gpu_snapshot: bool
    max_containers: int | None
    diagnostics: bool


def _flag(environ: Mapping[str, str], name: str) -> str:
    value = (environ.get(name) or "").strip().lower()
    if value not in ("", "true", "false"):
        raise ValueError(f"{name} 는 true 또는 false 여야 합니다: {value!r}")
    return value


def resolve_deploy_config(environ: Mapping[str, str], *, in_container: bool) -> DeployConfig:
    if in_container:
        stage = (environ.get(STAGE_VAR) or "production").strip().lower()
    else:
        unknown = sorted(name for name in environ if name.startswith("BODY_IMAGE_WORKER_") and name not in DEPLOY_VARS)
        if unknown:
            raise ValueError(
                f"지원하지 않는 배포 환경변수입니다: {unknown}. App·label·Secret 은 {STAGE_VAR} 로 정해지고 "
                "proxy 인증은 끌 수 없습니다. 셸에 남은 값이면 unset 한 뒤 다시 실행하세요."
            )
        stage = (environ.get(STAGE_VAR) or "").strip().lower()
        if not stage:
            raise ValueError(f"{STAGE_VAR}(staging|production)를 지정해야 배포할 수 있습니다. 기본값은 없습니다.")
    if stage not in DEPLOY_TARGETS:
        raise ValueError(f"{STAGE_VAR} 는 production 또는 staging 이어야 합니다: {stage!r}")

    gpu_flag = _flag(environ, GPU_SNAPSHOT_VAR)
    if stage == "staging" and gpu_flag == "false":
        raise ValueError("staging 은 GPU snapshot 을 끌 수 없습니다(검증 환경).")
    gpu_snapshot = stage == "staging" or gpu_flag == "true"

    max_containers = None
    diagnostics = False
    if not in_container:
        raw = (environ.get(MAX_CONTAINERS_VAR) or "").strip()
        if raw and not (raw.isdigit() and 1 <= int(raw) <= MAX_CONTAINERS_SANITY_LIMIT):
            raise ValueError(f"{MAX_CONTAINERS_VAR} 는 1~{MAX_CONTAINERS_SANITY_LIMIT} 정수여야 합니다: {raw!r}")
        max_containers = int(raw) if raw else None
        if stage == "staging" and max_containers not in (None, 1):
            raise ValueError("staging 의 max_containers 는 1 로 고정입니다.")
        if stage == "production" and gpu_snapshot and max_containers is None:
            raise ValueError(
                f"production GPU snapshot 배포에는 {MAX_CONTAINERS_VAR} 가 필요합니다(production AI Server task 수 기준)."
            )
        diagnostics = _flag(environ, DIAGNOSTICS_VAR) == "true"
        if diagnostics and stage != "staging":
            raise ValueError(f"{DIAGNOSTICS_VAR} 는 staging 배포에서만 켤 수 있습니다.")
    if stage == "staging":
        max_containers = 1  # 비용·테스트 통제
    return DeployConfig(stage=stage, **DEPLOY_TARGETS[stage], gpu_snapshot=gpu_snapshot,
                        max_containers=max_containers, diagnostics=diagnostics)


CONFIG = resolve_deploy_config(os.environ, in_container=LOCAL_PROJECT_ROOT is None)
if LOCAL_PROJECT_ROOT is not None:
    # 배포하는 사람이 Modal 로 보내기 전에 대상을 볼 수 있게 남긴다(비밀값 없음).
    print(f"body image worker deploy config: {CONFIG}", flush=True)

STAGE = CONFIG.stage
APP_NAME = CONFIG.app_name
ENDPOINT_LABEL = CONFIG.label
SECRET_NAME = CONFIG.secret_name
REQUIRES_PROXY_AUTH = True

# T4 benchmark 로 검증한 CUDA 조합. requirements.txt(EC2 CPU 이미지)와 별도로 여기서만 설치한다.
ONNXRUNTIME_GPU = "onnxruntime-gpu[cuda,cudnn]==1.30.0"
TORCH = ("torch==2.14.1", "torchvision==0.29.1")
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"

# 단일 요청 peak: CPU 2.92 core, RAM 6.46GiB, VRAM 1.37GiB. 컨테이너당 동시 요청은 기본값 1 이다
# (동시 요청 데코레이터를 쓰지 않는다). 공통 자원은 CPU memory snapshot 까지다. GPU snapshot 은 아래
# GPU_SNAPSHOT 일 때만 STAGE_SNAPSHOT 으로 켠다.
WORKER_RESOURCES = {
    "gpu": "T4",
    "cpu": 4,
    "memory": 8192,
    "scaledown_window": 30,
    "min_containers": 0,
    "enable_memory_snapshot": True,
}

# 컨테이너 수 상한(max_containers). 배포 시점에 Modal 에 등록되는 값이다(실행 중 변경은 Modal 대시보드·
# update_autoscaler 같은 별도 조작이 필요하고, 다음 배포 때 이 값으로 돌아간다).
MAX_CONTAINERS = CONFIG.max_containers
STAGE_SCALING = {"max_containers": MAX_CONTAINERS} if MAX_CONTAINERS else {}
# staging 도 T4 로 고정한다. GPU snapshot 을 production·benchmark 와 같은 GPU 조건에서 검증하기 위해서다.
# (L4 fallback 과 T4↔L4 snapshot 호환성은 별도로 검증한 뒤에 다시 켠다.)
WORKER_GPU = WORKER_RESOURCES["gpu"]
# GPU memory snapshot(experimental). 켜면 snap=True 단계에서 GPU 를 쓸 수 있어 CUDA·rembg CUDA session·nvJPEG·
# warm-up 까지 snapshot 에 담는다. 배포 시점에 Modal 에 등록되는 값이라 실행 중에는 바꿀 수 없다.
GPU_SNAPSHOT = CONFIG.gpu_snapshot
STAGE_SNAPSHOT = {"experimental_options": {"enable_gpu_snapshot": True}} if GPU_SNAPSHOT else {}
# 컨테이너는 이 파일을 다시 import 해 lifecycle 을 고른다. 로컬 환경변수는 컨테이너에 없으므로 GPU snapshot 배포는
# stage 와 GPU snapshot 여부를 컨테이너 환경변수로 넘긴다(CPU snapshot production 은 넘기지 않는다).
# staging 검증용 진단(요청별 단계 시간 로그 + rembg ONNX Runtime profiling). 배포할 때
# BODY_IMAGE_WORKER_DIAGNOSTICS=true 로만 켠다. profiling 은 snapshot 전 warm-up(1번째)과 복원 뒤 첫
# 요청(2번째)까지만 기록하고 끝낸다. production 에서는 켤 수 없다.
DIAGNOSTICS = CONFIG.diagnostics
DIAGNOSTIC_ENV = {"BODY_IMAGE_DIAGNOSTICS": "true", "BODY_IMAGE_REMBG_PROFILE_RUNS": "2"} if DIAGNOSTICS else {}
STAGE_ENV = {"env": {STAGE_VAR: STAGE, GPU_SNAPSHOT_VAR: "true", **DIAGNOSTIC_ENV}} if GPU_SNAPSHOT else {}

WORKER_ENV = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": "/root",
    "BODY_IMAGE_MODEL_DIR": MODEL_DIR,
    "REMBG_HOME": MODEL_DIR,
    "MPLCONFIGDIR": "/tmp/matplotlib",
    # 빌드 머신과 실행 머신의 CPU 가 달라도 PyMatting Numba cache 가 맞도록 generic 으로 컴파일한다.
    "NUMBA_CACHE_DIR": NUMBA_CACHE_DIR,
    "NUMBA_CPU_NAME": "generic",
    # T4 worker 에서만 켠다. 전처리 경로 플래그이고 rembg CUDA 와는 별개다.
    "BODY_IMAGE_GPU_PREPROCESSING": "true",
    "BODY_IMAGE_REMBG_PROVIDERS": "CUDAExecutionProvider,CPUExecutionProvider",
}

_CHECK_NUMBA_CACHE = (
    "python -c \"import glob, sys; n = len(glob.glob('"
    + NUMBA_CACHE_DIR
    + "/**/*.nbi', recursive=True)); print('numba cache index files:', n); sys.exit(0 if n else 1)\""
)

image = (
    modal.Image.from_registry("python:3.12-slim-bookworm")  # Dockerfile 과 같은 base
    .apt_install("libegl1", "libgl1", "libgles2", "libglib2.0-0", "libxcb1")
    .add_local_file(ROOT / "requirements.txt", "/build/requirements.txt", copy=True)
    .run_commands(
        "pip install --no-cache-dir --require-hashes -r /build/requirements.txt",
        # rembg[cpu] 가 설치한 CPU onnxruntime 을 GPU 빌드로 바꾼다(같은 1.30.0).
        "pip uninstall -y onnxruntime",
        f"pip install --no-cache-dir '{ONNXRUNTIME_GPU}'",
        f"pip install --no-cache-dir {' '.join(TORCH)} --index-url {TORCH_INDEX_URL}",
    )
    .add_local_file(
        ROOT / "scripts/download_body_image_models.py", "/build/download_body_image_models.py", copy=True
    )
    .run_commands(f"python /build/download_body_image_models.py {MODEL_DIR}")
    .env(WORKER_ENV)
    # rembg import 가 PyMatting 함수를 컴파일하며 Numba cache 를 만든다. 비어 있으면 빌드를 멈춘다.
    .run_commands(f"mkdir -p {NUMBA_CACHE_DIR}", 'python -c "import rembg"', _CHECK_NUMBA_CACHE)
)
# GPU snapshot 이미지에만 stage 와 GPU snapshot 여부를 굽는다. 함수 env(STAGE_ENV)와 다른 경로로 전달되는 값이라
# 컨테이너에서 둘을 비교해 한쪽이 빠지거나 다르면 멈춘다(resolve_gpu_snapshot). CPU snapshot production 이미지는
# 바꾸지 않는다. env 도 build step 이라 코드 layer(add_local_dir, copy 없음) 앞에 둔다(뒤에 두면 modal deploy 가 거절한다).
if GPU_SNAPSHOT:
    image = image.env({IMAGE_STAGE_VAR: STAGE, IMAGE_GPU_SNAPSHOT_VAR: "true"})
# 코드는 마지막 layer 다. 코드만 바뀌면 의존성·모델·cache layer 를 다시 만들지 않는다.
image = image.add_local_dir(ROOT / "app", "/root/app", ignore=["**/__pycache__", "**/*.pyc"])

app = modal.App(APP_NAME)


class StageConfigError(RuntimeError):
    """배포 stage 가 컨테이너에 일관되게 전달되지 않았다. 다른 경로(production)로 바꾸지 않고 멈춘다."""


class SnapshotStateError(RuntimeError):
    """GPU snapshot 에 담을 모델(또는 복원된 모델)이 없다. runtime 없이 시작하지 않고 멈춘다."""


def _rounded(timings: dict[str, float] | None) -> dict[str, float] | None:
    return None if timings is None else {name: round(value, 1) for name, value in timings.items()}


def _stage_from(name: str) -> str:
    return (os.environ.get(name) or "production").strip().lower()


def container_stage() -> str:
    """함수 env(@app.cls(env=...), Modal 이 Secret 으로 넣는다)로 받은 stage."""
    return _stage_from(STAGE_VAR)


def image_stage() -> str:
    """이미지(Image.env)에 구운 stage. 이미지 layer 라 컨테이너 시작부터 있다."""
    return _stage_from(IMAGE_STAGE_VAR)


def _flag_from(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() == "true"


def resolve_gpu_snapshot() -> bool:
    stages = {"image": image_stage(), "function": container_stage()}
    unknown = {where: stage for where, stage in stages.items() if stage not in DEPLOY_TARGETS}
    if unknown:
        raise StageConfigError(f"알 수 없는 stage 입니다: {unknown}")
    if stages["image"] != stages["function"]:
        raise StageConfigError(f"이미지 stage 와 함수 stage 가 다릅니다: {stages}")
    flags = {"image": _flag_from(IMAGE_GPU_SNAPSHOT_VAR), "function": _flag_from(GPU_SNAPSHOT_VAR)}
    if flags["image"] != flags["function"]:
        raise StageConfigError(f"이미지와 함수의 GPU snapshot 설정이 다릅니다: {flags}")
    gpu_snapshot = flags["function"]
    if stages["function"] == "staging" and not gpu_snapshot:
        raise StageConfigError("staging 컨테이너인데 GPU snapshot 설정이 없습니다.")
    print(f"body image worker lifecycle: stage={stages['function']} gpu_snapshot={gpu_snapshot}", flush=True)
    return gpu_snapshot


def _cuda_device_name() -> str | None:
    """실제로 붙은 GPU 이름(torch 공개 API). torch·CUDA 가 없으면 None."""
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - 진단 로그용. 실패해도 시작을 막지 않는다(CUDA 검사는 require_cuda 가 한다)
        return None


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    **{**WORKER_RESOURCES, "gpu": WORKER_GPU},
    **STAGE_SCALING,
    **STAGE_SNAPSHOT,
    **STAGE_ENV,
)
class BodyImageWorker:
    @modal.enter(snap=True)
    def prepare_snapshot(self) -> None:
        from app.body_image_validation import worker

        started = time.time()
        self.detectors = None
        self.models = self.warmup_ms = None
        self.gpu_snapshot = resolve_gpu_snapshot()
        gpu = None
        if self.gpu_snapshot:
            self.models, self.warmup_ms = worker.prepare_gpu_snapshot(require_cuda=True)
            if self.models is None:
                # 모델 파일 누락 등. GPU snapshot 에 빈 상태를 담고 runtime 없이 시작하지 않는다.
                raise SnapshotStateError("GPU snapshot 단계에서 모델을 준비하지 못했습니다.")
            gpu = _cuda_device_name()
        else:
            self.detectors = worker.prepare_cpu()
        print(
            f"body image worker lifecycle: snap=True done start={started:.3f} end={time.time():.3f} "
            f"gpu_snapshot={self.gpu_snapshot} gpu={gpu} warmup_ms={_rounded(self.warmup_ms)}",
            flush=True,
        )

    @modal.enter(snap=False)
    def start_after_restore(self) -> None:
        from app.body_image_validation import worker

        started = time.time()
        if resolve_gpu_snapshot() != self.gpu_snapshot:
            # 복원된 컨테이너의 stage 가 snapshot 을 만들 때와 다르다. 다른 경로로 바꾸지 않고 멈춘다.
            raise StageConfigError(f"복원 뒤 stage({container_stage()!r})가 snapshot 구성과 다릅니다.")
        if self.gpu_snapshot:
            if self.models is None:
                raise SnapshotStateError("복원된 snapshot 에 GPU 모델이 없습니다.")
            runtime = worker.start_restored_runtime(self.models, self.warmup_ms)
            timings = self.warmup_ms if runtime is not None else None
        else:
            timings = worker.start_gpu(*self.detectors)
        print(
            f"body image worker lifecycle: snap=False done start={started:.3f} end={time.time():.3f} "
            f"runtime={'ready' if timings is not None else 'unavailable'} warmup_ms={_rounded(timings)}",
            flush=True,
        )

    @modal.exit()
    def close(self) -> None:
        from app.body_image_validation import worker

        worker.close()

    @modal.asgi_app(label=ENDPOINT_LABEL, requires_proxy_auth=REQUIRES_PROXY_AUTH)
    def web(self):
        from app.body_image_validation import worker

        return worker.create_app()
