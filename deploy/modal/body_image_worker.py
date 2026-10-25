"""Modal T4 전신 이미지 검증 worker.

Backend 가 직접 부르는 HTTP API(EC2 AI Server 와 같은 endpoint·응답)를 Modal T4 컨테이너에서 띄운다.
검증 로직은 app/ 의 router·service 를 그대로 쓰고, 이 파일은 이미지·자원·초기화 순서만 정한다.

  배포:  modal deploy deploy/modal/body_image_worker.py   (저장소 루트에서)
  문서:  deploy/modal/README.md

자원과 cold start 설정은 T4 benchmark 로 정한 값이다. 바꾸기 전에 benchmark 를 다시 돌린다.
"""

import os
import time
from pathlib import Path

import modal

# 경로가 두 가지다.
# - 로컬(modal deploy 를 실행하는 쪽): 이 파일은 <저장소>/deploy/modal/body_image_worker.py 다.
#   이미지에 올릴 requirements.txt·scripts/·app/ 을 저장소 루트 기준으로 찾는다.
# - Modal 컨테이너: 이 파일만 /root/body_image_worker.py 로 올라온다. app 패키지는 이미지의
#   /root/app(PYTHONPATH=/root), 모델은 /app/models/body_image_validation 에 있다. 이미지는 이미
#   빌드돼 있어 로컬 경로를 읽지 않으므로, 저장소 구조를 가정해 경로를 계산하면 안 된다.
_THIS_FILE = Path(__file__).resolve()
_IN_REPOSITORY = _THIS_FILE.parent.name == "modal" and _THIS_FILE.parent.parent.name == "deploy"
LOCAL_PROJECT_ROOT = _THIS_FILE.parents[2] if _IN_REPOSITORY else None
# 컨테이너에서는 아래 이미지 정의의 로컬 경로를 쓰지 않는다. 계산이 실패하지 않도록 자리값만 둔다.
ROOT = LOCAL_PROJECT_ROOT or _THIS_FILE.parent
MODEL_DIR = "/app/models/body_image_validation"
NUMBA_CACHE_DIR = "/opt/numba_cache"

# 배포 대상. 기본은 production 이고, staging 은 이름 기본값·컨테이너 수 상한·GPU fallback·GPU snapshot 이 다르다.
STAGE = (os.environ.get("BODY_IMAGE_WORKER_STAGE") or "production").strip().lower()
_DEFAULT_NAMES = {
    # (app 이름, URL label, Modal Secret 이름)
    "production": ("lookddak-body-image-worker", "lookddak-body-image", "lookddak-body-image-worker"),
    "staging": (
        "lookddak-body-image-worker-staging",
        "lookddak-body-image-staging",
        "lookddak-body-image-worker-staging",
    ),
}
if STAGE not in _DEFAULT_NAMES:
    raise ValueError(f"BODY_IMAGE_WORKER_STAGE 는 production 또는 staging 이어야 합니다: {STAGE!r}")
_app_default, _label_default, _secret_default = _DEFAULT_NAMES[STAGE]

# 빈 문자열은 지정하지 않은 것으로 본다.
APP_NAME = os.environ.get("BODY_IMAGE_WORKER_APP_NAME") or _app_default
# INTERNAL_API_KEY, S3_BUCKET, AWS_REGION, AWS 자격증명을 담은 Modal Secret 이름.
SECRET_NAME = os.environ.get("BODY_IMAGE_WORKER_SECRET_NAME") or _secret_default
# Modal proxy auth(Modal-Key / Modal-Secret 헤더). 인증 없는 요청이 T4 컨테이너를 깨우지 않게
# Modal 앞단에서 막는다. INTERNAL_API_KEY Bearer 검사는 이와 별개로 앱에서 항상 한다.
REQUIRES_PROXY_AUTH = os.environ.get("BODY_IMAGE_WORKER_PROXY_AUTH", "true").lower() != "false"
ENDPOINT_LABEL = os.environ.get("BODY_IMAGE_WORKER_LABEL") or _label_default

_production_app, _production_label, _ = _DEFAULT_NAMES["production"]
if STAGE == "staging" and (APP_NAME == _production_app or ENDPOINT_LABEL == _production_label):
    # staging 배포가 production app·URL 을 덮어쓰지 않게 막는다.
    raise ValueError("staging 배포에 production app 이름이나 label 을 쓸 수 없습니다.")

# T4 benchmark 로 검증한 CUDA 조합. requirements.txt(EC2 CPU 이미지)와 별도로 여기서만 설치한다.
ONNXRUNTIME_GPU = "onnxruntime-gpu[cuda,cudnn]==1.30.0"
TORCH = ("torch==2.14.1", "torchvision==0.29.1")
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"

# 단일 요청 peak: CPU 2.92 core, RAM 6.46GiB, VRAM 1.37GiB. 컨테이너당 동시 요청은 기본값 1 이다
# (동시 요청 데코레이터를 쓰지 않는다). 공통 자원은 CPU memory snapshot 까지다. GPU snapshot 은 아래
# STAGE_SNAPSHOT 으로 staging 에서만 켠다.
WORKER_RESOURCES = {
    "gpu": "T4",
    "cpu": 4,
    "memory": 8192,
    "scaledown_window": 30,
    "min_containers": 0,
    "enable_memory_snapshot": True,
}

# staging 에서만 GPU 컨테이너를 1개로 제한한다(비용·테스트 통제). production 은 Modal 기본값 그대로다.
STAGE_SCALING = {"max_containers": 1} if STAGE == "staging" else {}
# staging 도 T4 로 고정한다. GPU snapshot 을 production·benchmark 와 같은 GPU 조건에서 검증하기 위해서다.
# (L4 fallback 과 T4↔L4 snapshot 호환성은 별도로 검증한 뒤에 다시 켠다.)
WORKER_GPU = WORKER_RESOURCES["gpu"]
# staging 에서만 GPU memory snapshot 을 켠다(experimental). 켜면 snap=True 단계에서 GPU 를 쓸 수 있어
# CUDA·rembg CUDA session·nvJPEG·warm-up 까지 snapshot 에 담는다. production 은 CPU memory snapshot 그대로다.
GPU_SNAPSHOT = STAGE == "staging"
STAGE_SNAPSHOT = {"experimental_options": {"enable_gpu_snapshot": True}} if GPU_SNAPSHOT else {}
# 컨테이너는 이 파일을 다시 import 해 lifecycle 을 고른다. 로컬 환경변수는 컨테이너에 없으므로 staging 은
# stage 를 컨테이너 환경변수로 넘긴다(production 은 기본값이 production 이라 넘기지 않는다).
STAGE_ENV = {"env": {"BODY_IMAGE_WORKER_STAGE": STAGE}} if GPU_SNAPSHOT else {}

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
    # 코드는 마지막 layer 다. 코드만 바뀌면 의존성·모델·cache layer 를 다시 만들지 않는다.
    .add_local_dir(ROOT / "app", "/root/app", ignore=["**/__pycache__", "**/*.pyc"])
)

app = modal.App(APP_NAME)


class StageConfigError(RuntimeError):
    """Modal 에 등록된 함수 구성과 컨테이너가 읽은 stage 가 맞지 않는다. 다른 경로로 바꾸지 않고 멈춘다."""


def registered_gpu_snapshot() -> tuple[bool, str] | None:
    """이 컨테이너가 Modal 서버에서 받은 함수 정의의 (GPU snapshot 사용 여부, GPU 종류). 컨테이너 밖이면 None.

    배포 때 experimental_options 로 등록된 구성이라 BODY_IMAGE_WORKER_STAGE(env)와 독립적이다.
    Modal 런타임 내부 객체(modal 1.6.1 의 _ContainerIOManager)를 읽으므로, 읽지 못하면 None 이다.
    """
    try:
        from modal._runtime.container_io_manager import _ContainerIOManager

        manager = _ContainerIOManager._singleton
        if manager is None:
            return None
        function_def = manager.function_def
        gpu_type = function_def.resources.gpu_config.gpu_type
        return bool(function_def._experimental_enable_gpu_snapshot and gpu_type), gpu_type
    except (ImportError, AttributeError):  # SDK 내부 구조가 바뀌면 읽지 못한 것으로 본다
        return None


def _rounded(timings: dict[str, float] | None) -> dict[str, float] | None:
    return None if timings is None else {name: round(value, 1) for name, value in timings.items()}


def container_stage() -> str:
    return (os.environ.get("BODY_IMAGE_WORKER_STAGE") or "production").strip().lower()


def resolve_gpu_snapshot() -> bool:
    """snap=True 에서 GPU snapshot 경로를 쓸지 정한다. 등록 구성과 컨테이너 stage 가 다르면 StageConfigError.

    - staging 으로 배포됐는데(GPU snapshot 등록) 컨테이너가 stage 를 못 받음 → 멈춘다(production 경로로 가지 않음).
    - 컨테이너는 staging 인데 GPU snapshot 이 등록되지 않음 → 멈춘다.
    - production(등록 안 됨, env 없음)은 기존 경로 그대로다. 구성을 읽지 못해도 production 은 그대로 진행한다.
    """
    stage = container_stage()
    wants_gpu_snapshot = stage == "staging"
    registered = registered_gpu_snapshot()
    if registered is None:
        if wants_gpu_snapshot:
            raise StageConfigError("staging 컨테이너인데 Modal 함수 구성을 읽지 못해 GPU snapshot 을 확인할 수 없습니다.")
        return False
    enabled, gpu_type = registered
    if enabled != wants_gpu_snapshot:
        raise StageConfigError(
            f"Modal 함수 구성(gpu_snapshot={enabled}, gpu={gpu_type})과 컨테이너 stage({stage!r})가 맞지 않습니다."
        )
    print(f"body image worker lifecycle: stage={stage} gpu_snapshot={enabled} gpu={gpu_type}", flush=True)
    return enabled


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
        """snapshot 에 담긴다. secret·S3 client·limiter 는 만들지 않는다.

        production: import·MediaPipe(CPU)만. CUDA 를 건드리지 않는다.
        staging(GPU snapshot): 여기에 더해 CUDA·rembg CUDA session·nvJPEG 생성과 warm-up 까지 한다.
        """
        from app.body_image_validation import worker

        started = time.time()
        self.detectors = None
        self.models = self.warmup_ms = None
        self.gpu_snapshot = resolve_gpu_snapshot()
        if self.gpu_snapshot:
            self.models, self.warmup_ms = worker.prepare_gpu_snapshot(require_cuda=True)
        else:
            self.detectors = worker.prepare_cpu()
        print(
            f"body image worker lifecycle: snap=True done start={started:.3f} end={time.time():.3f} "
            f"gpu_snapshot={self.gpu_snapshot} warmup_ms={_rounded(self.warmup_ms)}",
            flush=True,
        )

    @modal.enter(snap=False)
    def start_after_restore(self) -> None:
        """snapshot 복원 뒤 매번. secret 확인·S3 client·service(limiter)·runtime 등록은 여기서 새로 한다.

        snap=True 가 detector 만 만들었으면(production) 기존 start_gpu 로 GPU 초기화·warm-up 까지 한다.
        staging 은 snapshot 에 담긴 모델을 그대로 쓰고 다시 만들지 않는다.
        """
        from app.body_image_validation import worker

        started = time.time()
        if (container_stage() == "staging") != self.gpu_snapshot:
            # 복원된 컨테이너의 stage 가 snapshot 을 만들 때와 다르다. 다른 경로로 바꾸지 않고 멈춘다.
            raise StageConfigError(f"복원 뒤 stage({container_stage()!r})가 snapshot 구성과 다릅니다.")
        if self.detectors is not None:
            timings = worker.start_gpu(*self.detectors)
        else:
            runtime = worker.start_restored_runtime(self.models, self.warmup_ms)
            timings = self.warmup_ms if runtime is not None else None
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
