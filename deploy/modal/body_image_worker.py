"""Modal T4 전신 이미지 검증 worker.

Backend 가 직접 부르는 HTTP API(EC2 AI Server 와 같은 endpoint·응답)를 Modal T4 컨테이너에서 띄운다.
검증 로직은 app/ 의 router·service 를 그대로 쓰고, 이 파일은 이미지·자원·초기화 순서만 정한다.

  배포:  modal deploy deploy/modal/body_image_worker.py   (저장소 루트에서)
  문서:  deploy/modal/README.md

자원과 cold start 설정은 T4 benchmark 로 정한 값이다. 바꾸기 전에 benchmark 를 다시 돌린다.
"""

import os
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

# 배포 대상. 기본은 production 이고, staging 은 이름 기본값과 컨테이너 수 상한만 다르다.
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
# (동시 요청 데코레이터를 쓰지 않는다). GPU memory snapshot 은 쓰지 않는다(experimental_options 없음).
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
# staging 에서만 GPU fallback 을 시험한다. Modal 은 목록 순서를 우선순위로 쓴다(T4 우선, 없으면 L4).
# production 은 T4 그대로다. L4 에서도 같은 CUDA 13 / torch 2.14.1 / onnxruntime-gpu 1.30 조합이 동작했다.
WORKER_GPU = ["T4", "L4"] if STAGE == "staging" else WORKER_RESOURCES["gpu"]

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


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    **{**WORKER_RESOURCES, "gpu": WORKER_GPU},
    **STAGE_SCALING,
)
class BodyImageWorker:
    @modal.enter(snap=True)
    def prepare_cpu(self) -> None:
        """memory snapshot 에 담긴다. CUDA·secret 을 건드리지 않는다."""
        from app.body_image_validation import worker

        self.detectors = worker.prepare_cpu()

    @modal.enter(snap=False)
    def start_gpu(self) -> None:
        """snapshot 복원 뒤 매번. CUDA·rembg CUDA session·S3 client 를 만들고 warm-up 한다."""
        from app.body_image_validation import worker

        worker.start_gpu(*self.detectors)

    @modal.exit()
    def close(self) -> None:
        from app.body_image_validation import worker

        worker.close()

    @modal.asgi_app(label=ENDPOINT_LABEL, requires_proxy_auth=REQUIRES_PROXY_AUTH)
    def web(self):
        from app.body_image_validation import worker

        return worker.create_app()
