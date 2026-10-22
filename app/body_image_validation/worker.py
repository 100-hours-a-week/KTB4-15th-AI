"""Modal T4 worker 용 전신 이미지 검증 HTTP 앱과 초기화 순서.

EC2 AI Server(app.main)와 같은 router·service·runtime 을 그대로 쓰고, 이 모듈은 Modal 컨테이너에서
부르는 순서만 정한다. Modal 자체는 import 하지 않는다(정의는 deploy/modal/body_image_worker.py).

1. prepare_cpu()  — memory snapshot 에 담기는 단계. 라이브러리 import 와 MediaPipe(CPU) 모델 생성만
   한다. CUDA 를 건드리지 않고, secret(INTERNAL_API_KEY·AWS 자격증명)이 담긴 설정도 읽지 않는다.
2. start_gpu()    — snapshot 복원 뒤 매번 실행. CUDA context·rembg CUDA session·GPU 전처리·S3 client 를
   만들고, S3 없이 요청 경로를 한 번 돌려 첫 요청 전에 GPU 를 데운다.
3. create_app()   — 전신 이미지 검증 endpoint 와 /health 만 있는 FastAPI 앱.
"""

import importlib
import logging
import time
from io import BytesIO

from fastapi import FastAPI

from app.config import body_image_validation as config

logger = logging.getLogger(__name__)

# snapshot 전에 import 해 둘 무거운 라이브러리. rembg import 는 PyMatting Numba cache 를 읽는다.
_CPU_PRELOAD_MODULES = (
    "numpy",
    "PIL.Image",
    "cv2",
    "mediapipe",
    "onnxruntime",
    "rembg",
    "fastapi",
    "boto3",
)
# import 만 하고 CUDA 는 초기화하지 않는다(torch 는 첫 CUDA 호출 때 context 를 만든다).
_GPU_PRELOAD_MODULES = ("torch", "torchvision")
_CUDA_PROVIDER = "CUDAExecutionProvider"
WARMUP_IMAGE_SIZE = (1200, 1800)  # 긴 변이 MAX_PROCESSING_IMAGE_SIDE 보다 커서 resize 까지 거친다.


class WorkerConfigError(RuntimeError):
    """worker 를 띄우면 안 되는 설정."""


def preload_modules() -> None:
    for name in _CPU_PRELOAD_MODULES:
        importlib.import_module(name)
    if config.GPU_PREPROCESSING:
        for name in _GPU_PRELOAD_MODULES:
            importlib.import_module(name)


def prepare_cpu():
    """snapshot 단계. (person_detector, pose_detector) 를 돌려준다. 소유권은 start_gpu 로 넘긴다."""
    preload_modules()
    from app.body_image_validation.detectors import MediaPipePersonDetector, MediaPipePoseDetector

    person_detector = MediaPipePersonDetector(str(config.PERSON_DETECTOR_MODEL))
    try:
        pose_detector = MediaPipePoseDetector(str(config.POSE_LANDMARKER_MODEL))
    except BaseException:
        person_detector.close()
        raise
    return person_detector, pose_detector


def check_auth_settings() -> None:
    from app.config import settings

    if settings.AUTH_DISABLED:
        raise WorkerConfigError("Modal worker 에서는 AUTH_DISABLED 를 켤 수 없습니다.")
    if not settings.INTERNAL_API_KEY:
        # security.verify_internal_key 가 키 없이는 모든 요청을 401 로 거절한다. 열린 채로 뜨지 않는다.
        logger.error("INTERNAL_API_KEY is not set; every body image request will be rejected (401)")


def _preload_cuda_libraries() -> None:
    """pip 로 설치한 CUDA/cuDNN 라이브러리를 ONNX Runtime 이 찾도록 미리 올린다(onnxruntime-gpu)."""
    if _CUDA_PROVIDER not in config.REMBG_PROVIDERS:
        return
    import onnxruntime as ort

    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        preload()


def check_validation_mode() -> None:
    # worker 가 remote 모드로 뜨면 요청을 다시 Modal 로 보내 자기 자신을 부른다. 시작을 거부한다.
    if config.VALIDATION_MODE != "local":
        raise WorkerConfigError(
            f"Modal worker 는 BODY_IMAGE_VALIDATION_MODE=local 로만 실행합니다: {config.VALIDATION_MODE!r}"
        )


def start_gpu(person_detector, pose_detector) -> dict[str, float] | None:
    """snapshot 복원 뒤 단계. runtime 을 한 번만 만들고 warm-up 시간을 돌려준다(runtime 실패 시 None)."""
    from app.body_image_validation.runtime import runtime_manager

    check_validation_mode()
    check_auth_settings()
    _preload_cuda_libraries()
    runtime_manager.start(person_detector=person_detector, pose_detector=pose_detector)
    runtime = runtime_manager.runtime
    if runtime is None:
        # S3_BUCKET 누락·모델 파일 누락 등. 요청은 기존 router 가 500(S3_CONFIG_ERROR /
        # BODY_IMAGE_RUNTIME_UNAVAILABLE)으로 응답한다. 성공처럼 처리하지 않는다.
        logger.error("body image worker started without a runtime")
        return None
    providers = runtime.background_remover.providers
    if _CUDA_PROVIDER in config.REMBG_PROVIDERS and _CUDA_PROVIDER not in providers:
        logger.warning("rembg CUDA provider is not active, running on %s", providers)
    if config.GPU_PREPROCESSING and runtime.service.gpu_preprocessor is None:
        logger.warning("GPU preprocessing is enabled but unavailable, using Pillow only")
    timings = warm_up(runtime)
    logger.info(
        "body image worker ready: rembg_providers=%s gpu_preprocessing=%s warmup_ms=%s",
        providers,
        runtime.service.gpu_preprocessor is not None,
        {name: round(value, 1) for name, value in timings.items()},
    )
    return timings


def warmup_image_bytes() -> bytes:
    """사람 사진이 아닌 합성 JPEG. 사용자 이미지는 쓰지 않는다."""
    from PIL import Image

    width, height = WARMUP_IMAGE_SIZE
    image = Image.linear_gradient("L").resize((width, height)).convert("RGB")
    output = BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def warm_up(runtime) -> dict[str, float]:
    """요청과 같은 전처리·검출·rembg·PNG 단계를 한 번씩 돌린다. S3 에는 올리지 않는다."""
    from app.body_image_validation.background import encode_png
    from app.body_image_validation.detectors import DetectionImage
    from app.body_image_validation.image_io import open_image

    service = runtime.service
    body = warmup_image_bytes()
    timings = {}
    start = time.perf_counter()
    source = open_image(body)
    try:
        image = service._load(source, body)
    finally:
        source.close()
    timings["preprocess"] = (time.perf_counter() - start) * 1000
    try:
        start = time.perf_counter()
        detection = DetectionImage(image)
        service.person_detector.detect(detection)
        service.pose_detector.detect(detection)
        timings["person_pose"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        result = service.background_remover.remove(image)
        try:
            timings["rembg"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            encode_png(result)
            timings["png"] = (time.perf_counter() - start) * 1000
        finally:
            result.close()
    finally:
        image.close()
    return timings


def create_app() -> FastAPI:
    """EC2 앱(app.main)과 같은 router·오류 형식. 가상피팅·대화 router 와 그 lifespan 은 없다."""
    from app import health
    from app.body_image_validation import deadline, router
    from app.config.logging_config import configure_logging
    from app.errors import register_error_handlers

    configure_logging()
    app = FastAPI(title="LooKDDak Body Image Validation Worker")
    register_error_handlers(app)

    @app.middleware("http")
    async def request_deadline(request, call_next):
        # AI Server 가 보낸 deadline 을 이 요청의 context 에 둔다. sync endpoint 의 threadpool 에도 전해진다.
        token = deadline.set_deadline(deadline.parse_header(request.headers.get(deadline.HEADER)))
        try:
            return await call_next(request)
        finally:
            deadline.reset(token)

    app.include_router(health.router)
    app.include_router(router.router)
    return app


def close() -> None:
    from app.body_image_validation.runtime import runtime_manager

    runtime_manager.close()
