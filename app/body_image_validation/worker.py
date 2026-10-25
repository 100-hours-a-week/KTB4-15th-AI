"""Modal T4 worker 용 전신 이미지 검증 HTTP 앱과 초기화 순서.

EC2 AI Server(app.main)와 같은 router·service·runtime 을 그대로 쓰고, 이 모듈은 Modal 컨테이너에서
부르는 순서만 정한다. Modal 자체는 import 하지 않는다(정의는 deploy/modal/body_image_worker.py).

초기화는 세 단계로 나뉜다. 앞의 두 단계는 secret(INTERNAL_API_KEY·AWS 자격증명)이 담긴 설정과 S3 client 를
만들지 않으므로 snapshot 에 담겨도 된다. 세 번째 단계는 복원된 컨테이너마다 새로 해야 한다.

1. prepare_cpu()               — 라이브러리 import 와 MediaPipe Person/Pose(CPU) 생성. CUDA 를 건드리지 않는다.
2. prepare_gpu_for_snapshot()  — ort.preload_dlls·rembg CUDA session·CUDA context·nvJPEG decoder 를 만들고,
                                 S3 없이 요청 경로를 한 번 돌려(warm_up) GPU 를 데운다. → BodyImageModels
3. start_runtime(models)       — 실행 모드·INTERNAL_API_KEY 확인, S3 bucket 확인·S3 client, service 와
                                 limiter(Condition) 생성, runtime 등록.

Modal worker 의 호출 위치(deploy/modal/body_image_worker.py):
- production(CPU memory snapshot): snap=True 에서 1, snap=False 에서 2·3 을 start_gpu() 로.
- staging(GPU snapshot):           snap=True 에서 1·2 를 prepare_gpu_snapshot() 으로,
                                   snap=False 에서 3 을 start_restored_runtime() 으로.
create_app() 은 전신 이미지 검증 endpoint 와 /health 만 있는 FastAPI 앱이다.
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
    """(person_detector, pose_detector) 를 돌려준다. 소유권은 prepare_gpu_for_snapshot 이 넘겨받는다."""
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


def check_worker_settings() -> None:
    """실행 모드와 인증 설정. secret 을 읽으므로 snapshot 복원 뒤에 부른다."""
    check_validation_mode()
    check_auth_settings()


def prepare_gpu_for_snapshot(person_detector, pose_detector):
    """GPU 모델을 만들고 데운다. (BodyImageModels, warm-up 시간 ms) 를 돌려준다.

    S3 client·secret 설정·limiter 는 만들지 않는다(이 함수가 쓰는 모듈은 settings 를 import 하지 않는다).
    넘긴 detector 는 BodyImageModels 가 소유한다. 모델 생성 실패는 BodyImageRuntimeError 이며 detector 는
    이미 정리돼 있다. warm-up 이 실패하면 모델을 닫고 예외를 그대로 낸다.
    """
    from app.body_image_validation.inference import BodyImageModels

    _preload_cuda_libraries()
    models = BodyImageModels.load(person_detector=person_detector, pose_detector=pose_detector)
    try:
        _warn_inactive_gpu(models)
        timings = warm_up(models)
    except BaseException:
        models.close()
        raise
    return models, timings


def start_runtime(models):
    """복원된 컨테이너마다. 설정을 확인하고 S3 client·service(limiter)를 만들어 runtime 을 등록한다.

    runtime 을 돌려준다. S3_BUCKET 누락 등으로 만들지 못하면 None 이고 넘긴 모델은 닫는다(요청은 기존
    router 가 500 S3_CONFIG_ERROR / BODY_IMAGE_RUNTIME_UNAVAILABLE 로 응답한다).
    """
    check_worker_settings()
    return _register_runtime(models)


def prepare_gpu_snapshot(require_cuda: bool = False):
    """GPU snapshot(staging) 의 snap=True 단계. prepare_cpu → prepare_gpu_for_snapshot.

    (BodyImageModels, warm-up 시간 ms) 를 돌려준다. secret·S3 는 읽지 않는다. 실행 모드는 config 값이라
    여기서 먼저 확인해 GPU 작업 전에 멈춘다. 모델 파일 누락 등 BodyImageRuntimeError 면 (None, None) 이고
    detector 는 이미 정리돼 있다(요청은 기존처럼 500 BODY_IMAGE_RUNTIME_UNAVAILABLE).

    require_cuda 면 설정이 요구한 GPU(rembg CUDA provider·nvJPEG 전처리)를 이 단계에서 실제로 쓰지 못할 때
    모델을 닫고 WorkerConfigError 를 낸다. CPU 상태로 GPU snapshot 을 만든 채 정상 시작하지 않게 하기 위해서다.
    """
    from app.body_image_validation.inference import BodyImageRuntimeError

    check_validation_mode()
    person_detector, pose_detector = prepare_cpu()
    try:
        models, timings = prepare_gpu_for_snapshot(person_detector, pose_detector)
    except BodyImageRuntimeError as error:
        _log_no_runtime(error)
        return None, None
    if require_cuda:
        missing = _inactive_gpu(models)
        if missing:
            models.close()
            raise WorkerConfigError(f"GPU snapshot 단계에서 GPU 를 쓸 수 없습니다: {', '.join(missing)}")
    return models, timings


def start_restored_runtime(models, warmup_ms: dict[str, float] | None = None):
    """GPU snapshot(staging) 의 snap=False 단계. snapshot 에 담긴 모델로 start_runtime 만 한다(모델을 다시
    만들지 않는다). models 가 None(snapshot 단계 실패)이면 설정만 확인하고 runtime 없이 시작한다."""
    if models is None:
        check_worker_settings()
        logger.error("body image worker started without a runtime")
        return None
    runtime = start_runtime(models)
    if runtime is not None:
        _log_ready(models, warmup_ms or {})
    return runtime


def start_gpu(person_detector, pose_detector) -> dict[str, float] | None:
    """현재 production 순서(snap=False): prepare_gpu_for_snapshot → start_runtime 을 이어서 한다.

    기존처럼 설정(실행 모드·인증·S3_BUCKET)을 GPU 작업보다 먼저 확인해 잘못된 설정이면 모델을 만들지
    않는다. 이미 runtime 이 있으면 다시 만들지 않고 warm-up 만 한다. warm-up 시간을 돌려준다(runtime 이
    없으면 None).
    """
    from app.body_image_validation.inference import BodyImageRuntimeError
    from app.body_image_validation.runtime import runtime_manager
    from app.clients.s3 import S3ConfigError, get_s3_bucket

    check_worker_settings()
    runtime = runtime_manager.runtime
    if runtime is not None:
        _preload_cuda_libraries()
        _warn_inactive_gpu(runtime.models)
        timings = warm_up(runtime.models)
        _log_ready(runtime.models, timings)
        return timings
    # 넘겨받은 detector 는 이 함수가 소유한다. 여기서 멈추면 직접 닫고, prepare_gpu_for_snapshot 에 넘기면
    # 그쪽(BodyImageModels)이 성공·실패 모두 정리한다.
    try:
        get_s3_bucket()
    except S3ConfigError as error:
        _close_detectors(person_detector, pose_detector)
        _log_no_runtime(error)
        return None
    try:
        models, timings = prepare_gpu_for_snapshot(person_detector, pose_detector)
    except BodyImageRuntimeError as error:
        _log_no_runtime(error)
        return None
    if _register_runtime(models) is None:
        return None
    _log_ready(models, timings)
    return timings


def _log_no_runtime(error: Exception) -> None:
    # S3_BUCKET 누락·모델 파일 누락 등. 요청은 기존 router 가 500 으로 응답한다.
    logger.warning("body image validation runtime is unavailable: %s", error)
    logger.error("body image worker started without a runtime")


def _close_detectors(*detectors) -> None:
    for detector in detectors:
        try:
            detector.close()
        except Exception:
            logger.warning("failed to close %s", type(detector).__name__, exc_info=True)


def _register_runtime(models):
    from app.body_image_validation.runtime import runtime_manager

    runtime_manager.start(models=models)
    runtime = runtime_manager.runtime
    if runtime is None:
        # 성공처럼 처리하지 않는다. 등록하지 못한 모델은 아무도 닫지 않으므로 여기서 닫는다
        # (BodyImageRuntime.create 가 이미 닫았으면 BodyImageModels.close 는 다시 닫지 않는다).
        models.close()
        logger.error("body image worker started without a runtime")
        return None
    if getattr(runtime, "models", None) is not models:
        # 다른 모델로 runtime 이 이미 있었다. 넘긴 모델은 쓰이지 않으므로 닫는다.
        models.close()
    return runtime


def _inactive_gpu(models) -> list[str]:
    """설정이 요구했지만 실제로 붙지 않은 GPU 경로."""
    missing = []
    providers = models.background_remover.providers
    if _CUDA_PROVIDER in config.REMBG_PROVIDERS and _CUDA_PROVIDER not in providers:
        missing.append(f"rembg CUDA provider (active: {providers})")
    if config.GPU_PREPROCESSING and models.gpu_preprocessor is None:
        missing.append("GPU preprocessing (nvJPEG)")
    return missing


def _warn_inactive_gpu(models) -> None:
    providers = models.background_remover.providers
    if _CUDA_PROVIDER in config.REMBG_PROVIDERS and _CUDA_PROVIDER not in providers:
        logger.warning("rembg CUDA provider is not active, running on %s", providers)
    if config.GPU_PREPROCESSING and models.gpu_preprocessor is None:
        logger.warning("GPU preprocessing is enabled but unavailable, using Pillow only")


def _log_ready(models, timings: dict[str, float]) -> None:
    logger.info(
        "body image worker ready: rembg_providers=%s gpu_preprocessing=%s warmup_ms=%s",
        models.background_remover.providers,
        models.gpu_preprocessor is not None,
        {name: round(value, 1) for name, value in timings.items()},
    )


def warmup_image_bytes() -> bytes:
    """사람 사진이 아닌 합성 JPEG. 사용자 이미지는 쓰지 않는다."""
    from PIL import Image

    width, height = WARMUP_IMAGE_SIZE
    image = Image.linear_gradient("L").resize((width, height)).convert("RGB")
    output = BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def warm_up(models) -> dict[str, float]:
    """요청과 같은 전처리·검출·rembg·PNG 단계를 한 번씩 돌린다. 모델만 쓰고 S3·service 는 쓰지 않는다."""
    from app.body_image_validation.background import encode_png
    from app.body_image_validation.detectors import DetectionImage
    from app.body_image_validation.image_io import load_image, open_image

    body = warmup_image_bytes()
    timings = {}
    start = time.perf_counter()
    source = open_image(body)
    try:
        # service._load 와 같은 경로: GPU 로 처리할 수 있으면 GPU(nvJPEG), 아니면 Pillow.
        image = None
        if models.gpu_preprocessor is not None:
            image = models.gpu_preprocessor.try_load(source, body)
        if image is None:
            image = load_image(source)
    finally:
        source.close()
    timings["preprocess"] = (time.perf_counter() - start) * 1000
    try:
        start = time.perf_counter()
        detection = DetectionImage(image)
        models.person_detector.detect(detection)
        models.pose_detector.detect(detection)
        timings["person_pose"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        result = models.background_remover.remove(image)
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
