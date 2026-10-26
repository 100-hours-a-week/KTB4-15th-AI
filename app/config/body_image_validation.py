"""전신 이미지 검증 모델 경로와 초기 threshold."""

import os
from pathlib import Path

MODEL_DIR = Path(os.getenv("BODY_IMAGE_MODEL_DIR", "/app/models/body_image_validation"))
PERSON_DETECTOR_MODEL = Path(
    os.getenv("PERSON_DETECTOR_MODEL", str(MODEL_DIR / "efficientdet_lite0.tflite"))
)
POSE_LANDMARKER_MODEL = Path(
    os.getenv("POSE_LANDMARKER_MODEL", str(MODEL_DIR / "pose_landmarker_lite.task"))
)
REMBG_MODEL = Path(os.getenv("REMBG_MODEL", str(MODEL_DIR / "u2netp.onnx")))

# 파일 크기는 decode 메모리를 거의 바꾸지 않는다(Docker 측정: 같은 25MP 에서 1.2MB 와 10MB 의 peak
# 차이는 요청 간 편차 안). decode 메모리는 아래 MAX_IMAGE_PIXELS 가 막는다.
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
# 파일 크기와 별개로 decode 메모리를 막는다. 고압축 초고해상도 이미지는 파일이 작아도
# RGB decode 만으로 수백 MB 를 쓴다(150MP ≈ 450MB, 25MP ≈ 75MB).
MAX_IMAGE_PIXELS = int(os.getenv("BODY_IMAGE_MAX_PIXELS", str(25_000_000)))
MIN_SHORT_SIDE = int(os.getenv("BODY_IMAGE_MIN_SHORT_SIDE", "480"))
# 입력 검증을 통과한 이미지는 긴 변을 이 값 이하로 줄인 뒤 모델/rembg 에 넣는다(확대는 안 함).
MAX_PROCESSING_IMAGE_SIDE = int(os.getenv("MAX_PROCESSING_IMAGE_SIDE", "1600"))
# true 이면 JPEG decode·resize 를 GPU(nvJPEG)로 먼저 시도하고, 지원하지 않는 입력(PNG·CMYK 등)과
# GPU 실패는 기존 Pillow 경로로 처리한다. torch/torchvision(CUDA)이 설치된 GPU 배포에서만 켠다.
GPU_PREPROCESSING = os.getenv("BODY_IMAGE_GPU_PREPROCESSING", "false").lower() == "true"
# local: 이 프로세스가 모델을 올려 직접 검증한다(EC2 CPU 기존 방식, Modal worker).
# remote: 모델을 올리지 않고 Modal T4 worker(BODY_IMAGE_REMOTE_URL)에 HTTP 로 맡긴다(AI Server).
# Modal worker 는 local 로만 뜬다(remote 면 시작을 거부해 자기 자신을 다시 부르지 않는다).
VALIDATION_MODE = os.getenv("BODY_IMAGE_VALIDATION_MODE", "local").strip().lower()
# worker 의 base URL(https). 경로 /api/v1/validation/body-image 는 코드가 붙인다.
REMOTE_URL = os.getenv("BODY_IMAGE_REMOTE_URL", "").strip()
# worker 호출 시간 제한은 환경변수가 아닌 코드 상수로 둔다(바꾸려면 이 파일을 고쳐 재배포한다).
# 네트워크 timeout(초): 연결·전송·응답 대기·읽기 각 단계가 멈춰 있을 수 있는 최대 시간. 남은 전체 deadline 보다
# 길게 쓰이지 않는다.
REMOTE_TIMEOUT_SECONDS = 120.0
# 전체 deadline(초): endpoint 가 요청을 받은 시점부터 입력 검사·대기열 대기·Modal 호출·최종 응답 수신까지.
# redirect 를 따라가도 다시 시작하지 않고, 넘으면 재시도 없이 504 BODY_IMAGE_WORKER_TIMEOUT.
# Modal 의 303 redirect(150초)보다 짧으므로 정상적으로는 303 을 받기 전에 끝난다.
REMOTE_TOTAL_DEADLINE_SECONDS = 120.0
# 무거운 검증(모델 추론·rembg)의 프로세스 내 동시 실행 수와 대기 수. 넘치면 즉시 429.
MAX_CONCURRENT_BODY_VALIDATIONS = int(os.getenv("MAX_CONCURRENT_BODY_VALIDATIONS", "1"))
MAX_WAITING_BODY_VALIDATIONS = int(os.getenv("MAX_WAITING_BODY_VALIDATIONS", "1"))
# remote 모드에서 대기열에 들어간 요청이 실행 슬롯을 기다리는 최대 시간(초). 넘으면 429 SERVER_BUSY. 이 대기도
# 전체 deadline(REMOTE_TOTAL_DEADLINE_SECONDS) 안에 들어가고, deadline 이 먼저 끝나면 504.
# local 모드에는 적용하지 않는다(기존처럼 대기 시간 제한 없음).
MAX_QUEUE_WAIT_SECONDS = float(os.getenv("BODY_IMAGE_MAX_QUEUE_WAIT_SECONDS", "10"))
# rembg ONNX session 의 CPU memory arena. arena 는 요청 때 잡은 버퍼를 프로세스가 끝날 때까지
# 들고 있어 요청이 끝난 뒤 메모리가 높게 남는다(Docker 50회 측정: 끄면 idle −31%, latency 차이
# 없음). 문제가 생기면 true 로 되돌린다(ONNX Runtime 기본값).
REMBG_ENABLE_CPU_MEM_ARENA = os.getenv("REMBG_ENABLE_CPU_MEM_ARENA", "false").lower() == "true"
# rembg ONNX Runtime execution provider 순서(쉼표 구분). 비우면 rembg 기본 선택을 그대로 쓴다.
# Modal T4 worker 는 "CUDAExecutionProvider,CPUExecutionProvider" 로 둔다. GPU 전처리 플래그
# (BODY_IMAGE_GPU_PREPROCESSING)와는 별개다.
REMBG_PROVIDERS = tuple(
    name.strip() for name in os.getenv("BODY_IMAGE_REMBG_PROVIDERS", "").split(",") if name.strip()
)
# 진단(staging 검증용, 기본 꺼짐). 켜면 요청마다 단계별 처리 시간(ms)을 INFO 로그 한 줄로 남긴다.
# API 응답은 바꾸지 않는다.
DIAGNOSTICS = os.getenv("BODY_IMAGE_DIAGNOSTICS", "false").lower() == "true"
# 0 보다 크면 rembg ONNX Runtime session 을 profiling 을 켜고 만들고, 이 횟수만큼 실행한 뒤 profiling 을 끝내
# 실행별 provider(CUDA/CPU) 노드 수와 추론 시간을 로그로 남긴다. 그 뒤 실행은 profiling 하지 않는다(기본 0).
REMBG_PROFILE_RUNS = int(os.getenv("BODY_IMAGE_REMBG_PROFILE_RUNS", "0"))

PERSON_SCORE_THRESHOLD = float(os.getenv("PERSON_SCORE_THRESHOLD", "0.40"))
PERSON_HEIGHT_RATIO_THRESHOLD = float(os.getenv("PERSON_HEIGHT_RATIO_THRESHOLD", "0.55"))
POSE_DETECTION_CONFIDENCE = float(os.getenv("POSE_DETECTION_CONFIDENCE", "0.50"))
POSE_PRESENCE_CONFIDENCE = float(os.getenv("POSE_PRESENCE_CONFIDENCE", "0.50"))

NOSE_VISIBILITY = float(os.getenv("NOSE_VISIBILITY", "0.70"))
EYE_VISIBILITY = float(os.getenv("EYE_VISIBILITY", "0.50"))
SHOULDER_VISIBILITY = float(os.getenv("SHOULDER_VISIBILITY", "0.70"))
ELBOW_VISIBILITY = float(os.getenv("ELBOW_VISIBILITY", "0.60"))
WRIST_VISIBILITY = float(os.getenv("WRIST_VISIBILITY", "0.45"))
HIP_VISIBILITY = float(os.getenv("HIP_VISIBILITY", "0.70"))
KNEE_VISIBILITY = float(os.getenv("KNEE_VISIBILITY", "0.65"))
ANKLE_VISIBILITY = float(os.getenv("ANKLE_VISIBILITY", "0.60"))
HEEL_VISIBILITY = float(os.getenv("HEEL_VISIBILITY", "0.50"))
FOOT_INDEX_VISIBILITY = float(os.getenv("FOOT_INDEX_VISIBILITY", "0.50"))
BODY_BORDER_MARGIN = float(os.getenv("BODY_BORDER_MARGIN", "0.01"))

SHOULDER_WIDTH_RATIO = float(os.getenv("SHOULDER_WIDTH_RATIO", "0.25"))
HIP_WIDTH_RATIO = float(os.getenv("HIP_WIDTH_RATIO", "0.15"))
SHOULDER_VISIBILITY_DIFF = float(os.getenv("SHOULDER_VISIBILITY_DIFF", "0.35"))
DARK_THRESHOLD = float(os.getenv("DARK_THRESHOLD", "60"))
