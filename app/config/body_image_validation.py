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

MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "2"))
MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
# 파일 크기와 별개로 decode 메모리를 막는다. 고압축 초고해상도 이미지는 파일이 작아도
# RGB decode 만으로 수백 MB 를 쓴다(150MP ≈ 450MB, 25MP ≈ 75MB).
MAX_IMAGE_PIXELS = int(os.getenv("BODY_IMAGE_MAX_PIXELS", str(25_000_000)))
MIN_SHORT_SIDE = int(os.getenv("BODY_IMAGE_MIN_SHORT_SIDE", "480"))
# 입력 검증을 통과한 이미지는 긴 변을 이 값 이하로 줄인 뒤 모델/rembg 에 넣는다(확대는 안 함).
MAX_PROCESSING_IMAGE_SIDE = int(os.getenv("MAX_PROCESSING_IMAGE_SIDE", "1600"))
# 무거운 검증(모델 추론·rembg)의 프로세스 내 동시 실행 수와 대기 수. 넘치면 즉시 429.
MAX_CONCURRENT_BODY_VALIDATIONS = int(os.getenv("MAX_CONCURRENT_BODY_VALIDATIONS", "1"))
MAX_WAITING_BODY_VALIDATIONS = int(os.getenv("MAX_WAITING_BODY_VALIDATIONS", "1"))
# rembg ONNX session 의 CPU memory arena. arena 는 요청 때 잡은 버퍼를 프로세스가 끝날 때까지
# 들고 있어 요청이 끝난 뒤 메모리가 높게 남는다(Docker 50회 측정: 끄면 idle −31%, latency 차이
# 없음). 문제가 생기면 true 로 되돌린다(ONNX Runtime 기본값).
REMBG_ENABLE_CPU_MEM_ARENA = os.getenv("REMBG_ENABLE_CPU_MEM_ARENA", "false").lower() == "true"

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
