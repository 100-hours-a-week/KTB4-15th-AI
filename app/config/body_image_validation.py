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

MAX_IMAGE_BYTES = int(os.getenv("BODY_IMAGE_MAX_BYTES", str(10 * 1024 * 1024)))
MIN_SHORT_SIDE = int(os.getenv("BODY_IMAGE_MIN_SHORT_SIDE", "480"))

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
