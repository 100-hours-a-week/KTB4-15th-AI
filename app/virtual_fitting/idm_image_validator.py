"""IDM-VTON fallback 입력 검증: 상품 조합(상의 1개)과 상품 이미지 분류(의류 단독 이미지만 허용).

Pruna 상품 검증(product_selection.validate_and_sort_products)은 그대로 두고, 그 결과에 IDM-VTON 전용 조건만 더한다.

1. 조합: 상의 1개만(하의 1개·상의+하의는 미호출). 세부 분류는 Pruna 와 같은 영어 명칭 매핑이 있어야 한다.
   상품 이미지 URL 은 Modal 서비스(deploy/modal/idm_vton/service.py 의 _require_https_url)와 같은 조건(https, 호스트 있음)
   이어야 한다. 아니면 분류·호출 기회 선점 전에 미지원으로 끝낸다(Modal 이 invalid_request 로 거절할 입력을 보내지 않는다).
2. 상품 이미지 분류(IDM_VTON_*). GARMENT_ONLY 일 때만 Modal 을 부른다.
   - INVALID: 내려받기 4xx·형식(JPEG·PNG·WEBP 외)·디코딩 실패·짧은 변 256px 미만·전경 없음.
   - MODEL_WORN: 전신사진 검증과 같은 MediaPipe Pose Landmarker 가 사람 관절을 찾았고, 얼굴·어깨·팔꿈치·손목·엉덩이·
     무릎·발목 중 MIN_BODY_LANDMARKS 개 이상이 BODY_VISIBILITY 이상으로 보인다.
     Person Detector(EfficientDet)는 쓰지 않는다. 테스트 입력에서 평평하게 놓인 상의 사진 2장을 0.72·0.90 점수의
     person 으로 잡았다(Pose 는 의류 사진 7장 모두 없음, 착용 사진 2장 모두 0.96 이상). 02_image_validation.md 참고.
   - AMBIGUOUS: Pose 랜드마크는 있으나 기준 미만, 투명 배경(배경 제거 결과로 보임),
     가장자리가 한 가지 색이 아님(연출 사진), 전경 면적이 너무 작거나 큼.
   - GARMENT_ONLY: 위 어느 것에도 해당하지 않음 — 사람 근거가 없고, 단색 배경 위에 전경 하나가 적당한 크기로 있음.
   배경을 지운 결과만으로 의류 단독이라고 보지 않는다(투명 배경은 AMBIGUOUS). 판단이 서지 않으면 부르지 않는 쪽을 택한다.
3. 내려받기 timeout·연결 오류·5xx 는 분류하지 않고 IdmImageUnavailable 로 올린다(작업을 실패로 확정하지 않고 보류).

이미지 URL·바이트는 로그에 남기지 않는다. 모델 파일은 전신사진 검증과 같은 경로(AI Server 이미지에 포함)를 쓴다.
"""

import io
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.virtual_fitting.exceptions import UnsupportedSubCategoryError
from app.virtual_fitting.models import TOP_CATEGORY, FittingProduct
from app.virtual_fitting.subcategory_mapping import get_english_sub_category

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 15.0
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MIN_IMAGE_SIDE = 256
# 분류에 쓰는 긴 변 크기. MediaPipe 와 가장자리 통계에 충분하고 CPU 시간을 줄인다.
ANALYSIS_LONG_SIDE = 512
_FORMATS = {"JPEG", "PNG", "WEBP"}
_USER_AGENT = "LookDDak-AI/1.0"

# 투명(alpha < 250) 픽셀이 이 비율을 넘으면 배경 제거 결과로 본다.
TRANSPARENT_RATIO = 0.01
# 가장자리 띠(짧은 변의 4%)에서 배경색(중앙값)과 이 거리 이내인 픽셀 비율이 기준 이상이어야 단색 배경으로 본다.
BORDER_RATIO = 0.04
BACKGROUND_DISTANCE = 28.0
MIN_UNIFORM_BORDER = 0.85
# 배경색과 다른 픽셀(전경)이 이미지에서 차지하는 비율 범위.
MIN_FOREGROUND_RATIO = 0.08
MAX_FOREGROUND_RATIO = 0.85
# Pose Landmarker 근거. 아래 관절 중 이 개수 이상이 이 visibility 이상이면 착용(사람) 사진으로 본다.
BODY_VISIBILITY = 0.6
MIN_BODY_LANDMARKS = 4
_BODY_LANDMARKS = (
    "NOSE",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
)


class IdmImageClass(str, Enum):
    GARMENT_ONLY = "IDM_VTON_GARMENT_ONLY"
    MODEL_WORN = "IDM_VTON_MODEL_WORN"
    AMBIGUOUS = "IDM_VTON_AMBIGUOUS"
    INVALID = "IDM_VTON_INVALID"


@dataclass(frozen=True)
class GarmentImageCheck:
    classification: IdmImageClass
    # 로그용 짧은 사유(고정 문자열). URL·픽셀 값은 넣지 않는다.
    reason: str
    details: Mapping[str, Any] | None = None

    @property
    def allowed(self) -> bool:
        return self.classification is IdmImageClass.GARMENT_ONLY


class IdmImageUnavailable(RuntimeError):
    """일시적으로 상품 이미지를 내려받지 못했다(timeout·연결 오류·5xx·429). 분류하지 않는다."""


@dataclass(frozen=True)
class IdmProductSupport:
    supported: bool
    reason: str
    product: FittingProduct | None = None
    garment_name: str | None = None


def is_https_url(url: Any) -> bool:
    """Modal 서비스가 받는 이미지 URL 조건(https 이고 호스트가 있음). 서비스의 _require_https_url 과 같다."""
    if not isinstance(url, str) or not url:
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme == "https" and bool(parts.netloc)


def check_idm_products(products: Sequence[FittingProduct]) -> IdmProductSupport:
    """Pruna 검증을 통과한 상품 목록에 IDM-VTON 조합 조건(상의 1개)을 적용한다."""
    if len(products) != 1:
        return IdmProductSupport(False, "not_single_product")
    product = products[0]
    if product.main_category != TOP_CATEGORY:
        return IdmProductSupport(False, "not_top")
    if not product.image_url or not product.image_url.strip():
        return IdmProductSupport(False, "image_missing")
    if not is_https_url(product.image_url):
        return IdmProductSupport(False, "image_url_not_https")
    try:
        name = get_english_sub_category(product.sub_category)
    except UnsupportedSubCategoryError:
        return IdmProductSupport(False, "sub_category_unmapped")
    return IdmProductSupport(True, "single_top", product, name)


# 상품 DB color(크롤러 색상 계약)의 영어 이름. 없는 값은 설명에서 뺀다(번역을 추측하지 않는다).
COLOR_ENGLISH_NAMES = {
    "블랙": "black",
    "블루": "blue",
    "네이비": "navy",
    "화이트": "white",
    "그레이": "gray",
    "아이보리": "ivory",
    "다크그레이": "dark gray",
    "브라운": "brown",
    "베이지": "beige",
    "그린": "green",
    "카키": "khaki",
    "레드": "red",
    "핑크": "pink",
    "옐로우": "yellow",
    "스카이블루": "sky blue",
    "민트": "mint",
    "다크네이비": "dark navy",
    "오렌지": "orange",
    "퍼플": "purple",
    "브릭": "brick red",
    "다크블루": "dark blue",
    "라이트그레이": "light gray",
    "라이트그린": "light green",
    "머스타드": "mustard",
    "버건디": "burgundy",
    "다크브라운": "dark brown",
    "라벤더": "lavender",
    "카멜": "camel",
    "오트밀": "oatmeal",
    "피치": "peach",
    "다크그린": "dark green",
    "라임": "lime",
    "다크오렌지": "dark orange",
    "올리브그린": "olive green",
    "라이트핑크": "light pink",
    "라이트옐로우": "light yellow",
    "다크베이지": "dark beige",
    "라이트브라운": "light brown",
    "딥레드": "deep red",
    "다크핑크": "dark pink",
}


def garment_description(garment_name: str, color: str | None) -> str:
    """IDM-VTON 프롬프트("model is wearing …")에 넣을 영어 설명. 상품 DB 의 색상·세부 분류만 쓴다."""
    english = COLOR_ENGLISH_NAMES.get((color or "").strip())
    return f"{english} {garment_name}" if english else garment_name


# --- 이미지 분류 ---


def download_image(url: str, *, opener: Callable[..., Any] = urlopen) -> bytes:
    """외부 상품 이미지를 시간·크기 제한 안에서 받는다. 4xx 는 INVALID 근거로, 그 밖의 실패는 IdmImageUnavailable."""
    if not is_https_url(url):
        raise ValueError("상품 이미지 URL 은 https 여야 합니다.")
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with opener(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            body = response.read(MAX_IMAGE_BYTES + 1)
    except HTTPError as error:
        if 400 <= error.code < 500 and error.code not in (408, 429):
            raise ValueError(f"상품 이미지 HTTP {error.code}") from None
        raise IdmImageUnavailable(f"상품 이미지 HTTP {error.code}") from None
    except (URLError, TimeoutError, OSError) as error:
        raise IdmImageUnavailable(f"상품 이미지 {type(error).__name__}") from None
    if not body or len(body) > MAX_IMAGE_BYTES:
        raise ValueError("상품 이미지 크기가 허용 범위를 벗어났습니다.")
    return body


def _decode(body: bytes) -> tuple[Image.Image, float]:
    """(RGB 이미지, 투명 픽셀 비율). 형식·크기 조건이 맞지 않으면 ValueError."""
    try:
        with Image.open(io.BytesIO(body)) as opened:
            if opened.format not in _FORMATS:
                raise ValueError(f"format_{opened.format}")
            opened.load()
            upright = ImageOps.exif_transpose(opened)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"decode_{type(error).__name__}") from None
    if min(upright.size) < MIN_IMAGE_SIDE:
        raise ValueError("too_small")
    transparent = 0.0
    if upright.mode in ("RGBA", "LA", "PA") or "transparency" in upright.info:
        alpha = np.asarray(upright.convert("RGBA").getchannel("A"))
        transparent = float((alpha < 250).mean())
    rgb = upright.convert("RGB")
    rgb.thumbnail((ANALYSIS_LONG_SIDE, ANALYSIS_LONG_SIDE))
    return rgb, transparent


def background_stats(rgb: Image.Image) -> dict[str, float]:
    """가장자리 띠의 배경색 일치 비율과, 배경색과 다른 픽셀(전경) 비율."""
    array = np.asarray(rgb, dtype=np.float32)
    height, width = array.shape[:2]
    band = max(2, round(min(width, height) * BORDER_RATIO))
    border = np.concatenate(
        [
            array[:band].reshape(-1, 3),
            array[-band:].reshape(-1, 3),
            array[:, :band].reshape(-1, 3),
            array[:, -band:].reshape(-1, 3),
        ]
    )
    color = np.median(border, axis=0)
    border_close = np.linalg.norm(border - color, axis=1) <= BACKGROUND_DISTANCE
    foreground = np.linalg.norm(array - color, axis=2) > BACKGROUND_DISTANCE
    return {
        "uniform_border": round(float(border_close.mean()), 4),
        "foreground_ratio": round(float(foreground.mean()), 4),
    }


def _visible(landmarks: Mapping[str, Any], names: Sequence[str], threshold: float) -> int:
    return sum(
        1
        for name in names
        if name in landmarks and float(getattr(landmarks[name], "visibility", 0.0)) >= threshold
    )


class IdmGarmentImageClassifier:
    """상품 이미지 한 장을 IdmImageClass 로 나눈다. 탐지기는 처음 쓸 때 만든다(thread 안전, 프로세스당 한 벌)."""

    def __init__(
        self,
        *,
        pose_detector: Any = None,
        downloader: Callable[[str], bytes] = download_image,
    ) -> None:
        self._pose = pose_detector
        self._download = downloader
        self._lock = threading.Lock()

    def _pose_detector(self) -> Any:
        with self._lock:
            if self._pose is None:
                from app.body_image_validation.detectors import MediaPipePoseDetector
                from app.config import body_image_validation as config

                self._pose = MediaPipePoseDetector(str(config.POSE_LANDMARKER_MODEL))
            return self._pose

    def classify_url(self, url: str) -> GarmentImageCheck:
        """내려받아 분류한다. 일시 오류는 IdmImageUnavailable 로 올린다."""
        try:
            body = self._download(url)
        except ValueError as error:
            return GarmentImageCheck(IdmImageClass.INVALID, f"download_{error}")
        return self.classify_bytes(body)

    def classify_bytes(self, body: bytes) -> GarmentImageCheck:
        try:
            rgb, transparent = _decode(body)
        except ValueError as error:
            return GarmentImageCheck(IdmImageClass.INVALID, str(error))
        details: dict[str, Any] = {"size": list(rgb.size), "transparent_ratio": transparent}
        if transparent > TRANSPARENT_RATIO:
            # 배경을 지운 이미지. 착용 사진에서 의류만 오려낸 결과일 수 있어 의류 단독으로 보지 않는다.
            return GarmentImageCheck(IdmImageClass.AMBIGUOUS, "transparent_background", details)

        from app.body_image_validation.detectors import DetectionImage

        landmarks = self._pose_detector().detect(DetectionImage(rgb))
        if landmarks:
            visible = _visible(landmarks, _BODY_LANDMARKS, BODY_VISIBILITY)
            details["visible_body_landmarks"] = visible
            if visible >= MIN_BODY_LANDMARKS:
                return GarmentImageCheck(IdmImageClass.MODEL_WORN, "pose_detected", details)
            return GarmentImageCheck(IdmImageClass.AMBIGUOUS, "weak_pose", details)

        stats = background_stats(rgb)
        details.update(stats)
        if stats["foreground_ratio"] < 0.005:
            return GarmentImageCheck(IdmImageClass.INVALID, "no_foreground", details)
        if stats["uniform_border"] < MIN_UNIFORM_BORDER:
            return GarmentImageCheck(IdmImageClass.AMBIGUOUS, "busy_background", details)
        if not MIN_FOREGROUND_RATIO <= stats["foreground_ratio"] <= MAX_FOREGROUND_RATIO:
            return GarmentImageCheck(IdmImageClass.AMBIGUOUS, "foreground_size", details)
        return GarmentImageCheck(IdmImageClass.GARMENT_ONLY, "garment_only", details)
