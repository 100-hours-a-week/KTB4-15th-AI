"""모델 출력에 적용하는 순수 rule 기반 검증."""

from collections.abc import Mapping, Sequence

from app.body_image_validation.exceptions import user_error
from app.body_image_validation.models import BoundingBox, Landmark, PersonDetection
from app.config import body_image_validation as config

_FLOAT_EPSILON = 1e-9


def select_single_person(
    detections: Sequence[PersonDetection], image_height: int
) -> BoundingBox:
    if not detections:
        raise user_error("PERSON_NOT_FOUND")
    if len(detections) > 1:
        raise user_error("MULTIPLE_PERSONS")
    box = detections[0].bounding_box
    if box.height / image_height < config.PERSON_HEIGHT_RATIO_THRESHOLD:
        raise user_error("PERSON_TOO_SMALL")
    return box


def _visible(landmarks: Mapping[str, Landmark], names: Sequence[str], threshold: float) -> bool:
    return all(
        name in landmarks
        and landmarks[name].visibility >= threshold
        and landmarks[name].presence >= config.POSE_PRESENCE_CONFIDENCE
        for name in names
    )


def _any_visible(
    landmarks: Mapping[str, Landmark], names: Sequence[str], threshold: float
) -> bool:
    return any(_visible(landmarks, (name,), threshold) for name in names)


def validate_full_body(landmarks: Mapping[str, Landmark]) -> None:
    rules = (
        (("NOSE",), config.NOSE_VISIBILITY),
        (("LEFT_SHOULDER", "RIGHT_SHOULDER"), config.SHOULDER_VISIBILITY),
        (("LEFT_HIP", "RIGHT_HIP"), config.HIP_VISIBILITY),
        (("LEFT_KNEE", "RIGHT_KNEE"), config.KNEE_VISIBILITY),
        (("LEFT_ANKLE", "RIGHT_ANKLE"), config.ANKLE_VISIBILITY),
        (("LEFT_HEEL", "RIGHT_HEEL"), config.HEEL_VISIBILITY),
        (("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"), config.FOOT_INDEX_VISIBILITY),
    )
    # 전신의 위아래 범위를 확인하는 단계이므로 양측 관절 중 한쪽이 보이면 통과한다.
    # 뒤의 팔/다리 규칙에서 양측 관절을 별도로 요구해 원인 코드를 구분한다.
    if not all(_any_visible(landmarks, names, threshold) for names, threshold in rules):
        raise user_error("FULL_BODY_NOT_VISIBLE")
    nose = landmarks["NOSE"]
    foot_rules = (
        (("LEFT_HEEL", "RIGHT_HEEL"), config.HEEL_VISIBILITY),
        (("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"), config.FOOT_INDEX_VISIBILITY),
    )
    feet = [
        landmarks[name]
        for names, threshold in foot_rules
        for name in names
        if _visible(landmarks, (name,), threshold)
    ]
    if nose.y <= config.BODY_BORDER_MARGIN or max(point.y for point in feet) >= (
        1 - config.BODY_BORDER_MARGIN
    ):
        raise user_error("FULL_BODY_NOT_VISIBLE")


def validate_arms(landmarks: Mapping[str, Landmark]) -> None:
    rules = (
        (("LEFT_SHOULDER", "RIGHT_SHOULDER"), config.SHOULDER_VISIBILITY),
        (("LEFT_ELBOW", "RIGHT_ELBOW"), config.ELBOW_VISIBILITY),
        (("LEFT_WRIST", "RIGHT_WRIST"), config.WRIST_VISIBILITY),
    )
    if not all(_visible(landmarks, names, threshold) for names, threshold in rules):
        raise user_error("ARMS_NOT_VISIBLE")


def validate_legs(landmarks: Mapping[str, Landmark]) -> None:
    rules = (
        (("LEFT_HIP", "RIGHT_HIP"), config.HIP_VISIBILITY),
        (("LEFT_KNEE", "RIGHT_KNEE"), config.KNEE_VISIBILITY),
        (("LEFT_ANKLE", "RIGHT_ANKLE"), config.ANKLE_VISIBILITY),
    )
    if not all(_visible(landmarks, names, threshold) for names, threshold in rules):
        raise user_error("LEGS_NOT_VISIBLE")


def validate_frontal_pose(
    landmarks: Mapping[str, Landmark], person_box: BoundingBox, image_width: int
) -> None:
    """얼굴·어깨·골반의 대칭과 폭을 이용한 rule-based 정면 추정이다."""
    face_visible = _visible(landmarks, ("NOSE",), config.NOSE_VISIBILITY) and _visible(
        landmarks, ("LEFT_EYE", "RIGHT_EYE"), config.EYE_VISIBILITY
    )
    shoulders_visible = _visible(
        landmarks, ("LEFT_SHOULDER", "RIGHT_SHOULDER"), config.SHOULDER_VISIBILITY
    )
    hips_visible = _visible(landmarks, ("LEFT_HIP", "RIGHT_HIP"), config.HIP_VISIBILITY)
    if not (face_visible and shoulders_visible and hips_visible) or person_box.width <= 0:
        raise user_error("NOT_FRONTAL")

    left_shoulder = landmarks["LEFT_SHOULDER"]
    right_shoulder = landmarks["RIGHT_SHOULDER"]
    shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * image_width
    hip_width_px = abs(landmarks["LEFT_HIP"].x - landmarks["RIGHT_HIP"].x) * image_width
    if (
        shoulder_width_px / person_box.width + _FLOAT_EPSILON < config.SHOULDER_WIDTH_RATIO
        or hip_width_px / person_box.width + _FLOAT_EPSILON < config.HIP_WIDTH_RATIO
        or abs(left_shoulder.visibility - right_shoulder.visibility)
        > config.SHOULDER_VISIBILITY_DIFF
    ):
        raise user_error("NOT_FRONTAL")
