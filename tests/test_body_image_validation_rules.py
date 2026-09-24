import pytest
from PIL import Image

from app.body_image_validation.brightness import brightness_warnings
from app.body_image_validation.exceptions import UserImageValidationError
from app.body_image_validation.models import BoundingBox, Landmark, PersonDetection
from app.body_image_validation.validation_rules import (
    select_single_person,
    validate_arms,
    validate_frontal_pose,
    validate_full_body,
    validate_legs,
)
from app.config import body_image_validation as config


def point(x=0.5, y=0.5, visibility=1.0, presence=1.0):
    return Landmark(x=x, y=y, visibility=visibility, presence=presence)


def valid_landmarks():
    names = {
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
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX",
    }
    landmarks = {name: point() for name in names}
    landmarks.update(
        {
            "NOSE": point(y=0.10),
            "LEFT_EYE": point(x=0.46, y=0.09),
            "RIGHT_EYE": point(x=0.54, y=0.09),
            "LEFT_SHOULDER": point(x=0.35, y=0.25),
            "RIGHT_SHOULDER": point(x=0.65, y=0.25),
            "LEFT_HIP": point(x=0.40, y=0.50),
            "RIGHT_HIP": point(x=0.60, y=0.50),
            "LEFT_HEEL": point(x=0.43, y=0.90),
            "RIGHT_HEEL": point(x=0.57, y=0.90),
            "LEFT_FOOT_INDEX": point(x=0.42, y=0.92),
            "RIGHT_FOOT_INDEX": point(x=0.58, y=0.92),
        }
    )
    return landmarks


def reason_from(call):
    with pytest.raises(UserImageValidationError) as exc_info:
        call()
    return exc_info.value.reason_code


@pytest.mark.parametrize(
    ("detections", "reason_code"),
    [([], "PERSON_NOT_FOUND"), ([PersonDetection(BoundingBox(0, 0, 100, 600), 0.9)] * 2, "MULTIPLE_PERSONS")],
)
def test_person_count_failures(detections, reason_code):
    assert reason_from(lambda: select_single_person(detections, 1000)) == reason_code


def test_person_height_ratio_boundary_054_055():
    too_small = [PersonDetection(BoundingBox(0, 0, 100, 540), 0.9)]
    accepted = [PersonDetection(BoundingBox(0, 0, 100, 550), 0.9)]

    assert reason_from(lambda: select_single_person(too_small, 1000)) == "PERSON_TOO_SMALL"
    assert select_single_person(accepted, 1000).height == 550


def test_wrist_visibility_boundary_044_045():
    landmarks = valid_landmarks()
    landmarks["LEFT_WRIST"] = point(visibility=0.44)
    assert reason_from(lambda: validate_arms(landmarks)) == "ARMS_NOT_VISIBLE"

    landmarks["LEFT_WRIST"] = point(visibility=0.45)
    validate_arms(landmarks)


def test_body_and_leg_thresholds_are_inclusive():
    landmarks = valid_landmarks()
    landmarks["LEFT_KNEE"] = point(visibility=0.65)
    landmarks["LEFT_ANKLE"] = point(visibility=0.60)
    landmarks["LEFT_HEEL"] = point(y=0.90, visibility=0.50)
    landmarks["LEFT_FOOT_INDEX"] = point(y=0.92, visibility=0.50)

    validate_full_body(landmarks)
    validate_legs(landmarks)


def test_one_hidden_leg_passes_body_span_but_fails_leg_visibility():
    landmarks = valid_landmarks()
    landmarks["LEFT_KNEE"] = point(visibility=0.64)

    validate_full_body(landmarks)
    assert reason_from(lambda: validate_legs(landmarks)) == "LEGS_NOT_VISIBLE"


def test_head_and_foot_border_margin_is_fail_fast():
    landmarks = valid_landmarks()
    landmarks["NOSE"] = point(y=config.BODY_BORDER_MARGIN)
    assert reason_from(lambda: validate_full_body(landmarks)) == "FULL_BODY_NOT_VISIBLE"

    landmarks = valid_landmarks()
    landmarks["RIGHT_FOOT_INDEX"] = point(y=1 - config.BODY_BORDER_MARGIN)
    assert reason_from(lambda: validate_full_body(landmarks)) == "FULL_BODY_NOT_VISIBLE"


def test_frontal_rule_width_boundaries(monkeypatch):
    landmarks = valid_landmarks()
    box = BoundingBox(100, 0, 400, 700)
    monkeypatch.setattr(config, "SHOULDER_WIDTH_RATIO", 0.45)
    monkeypatch.setattr(config, "HIP_WIDTH_RATIO", 0.30)

    validate_frontal_pose(landmarks, box, 600)

    monkeypatch.setattr(config, "SHOULDER_WIDTH_RATIO", 0.451)
    assert reason_from(lambda: validate_frontal_pose(landmarks, box, 600)) == "NOT_FRONTAL"


def test_frontal_rule_rejects_shoulder_visibility_asymmetry():
    landmarks = valid_landmarks()
    landmarks["LEFT_SHOULDER"] = point(x=0.35, visibility=1.0)
    landmarks["RIGHT_SHOULDER"] = point(x=0.65, visibility=0.64)

    assert (
        reason_from(lambda: validate_frontal_pose(landmarks, BoundingBox(0, 0, 400, 700), 600))
        == "NOT_FRONTAL"
    )


def test_brightness_boundary_59_60():
    box = BoundingBox(0, 0, 20, 20)

    assert brightness_warnings(Image.new("RGB", (20, 20), (59, 59, 59)), box) == [
        "IMAGE_TOO_DARK"
    ]
    assert brightness_warnings(Image.new("RGB", (20, 20), (60, 60, 60)), box) == []
