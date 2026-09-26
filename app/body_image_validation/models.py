"""외부 라이브러리에 의존하지 않는 전신 이미지 내부 타입."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    origin_x: int
    origin_y: int
    width: int
    height: int


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float
    presence: float


@dataclass(frozen=True)
class PersonDetection:
    bounding_box: BoundingBox
    score: float


@dataclass(frozen=True)
class BodyImageValidationResult:
    s3_key: str
