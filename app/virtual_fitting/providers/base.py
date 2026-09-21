"""가상피팅 Provider 공통 타입과 인터페이스."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class FittingInput:
    """garment_image_urls 의 순서가 prompt 의 N번째 garment image 순서와 같아야 한다."""

    person_image_url: str
    garment_image_urls: Sequence[str]
    prompt: str


@dataclass(frozen=True)
class FittingResult:
    result_image_url: str


@runtime_checkable
class FittingProvider(Protocol):
    """실패는 VirtualFittingError 하위 예외(FittingModelError, FittingTimeoutError)로 던진다."""
    def try_on(self, fitting_input: FittingInput) -> FittingResult: ...
