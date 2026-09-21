"""가상피팅 Provider 공통 타입과 인터페이스.

Service 는 이 모듈의 타입에만 의존하고, Pruna/Runware 같은 구체 Provider 는
주입받아 사용한다.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class FittingInput:
    """garment_image_urls 의 순서가 prompt 의 Garment N 순서와 같아야 한다."""

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
