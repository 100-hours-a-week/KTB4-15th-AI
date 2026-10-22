"""가상피팅 Provider 공통 타입과 인터페이스."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class FittingInput:
    """garment_image_urls 의 순서가 prompt 의 N번째 garment image 순서와 같아야 한다."""

    person_image_url: str
    garment_image_urls: Sequence[str]
    prompt: str


@dataclass(frozen=True)
class FittingResult:
    # 외부(Runware CDN) 임시 URL. 저장·로그하지 않고 결과 이미지를 S3 로 옮길 때만 쓴다.
    result_image_url: str
    # 이 호출로 차감된 금액(USD). 비용을 알려주지 않는 Provider 는 None 이다(0 으로 보지 않는다).
    cost: Decimal | None = None


@runtime_checkable
class FittingProvider(Protocol):
    """실패는 VirtualFittingError 하위 예외(FittingModelError, FittingTimeoutError)로 던진다."""
    def try_on(self, fitting_input: FittingInput) -> FittingResult: ...
