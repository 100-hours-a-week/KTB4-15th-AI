"""배포 liveness 확인용 endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str = "ok"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """외부 의존성을 검사하지 않고 프로세스와 HTTP server 생존 여부만 반환한다."""
    return HealthResponse()
