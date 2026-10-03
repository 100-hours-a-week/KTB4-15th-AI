"""가상피팅 API 의 처리 로직.

라우터는 HTTP 를 다루고, Service 호출과 응답 변환은 여기서 한다.
"""

from app.virtual_fitting.concurrency import FittingLimiters
from app.virtual_fitting.schemas import SyncFittingData, SyncFittingRequest, SyncFittingResponse
from app.virtual_fitting.service import VirtualFittingService


async def sync_fit(
    service: VirtualFittingService, request: SyncFittingRequest, limiters: FittingLimiters
) -> SyncFittingResponse:
    """VirtualFittingError 는 변환하지 않고 그대로 올린다. HTTP 응답으로 바꾸는 것은 라우터다."""
    result = await service.fit_with_limiters(request, limiters)
    return SyncFittingResponse(
        data=SyncFittingData(
            result_image_key=result.result_image_key,
            llm_title=result.llm_title,
            llm_comment=result.llm_comment,
        )
    )
