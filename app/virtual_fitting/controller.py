"""가상피팅 API 의 처리 로직.

라우터는 HTTP 를 다루고, Service 호출과 응답 변환은 여기서 한다.
"""

from app.virtual_fitting.schemas import SyncFittingData, SyncFittingRequest, SyncFittingResponse
from app.virtual_fitting.service import VirtualFittingService


def sync_fit(service: VirtualFittingService, request: SyncFittingRequest) -> SyncFittingResponse:
    """VirtualFittingError 는 변환하지 않고 그대로 올린다. HTTP 응답으로 바꾸는 것은 라우터다."""
    result = service.fit(request)
    return SyncFittingResponse(
        data=SyncFittingData(
            result_image_url=result.result_image_url,
            llm_title=result.llm_title,
            llm_comment=result.llm_comment,
        )
    )
