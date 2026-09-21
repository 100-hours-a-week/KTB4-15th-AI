"""서버 간 인증 (단계1 §9).

AI Server 는 사용자가 직접 호출하지 않고 Backend 를 통해서만 호출되는 내부 API 다.
Request Body 의 `user_id` 로 사용자를 인증하지 않으며, 신뢰할 수 있는 Backend 에서
온 요청인지만 확인한다. 소유권 검증은 Backend 가 한다.

**키가 설정되지 않으면 모든 요청을 거절한다.** 키가 없을 때 그냥 통과시키면, 설정을
빠뜨린 채 배포된 서버가 아무 검사 없이 열린 상태로 조용히 돌아간다. 로컬에서
인증 없이 띄우려면 AUTH_DISABLED=1 을 명시적으로 켠다.
"""

from fastapi import Header, HTTPException, status

from app.config import settings

SCHEME = "Bearer "


def verify_internal_key(authorization: str | None = Header(default=None)) -> None:
    """`Authorization: Bearer <key>` 를 확인한다.

    헤더 이름과 형식은 Backend 와 합의가 필요하다. 단계1 §9 는 "내부 API Key 또는
    Bearer Token" 까지만 정해 두었다.
    """
    if settings.AUTH_DISABLED:
        return

    if not settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        )

    if authorization is None or not authorization.startswith(SCHEME):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        )

    if authorization[len(SCHEME) :] != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        )


# 단계1 §6 의 403 forbidden 은 "인증은 되었지만 호출 권한이 없는 경우" 다.
# 지금은 Backend 하나가 공유 키 하나를 쓰므로 그 상태가 생기지 않는다.
# 호출 주체가 여러 개로 늘어나면 그때 키별 권한을 구분한다.
