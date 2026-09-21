# builder/test/runtime이 같은 Python 이미지와 경로를 쓰도록 한 곳에서 관리한다.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# =========================
# 1. Builder stage
# - 운영 의존성 설치
# - venv 생성
# =========================
FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# runtime에 그대로 복사할 운영 venv를 만든다.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# 운영 의존성은 해시 고정된 requirements.txt 기준으로 설치한다.
COPY requirements.txt .
RUN pip install --require-hashes -r requirements.txt

# =========================
# 2. Test stage
# - 코드 검사
# - 테스트 실행
# =========================
FROM builder AS test

# 테스트/린트용 dev 의존성은 test stage에만 설치한다.
COPY requirements-dev.txt .
RUN pip install --require-hashes -r requirements-dev.txt

WORKDIR /app

# pyproject.toml의 ruff/pytest 설정을 로컬 개발 환경과 동일하게 적용한다.
COPY pyproject.toml .
COPY app ./app
COPY tests ./tests

# 최종 이미지 빌드 시 테스트가 반드시 실행되도록 통과 표시 파일을 만든다.
RUN ruff check app tests \
 && pytest -q \
 && touch /tmp/.tests-passed

# =========================
# 3. Runtime stage
# - 운영용 최종 이미지
# - dev 의존성 없이 운영 venv와 app 코드만 포함
# =========================
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# FastAPI 애플리케이션은 root 권한이 필요 없으므로 non-root 사용자로 실행한다.
RUN groupadd --system --gid 1001 app \
 && useradd --system --uid 1001 --gid app app

# test stage가 성공해야만 최종 이미지가 만들어지도록 한다.
COPY --from=test /tmp/.tests-passed /tmp/.tests-passed

# 운영 의존성만 설치된 venv를 복사한다.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# 운영 이미지에는 테스트 코드를 포함하지 않는다.
COPY app ./app

USER app

# 문서화용 포트 선언. 실제 노출은 compose/ECS 설정에서 결정한다.
EXPOSE 8000

# workers는 v1 메모리 예산을 고려해 1개로 제한한다.
# graceful shutdown 시간은 compose/ECS stop timeout보다 짧게 맞춘다.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]