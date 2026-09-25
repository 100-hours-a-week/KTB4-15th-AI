"""설정값 한 곳 모음. 모델명·상한·DB 접속 정보는 여기만 본다."""

import os

# 로컬 실행(`python -m app.main`)에서만 쓴다. 컨테이너는 Dockerfile 의 CMD 가
# uvicorn 을 직접 부르므로 이 값들을 보지 않는다.
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = os.getenv("RELOAD", "1") == "1"

# 서버 간 인증 (단계1 §9). 키가 비어 있으면 모든 요청을 거절한다.
# 로컬에서 인증 없이 띄우려면 AUTH_DISABLED=1 을 명시적으로 켠다.
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
AUTH_DISABLED = os.getenv("AUTH_DISABLED", "") == "1"

# 단계2 §4.2 — 8개 사용처 전부 gpt-5.6-luna
MODEL = os.getenv("CHAT_MODEL", "gpt-5.6-luna")

# 단계2 §4.8 — 상한은 "보이는 출력"이 아니라 "추론 + 출력"으로 잡아야 한다.
# analyze는 추론을 끄므로(effort=none) 출력 길이만 고려하면 된다.
ANALYZE_MAX_OUTPUT_TOKENS = int(os.getenv("ANALYZE_MAX_OUTPUT_TOKENS", "300"))
ANALYZE_REASONING_EFFORT = os.getenv("ANALYZE_REASONING_EFFORT", "none")

CHAT_MAX_OUTPUT_TOKENS = int(os.getenv("CHAT_MAX_OUTPUT_TOKENS", "600"))
SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("SUMMARY_MAX_OUTPUT_TOKENS", "300"))

# 검색 기반 추천은 Top 3 (단계1 §7, 단계5 §3.2)
TOP_K = int(os.getenv("TOP_K", "3"))

# 프롬프트에 남겨둘 추천 목록의 개수. 오래된 추천까지 계속 끌고 가면
# analyze 입력이 길어져 TTFT 예산을 먹는다.
RECOMMENDATION_MEMORY_TURNS = int(os.getenv("RECOMMENDATION_MEMORY_TURNS", "2"))

# 2026-09-17 결정 — 대화 내역은 Backend와 AI 양쪽이 보관한다.
# AI 쪽 보관은 LangGraph checkpointer가 맡으므로 운영에서는 이 DSN이 필수다.
# 비어 있으면 인메모리로 뜨며, 이는 로컬 개발용이다.
#
# 상품 데이터와 같은 AI PostgreSQL을 쓰되, checkpointer가 만드는 네 테이블이
# 상품 테이블과 섞이지 않도록 DSN에서 전용 스키마를 지정한다.
#   postgresql://user:password@host:5432/ai?options=-csearch_path%3Dlanggraph
CHECKPOINT_DSN = os.getenv("CHECKPOINT_DSN", "")

# 공용 AWS/S3 설정. boto3 자격증명은 EC2 IAM Role(로컬에서는 표준 AWS credential
# chain)에서 읽고, 코드나 환경변수에 access key를 직접 두지 않는다.
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET = os.getenv("S3_BUCKET", "")

# PostgreSQL pool 크기. 가상피팅 상품 조회는 짧은 SELECT만 수행하고 connection을 즉시
# 반환한다. 외부 모델 응답을 기다리는 동안 connection을 점유하지 않는다.
DATABASE_POOL_MIN_SIZE = int(os.getenv("DATABASE_POOL_MIN_SIZE", "1"))
DATABASE_POOL_MAX_SIZE = int(os.getenv("DATABASE_POOL_MAX_SIZE", "10"))
