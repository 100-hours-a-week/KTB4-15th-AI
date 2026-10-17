"""설정값 한 곳 모음. 모델명·상한·DB 접속 정보는 여기만 본다."""

import os
from decimal import Decimal

# 로컬 실행(`python -m app.main`)에서만 쓴다. 컨테이너는 Dockerfile 의 CMD 가
# uvicorn 을 직접 부르므로 이 값들을 보지 않는다.
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = os.getenv("RELOAD", "1") == "1"

# 서버 간 인증 (단계1 §9). 키가 비어 있으면 모든 요청을 거절한다.
# 로컬에서 인증 없이 띄우려면 AUTH_DISABLED=1 을 명시적으로 켠다.
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
AUTH_DISABLED = os.getenv("AUTH_DISABLED", "") == "1"

# 전신 이미지 검증을 Modal worker 에 맡길 때(BODY_IMAGE_VALIDATION_MODE=remote) 쓰는 인증 정보.
# Modal proxy auth token(Modal-Key / Modal-Secret 헤더)과 worker 의 INTERNAL_API_KEY(Bearer).
BODY_IMAGE_REMOTE_PROXY_KEY = os.getenv("BODY_IMAGE_REMOTE_PROXY_KEY", "")
BODY_IMAGE_REMOTE_PROXY_SECRET = os.getenv("BODY_IMAGE_REMOTE_PROXY_SECRET", "")
BODY_IMAGE_REMOTE_API_KEY = os.getenv("BODY_IMAGE_REMOTE_API_KEY", "")

# 2026-10-07 결정 — 채팅 LLM 은 Runware 의 OpenAI 호환 API 로 부른다(키는 RUNWARE_LLM_API_KEY).
# 모델명은 Runware AIR 형식이다. 임베딩은 Runware 가 제공하지 않아 OpenAI 그대로다.
# gpt@6-luna 는 되묻기 직후 라우팅(103/114)과 첫 말풍선 지연(최댓값 5.19·6.26s)에서 합격선을
# 넘지 못해 5.6 을 쓴다 (2026-10-07 로컬 측정).
MODEL = os.getenv("CHAT_MODEL", "openai:gpt@5.6-luna")
LLM_BASE_URL = "https://api.runware.ai/v1"
# 채팅 LLM(Runware) HTTP timeout(초). 지정하지 않으면 SDK 기본값 600초라 Runware 가 멈추면
# 사용자는 10분 동안 빈 말풍선을 본다. Runware + 5.6 의 첫 토큰까지 최대 3.81초(2026-10-07
# 로컬 측정)에 여유를 두어 6초로 잡았다 (2026-10-12 결정). 임베딩 클라이언트에는 걸지 않았다.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "6"))
# SDK 자동 재시도 횟수. 기본값 2회면 Runware 가 멈췄을 때 6 + 0.5 + 6 + 1 + 6 = 최악 19.5초를
# 기다린 뒤에야 error 를 보낸다. 1회로 줄여 6 + 0.5 + 6 = 최악 12.5초 (2026-10-12 결정).
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))

# 단계2 §4.8 — 상한은 "보이는 출력"이 아니라 "추론 + 출력"으로 잡아야 한다.
# analyze는 추론을 끄므로(effort=none) 출력 길이만 고려하면 된다.
ANALYZE_MAX_OUTPUT_TOKENS = int(os.getenv("ANALYZE_MAX_OUTPUT_TOKENS", "300"))
ANALYZE_REASONING_EFFORT = os.getenv("ANALYZE_REASONING_EFFORT", "none")

CHAT_MAX_OUTPUT_TOKENS = int(os.getenv("CHAT_MAX_OUTPUT_TOKENS", "600"))
SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("SUMMARY_MAX_OUTPUT_TOKENS", "300"))

# 검색 기반 추천은 Top 3 (단계1 §7, 단계5 §3.2)
TOP_K = int(os.getenv("TOP_K", "3"))

# 검색 질의와 상품 설명을 같은 모델로 임베딩해야 거리가 의미를 갖는다 (단계5 §3.5).
# 차원은 scripts/db/schema.sql 의 embedding vector(1536) 과 한 몸이다. 적재 스크립트
# scripts/pipeline/embed_products.py 도 이 값을 읽는다.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

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

# V2 비동기 가상피팅 Presigned URL 유효기간(초). 버킷은 위 S3_BUCKET 하나를 그대로 쓴다.
# - 사용자 원본 GET: Pruna 호출 직전에 발급한다. Pruna 가 실제로 읽을 때까지 유효해야 한다.
# - 결과 PUT(uploadEndpoint): Pruna 가 생성을 마치고 업로드할 때까지 유효해야 한다.
# - 결과 GET: OpenAI 후처리에 넘길 때 발급한다.
# IAM Role(임시 자격증명)으로 서명하므로 실제 만료는 이 값과 자격증명 만료 중 이른 쪽이다.
VIRTUAL_FITTING_USER_IMAGE_URL_EXPIRES_SECONDS = int(
    os.getenv("VIRTUAL_FITTING_USER_IMAGE_URL_EXPIRES_SECONDS", "1800")
)
VIRTUAL_FITTING_RESULT_UPLOAD_URL_EXPIRES_SECONDS = int(
    os.getenv("VIRTUAL_FITTING_RESULT_UPLOAD_URL_EXPIRES_SECONDS", "3600")
)
VIRTUAL_FITTING_RESULT_IMAGE_URL_EXPIRES_SECONDS = int(
    os.getenv("VIRTUAL_FITTING_RESULT_IMAGE_URL_EXPIRES_SECONDS", "600")
)

# V2 비동기 가상피팅 SQS Standard 큐 URL. 리전은 AWS_REGION 과 같아야 한다(app/clients/sqs.py 가 검사).
# 큐·DLQ 생성과 IAM 설정은 Cloud 담당이다. 비어 있으면 그 큐를 쓰는 시점에 SQS_CONFIG_ERROR.
# - be-fitting-request : BE 가 발행, AI 요청 Worker 가 수신·삭제·Visibility 연장
# - ai-fitting-response: AI Webhook·요청 Worker·Watchdog 이 발행, AI 후처리 Worker 가 수신
# - be-fitting-response: AI 가 최종 결과 발행, BE 결과 Worker 가 소비
BE_FITTING_REQUEST_QUEUE_URL = os.getenv("BE_FITTING_REQUEST_QUEUE_URL", "")
AI_FITTING_RESPONSE_QUEUE_URL = os.getenv("AI_FITTING_RESPONSE_QUEUE_URL", "")
BE_FITTING_RESPONSE_QUEUE_URL = os.getenv("BE_FITTING_RESPONSE_QUEUE_URL", "")

# PostgreSQL pool 크기. 가상피팅 상품 조회는 짧은 SELECT만 수행하고 connection을 즉시
# 반환한다. 외부 모델 응답을 기다리는 동안 connection을 점유하지 않는다.
DATABASE_POOL_MIN_SIZE = int(os.getenv("DATABASE_POOL_MIN_SIZE", "1"))
DATABASE_POOL_MAX_SIZE = int(os.getenv("DATABASE_POOL_MAX_SIZE", "10"))

# 검색 기반 추천 전용 async pool 크기. 검색은 한 턴에 짧은 SELECT 한 번이고, 임베딩 API 를
# 기다리는 동안에는 connection 을 잡지 않는다. 검색이 실제로 많아지면 값만 올린다.
SEARCH_DB_POOL_MIN_SIZE = int(os.getenv("SEARCH_DB_POOL_MIN_SIZE", "1"))
SEARCH_DB_POOL_MAX_SIZE = int(os.getenv("SEARCH_DB_POOL_MAX_SIZE", "5"))

# 2026-09-28 결정 — 가상피팅 Runware 하루 사용액 상한(USD). 코드에 고정한다.
# "하루"는 UTC 날짜라 KST 09:00 에 초기화된다 (scripts/db/schema.sql 의 usage_amount).
FITTING_DAILY_BUDGET_USD = Decimal(5)
# 우리 합계를 Runware 집계(getUsageActivity)로 보정하는 간격. 스케줄러 없이, 이 시간이 지난 뒤
# 들어온 첫 피팅 요청이 보정한다.
FITTING_USAGE_RECONCILE_SECONDS = 600

# 가상피팅 동시 실행 제한 (app/virtual_fitting/concurrency.py). 값 검증은 limiter 생성자가 하고,
# limiter 는 lifespan 에서 만들므로 잘못된 값이면 서버가 뜨지 않는다.
# 메모리 limiter 라 프로세스마다 따로 적용된다. uvicorn worker 나 인스턴스가 늘면 전체 한도도 그만큼 는다.
# - VTON: 실제 Runware try-on 요청 동시 실행 수, 그 슬롯을 기다릴 수 있는 요청 수, 기다리는 최대 시간(초).
#   실행과 대기가 모두 차면 즉시 429 FITTING_SERVER_BUSY, 대기 시간을 넘기면 429 FITTING_QUEUE_TIMEOUT.
# - LLM: 실제 Runware LLM(comment / title) 요청 동시 실행 수와, 그 슬롯을 기다리는 최대 시간(초).
#   대기 수는 제한하지 않는다. 대기 시간은 LLM HTTP timeout 과 별개이고, 넘기면 LLM 을 부르지 않고
#   fallback comment / title 로 정상 응답한다. comment 와 title 의 슬롯 대기에 각각 적용된다.
VIRTUAL_FITTING_VTON_CONCURRENCY = int(os.getenv("VIRTUAL_FITTING_VTON_CONCURRENCY", "4"))
VIRTUAL_FITTING_VTON_QUEUE_SIZE = int(os.getenv("VIRTUAL_FITTING_VTON_QUEUE_SIZE", "4"))
VIRTUAL_FITTING_VTON_QUEUE_TIMEOUT_SECONDS = float(
    os.getenv("VIRTUAL_FITTING_VTON_QUEUE_TIMEOUT_SECONDS", "30")
)
VIRTUAL_FITTING_LLM_CONCURRENCY = int(os.getenv("VIRTUAL_FITTING_LLM_CONCURRENCY", "4"))
VIRTUAL_FITTING_LLM_SLOT_WAIT_TIMEOUT_SECONDS = float(
    os.getenv("VIRTUAL_FITTING_LLM_SLOT_WAIT_TIMEOUT_SECONDS", "15")
)

# V2 비동기 가상피팅 요청 Worker (app/virtual_fitting/request_worker.py). 별도 실행 명령 없이 FastAPI lifespan 이
# 시작·종료한다(구조는 구현됨). 기본은 꺼짐: 꺼져 있으면 Worker 를 만들지 않으므로 Worker 용 SQS client·DB pool
# 조립도 하지 않는다.
# 주의: 7단계에는 Pruna 호출(8단계)과 결과 발행(14단계)이 없어 메시지를 거의 삭제하지 않는다. 켜면 같은 메시지가
# Visibility Timeout 마다 다시 오고 maxReceiveCount(3) 뒤 DLQ 로 갈 수 있다. 운영 활성화는 Pruna 제출·Webhook·
# 후처리·결과 발행·복구 경로를 함께 확인한 뒤 정한다(8단계 완료만으로 켜지 않는다).
VIRTUAL_FITTING_REQUEST_WORKER_ENABLED = (
    os.getenv("VIRTUAL_FITTING_REQUEST_WORKER_ENABLED", "false").strip().lower() == "true"
)
# 작업 실행 권한(DB lease)과 처리 중인 메시지의 숨김 시간(SQS Visibility Timeout)은 따로 연장한다(초).
# 연장은 처리 단계 사이에서만 한다. 긴 외부 호출 하나가 진행되는 동안 주기적으로 연장하지는 않는다.
VIRTUAL_FITTING_REQUEST_LEASE_SECONDS = float(
    os.getenv("VIRTUAL_FITTING_REQUEST_LEASE_SECONDS", "120")
)
VIRTUAL_FITTING_REQUEST_VISIBILITY_SECONDS = int(
    os.getenv("VIRTUAL_FITTING_REQUEST_VISIBILITY_SECONDS", "120")
)
# 종료할 때 Worker 가 스스로 끝나기를 기다리는 시간(초). 전체 종료 시간의 상한이 아니다. 넘으면 수신 대기(long
# polling)는 버리지만, DB·S3·Runware 를 쓰는 처리 단계는 버리지 않고 끝날 때까지 추가로 기다린다(제한 없음).
VIRTUAL_FITTING_REQUEST_SHUTDOWN_TIMEOUT_SECONDS = float(
    os.getenv("VIRTUAL_FITTING_REQUEST_SHUTDOWN_TIMEOUT_SECONDS", "30")
)
