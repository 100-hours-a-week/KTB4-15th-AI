# 로컬 BE·AI 비동기 통합 검증 결과

2026-10-18 KST 최종 실행 `bia_a73d6bfd40fc`: **66/66 검증 통과**, BE JUnit 1건 통과, `BUILD SUCCESSFUL`. [요약](runs/bia_a73d6bfd40fc/summary.json) · [검증별 증거](runs/bia_a73d6bfd40fc/checks.json) · [BE 실행 로그](runs/bia_a73d6bfd40fc/be-gradle.log)

실제 BE 요청 서비스가 Job·상품 연결·Outbox를 트랜잭션으로 저장하고 Outbox가 발행한 payload를 AI가 처리했다. 자동 실행 시나리오는 실제 AI lifespan 루프와 독립 BE 주기 실행으로 결과 저장·상태 조회까지 완료했다. 최초 요청을 테스트가 대신 큐에 생성하지 않았다.

## 실행 코드와 보존한 변경

| 저장소 | 브랜치 | 실제 커밋 |
|---|---|---|
| AI `/Users/ijaein/Documents/LooKDDak/ai` | `feature/virtual-fitting-async` | `e07dbd85207e54cc510d39d1d698a156797435e4` |
| BE `/Users/ijaein/Documents/lookddak-be-local` | `feature/fitting-mq-v2` | `c1eb0202d06becfd384a03ec4e97693b721264e4` |

시작 전 AI의 미추적 `app/db_migrations/`, `scripts/db/MIGRATIONS.md`, `tests/test_db_migrations.py`와 BE의 미추적 `compose.local-integration.yaml`을 확인하고 보존했다. 종료 시 기존 추적 파일 diff는 양쪽 모두 없다. commit·push·stash·브랜치 전환을 하지 않았다. AI에 이 테스트 디렉터리만 추가했다. 대소문자가 다른 사용자 제공 AI 경로는 이 macOS 파일시스템에서 동일 디렉터리로 확인됐다.

## 최종 실제 DB 결과

[MySQL 직접 재조회](runs/bia_a73d6bfd40fc/be-db-verification.txt) · [PostgreSQL 직접 재조회](runs/bia_a73d6bfd40fc/ai-db-verification.txt)

`결과/알림`은 BE 행 개수이고, `S3/LLM/FB/전송`은 실제 AI DB의 각 단계 시도 카운터다.

| 시나리오 | Job | BE 상태 | AI 상태 / 결과 전송 상태 | 결과/알림 | S3/LLM/FB/전송 |
|---|---:|---|---|---|---|
| 정상 성공·중복·DB 저장 복구 | 14 | COMPLETED | COMPLETED / PUBLISHED | 1 / 1 | 1/1/0/1 |
| 업무 실패 | 15 | FAILED | FAILED / PUBLISHED | 0 / 1 | 0/0/0/1 |
| 확인된 fallback | 16 | COMPLETED | COMPLETED / PUBLISHED | 1 / 1 | 0/1/1/1 |
| Outbox·내부 큐·결과 큐 재시도 | 17 | COMPLETED | COMPLETED / PUBLISHED | 1 / 1 | 1/1/0/2 |
| LLM 시도 소진 | 18 | COMPLETED | COMPLETED / PUBLISHED | 1 / 1 | 1/3/0/1 |
| 결과 전송 시도 소진 | 19 | PENDING | RECEIVED / RETRY_EXHAUSTED | 0 / 0 | 0/0/0/3 |
| 자동 Outbox·AI lifespan·BE Consumer | 20 | COMPLETED | COMPLETED / PUBLISHED | 1 / 1 | 1/1/0/1 |

일곱 Outbox 모두 `PUBLISHED`이고, Job 17만 `retry_count=1`, 나머지는 0이다. 모든 테스트 회원의 `fitting_request_count=1`이다.

## 시나리오별 확인

1. **정상 성공**: 실제 요청 생성 직후 BE Job·Outbox는 PENDING. AI 접수 후 AI는 WAITING_PRUNA, 같은 시점 BE는 PENDING. Mock Runware가 접수 응답을 반환하고 결과 이미지 bytes를 저장한 후 실제 Webhook Router를 호출했다. 후처리·결과 큐·BE 저장·실제 상태 조회가 COMPLETED까지 연결됐다. [증거](runs/bia_a73d6bfd40fc/success.json)
2. **메타데이터**: BE Outbox의 `requested_at`을 timezone 포함 동일 시각으로 AI가 보존했다. `fitting_job_id`, 상품 코드 문자열 `800016499632` 또는 실패 상품 `800016499633`도 모든 요청에서 일치했다. 코드는 최초 BE 상품에서 생성됐다. [원본 Outbox](runs/bia_a73d6bfd40fc/success-created.json)
3. **이미지 → 코멘트 → 제목**: 실제 comment Provider 입력에 생성 결과 이미지 Key가 포함되고, 다음 title Provider 입력은 실제 comment 응답을 포함했다. 성공 title은 `편안한 데일리 캐주얼`. BE `result_image_key`, `outfit_name`, `ai_comment`는 AI payload의 Key·`llm_title`·`llm_comment`와 정확히 같았다. Mock 문구의 품질은 검증 대상이 아니다. [Provider 요청 기록](runs/bia_a73d6bfd40fc/runware.json)
4. **업무 실패**: BE에는 존재하지만 AI catalog에 없는 테스트 상품을 요청했다. AI가 `PRODUCT_NOT_FOUND`와 `요청한 상품을 찾을 수 없습니다.`를 발행하고 BE가 동일 오류 코드·메시지를 저장했다. 이미지 결과 0, 실패 알림 1. [증거](runs/bia_a73d6bfd40fc/business-failure.json)
5. **fallback**: 실제 async Provider가 `providerRateLimitExceeded` 거절 응답을 해석했다. 실제 RUN_FALLBACK → sync Provider → 이미지 다운로드·저장·후처리를 거쳐 `virtual-fitting/results/16/pruna-sync.jpg`가 BE에 저장됐다. fallback 1회, async/fallback task UUID 분리, 중복 요청과 Worker 재생성에도 카운터·Key·생성 횟수 유지. [증거](runs/bia_a73d6bfd40fc/fallback.json)
6. **중복 요청·결과·내부 메시지**: 생성 진행 중과 완료 후에 실제 BE 원본 payload를 중복 전달했다. async 접수 중복 없음. 동일 최종 결과를 두 번 더 전달해도 BE 결과·알림 각 1, 회원 카운터 1, AI 저장 상태·카운터 전체가 동일했다. 내부 메시지 중복에도 LLM 작업 한 번. [증거](runs/bia_a73d6bfd40fc/success-duplicates.json)
7. **BE Outbox 발행 실패**: 공유 큐가 확정 실패를 반환하면 Outbox PENDING·retry_count=1로 보존됐다. 실제 backoff 후 재발행해 PUBLISHED. [실패 시점](runs/bia_a73d6bfd40fc/outbox-failure.json)
8. **내부 큐 발행 실패**: 첫 발행 throttling 실패는 Webhook 503, DB next_action=POSTPROCESS 보존, 큐 저장 없음. 다음 발행은 저장 후 timeout으로 다시 503. 세 번째 동일 Webhook 재전달은 200. 실제 내부 메시지 두 개에도 LLM 한 번, task 비용 중복 없음. [검증 기록](runs/bia_a73d6bfd40fc/checks.json)
9. **결과 큐 발행 실패**: 확정 실패 후 이미지·성공 payload·카운터 1을 보존했다. 실제 5초 backoff 후 동일 payload로 발행 성공, 전송 카운터 2. [실패 시점](runs/bia_a73d6bfd40fc/result-publish-failure.json) · [복구 후](runs/bia_a73d6bfd40fc/queue-retries.json)
10. **BE 실제 DB 저장 실패**: 테스트 Job 한 개의 결과 INSERT를 MySQL 트리거로 거부했다. 실제 트랜잭션 rollback으로 BE PENDING·결과0·알림0이며 메시지는 삭제되지 않았다. 트리거 제거와 실제 visibility 2초 경과 뒤 **같은 메시지 receive_count=2**로 복구, 결과·알림 각1. 장애 트리거는 모두 제거됐다. [실패 시점](runs/bia_a73d6bfd40fc/success-save-failure.json)
11. **횟수 한도**: 제목 응답 3회 실패 시 comment/title 3쌍만 호출하고 기본 title `가상 피팅 결과`, comment `선택하신 상품으로 완성한 가상 피팅 결과입니다.`를 저장했다. 이미지 성공은 유지됐다. pool·Worker 재생성 및 중복 수신에도 LLM 카운터3 유지, 7번째 호출 없음. [증거](runs/bia_a73d6bfd40fc/llm-limit.json)
12. **결과 전송 소진**: 5초·30초 backoff를 실제로 기다리며 세 번 실패시켰다. pool·Worker 재생성과 추가 발행 사이클에도 4번째 발행 없음. AI RETRY_EXHAUSTED와 Watchdog `result_publish_exhausted` 알림이 남고 BE는 PENDING. [증거](runs/bia_a73d6bfd40fc/result-limit.json)
13. **자동 실행**: BE timer가 실제 Outbox scheduler·Consumer를 실행하고, 실제 AI `app.main` lifespan이 request/postprocess/result-publisher/Watchdog를 시작했다. 테스트가 직접 각 단계 cycle을 부르지 않고 Job 20이 COMPLETED까지 도달했다. [증거](runs/bia_a73d6bfd40fc/automatic-lifespan.json)

## 큐·상태·비용 계약

공유 큐는 실제 두 프로세스가 HTTP로 접속하는 테스트 큐 서버다. AWS 형식 큐 URL 검증은 그대로 유지했고 양쪽 SDK 경계에서만 클라이언트를 주입했다. S3 및 Runware·LLM은 Mock이며 실제 Provider의 JSON 생성·해석과 실제 DB 관리 코드를 재사용했다. 구체적 Mock 경계·실행 명령은 [README](README.md)에 있다.

최종 [큐 덤프](runs/bia_a73d6bfd40fc/queue.json): 요청 큐 메시지 총10개 중 미삭제2, 내부 큐9개 모두 삭제, 결과 큐8개 모두 삭제. 미삭제 요청은 Job 15와 19다. Job 15는 실패 결과를 나중에 발행한 뒤 기존 요청의 visibility가 아직 남은 상태에서 테스트가 종료돼, 최종 상태 확인용 재수신 전이다. Job 19는 결과 발행 소진으로 보존된다. 이 두 메시지도 메모리 큐 프로세스 종료로 해제됐으며 덤프에 상태를 남겼다. 큐가 비었다고 보고하지 않는다.

BE 상태명을 AI 상태명과 동일하게 해석하지 않았다. AI의 `PUBLISHED`는 큐 접수 성공이며 BE DB 반영 완료가 아니다. 실제 저장 실패 시 이미 AI PUBLISHED여도 BE는 PENDING이고, 큐 재수신으로만 복구됨을 확인했다. Job 19의 실패 payload는 존재하지만 발행 완료가 아니므로 AI status도 RECEIVED에 남는다.

실제 비용 Repository에는 고유 task 5개가 `0.023 + 0.015 + 0.023 + 0.023 + 0.023 = 0.107`로 기록됐다. `usage_amount.recorded_amount=amount=0.107`, `legacy_amount=0`, `reconciled_amount=0`. 이 금액은 **Mock 응답의 테스트 값**이며 실제 과금은 없다. async 접수 시도5(확정 거절1 포함), sync fallback1, comment7·title7 호출이다. 중복 Webhook과 메시지로 비용이 증가하지 않았다. S3 단계≤2·LLM 단계≤3·fallback≤1·전송≤3을 모든 Job에서 확인했다.

기존 AI public 9개 테이블의 모든 행은 [실행 전](runs/bia_a73d6bfd40fc/public-before.json)·[실행 후](runs/bia_a73d6bfd40fc/public-after.json) 동일하다. migration apply·이력 등록을 하지 않았다. 실제 외부 API·Dev/운영 DB·배포·운영 활성화는 실행하지 않았다.

## 발견 사항과 검증 범위

검증한 경로에서 **업무 기능 결함이나 양쪽 payload 계약 불일치는 재현되지 않았다**. 기존 구현에 맞추려고 기대값을 변경하거나 업무 로직을 수정하지 않았다. 결과 발행 3회 소진 시 BE가 PENDING에 남는 것은 확인된 현재 복구 계약이며 자동으로 BE FAILED가 되는 것으로 가정하지 않았다. Watchdog 알림 후 운영자 복구가 필요한 상태라는 사실을 증거에 남겼다.

초기 실패는 테스트 환경 구성 문제였다: macOS sandbox 아래 Gradle IPv6 IPC, 테스트가 불변 설정 클래스에 setter를 사용한 컴파일 오류, 로컬 MySQL trigger 생성의 SUPER 요구. 각각 테스트 전용 JVM 옵션·설정 주입·Compose-local root의 트리거 전용 사용으로 해결했다. 중간 실행에는 Consumer 초기 scheduled tick이 전송 Mock 준비 전에 수행된 테스트 조립 race도 있었고, 최종 fixture는 준비 전 disabled 설정을 보게 수정했다. 업무 코드는 바꾸지 않았다. 초기 실패 증거도 보존했다.

실제 AWS transport/IAM, 실제 Provider 응답의 현행 호환성·이미지 품질, BE REST 인증·Controller 경로는 범위 밖이다. 최종 검증은 지정된 로컬 DB와 테스트용 공유 큐 위의 실제 양쪽 업무 코드에 한정된다.

## 변경 파일과 정리

추가 파일: `run.py`, `be-test.init.gradle`, `localhost.sb`, `java/com/ktb/lookddak/integration/LocalBeAiBridgeTest.java`, `prepare_cleanup.py`, `README.md`, 이 보고서, `resource-inventory.json`, 실행별 증거·cleanup SQL. 기존 앱·운영 설정·큐 검증 코드는 변경하지 않았다.

[전체 자원 목록](resource-inventory.json)은 실패 실행까지 포함한다. 이번 작업 전체에서 BE Job·회원1~20, 고유 상품7개, AI 전용 스키마7개가 생성됐다. 최종 실행은 Job·회원14~20, 상품ID6·7, 스키마 `bia_a73d6bfd40fc`다. BE 테이블17개는 처음 빈 로컬 테스트 DB에 생성했다. DB 행과 스키마는 재검토를 위해 보존했다.

큐 서버·BE 제어 서버·AI Worker·BE 테스트 JVM은 종료됐고 18761·18762 리스너가 남아 있지 않다. 장애 트리거는 제거됐고 기존 DB 컨테이너2개·볼륨은 보존했다. [정리 방법](README.md#이번-테스트-자원만-정리)의 SQL은 각 실행의 기록된 ID·스키마만 대상으로 한다. **전체 DB 초기화나 Compose down -v는 필요하지 않다.**
