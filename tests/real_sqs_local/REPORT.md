# 실제 SQS 로컬 통합 테스트: 송수신 전 보류

2026-10-18 KST. **정상·중복 통합 테스트는 실행하지 않았다.** 실제 SQS에 테스트 메시지를 보내기 전 대상 격리 점검에서 중단했다. 현재 결과를 실제 SQS 업무 흐름 검증 성공으로 보고하지 않는다.

## 중단 근거

`.env.sqs-local`에 지정된 큐 이름은 다음과 같다.

| 역할 | 큐 이름 | 최초 읽기 점검 | 이후 재점검 |
|---|---|---|---|
| BE 요청 | lookddak-dev-v2-be-fitting-request | visible/inflight/delayed 0/0/0 | **0/1/0** |
| AI 후처리 | lookddak-dev-v2-ai-fitting-response | 0/0/0 | 0/0/0 |
| BE 결과 | lookddak-dev-v2-be-fitting-response | 0/0/0 | 0/0/0 |

적재량은 SQS의 approximate 값이며 점검 시점의 관측이다. 이번 작업에서는 SendMessage·ReceiveMessage를 한 번도 호출하지 않았다. 따라서 관측된 요청 큐 처리 중 메시지 1개는 이번 테스트가 만든 메시지가 아니다. 큐의 단독 사용 및 다른 Consumer 부재를 확인할 수 없어 송수신을 보류했다. 해당 메시지의 본문·MessageId를 알아내기 위한 수신도 하지 않았다.

큐가 비어 있다는 사실만으로 다른 Consumer가 없다고 판단하지 않았다. 사용자에게 세 큐의 단독 사용 여부를 확인 요청했으며, 재점검에서 발견된 기존 처리 중 메시지도 전달했다. 확인 없이 진행하지 않는다.

[최초 API별 읽기 점검](preflight.json) · [재점검 적재량](queue-recheck.json) · [신규 실행기 preflight 결과](runs/sqs_02a094703975/summary.json)

## 실제 확인한 AWS 연결과 권한

- 양쪽 테스트용 프로필은 `lookddak-sqs-local`, 리전은 `ap-northeast-2`로 구성했다. 실제 STS GetCallerIdentity는 성공했고 IAM user 주체임을 확인했다. 전체 ARN·자격증명은 출력하지 않았다.
- 세 큐 모두 실제 boto3 `GetQueueAttributes`에 성공했다. 설정의 세 큐 URL은 동일 계정·리전이며 신규 preflight에서 요청 큐 ARN과 STS 계정·리전 일치도 확인했다.
- 세 큐 모두 `ListQueueTags`가 HTTP 403, `AccessDenied`로 실패했다. 큐 태그로 용도를 추가 확인할 수 없었다. IAM 변경은 하지 않았다. 이 권한 거부만으로 Send/Receive/Delete 권한이 없다고 판단하지 않았다.
- 세 큐의 visibility는 각각 120/300/60초이고 redrive 설정이 존재했다. 속성은 읽기만 했으며 변경하지 않았다.
- **SendMessage, ReceiveMessage, ChangeMessageVisibility, DeleteMessage 권한과 실제 동작은 미검증**이다. Java SDK의 실제 AWS 요청도 아직 실행하지 않았다.

신규 실행기 journal에는 `GetQueueAttributes`만 있다. 별도 읽기 점검에서는 STS GetCallerIdentity, 세 큐의 GetQueueAttributes 및 ListQueueTags만 호출했다.

## 실행 코드와 기존 상태 보존

| 저장소 | 브랜치 | 커밋 |
|---|---|---|
| AI | feature/virtual-fitting-async | e07dbd85207e54cc510d39d1d698a156797435e4 |
| BE | feature/fitting-mq-v2 | c1eb0202d06becfd384a03ec4e97693b721264e4 |

작업 전 AI 기존 파일587개·BE 기존 파일793개의 SHA-256과 Git staged/unstaged/untracked 상태를 기록했다. `.git` 내부·가상환경·node_modules는 파일 해시 대상에서 제외했다. 기존 소스·설정·SQL·테스트·이전 통합 테스트 결과·기존 build 결과는 포함됐다.

종료 확인 결과 **기존 파일 변경0·누락0, HEAD 및 기존 staged/unstaged/untracked 상태 동일**이다. [해시/상태 확인 결과](existing-files-final.json)

기존 AI 미추적 `app/db_migrations/`, `scripts/db/MIGRATIONS.md`, `tests/test_db_migrations.py`, `tests/local_be_ai/` 및 BE 미추적 Compose 파일은 이번 변경으로 집계하지 않았다. commit·push·stash·브랜치 전환을 하지 않았다. `.env.sqs-local`은 기존 Git ignore 규칙에 의해 제외돼 있었으며 `.gitignore`를 수정하지 않았다.

## 신규 구성과 확인 범위

신규 소스/문서:

- `tests/real_sqs_local/run.py`
- `tests/real_sqs_local/be.init.gradle`
- `tests/real_sqs_local/java/com/ktb/lookddak/integration/RealSqsLocalTest.java`
- `tests/real_sqs_local/README.md`
- 이 보고서 및 신규 baseline/preflight/증거 JSON 파일

신규 Java fixture는 **오프라인 compileTestJava 성공**(`BUILD SUCCESSFUL in 7s`). Python 소스 문법 확인도 통과했다. BE JUnit을 실행한 것은 아니다. 빌드/프로젝트 캐시를 신규 `build_check/`에 두어 기존 결과를 덮어쓰지 않았다.

구성은 기존 `tests/local_be_ai/`의 실제 BE 업무 서비스·Outbox·Consumer 호출 방식을 재사용한다. SQS는 실제 Java `DefaultSqsClient` 및 boto3 client로 바꾼 별도 신규 fixture다. SDK interceptor/event handler는 실제 응답을 관찰하며 가짜 성공 응답을 반환하지 않는다. 기존 메모리 큐를 사용하지 않는다. S3·Runware·LLM만 기존 Mock을 재사용한다.

하지만 **실제 Java 인증·SQS 송수신, BE Job/Outbox 생성, AI Worker 처리, Mock 완료/Webhook, 결과 저장·중복·DeleteMessage·Visibility·잔여 메시지 cleanup 경로는 아직 실행 검증되지 않았다.** 미실행 상태의 새 도구를 성공 결과와 혼동하지 않는다. 업무 결함이나 메시지 계약 오류가 재현된 것은 없다.

## 재실행

읽기 전용 점검:

```sh
cd /Users/ijaein/Documents/LooKDDak/ai
.venv/bin/python -B tests/real_sqs_local/run.py --preflight-only
```

기존 메시지와 다른 Consumer 여부를 사용자가 확인하고 **이번 테스트가 세 큐를 단독으로 쓰는 상태**가 된 후에만:

```sh
cd /Users/ijaein/Documents/LooKDDak/ai
.venv/bin/python -B tests/real_sqs_local/run.py --exclusive-queues-confirmed
```

플래그를 주더라도 기존 메시지 적재량이 0이 아니면 도구는 발행·수신 전에 중단한다. 기존 메시지를 임의로 삭제하거나 실패 기대값을 바꾸지 않는다. 실행/Mock 범위·예정 종료 처리는 [README](README.md)에 설명했다.

## 프로세스와 자원 정리

- 신규 BE/AI 업무 테스트 프로세스·Worker·Consumer·publisher·Watchdog은 시작하지 않았다.
- 오프라인 컴파일 프로세스와 읽기 점검 Python 프로세스는 종료됐다. 새 제어 서버18772 리스너도 없다.
- 이번 작업의 SQS 발행/수신 메시지0, 정상 DeleteMessage0, 종료 강제 DeleteMessage0. **이번 테스트의 잔여 메시지는 없다.** 관측된 기존 메시지1개는 삭제·수신·visibility 변경을 하지 않았다.
- 테스트용 DB 회원·상품·Job·Outbox·AI 행 생성0. DB·컨테이너·볼륨·스키마·마이그레이션·큐·IAM 변경0.
- 신규 `run.py`의 finally에는 증거 저장 → outcome 확정 → BE timer·AI lifespan·BE JVM 종료 → 이번 Job/MessageId만 main 큐·DLQ에서 정리 → 잔여/권한/시간 제한 보고 순서를 구현했다. **이 cleanup 경로는 이번 점검에서는 `no job created/no message sends possible`로 생략됐고, 실제 삭제 동작은 미검증**이다. [실제 종료 기록](runs/sqs_02a094703975/process-shutdown.json) · [실제 cleanup 기록](runs/sqs_02a094703975/cleanup.json)

실제 Runware·LLM·S3·Dev/운영 DB 호출, 배포·운영 활성화는 하지 않았다. 현재 막힌 것은 큐 대상 격리 확인이며 기존 코드 수정이 필요한 상황은 아니다.
