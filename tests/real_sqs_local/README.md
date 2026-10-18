# 실제 SQS를 사용하는 로컬 BE·AI 테스트

이 디렉터리는 2026-10-18 신규 추가 도구다. 기존 `tests/local_be_ai/`를 읽고 실제 BE service/scheduler/consumer 구성 및 기존 `tests/v2_integration_harness.py`의 S3·Runware·LLM Mock을 재사용한다. 기존 파일은 수정하지 않는다.

## 현재 실행 결과

**전체 흐름은 아직 실행하지 않았다.** [REPORT.md](REPORT.md)의 큐 격리 확인 문제가 해결되기 전에는 실제 송수신을 진행하지 않는다. 오프라인 Java 컴파일과 실제 STS·SQS 읽기 점검만 수행했다.

## 실행 명령

```sh
cd /Users/ijaein/Documents/LooKDDak/ai
.venv/bin/python -B tests/real_sqs_local/run.py --preflight-only
```

세 큐가 이번 테스트의 단독 대상이고 다른 Consumer가 없다는 사용자 확인을 받은 후에만 다음 명령을 사용한다. 큐에 기존 메시지가 있으면 스크립트는 송수신 전에 중단한다. 기존 메시지를 비우는 기능은 없다.

```sh
.venv/bin/python -B tests/real_sqs_local/run.py --exclusive-queues-confirmed
```

이 플래그는 사용자 확인을 기록하는 용도다. 현재 적재량 검사나 큐 계정·리전 검사를 우회하지 않는다. AWS 프로필은 양쪽 모두 `lookddak-sqs-local`, 큐/리전 설정은 기존 `.env.sqs-local`에서 읽는다. 파일은 `shlex`로 값만 파싱하며 source/eval/shell 명령 실행을 하지 않는다. 프로세스 환경만 설정한다. 실 API 키는 필요 없다.

빌드에는 신규 `be.init.gradle`만 전달한다. 기존 Gradle 파일을 바꾸지 않고 source path·test filter·build directory를 테스트 프로세스에서 지정한다. 기존 BE 테스트는 실행하지 않으며 `--project-cache-dir`도 실행별 신규 디렉터리로 지정한다. 따라서 기존 build/test-results와 이전 66개 검증 결과를 덮어쓰지 않는다. MySQL은 `ddl-auto=validate`; AI도 준비된 public 테이블에 테스트 행만 추가한다. DDL·마이그레이션·기존 SQL을 실행하지 않는다.

## 구성과 예정 검증

- BE의 실제 `FittingJobService.createFittingJob`이 회원/프로필/상품/후보 검증, Job·JobProduct·Outbox·회원 카운터를 저장한다. fixture는 회원·상품 등의 사전 데이터만 준비한다.
- 신규 Spring `@TestBean`으로 **실제 `DefaultSqsClient`**를 주입한다. 명시적 ProfileCredentialsProvider와 기존 SDK의 기본 AWS endpoint를 사용한다. `ExecutionInterceptor`는 결과를 바꾸지 않고 실제 API 응답·MessageId·HTTP 상태를 기록한다.
- BE의 실제 Outbox scheduler·claim/publication 서비스와 실제 결과 Consumer·handler·repository를 사용한다. 테스트 timer는 서비스 호출 시점만 제어한다.
- AI `app.main`의 실제 lifespan에서 request/postprocess Worker, result publisher, Watchdog을 시작한다. boto3 SQS client는 실제 AWS를 호출하며 botocore event handler가 요청/응답을 관찰한다. 메모리 SQS backend나 HTTP queue bridge를 쓰지 않는다.
- S3 boto3 경계, Runware 생성/LLM HTTP opener, BE 이미지 presign 경계만 기존 Mock으로 대체한다. 실제 Provider JSON 생성·해석과 PostgreSQL repository·상태·비용 기록은 재사용한다.
- Mock 접수 응답을 43초 지연한다. 실제 Worker의 기존 120초 lease·40초 keepalive 간격을 유지하면서 실제 SQS ChangeMessageVisibility 호출을 확인한다. 시도 횟수·재시도 정책은 변경하지 않는다.
- 정상 1건의 메타데이터·생성 이미지→코멘트→제목·AI PUBLISHED·BE COMPLETED·세 큐의 정상 DeleteMessage를 각각 확인한 뒤 동일 결과 payload를 실제 결과 큐에 한 번 더 발행한다. 결과·알림·기존 결과 필드·카운터 불변을 검증한다.
- Python은 Mock을 거치지 않는 AWS service 생성과 비 SQS 네트워크 연결을 차단한다. Java SDK interceptor도 실제 regional SQS endpoint 외 전송을 거부한다. 실제 Runware·LLM·S3는 호출하지 않는다.

## 종료 처리

`run.py`의 finally는 다음 순서를 따른다.

1. 실패 단계/종류·실제 SDK journal·양쪽 DB 상태를 저장하고 정상 테스트 결과를 확정한다.
2. BE timer를 멈추고, AI lifespan 종료를 기다려 Worker·publisher·Watchdog을 종료한다. BE 테스트 JVM도 종료한다.
3. 실제 수신 receipt를 프로세스 메모리에서 사용하거나 visibility 만료 후 재수신하여 이번 Job·MessageId의 메시지만 삭제한다. 설정된 DLQ도 같은 방식으로 확인한다.
4. 정상 처리 DeleteMessage와 종료 시 cleanup DeleteMessage를 별도 phase로 기록한다. 다른 메시지는 삭제하지 않는다. 권한 부족·제한 시간 종료·남은 대상은 cleanup 보고서에 남기며 성공으로 숨기지 않는다.

ReceiptHandle·자격증명은 journal/report에 기록하지 않는다. 큐/DB URL도 콘솔과 BE 로그에서 가린다. DB 증거는 삭제하지 않는다. 큐 Purge·생성·삭제·속성 변경·IAM 변경 기능이 없다.

SIGINT/SIGTERM은 중단 요청으로 받아 위 종료 처리로 진행한다. SIGKILL이나 OS 강제 종료 시 finally 실행은 보장할 수 없다. cleanup은 최대 360초 동안 수행하며 실패 시 남은 MessageId/큐 역할을 보고한다.

## 증거 파일

`baseline.json`은 작업 전 기존 파일 해시와 Git 상태를 저장한다(.git 내부·가상환경·node_modules 제외). 실행 결과는 신규 `runs/<run>/`에만 쓴다. `existing-files-verification.json`으로 기존 파일의 해시·HEAD·기존 staged/unstaged/untracked 상태 보존을 확인한다.

실제 SQS 정상 흐름·중복·종료 cleanup은 아직 실행 검증되지 않았다. 현재의 컴파일·preflight 결과와 예정 검증을 구분해야 한다.
