# 로컬 BE ↔ AI 비동기 통합 테스트

이 디렉터리의 테스트는 **실제 BE 업무 코드가 요청·Job·Outbox를 생성**한다. 최초 요청을 테스트가 큐에 직접 만들어 넣지 않는다. 중복 전달 재현에만 기존 BE payload를 다시 넣는다.

## 재현

전제: Docker Desktop의 기존 `lookddak-v2-local-test` MySQL(13306)·PostgreSQL(15432)이 실행 중이고, AI `.venv` 및 BE Gradle 의존성 캐시와 JDK 25가 준비되어 있어야 한다. 외부 API 키는 필요 없다. 18761·18762 포트는 비어 있어야 한다.

```sh
cd /Users/ijaein/Documents/LooKDDak/ai
/usr/bin/sandbox-exec -f tests/local_be_ai/localhost.sb \
  .venv/bin/python tests/local_be_ai/run.py
.venv/bin/python tests/local_be_ai/prepare_cleanup.py
```

`run.py`는 `.env`를 읽거나 변경하지 않는다. 앱 import 전에 환경을 제한하고 테스트용 더미 자격증명만 넣는다. macOS sandbox는 Python·libpq·Gradle·Java 자식 프로세스의 loopback 외 outbound 통신을 막는다. Python과 Java에서 각각 비로컬 socket 접속이 OS에 의해 거부되는지 먼저 검사한다. 실제 외부 서비스에 요청을 보내기 위한 검사가 아니다.

BE 내부 실행 명령은 다음과 같다. 스크립트가 테스트 전용 환경과 아래 경로를 전달한다.

```sh
./gradlew --offline --no-daemon --console=plain \
  -I /Users/ijaein/Documents/LooKDDak/ai/tests/local_be_ai/be-test.init.gradle test
```

스크립트는 `JAVA_HOME=/opt/homebrew/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home` 및 `JAVA_TOOL_OPTIONS=-Djava.net.preferIPv4Stack=true`를 설정한다. IPv4 옵션은 macOS sandbox 아래 Gradle loopback IPC에 필요하다. Gradle init script는 이 디렉터리의 Java 테스트 소스 경로와 테스트 필터만 주입한다. BE의 build.gradle·src·기존 설정은 수정하지 않는다. 기존 BE 통합 테스트들의 전체 삭제 setup이 실행되지 않도록 이 fixture 하나만 실행한다.

## 실제 사용하는 코드와 Mock 경계

| 구간 | 구성 |
|---|---|
| BE 요청 접수 | 실제 `FittingJobService.createFittingJob`: 회원·프로필·상품·후보 검증, 회원 요청 카운터, Job·JobProduct·Outbox 트랜잭션 |
| BE Outbox | 실제 `OutboxEventScheduler`, `OutboxEventClaimService`, `OutboxEventPublisher`, publication repository·서비스 |
| 공유 큐 | 127.0.0.1:18761 HTTP 서버. 기존 `FakeSqsBackend`의 수신·receipt·visibility·삭제·고장 모델 재사용. Java `SqsClient`와 AI boto3 경계 **모두 실제 HTTP로 이 서버에 접속** |
| AI | 실제 request/postprocess Worker, Submitter, async/sync Runware Provider, comment/title Provider, Webhook Router, result publisher, Watchdog, PostgreSQL Repository·lease·상태·비용 기록 |
| Runware·LLM | Provider의 `urlopen`만 기존 `FakeRunware`로 주입. 실제 Provider가 요청 JSON을 생성하고 Mock 응답을 해석. 접수 응답·완료 이미지 저장·Webhook 본문을 Mock으로 생성하고 실제 Router에 전달 |
| S3 | AI boto3 S3 경계는 메모리 객체 저장소. 원본/생성 이미지 bytes, head, put, presign 모델. BE 조회 시 presign 제공자만 로컬 URL로 대체 |
| BE 결과 반영 | 실제 `FittingResultConsumer` → handler → `FittingGenerationResultService` → MySQL Job·결과·알림 트랜잭션 |
| BE 상태 조회 | 실제 `FittingJobService.getFittingJobStatus` 호출. 별도의 로컬 테스트 제어 HTTP 서버(18762)로 실행 |

세 큐의 AWS 형식 URL은 **그대로 유지**한다. URL을 localhost로 바꾸지 않는다. 운영 URL 검증을 통과한 뒤 테스트 클라이언트 주입으로 로컬 서버에 연결한다. AWS endpoint/DNS 변경이나 운영 검증 완화는 없다.

장애 시나리오는 실제 Worker의 수신·처리·삭제 사이클과 실제 BE scheduler/consumer 사이클을 명시적으로 진행한다. 별도의 자동 시나리오는 **실제 FastAPI lifespan에서 네 루프를 시작**하며, 독립 BE 타이머가 실제 Outbox scheduler와 Consumer를 주기적으로 실행한다. 자동 시나리오에는 직접 Worker cycle 호출이 없다.

테스트 제어용 HTTP 서버는 인증 없이 loopback에만 바인딩되며 테스트 종료 시 함께 닫힌다. BE 설정 중 Consumer의 on/off·poll 값만 테스트 설정 Mock으로 주입한다. Consumer와 결과 저장 Repository는 실제 객체다.

## 데이터 격리 및 장애 주입

- AI: 이미 준비된 **public의 9개 테이블 정의**를 `CREATE TABLE ... LIKE public... INCLUDING ALL`로 매 실행 고유 `bia_<12자리 hex>` 스키마에 복제한다. limiter singleton만 복사하고, 고유 테스트 상품 한 개를 적재한다. `apply`나 변경 SQL 파일은 실행하지 않는다. 전후 public의 전체 행 스냅샷을 비교한다.
- BE: 지정된 `lookddak_be_test`만 연결한다. 처음 확인했을 때 빈 DB였으며 Hibernate `update`가 17개 테이블을 생성했다. `create/drop/truncate`로 기존 데이터를 초기화하지 않는다. 이후 실행은 각 실행의 고유 이메일·상품 코드·자동 생성 ID를 사용한다.
- 회원 이메일: `bia_<run>-<scenario>@local.invalid`. 회원별 프로필과 후보 상품을 저장하고 실제 요청 서비스를 호출한다. 원본 이미지 Key는 메모리 S3에 준비한 `users/10/body.png`이다.
- DB 저장 실패: `local_fail_<run>` 트리거가 해당 테스트 Job의 결과 INSERT에만 SQLSTATE 45000을 발생시킨다. MySQL binary log 설정 때문에 생성·제거에만 **Compose의 로컬 root 연결**을 사용한다. 일반 업무 트랜잭션은 `be_test`다. 전역 MySQL 설정·권한은 변경하지 않는다. 트리거 제거 후 실제 2초 visibility timeout을 기다려 같은 메시지를 재수신한다.
- Queue 고장: Outbox 발행 실패, 내부 큐의 확정 실패와 저장 후 timeout, 결과 큐 확정 실패·backoff·3회 소진을 주입한다. 실제 상태·카운터·재시도 로직을 바꾸지 않는다.
- LLM: 제목 호출을 3번 실패시켜 실제 3쌍(최대 6 HTTP 호출) 한도와 기본 문구 저장을 확인한다. Worker와 DB pool 재생성 후 카운터 유지·추가 호출 방지를 확인한다.

이 테스트는 이미지 생성 품질이나 실제 SQS SDK wire protocol·IAM·AWS 전달 보장, 외부 Provider 호환성, BE REST 인증·Controller 라우팅을 검증하지 않는다. Mock 큐 모델 위에서 양쪽 실제 업무 코드의 계약과 복구를 검증한다. body-image 모델 부재 및 검색 DB pool 비활성 경고는 이 피팅 흐름의 사전 생성 프로필/이미지 fixture와 무관하며 해당 기능은 호출하지 않는다.

## 결과와 증거

최종 실행 결과는 [REPORT.md](REPORT.md)에 기록한다. 실행별 증거는 `runs/<run>/`에 보존한다.

- `manifest.json`: 커밋, 큐, 상품 코드, 구성
- `checks.json`, `summary.json`: 검증별 성공 여부와 증거
- `*-created.json`: 실제 BE 서비스 응답과 저장된 원본 Outbox payload
- 시나리오별 `.json`: 양쪽 DB 상태·BE 상태 조회 결과·결과/알림·큐 보존 개수
- `queue.json`: 메시지별 body·receipt·수신 횟수·삭제 여부
- `runware.json`: Provider 요청/호출 기록, LLM 입력, Mock S3 객체 Key
- `ai-tables.json`: 테스트 스키마 최종 DB 행
- `public-before.json`, `public-after.json`: 기존 AI public 보존 증거
- `be-gradle.log`: BE JUnit 실행, 실제 INSERT 실패와 복구, Gradle 최종 결과

초기 환경 구성 실패 실행도 버리지 않는다. `failure.txt`와 `summary.json`이 원인을 기록한다. 통합 성공으로 집계하지 않는다.

## 이번 테스트 자원만 정리

```sh
.venv/bin/python tests/local_be_ai/prepare_cleanup.py
```

이 명령은 SQL과 [resource-inventory.json](resource-inventory.json)을 **생성만** 한다. DB 삭제를 실행하지 않는다. 각 실행 디렉터리의 `cleanup-be.sql`은 기록된 Job·회원·상품 ID만 삭제하며, `cleanup-ai.sql`은 해당 실행의 `bia_*` 스키마 하나만 삭제한다. 초기 실패 실행도 inventory에 포함된다. 적용 전에 파일 내용을 확인할 수 있다.

로컬 인증정보를 프로세스 명령줄에 넣지 않고, 연결된 로컬 DB 클라이언트에서 다음 파일들을 실행한다.

```text
MySQL 127.0.0.1:13306 / lookddak_be_test:
  runs/<대상 run>/cleanup-be.sql
PostgreSQL 127.0.0.1:15432 / lookddak_ai_test:
  runs/<대상 run>/cleanup-ai.sql
```

BE 테이블 구조는 보존한다. 기존 Compose 컨테이너·볼륨·AI public·마이그레이션 이력은 삭제하지 않는다. 큐 서버·S3 Mock은 메모리 자원이므로 프로세스 종료로 해제되며 메시지 덤프는 증거 파일에 남는다. 파일까지 정리하려면 증거 보관 후 이 테스트 디렉터리 또는 선택한 `runs/<run>` 디렉터리만 제거한다. 기존 `.env`, 마이그레이션 관련 미추적 파일, Compose 파일은 대상이 아니다.
