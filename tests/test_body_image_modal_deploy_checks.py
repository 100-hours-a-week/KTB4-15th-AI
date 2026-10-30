import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("modal")
CHECKS_FILE = Path(__file__).resolve().parents[1] / "deploy/modal/deploy_checks.py"
if not CHECKS_FILE.is_file():
    pytest.skip("deploy/ 가 없는 환경(Docker test stage)", allow_module_level=True)

spec = importlib.util.spec_from_file_location("deploy_checks", CHECKS_FILE)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)

SINCE = datetime(2026, 10, 25, 9, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def clean_deploy_env(monkeypatch, tmp_path):
    """테스트가 개발자 셸·저장소 .env 의 값에 기대지 않게 한다."""
    import os

    for name in list(os.environ):
        if name.startswith("BODY_IMAGE_WORKER_") or name in dc.MODAL_TARGET_OVERRIDES:
            monkeypatch.delenv(name)
    monkeypatch.setattr(dc, "DOTENV_FILE", tmp_path / "missing.env")

# 4-C: /health 로 GPU snapshot 생성(정상)
WARMUP_OK = """\
2026-10-25 18:35:34+09:00 ta-01M5S6MM756AKMZ7HHM8EB4B5R Creating GPU memory snapshot for Function.
2026-10-25 18:35:36+09:00 ta-01M5S6MM756AKMZ7HHM8EB4B5R body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-25 18:35:48+09:00 ta-01M5S6MM756AKMZ7HHM8EB4B5R body image worker lifecycle: snap=True done start=1792920936.974 end=1792920948.535 gpu_snapshot=True gpu=Tesla T4 warmup_ms={'preprocess': 149.0, 'person_pose': 59.3, 'rembg': 1190.0, 'png': 71.4}
2026-10-25 18:35:50+09:00 ta-01M5S6MM756AKMZ7HHM8EB4B5R body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-25 18:35:50+09:00 ta-01M5S6MM756AKMZ7HHM8EB4B5R body image worker lifecycle: snap=False done start=1792920950.361 end=1792920950.704 runtime=ready warmup_ms={'preprocess': 149.0, 'person_pose': 59.3, 'rembg': 1190.0, 'png': 71.4}
2026-10-25 18:35:51+09:00 ta-01M5S6MM756AKMZ7HHM8EB4B5R     GET /health -> 200 OK  (duration: 19.0 s, execution: 105.8 ms)
""".splitlines()

# 4-C: 컨테이너 0 이후 검증 요청(HIT, 진단 모드 profile 포함)
VALIDATION_HIT = """\
2026-10-25 18:37:12+09:00 ta-01M5S6QKJGMY9TS6F0MW2B6X9R Restoring Function from memory snapshot.
2026-10-25 18:37:13+09:00 ta-01M5S6QKJGMY9TS6F0MW2B6X9R body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-25 18:37:13+09:00 ta-01M5S6QKJGMY9TS6F0MW2B6X9R body image worker lifecycle: snap=False done start=1792921033.464 end=1792921033.805 runtime=ready warmup_ms={'preprocess': 149.0}
2026-10-25 18:37:15+09:00 ta-01M5S6QKJGMY9TS6F0MW2B6X9R 2026-10-25T18:37:15.655+09:00 INFO app.body_image_validation.background - rembg onnxruntime profile: runs=[{"run": 1, "model_run_ms": 1118.15, "nodes": {"CUDAExecutionProvider": 371}, "node_ms": {"CUDAExecutionProvider": 1102.31}}, {"run": 2, "model_run_ms": 210.2, "nodes": {"CUDAExecutionProvider": 371}, "node_ms": {"CUDAExecutionProvider": 76.0}}] parse_ms=21.5
2026-10-25 18:37:17+09:00 ta-01M5S6QKJGMY9TS6F0MW2B6X9R    POST /api/v1/validation/body-image -> 200 OK  (duration: 7.37 s, execution: 2.71 s)
""".splitlines()

# 5단계 A0(2026-10-30 T4): 배포 뒤 첫 검증 요청이 snapshot 을 처음 만든 경우(MISS). 현재 lifecycle 로그 형식.
# (4-A 의 옛 fixture 는 snap=True·runtime=ready 줄이 없어 이제 incomplete 로 판정된다.)
VALIDATION_MISS = """\
2026-10-30 09:07:14+09:00 ta-01M6523JZA17Q90AJ6H2H6ZJWR Creating GPU memory snapshot for Function.
2026-10-30 09:07:17+09:00 ta-01M6523JZA17Q90AJ6H2H6ZJWR body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-30 09:07:28+09:00 ta-01M6523JZA17Q90AJ6H2H6ZJWR body image worker lifecycle: snap=True done start=1793318837.403 end=1793318848.776 gpu_snapshot=True gpu=Tesla T4 warmup_ms={'preprocess': 132.8, 'person_pose': 59.4, 'rembg': 1098.6, 'png': 73.5}
2026-10-30 09:07:30+09:00 ta-01M6523JZA17Q90AJ6H2H6ZJWR body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-30 09:07:30+09:00 ta-01M6523JZA17Q90AJ6H2H6ZJWR body image worker lifecycle: snap=False done start=1793318850.493 end=1793318850.796 runtime=ready warmup_ms={'preprocess': 132.8, 'person_pose': 59.4, 'rembg': 1098.6, 'png': 73.5}
2026-10-30 09:07:33+09:00 ta-01M6523JZA17Q90AJ6H2H6ZJWR    POST /api/v1/validation/body-image -> 200 OK  (duration: 20.6 s, execution: 2.20 s)
""".splitlines()

# 5단계 A1: 컨테이너 0 뒤 검증 요청이 GPU snapshot 에서 복원(HIT).
RESTORED_A1 = """\
2026-10-30 09:08:58+09:00 ta-01M6526R9X0STAYGBHZM5KT38R Restoring Function from memory snapshot.
2026-10-30 09:09:01+09:00 ta-01M6526R9X0STAYGBHZM5KT38R body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-30 09:09:01+09:00 ta-01M6526R9X0STAYGBHZM5KT38R body image worker lifecycle: snap=False done start=1793318941.298 end=1793318941.665 runtime=ready warmup_ms={'preprocess': 132.8, 'person_pose': 59.4, 'rembg': 1098.6, 'png': 73.5}
2026-10-30 09:09:05+09:00 ta-01M6526R9X0STAYGBHZM5KT38R    POST /api/v1/validation/body-image -> 200 OK  (duration: 8.74 s, execution: 2.75 s)
""".splitlines()

# 5단계 B1: A1~A3 에서 복원되던 snapshot 이 있는데도 /health 컨테이너가 snapshot 을 다시 만들었다(재생성 MISS).
RECREATED_B1 = """\
2026-10-30 09:10:51+09:00 ta-01M652A69TT0CATRSX2RY93P0R Creating GPU memory snapshot for Function.
2026-10-30 09:10:54+09:00 ta-01M652A69TT0CATRSX2RY93P0R body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-30 09:11:07+09:00 ta-01M652A69TT0CATRSX2RY93P0R body image worker lifecycle: snap=True done start=1793319054.808 end=1793319067.097 gpu_snapshot=True gpu=Tesla T4 warmup_ms={'preprocess': 125.4, 'person_pose': 101.3, 'rembg': 1233.7, 'png': 83.0}
2026-10-30 09:11:09+09:00 ta-01M652A69TT0CATRSX2RY93P0R body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-30 09:11:09+09:00 ta-01M652A69TT0CATRSX2RY93P0R body image worker lifecycle: snap=False done start=1793319069.245 end=1793319069.728 runtime=ready warmup_ms={'preprocess': 125.4, 'person_pose': 101.3, 'rembg': 1233.7, 'png': 83.0}
2026-10-30 09:11:10+09:00 ta-01M652A69TT0CATRSX2RY93P0R     GET /health -> 200 OK  (duration: 51.6 s, execution: 269.5 ms)
2026-10-30 09:11:14+09:00 ta-01M652A69TT0CATRSX2RY93P0R    POST /api/v1/validation/body-image -> 200 OK  (duration: 3.05 s, execution: 2.78 s)
""".splitlines()

# 5단계 SH: 복원 시도 → Transient snapshot error(exit 137) → 같은 컨테이너가 snapshot 없이 재초기화 → /health·검증 200.
# 기존 판정은 'Restoring' 줄만 보고 HIT 로 잘못 분류했다.
SH_RESTORE_FALLBACK = """\
2026-10-30 09:23:02+09:00 ta-01M6530G7TAJPWTWD0BQVRVCAR Restoring Function from memory snapshot.
2026-10-30 09:23:03+09:00 ta-01M6530G7TAJPWTWD0BQVRVCAR Transient snapshot error: failed to restore container from snapshot with exit code 137. Will retry with no snapshots.
2026-10-30 09:23:05+09:00 ta-01M6530G7TAJPWTWD0BQVRVCAR body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-30 09:23:12+09:00 ta-01M6530G7TAJPWTWD0BQVRVCAR WARNING: Logging before InitGoogle() is written to STDERR
2026-10-30 09:23:13+09:00 ta-01M6530G7TAJPWTWD0BQVRVCAR body image worker lifecycle: snap=True done start=1793319785.772 end=1793319793.937 gpu_snapshot=True gpu=Tesla T4 warmup_ms={'preprocess': 77.3, 'person_pose': 55.4, 'rembg': 713.4, 'png': 72.6}
2026-10-30 09:23:13+09:00 ta-01M6530G7TAJPWTWD0BQVRVCAR body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-30 09:23:14+09:00 ta-01M6530G7TAJPWTWD0BQVRVCAR body image worker lifecycle: snap=False done start=1793319793.938 end=1793319794.210 runtime=ready warmup_ms={'preprocess': 77.3, 'person_pose': 55.4, 'rembg': 713.4, 'png': 72.6}
2026-10-30 09:23:14+09:00 ta-01M6530G7TAJPWTWD0BQVRVCAR     GET /health -> 200 OK  (duration: 14.3 s, execution: 95.3 ms)
2026-10-30 09:23:18+09:00 ta-01M6530G7TAJPWTWD0BQVRVCAR    POST /api/v1/validation/body-image -> 200 OK  (duration: 2.99 s, execution: 2.90 s)
""".splitlines()
SH_CONTAINER = "ta-01M6530G7TAJPWTWD0BQVRVCAR"
SINCE_1030 = datetime(2026, 10, 30, 0, 0, tzinfo=UTC)  # 09:00 KST

# 4-C v5: GPU snapshot 옵션 누락 → 1회차 실패, Modal 이 snapshot 없이 다시 띄워 /health 200
SNAPSHOT_OPTION_MISSING = """\
2026-10-25 18:47:10+09:00 ta-01M5S79WQK9J1CB38C31JWRV0R Creating CPU memory snapshot for Function.
2026-10-25 18:47:14+09:00 ta-01M5S79WQK9J1CB38C31JWRV0R body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-25 18:47:25+09:00 ta-01M5S79WQK9J1CB38C31JWRV0R Runner failed with exception: WorkerConfigError("GPU snapshot 단계에서 GPU 를 쓸 수 없습니다: rembg CUDA provider (active: ['CPUExecutionProvider']), GPU preprocessing (nvJPEG)")
2026-10-25 18:47:29+09:00 ta-01M5S7AC99YN495HXNPDEK83MR body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-25 18:47:41+09:00 ta-01M5S7AC99YN495HXNPDEK83MR body image worker lifecycle: snap=True done start=1792921649.957 end=1792921661.169 gpu_snapshot=True gpu=Tesla T4 warmup_ms={'preprocess': 111.4}
2026-10-25 18:47:41+09:00 ta-01M5S7AC99YN495HXNPDEK83MR body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-25 18:47:41+09:00 ta-01M5S7AC99YN495HXNPDEK83MR     GET /health -> 200 OK  (duration: 32.8 s, execution: 31.1 ms)
""".splitlines()


# 3-1단계 L4(8GiB): snap=True 는 끝났지만 snapshot 생성 실패 → SIGKILL. 같은 일이 4번 반복됐다.
L4_SNAPSHOT_FAILURE = """\
2026-10-25 22:15:34+09:00 ta-01M5SK7F6GBYGG8XZJXK6AXZ9R Creating GPU memory snapshot for Function.
2026-10-25 22:15:37+09:00 ta-01M5SK7F6GBYGG8XZJXK6AXZ9R body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-25 22:15:51+09:00 ta-01M5SK7F6GBYGG8XZJXK6AXZ9R body image worker lifecycle: snap=True done start=1792934137.371 end=1792934151.199 gpu_snapshot=True gpu=NVIDIA L4 warmup_ms={'preprocess': 540.1, 'person_pose': 66.7, 'rembg': 1684.3, 'png': 73.5}
2026-10-25 22:15:52+09:00 ta-01M5SK7F6GBYGG8XZJXK6AXZ9R Failed creating Function memory snapshot. Please disable memory snapshots if failures persist and contact us at support@modal.com or modal.com/slack for assistance.
2026-10-25 22:15:52+09:00 ta-01M5SK7F6GBYGG8XZJXK6AXZ9R Not all applications are supported by memory snapshots. For known issues see modal.com/docs/guide/memory-snapshot
2026-10-25 22:15:53+09:00 ta-01M5SK7F6GBYGG8XZJXK6AXZ9R Runner killed (SIGKILL), exit code: 137. Most often this happens when the process runs out of memory. Consider raising the memory limit to avoid unexpected termination: https://modal.com/docs/guide/resources
""".splitlines()

# 3-1단계 gpu=["T4","L4"]: /health 는 T4 에서 snapshot 생성·200, 검증 요청은 L4 에 배정돼 snapshot 생성 실패
# (Runner killed 줄 없이 로그가 끝났다).
LIST_T4_HEALTH = """\
2026-10-25 22:21:18+09:00 ta-01M5SKHZTYBR9TV3TXBB2X83JR Creating GPU memory snapshot for Function.
2026-10-25 22:21:21+09:00 ta-01M5SKHZTYBR9TV3TXBB2X83JR body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-25 22:21:32+09:00 ta-01M5SKHZTYBR9TV3TXBB2X83JR body image worker lifecycle: snap=True done start=1792934481.349 end=1792934492.469 gpu_snapshot=True gpu=Tesla T4 warmup_ms={'preprocess': 129.7, 'person_pose': 59.5, 'rembg': 1122.6, 'png': 73.5}
2026-10-25 22:21:34+09:00 ta-01M5SKHZTYBR9TV3TXBB2X83JR body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-25 22:21:36+09:00 ta-01M5SKHZTYBR9TV3TXBB2X83JR body image worker lifecycle: snap=False done start=1792934494.862 end=1792934496.036 runtime=ready warmup_ms={'preprocess': 129.7, 'person_pose': 59.5, 'rembg': 1122.6, 'png': 73.5}
2026-10-25 22:21:37+09:00 ta-01M5SKHZTYBR9TV3TXBB2X83JR     GET /health -> 200 OK  (duration: 19.8 s, execution: 51.3 ms)
""".splitlines()
LIST_L4_REQUEST = """\
2026-10-25 22:23:19+09:00 ta-01M5SKNPHB0X3STE6ETQ4F6RTR Creating GPU memory snapshot for Function.
2026-10-25 22:23:22+09:00 ta-01M5SKNPHB0X3STE6ETQ4F6RTR body image worker lifecycle: stage=staging gpu_snapshot=True
2026-10-25 22:23:32+09:00 ta-01M5SKNPHB0X3STE6ETQ4F6RTR body image worker lifecycle: snap=True done start=1792934602.342 end=1792934612.759 gpu_snapshot=True gpu=NVIDIA L4 warmup_ms={'preprocess': 359.3, 'person_pose': 65.0, 'rembg': 1312.1, 'png': 49.2}
2026-10-25 22:23:33+09:00 ta-01M5SKNPHB0X3STE6ETQ4F6RTR Failed creating Function memory snapshot. Please disable memory snapshots if failures persist and contact us at support@modal.com or modal.com/slack for assistance.
""".splitlines()


# --- 배포 전 검사 ---


@pytest.mark.parametrize("stage", ["staging", "production"])
def test_current_deploy_file_passes_the_config_checks(stage):
    module, calls = dc.load_worker(stage)

    checks = dc.check_deploy_config(module, calls, stage)

    assert all(check.ok for check in checks), [c for c in checks if not c.ok]


def test_a_staging_deploy_without_gpu_snapshot_fails_the_config_check():
    module, calls = dc.load_worker("staging")
    module.STAGE_SNAPSHOT = {}

    failed = [c.name for c in dc.check_deploy_config(module, calls, "staging") if not c.ok]

    assert failed == ["gpu snapshot"]


def test_a_build_step_after_the_code_layer_fails_the_config_check():
    module, calls = dc.load_worker("staging")

    failed = [c.name for c in dc.check_deploy_config(module, calls + [("env", {"X": "1"})], "staging") if not c.ok]

    assert "code layer last" in failed


def test_production_with_diagnostics_cannot_be_loaded():
    with pytest.raises(ValueError):
        dc.load_worker("production", diagnostics=True)


def test_lifecycle_hooks_and_private_api_checks_pass_on_the_real_file():
    assert dc.check_lifecycle_decorators().ok
    assert dc.check_no_private_modal_api().ok


def test_lifecycle_check_rejects_an_extra_modal_method(tmp_path):
    source = dc.WORKER_FILE.read_text() + "\n\nclass X:\n    @modal.method()\n    def warmup(self):\n        pass\n"
    path = tmp_path / "worker.py"
    path.write_text(source)

    assert not dc.check_lifecycle_decorators(path).ok


def test_private_api_check_rejects_internal_imports(tmp_path):
    path = tmp_path / "worker.py"
    path.write_text("from modal._runtime.container_io_manager import _ContainerIOManager\n")

    assert not dc.check_no_private_modal_api(path).ok


@pytest.mark.parametrize(
    ("env_profile", "expected", "ok"),
    [(None, "lookddak", True), (None, "leejein", False), ("leejein", "leejein", True), ("missing", "leejein", False)],
)
def test_profile_must_point_at_the_expected_workspace(env_profile, expected, ok):
    profiles = [{"name": "leejein", "workspace": "leejein", "active": False},
                {"name": "lookddak", "workspace": "lookddak", "active": True}]

    _, check = dc.resolve_profile(expected, profiles, env_profile)

    assert check.ok is ok


def test_preflight_stops_before_the_secret_check_on_a_workspace_mismatch(monkeypatch):
    profiles = [{"name": "lookddak", "workspace": "lookddak", "active": True}]
    monkeypatch.setattr(dc, "modal_cli", lambda *args, **kwargs: json.dumps(profiles))
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    monkeypatch.setattr(dc, "check_secret", lambda name: pytest.fail("secret 을 조회하면 안 된다"))

    _, checks = dc.preflight("staging", "leejein", False)

    assert [c.name for c in checks if not c.ok] == ["workspace"]


def test_deploy_command_does_not_deploy_when_preflight_fails(monkeypatch):
    monkeypatch.setattr(dc, "preflight", lambda *args: ("p", [dc.Check("workspace", False, "x")]))
    monkeypatch.setattr(dc.subprocess, "run", lambda *a, **k: pytest.fail("배포하면 안 된다"))

    assert dc.main(["deploy", "--stage", "staging", "--workspace", "leejein"]) == 1


# --- smoke 판정 ---


def test_warmup_with_gpu_snapshot_creation_is_ok():
    status, reason = dc.judge_warmup(dc.summarize_logs(WARMUP_OK, SINCE))

    assert status == "ok" and "Tesla T4" in reason


def test_health_restored_from_an_existing_snapshot_is_ok():
    # 같은 배포에서 smoke 를 다시 돌리면 /health 도 snapshot 에서 복원된다.
    lines = [line.replace("POST /api/v1/validation/body-image", "GET /health") for line in VALIDATION_HIT]

    status, reason = dc.judge_warmup(dc.summarize_logs(lines, SINCE))

    assert status == "ok" and "restored" in reason


def test_init_failure_followed_by_a_snapshotless_200_is_not_ok():
    # Modal 이 snapshot 없이 다시 띄운 컨테이너의 200 을 성공으로 보지 않는다.
    status, _ = dc.judge_warmup(dc.summarize_logs(SNAPSHOT_OPTION_MISSING, SINCE))

    assert status == "init_failure"


def test_health_without_gpu_snapshot_creation_is_snapshot_disabled():
    lines = [line for line in WARMUP_OK if "Creating GPU" not in line]

    assert dc.judge_warmup(dc.summarize_logs(lines, SINCE))[0] == "snapshot_disabled"


def test_cpu_snapshot_on_staging_is_snapshot_disabled():
    lines = [line.replace("Creating GPU memory", "Creating CPU memory") for line in WARMUP_OK]

    assert dc.judge_warmup(dc.summarize_logs(lines, SINCE))[0] == "snapshot_disabled"


def test_validation_hit_miss_and_disabled():
    assert dc.judge_validation(dc.summarize_logs(VALIDATION_HIT, SINCE))[0] == "hit"
    assert dc.judge_validation(dc.summarize_logs(VALIDATION_MISS, SINCE))[0] == "miss"
    no_snapshot = [line for line in VALIDATION_HIT if "Restoring" not in line]
    assert dc.judge_validation(dc.summarize_logs(no_snapshot, SINCE))[0] == "snapshot_disabled"


def test_modal_request_times_separate_waiting_from_execution():
    log = next(iter(dc.summarize_logs(VALIDATION_HIT, SINCE).values()))
    assert log.request_times["POST /api/v1/validation/body-image"] == {"duration": 7.37, "execution": 2.71}

    health = next(iter(dc.summarize_logs(WARMUP_OK, SINCE).values()))
    assert health.request_times["GET /health"] == pytest.approx({"duration": 19.0, "execution": 0.1058})


def test_lines_before_the_test_started_are_ignored():
    later = datetime(2026, 10, 25, 9, 36, tzinfo=UTC)  # 18:36 KST, /health 로그(18:35) 이후

    assert dc.summarize_logs(WARMUP_OK, later) == {}


def test_cuda_check_reads_the_restored_request_profile():
    ok, detail = dc.cuda_check(dc.summarize_logs(VALIDATION_HIT, SINCE))
    assert ok is True and "run 2" in detail

    cpu = [line.replace('{"run": 2, "model_run_ms": 210.2, "nodes": {"CUDAExecutionProvider": 371}',
                        '{"run": 2, "model_run_ms": 210.2, "nodes": {"CUDAExecutionProvider": 300, '
                        '"CPUExecutionProvider": 71}') for line in VALIDATION_HIT]
    assert dc.cuda_check(dc.summarize_logs(cpu, SINCE))[0] is False
    no_profile = [line for line in VALIDATION_HIT if "profile" not in line]
    assert dc.cuda_check(dc.summarize_logs(no_profile, SINCE))[0] is None


# --- smoke orchestration (fake Modal·HTTP) ---


class FakeOps:
    """phases: [/health 때 로그, 1번째 검증 요청 때 로그, 2번째 …]. 요청이 갈 때마다 다음 단계 로그를 보여 준다
    (실제로는 smoke 가 요청 시각 이후 로그만 보므로 이전 단계 로그는 섞이지 않는다)."""

    def __init__(self, phases):
        self.phases = list(phases)
        self.current = []
        self.actions = []

    def next_phase(self):
        self.current = self.phases.pop(0) if len(self.phases) > 1 else self.phases[0]

    def app_info(self, app_name):
        return {"app_id": "ap-test", "lifecycle": {"version": 11},
                "functions": {"BodyImageWorker.*": {"id": "fu-test"}}}

    def function_info(self, function_id):
        return {"gpus": [["T4", 1]], "max_containers": 1,
                "method_details": {"web": {"web_url": "https://example.modal.run", "unauthenticated": False}}}

    def live_containers(self, app_id):
        return []

    def logs(self, app_id):
        return self.current

    def rollback(self, app_id):
        self.actions.append(("rollback", app_id))

    def stop(self, app_id):
        self.actions.append(("stop", app_id))


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body


class FakeHttp:
    def __init__(self, ops, health=200, post=200, delay=0.0):
        self.ops, self.health, self.post_status, self.delay = ops, health, post, delay

    def get(self, url, timeout=None):
        import time

        self.ops.next_phase()
        time.sleep(self.delay)
        return FakeResponse(self.health)

    def post(self, url, files=None, headers=None, timeout=None):
        self.ops.next_phase()
        return FakeResponse(self.post_status, {"data": {"s3_key": "body-images/x.png"}})


FAST = dc.SmokeLimits(poll_interval_s=0.01, log_settle_s=0, max_init_failures=2)
CREDENTIALS = {"modal_key": "k", "modal_secret": "s", "api_key": "a"}


def run_smoke(ops, http, tmp_path, s3=None):
    http = http(ops) if isinstance(http, type) or not hasattr(http, "get") else http  # 클래스·factory 면 만든다
    image = tmp_path / "body.jpg"
    image.write_bytes(b"jpeg")
    return dc.smoke(ops, "app", CREDENTIALS, image, FAST, http=http,
                    s3_checker=s3 or (lambda secret, key: {"format": "PNG"}), secret_name="secret",
                    clock=lambda: SINCE.timestamp())


def test_smoke_passes_on_snapshot_creation_then_hit(tmp_path):
    report = run_smoke(FakeOps([WARMUP_OK, VALIDATION_HIT]), FakeHttp, tmp_path)

    assert report["result"] == "PASS"
    assert [a["snapshot"] for a in report["validation"]] == ["hit"]
    assert report["validation"][0]["cuda"] is True


def test_smoke_retries_a_transient_miss_and_then_passes(tmp_path):
    report = run_smoke(FakeOps([WARMUP_OK, VALIDATION_MISS, VALIDATION_HIT]), FakeHttp, tmp_path)

    assert [a["snapshot"] for a in report["validation"]] == ["miss", "hit"]


def test_smoke_fails_after_only_misses(tmp_path):
    with pytest.raises(dc.SmokeFailure) as failure:
        run_smoke(FakeOps([WARMUP_OK, VALIDATION_MISS]), FakeHttp, tmp_path)

    assert failure.value.stage == "restore"


def test_smoke_stops_waiting_when_init_failures_repeat(tmp_path):
    failing = SNAPSHOT_OPTION_MISSING[:3] + [
        SNAPSHOT_OPTION_MISSING[2].replace("ta-01M5S79WQK9J1CB38C31JWRV0R", "ta-01M5S7AC99YN495HXNPDEK83MR")
    ]
    with pytest.raises(dc.SmokeFailure) as failure:
        run_smoke(FakeOps([failing]), lambda ops: FakeHttp(ops, delay=5), tmp_path)  # /health 응답보다 먼저 멈춘다

    assert failure.value.stage == "init"


def test_smoke_fails_when_modal_ran_without_a_snapshot(tmp_path):
    with pytest.raises(dc.SmokeFailure) as failure:
        run_smoke(FakeOps([SNAPSHOT_OPTION_MISSING]), FakeHttp, tmp_path)

    assert failure.value.stage == "warmup"


def test_smoke_rejects_a_non_t4_function(tmp_path):
    ops = FakeOps([WARMUP_OK])
    ops.function_info = lambda function_id: {"gpus": [["L4", 1]],
                                             "method_details": {"web": {"web_url": "u", "unauthenticated": False}}}

    with pytest.raises(dc.SmokeFailure) as failure:
        run_smoke(ops, FakeHttp, tmp_path)

    assert failure.value.stage == "config"


@pytest.mark.parametrize(
    ("action", "workspace", "stage", "expected"),
    [
        ("rollback", "leejein", "staging", [("rollback", "ap-1")]),
        ("stop", "leejein", "staging", [("stop", "ap-1")]),
        ("rollback", "lookddak", "staging", []),  # 팀 Workspace 는 자동 조치하지 않는다
        ("stop", "leejein", "production", []),
        ("none", "leejein", "staging", []),
    ],
)
def test_failure_actions_only_touch_personal_staging(action, workspace, stage, expected):
    ops = FakeOps([[]])

    dc.on_failure(ops, action, "ap-1", workspace, stage)

    assert ops.actions == expected


# --- snapshot 생성 실패·Runner killed 분류 (3-1단계 실제 L4 로그) ---


def kinds(lines, since=SINCE):
    return [f["kind"] for f in dc.startup_failures(dc.summarize_logs(lines, since))]


def test_l4_snapshot_failure_followed_by_sigkill_is_one_snapshot_creation_failure():
    [failure] = dc.startup_failures(dc.summarize_logs(L4_SNAPSHOT_FAILURE, SINCE))

    assert failure["kind"] == "snapshot_creation_failed"
    assert "NVIDIA L4" in failure["detail"] and "SIGKILL exit 137" in failure["detail"]


def test_snapshot_failure_without_a_runner_killed_line_is_still_detected():
    assert kinds(LIST_L4_REQUEST) == ["snapshot_creation_failed"]


def test_runner_killed_alone_is_not_called_a_snapshot_failure():
    killed_only = [line for line in L4_SNAPSHOT_FAILURE if "Failed creating" not in line]

    [failure] = dc.startup_failures(dc.summarize_logs(killed_only, SINCE))
    assert failure["kind"] == "runner_killed" and "not confirmed" in failure["detail"]


def test_a_normal_miss_is_not_a_failure():
    # 새 snapshot 을 만들고 정상 응답한 컨테이너(MISS)는 실패가 아니다.
    assert kinds(VALIDATION_MISS) == []
    assert dc.judge_validation(dc.summarize_logs(VALIDATION_MISS, SINCE))[0] == "miss"


def test_judges_report_snapshot_failure_separately_from_init_failure():
    assert dc.judge_warmup(dc.summarize_logs(L4_SNAPSHOT_FAILURE, SINCE))[0] == "snapshot_failed"
    assert dc.judge_validation(dc.summarize_logs(LIST_L4_REQUEST, SINCE))[0] == "snapshot_failed"
    # 사용자 코드 예외는 init_failure 그대로
    assert dc.judge_warmup(dc.summarize_logs(SNAPSHOT_OPTION_MISSING, SINCE))[0] == "init_failure"


def test_a_container_from_before_the_request_is_ignored_even_if_it_is_killed_later():
    # 이전 단계 컨테이너가 요청 뒤에 종료(Runner killed)돼도 이번 판정에 넣지 않는다.
    later = datetime(2026, 10, 25, 13, 15, 52, tzinfo=UTC)  # 22:15:52 KST: snapshot 실패 줄부터
    old_container_late_lines = L4_SNAPSHOT_FAILURE  # 22:15:34 부터 로그가 있던 컨테이너

    assert kinds(old_container_late_lines, since=later) == []


def test_lines_from_an_earlier_deploy_are_ignored():
    after_failure = datetime(2026, 10, 25, 13, 20, tzinfo=UTC)

    assert kinds(L4_SNAPSHOT_FAILURE + LIST_T4_HEALTH, since=after_failure) == []


def test_smoke_fails_fast_on_the_first_snapshot_creation_failure(tmp_path):
    import time

    started = time.monotonic()
    with pytest.raises(dc.SmokeFailure) as failure:
        # /health 응답은 5초 뒤에 오지만, 첫 snapshot 생성 실패에서 바로 멈춘다.
        run_smoke(FakeOps([L4_SNAPSHOT_FAILURE]), lambda ops: FakeHttp(ops, delay=5), tmp_path)

    assert failure.value.stage == "snapshot"
    assert "NVIDIA L4" in failure.value.reason
    assert time.monotonic() - started < 2


def test_smoke_stops_when_the_validation_container_fails_to_snapshot(tmp_path):
    # gpu=["T4","L4"] 실측: /health 는 T4 에서 정상, 검증 요청은 L4 로 가서 snapshot 생성 실패.
    with pytest.raises(dc.SmokeFailure) as failure:
        run_smoke(FakeOps([LIST_T4_HEALTH, LIST_L4_REQUEST]), lambda ops: FakeHttp(ops, delay=5), tmp_path)

    assert failure.value.stage == "snapshot"


def test_unconfirmed_runner_kills_stop_at_the_shared_failure_limit(tmp_path):
    killed = [line for line in L4_SNAPSHOT_FAILURE if "Failed creating" not in line]
    second = [line.replace("ta-01M5SK7F6GBYGG8XZJXK6AXZ9R", "ta-01M5SK837W23T582X03CAVNARR") for line in killed]

    with pytest.raises(dc.SmokeFailure) as one_more:
        run_smoke(FakeOps([killed + second]), lambda ops: FakeHttp(ops, delay=5), tmp_path)
    assert one_more.value.stage == "init"  # snapshot 실패가 아니라 시작 실패로 보고한다


def test_miss_retries_and_limits_are_unchanged():
    limits = dc.SmokeLimits()

    assert (limits.max_miss_retries, limits.health_timeout_s, limits.validation_timeout_s) == (2, 150, 120)
    assert (limits.max_init_failures, limits.max_snapshot_failures) == (2, 1)


# --- snapshot 상태 4분류 (5단계 Staging 실측 로그) ---


def states(lines, since=SINCE_1030):
    return {c: dc.snapshot_state(log)[0] for c, log in dc.summarize_logs(lines, since).items()}


def without(lines, *markers):
    return [line for line in lines if not any(marker in line for marker in markers)]


def test_a_real_gpu_snapshot_restore_is_a_hit():
    assert states(RESTORED_A1) == {"ta-01M6526R9X0STAYGBHZM5KT38R": "snapshot_hit"}
    assert dc.judge_validation(dc.summarize_logs(RESTORED_A1, SINCE_1030))[0] == "hit"


def test_the_first_snapshot_creation_is_a_miss():
    assert states(VALIDATION_MISS) == {"ta-01M6523JZA17Q90AJ6H2H6ZJWR": "snapshot_miss"}


def test_recreating_an_existing_snapshot_is_a_miss():
    # A1 의 복원 컨테이너는 측정 구간(09:10 KST) 전에 로그를 남겼으므로 판정에서 빠진다.
    containers = dc.summarize_logs(RESTORED_A1 + RECREATED_B1, datetime(2026, 10, 30, 0, 10, tzinfo=UTC))

    assert {c: dc.snapshot_state(log)[0] for c, log in containers.items()} == {
        "ta-01M652A69TT0CATRSX2RY93P0R": "snapshot_miss"}
    assert dc.judge_warmup(containers)[0] == "ok"
    assert dc.judge_validation(containers)[0] == "miss"


def test_sh_restore_failure_then_reinit_is_a_fallback_not_a_hit():
    # 회귀: 기존 로직은 'Restoring' 줄만 보고 hit 로 판정했다.
    containers = dc.summarize_logs(SH_RESTORE_FALLBACK, SINCE_1030)

    state, reason = dc.snapshot_state(containers[SH_CONTAINER])
    assert state == "restore_failed_fallback" and "exit code 137" in reason
    assert dc.judge_validation(containers)[0] == "restore_failed_fallback"
    assert dc.judge_warmup(containers)[0] == "restore_failed_fallback"
    assert dc.startup_failures(containers) == []  # Modal 이 스스로 복구했다(시작 실패가 아니다)


def test_snapshot_creation_failure_is_its_own_state():
    assert states(L4_SNAPSHOT_FAILURE, SINCE) == {"ta-01M5SK7F6GBYGG8XZJXK6AXZ9R": "snapshot_creation_failed"}
    assert dc.judge_validation(dc.summarize_logs(LIST_L4_REQUEST, SINCE))[0] == "snapshot_failed"


@pytest.mark.parametrize(
    "failure_line",
    [
        f"2026-10-30 09:23:10+09:00 {SH_CONTAINER} Runner failed with exception: RuntimeError('model load failed')",
        f"2026-10-30 09:23:10+09:00 {SH_CONTAINER} Runner killed (SIGKILL), exit code: 137.",
    ],
)
def test_a_failed_reinit_after_a_restore_failure_is_not_a_fallback(failure_line):
    lines = SH_RESTORE_FALLBACK[:3] + [failure_line]
    containers = dc.summarize_logs(lines, SINCE_1030)

    state, reason = dc.snapshot_state(containers[SH_CONTAINER])
    assert state == "init_failure" and "복원 실패 뒤" in reason
    [failure] = dc.startup_failures(containers)
    assert "복원 실패 뒤" in failure["detail"]
    # 재초기화가 실패했다면 뒤에 다른 컨테이너가 200 을 내도 정상으로 보지 않는다.
    retried = [line.replace(SH_CONTAINER, "ta-01M6530RETRY0000000000000R") for line in SH_RESTORE_FALLBACK[2:]]
    assert dc.judge_validation(dc.summarize_logs(lines + retried, SINCE_1030))[0] == "init_failure"


def test_restore_error_and_readiness_from_different_containers_are_not_combined():
    other = "ta-01M6530OTHER000000000000R"
    # X 는 복원 실패만, Y 는 (복원 표시 없이) 준비 완료·200 만 남겼다. 합쳐서 fallback 으로 보지 않는다.
    mixed = SH_RESTORE_FALLBACK[:2] + [line.replace(SH_CONTAINER, other) for line in SH_RESTORE_FALLBACK[2:]]
    containers = dc.summarize_logs(mixed, SINCE_1030)

    assert states(mixed) == {SH_CONTAINER: "incomplete", other: "snapshot_disabled"}
    status, reason = dc.judge_validation(containers)
    assert status == "snapshot_disabled" and SH_CONTAINER in reason  # 다른 컨테이너의 복원 실패도 이유에 남긴다


def test_a_clean_restore_is_still_a_hit_next_to_another_containers_restore_failure():
    lines = SH_RESTORE_FALLBACK[:2] + RESTORED_A1

    status, reason = dc.judge_validation(dc.summarize_logs(lines, SINCE_1030))

    assert status == "hit" and f"복원 실패: ['{SH_CONTAINER}']" in reason


@pytest.mark.parametrize(
    ("lines", "removed"),
    [
        (SH_RESTORE_FALLBACK, ("Transient snapshot error",)),  # 오류 줄이 빠지면 hit 로 추측하지 않는다
        (SH_RESTORE_FALLBACK, ("runtime=ready",)),  # 재초기화 완료를 확인할 수 없다
        (SH_RESTORE_FALLBACK, ("Restoring Function",)),  # 복원 시도를 확인할 수 없다
        (RESTORED_A1, ("runtime=ready",)),  # 복원 뒤 준비 완료를 확인할 수 없다
        (VALIDATION_MISS, ("snap=True done", "snap=False done")),  # 4-A 의 옛 형식처럼 lifecycle 줄이 없다
        (VALIDATION_MISS, ("runtime=ready",)),
    ],
)
def test_missing_lines_never_produce_a_normal_state(lines, removed):
    partial = without(lines, *removed)

    assert set(states(partial).values()) == {"incomplete"}
    assert dc.judge_validation(dc.summarize_logs(partial, SINCE_1030))[0] == "incomplete"


def test_readiness_logged_before_a_restore_error_does_not_count():
    # 복원 오류 앞에 찍힌 준비 완료 줄은 재초기화 결과가 아니다.
    ready = next(line for line in SH_RESTORE_FALLBACK if "runtime=ready" in line).replace("09:23:14", "09:23:02")
    lines = [SH_RESTORE_FALLBACK[0], ready, SH_RESTORE_FALLBACK[1], SH_RESTORE_FALLBACK[-1]]

    assert states(lines) == {SH_CONTAINER: "incomplete"}


def test_smoke_retries_after_a_restore_fallback_and_then_passes(tmp_path):
    report = run_smoke(FakeOps([WARMUP_OK, SH_RESTORE_FALLBACK, VALIDATION_HIT]), FakeHttp, tmp_path)

    assert report["result"] == "PASS"
    assert [a["snapshot"] for a in report["validation"]] == ["restore_failed_fallback", "hit"]


def test_smoke_continues_when_the_health_container_fell_back(tmp_path):
    # /health 가 fallback 이면 snapshot 은 있었다. 쓸 수 있는지는 검증 요청의 HIT 로 확인한다.
    report = run_smoke(FakeOps([SH_RESTORE_FALLBACK, VALIDATION_HIT]), FakeHttp, tmp_path)

    assert report["snapshot_creation"]["status"] == "restore_failed_fallback"
    assert report["result"] == "PASS"


def test_smoke_fails_when_every_attempt_falls_back(tmp_path):
    with pytest.raises(dc.SmokeFailure) as failure:
        run_smoke(FakeOps([WARMUP_OK, SH_RESTORE_FALLBACK]), FakeHttp, tmp_path)

    assert failure.value.stage == "restore" and "restore_failed_fallback" in failure.value.reason


# --- production preflight (읽기 전용. production 은 deploy 하지 않는다) ---


def test_production_gpu_snapshot_config_passes_when_explicitly_enabled():
    module, calls = dc.load_worker("production", gpu_snapshot=True, max_containers=2)

    checks = dc.check_deploy_config(module, calls, "production", gpu_snapshot=True, max_containers=2)

    assert all(check.ok for check in checks), [c for c in checks if not c.ok]


def test_production_mode_confusion_fails_the_config_check():
    cpu_module, cpu_calls = dc.load_worker("production")
    gpu_module, gpu_calls = dc.load_worker("production", gpu_snapshot=True, max_containers=2)

    # CPU 로 불러온 배포를 GPU 로 검사하거나 그 반대면 실패한다.
    assert "gpu snapshot mode" in [
        c.name for c in dc.check_deploy_config(cpu_module, cpu_calls, "production", True, 2) if not c.ok]
    assert "gpu snapshot mode" in [
        c.name for c in dc.check_deploy_config(gpu_module, gpu_calls, "production", False, 2) if not c.ok]


def test_production_preflight_requires_the_production_workspace(monkeypatch):
    profiles = [{"name": "leejein", "workspace": "leejein", "active": True}]
    monkeypatch.setattr(dc, "modal_cli", lambda *args, **kwargs: json.dumps(profiles))
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    monkeypatch.setattr(dc, "check_secret", lambda name: pytest.fail("secret 을 조회하면 안 된다"))

    _, checks = dc.preflight("production", "leejein", gpu_snapshot=True)

    assert "production workspace" in [c.name for c in checks if not c.ok]


def test_production_preflight_passes_in_lookddak_with_the_production_secret(monkeypatch):
    profiles = [{"name": "lookddak", "workspace": "lookddak", "active": True}]
    monkeypatch.setattr(dc, "modal_cli", lambda *args, **kwargs: json.dumps(profiles))
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    seen = []
    monkeypatch.setattr(dc, "check_secret", lambda name: seen.append(name) or dc.Check("secret", True, name))

    _, checks = dc.preflight("production", "lookddak", gpu_snapshot=True, max_containers=2)

    assert all(c.ok for c in checks), [c for c in checks if not c.ok]
    assert seen == ["lookddak-body-image-worker"]  # staging Secret 이 아니다


@pytest.mark.parametrize("gpu", [[], ["--gpu-snapshot"]])
def test_deploy_command_refuses_production(monkeypatch, capsys, gpu):
    monkeypatch.setattr(dc, "preflight", lambda *args: pytest.fail("production deploy 는 검사·배포를 시작하지 않는다"))
    monkeypatch.setattr(dc.subprocess, "run", lambda *a, **k: pytest.fail("배포하면 안 된다"))

    assert dc.main(["deploy", "--stage", "production", "--workspace", "lookddak", *gpu]) == 2
    assert "preflight" in capsys.readouterr().out



# --- production max_containers 검사 ---


def test_production_gpu_preflight_fails_without_a_container_cap(monkeypatch):
    profiles = [{"name": "lookddak", "workspace": "lookddak", "active": True}]
    monkeypatch.setattr(dc, "modal_cli", lambda *args, **kwargs: json.dumps(profiles))
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    monkeypatch.setattr(dc, "check_secret", lambda name: pytest.fail("설정이 틀리면 secret 을 조회하지 않아도 된다"))

    _, checks = dc.preflight("production", "lookddak", gpu_snapshot=True)  # --max-containers 없음

    failed = [c for c in checks if not c.ok]
    assert [c.name for c in failed] == ["load deploy file"]
    assert "BODY_IMAGE_WORKER_MAX_CONTAINERS" in failed[0].detail


def test_production_gpu_cap_must_match_the_expected_value():
    module, calls = dc.load_worker("production", gpu_snapshot=True, max_containers=3)

    failed = [c.name for c in dc.check_deploy_config(module, calls, "production", True, 2) if not c.ok]

    assert failed == ["production max containers"]


def test_production_cpu_cap_is_optional_and_checked_when_given():
    module, calls = dc.load_worker("production")
    assert all(c.ok for c in dc.check_deploy_config(module, calls, "production"))

    capped, capped_calls = dc.load_worker("production", max_containers=3)
    assert all(c.ok for c in dc.check_deploy_config(capped, capped_calls, "production", max_containers=3))
    assert "production max containers" in [
        c.name for c in dc.check_deploy_config(capped, capped_calls, "production") if not c.ok]


def test_deploy_refusal_prints_the_cap_in_the_manual_command(monkeypatch, capsys):
    monkeypatch.setattr(dc.subprocess, "run", lambda *a, **k: pytest.fail("배포하면 안 된다"))

    code = dc.main(["deploy", "--stage", "production", "--workspace", "lookddak", "--gpu-snapshot",
                    "--max-containers", "2"])

    out = capsys.readouterr().out
    assert code == 2
    assert ("BODY_IMAGE_WORKER_STAGE=production BODY_IMAGE_WORKER_GPU_SNAPSHOT=true "
            "BODY_IMAGE_WORKER_MAX_CONTAINERS=2 BODY_IMAGE_WORKER_DIAGNOSTICS=false modal deploy") in out


# --- 검사한 설정 = 실제 배포 설정 ---


def lookddak_profile(monkeypatch, secret_ok=True):
    profiles = [{"name": "lookddak", "workspace": "lookddak", "active": True}]
    monkeypatch.setattr(dc, "modal_cli", lambda *args, **kwargs: json.dumps(profiles))
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    seen = []
    monkeypatch.setattr(dc, "check_secret", lambda name: seen.append(name) or dc.Check("secret", secret_ok, name))
    return seen


def resolved(env):
    """배포 파일의 resolve_deploy_config 로 env 를 해석한 값(필드 dict. 모듈을 다시 불러와도 비교할 수 있게)."""
    from dataclasses import asdict

    module, _ = dc.load_worker("production")
    return asdict(module.resolve_deploy_config(env, in_container=False))


def config_of(module):
    from dataclasses import asdict

    return asdict(module.CONFIG)


def test_deploy_env_states_every_deploy_variable():
    module, _ = dc.load_worker("staging")
    assert set(dc.deploy_env("staging")) == set(module.DEPLOY_VARS)
    assert dc.deploy_env("production") == {
        "BODY_IMAGE_WORKER_STAGE": "production", "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "false",
        "BODY_IMAGE_WORKER_MAX_CONTAINERS": "", "BODY_IMAGE_WORKER_DIAGNOSTICS": "false",
    }


def test_staging_deploy_runs_modal_with_exactly_the_checked_settings(monkeypatch):
    lookddak_profile(monkeypatch)
    # 셸에 남은 값: 같은 값의 STAGE 는 통과, 배포 환경에는 검사값만 들어가야 한다
    monkeypatch.setenv("BODY_IMAGE_WORKER_STAGE", "staging")
    ran = {}
    monkeypatch.setattr(dc.subprocess, "run",
                        lambda cmd, env, check: ran.update(cmd=cmd, env=env) or type("R", (), {"returncode": 0}))

    assert dc.main(["deploy", "--stage", "staging", "--workspace", "lookddak"]) == 0

    deployed = {k: v for k, v in ran["env"].items() if k.startswith("BODY_IMAGE_WORKER_")}
    assert deployed == dc.deploy_env("staging")
    assert ran["env"]["MODAL_PROFILE"] == "lookddak"
    module, _ = dc.load_worker("staging")
    assert resolved(deployed) == config_of(module)  # 같은 함수·같은 값 → 같은 설정


def test_shell_leftovers_are_not_passed_to_modal_deploy():
    env = dc.deploy_process_env(dc.deploy_env("staging"), "p", environ={
        "PATH": "/bin", "BODY_IMAGE_WORKER_SECRET_NAME": "lookddak-body-image-worker",
        "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "false"})

    assert env["PATH"] == "/bin" and "BODY_IMAGE_WORKER_SECRET_NAME" not in env
    assert env["BODY_IMAGE_WORKER_GPU_SNAPSHOT"] == "true"


@pytest.mark.parametrize(("name", "value"), [
    ("BODY_IMAGE_WORKER_STAGE", "production"),  # 다른 stage
    ("BODY_IMAGE_WORKER_SECRET_NAME", "lookddak-body-image-worker"),  # staging 에 production Secret
    ("BODY_IMAGE_WORKER_PROXY_AUTH", "false"),
    ("BODY_IMAGE_WORKER_MAX_CONTAINERS", "3"),
])
def test_conflicting_shell_values_fail_preflight_before_the_secret_check(monkeypatch, name, value):
    seen = lookddak_profile(monkeypatch)
    monkeypatch.setenv(name, value)

    _, checks = dc.preflight("staging", "lookddak")

    assert [c.name for c in checks if not c.ok] == ["deploy env sources"]
    assert seen == []  # 대상이 확인되지 않으면 Secret 을 조회하지 않는다


def test_conflicting_dotenv_values_fail_preflight(monkeypatch, tmp_path):
    lookddak_profile(monkeypatch)
    dotenv = tmp_path / ".env"
    dotenv.write_text("DATABASE_URL=postgresql://user:pw@h/db\nexport BODY_IMAGE_WORKER_STAGE='staging'\n")
    monkeypatch.setattr(dc, "DOTENV_FILE", dotenv)

    _, checks = dc.preflight("production", "lookddak")

    [failed] = [c for c in checks if not c.ok]
    assert failed.name == "deploy env sources" and ".env:BODY_IMAGE_WORKER_STAGE" in failed.detail
    assert "pw" not in failed.detail  # 다른 .env 값은 읽어 보여 주지 않는다


def test_matching_dotenv_and_shell_values_pass(monkeypatch, tmp_path):
    lookddak_profile(monkeypatch)
    dotenv = tmp_path / ".env"
    dotenv.write_text("BODY_IMAGE_WORKER_STAGE=production\n")
    monkeypatch.setattr(dc, "DOTENV_FILE", dotenv)
    monkeypatch.setenv("BODY_IMAGE_WORKER_GPU_SNAPSHOT", "true")
    monkeypatch.setenv("BODY_IMAGE_WORKER_MAX_CONTAINERS", "2")

    _, checks = dc.preflight("production", "lookddak", gpu_snapshot=True, max_containers=2)

    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


@pytest.mark.parametrize("name", ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_ENVIRONMENT"])
def test_workspace_that_cannot_be_confirmed_fails_preflight(monkeypatch, name):
    seen = lookddak_profile(monkeypatch)
    monkeypatch.setenv(name, "x")

    _, checks = dc.preflight("production", "lookddak")

    assert [c.name for c in checks if not c.ok] == ["modal target env"]
    assert "x" not in checks[1].detail.replace("unset", "") and seen == []


def test_disabled_proxy_auth_fails_the_config_check():
    module, calls = dc.load_worker("staging")
    module.REQUIRES_PROXY_AUTH = False

    assert [c.name for c in dc.check_deploy_config(module, calls, "staging") if not c.ok] == ["proxy auth"]


def test_production_preflight_prints_a_manual_command_that_reproduces_the_checked_config(monkeypatch, capsys):
    lookddak_profile(monkeypatch)
    monkeypatch.setattr(dc.subprocess, "run", lambda *a, **k: pytest.fail("production 은 배포하지 않는다"))

    assert dc.main(["preflight", "--stage", "production", "--workspace", "lookddak", "--gpu-snapshot",
                    "--max-containers", "2"]) == 0

    command = capsys.readouterr().out.splitlines()[-1]
    assigns = dict(word.split("=", 1) for word in command.split() if word.startswith("BODY_IMAGE_WORKER_"))
    checked, _ = dc.load_worker("production", gpu_snapshot=True, max_containers=2)
    assert resolved(assigns) == config_of(checked)


def test_cpu_manual_command_overrides_a_leftover_gpu_flag():
    command = dc.manual_deploy_command(dc.deploy_env("production"))
    # 셸에 GPU_SNAPSHOT=true·MAX_CONTAINERS 가 남아 있어도 명령의 값이 덮어쓴다
    assert "BODY_IMAGE_WORKER_GPU_SNAPSHOT=false" in command and "BODY_IMAGE_WORKER_MAX_CONTAINERS=''" in command
    assert resolved({"BODY_IMAGE_WORKER_STAGE": "production", "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "false",
                     "BODY_IMAGE_WORKER_MAX_CONTAINERS": "", "BODY_IMAGE_WORKER_DIAGNOSTICS": "false"})["gpu_snapshot"] is False
