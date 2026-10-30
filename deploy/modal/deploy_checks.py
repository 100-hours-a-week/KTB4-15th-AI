import argparse
import ast
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

WORKER_FILE = Path(__file__).resolve().parent / "body_image_worker.py"
DOTENV_FILE = Path(__file__).resolve().parents[2] / ".env"
APP_NAMES = {"production": "lookddak-body-image-worker", "staging": "lookddak-body-image-worker-staging"}
LABELS = {"production": "lookddak-body-image", "staging": "lookddak-body-image-staging"}
SECRET_NAMES = {"production": "lookddak-body-image-worker", "staging": "lookddak-body-image-worker-staging"}
PRODUCTION_WORKSPACE = "lookddak" 
EXPECTED_GPU = "T4"
PROTECTED_WORKSPACES = frozenset({"lookddak"})
REQUIRED_SECRET_KEYS = ("INTERNAL_API_KEY", "S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
STAGING_BUCKET_PREFIX = "lookddak-dev-images"
DEPLOY_VAR_PREFIX = "BODY_IMAGE_WORKER_"
MODAL_TARGET_OVERRIDES = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_ENVIRONMENT")
_IMAGE_METHODS = ("env", "run_commands", "pip_install", "apt_install", "add_local_file", "add_local_dir")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


# ---------------------------------------------------------------- preflight


def deploy_env(stage: str, diagnostics: bool = False, gpu_snapshot: bool = False,
               max_containers: int | None = None) -> dict[str, str]:
    """배포에 쓸 BODY_IMAGE_WORKER_* 전부. 검사(load_worker)·staging 배포·production 수동 명령이 이 값을 같이 쓴다.

    네 값을 모두 명시한다. 셸에 같은 이름이 남아 있어도 이 값이 덮어쓴다(빈 상한은 "상한 없음").
    """
    return {
        "BODY_IMAGE_WORKER_STAGE": stage,
        "BODY_IMAGE_WORKER_GPU_SNAPSHOT": "true" if stage == "staging" or gpu_snapshot else "false",
        "BODY_IMAGE_WORKER_MAX_CONTAINERS": "" if max_containers is None else str(max_containers),
        "BODY_IMAGE_WORKER_DIAGNOSTICS": "true" if diagnostics else "false",
    }


def deploy_process_env(wanted: dict[str, str], profile: str, environ=None) -> dict[str, str]:
    """`modal deploy` 에 넘길 환경. 셸의 BODY_IMAGE_WORKER_* 는 버리고 검사한 값(wanted)만 넣는다."""
    environ = os.environ if environ is None else environ
    base = {name: value for name, value in environ.items() if not name.startswith(DEPLOY_VAR_PREFIX)}
    return {**base, **wanted, "MODAL_PROFILE": profile}


def manual_deploy_command(wanted: dict[str, str]) -> str:
    """production 수동 배포 명령. 검사한 네 값을 모두 명시한다(셸에 남은 값이 끼어들지 않게)."""
    assigns = " ".join(f"{name}={shlex.quote(value)}" for name, value in wanted.items())
    return f"MODAL_PROFILE=<{PRODUCTION_WORKSPACE} profile> {assigns} modal deploy deploy/modal/body_image_worker.py"


def load_worker(stage: str, diagnostics: bool = False, gpu_snapshot: bool = False, max_containers: int | None = None):
    import modal

    calls: list[tuple[str, object]] = []
    originals = {name: getattr(modal.Image, name) for name in _IMAGE_METHODS}

    def make_spy(name, original):
        def spy(self, *args, **kwargs):
            calls.append((name, dict(args[0]) if name == "env" and args else None))
            return original(self, *args, **kwargs)
        return spy

    saved = {name: os.environ.pop(name) for name in list(os.environ) if name.startswith(DEPLOY_VAR_PREFIX)}
    os.environ.update(deploy_env(stage, diagnostics, gpu_snapshot, max_containers))
    try:
        for name, original in originals.items():
            setattr(modal.Image, name, make_spy(name, original))
        spec = importlib.util.spec_from_file_location("body_image_worker_preflight", WORKER_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, calls
    finally:
        for name, original in originals.items():
            setattr(modal.Image, name, original)
        for name in [name for name in os.environ if name.startswith(DEPLOY_VAR_PREFIX)]:
            del os.environ[name]
        os.environ.update(saved)


def check_deploy_config(module, calls, stage: str, gpu_snapshot: bool = False,
                        max_containers: int | None = None) -> list[Check]:
    expect_gpu = stage == "staging" or gpu_snapshot
    checks = [
        Check("stage", module.STAGE == stage, f"STAGE={module.STAGE!r}"),
        Check("app name", module.APP_NAME == APP_NAMES[stage], f"APP_NAME={module.APP_NAME!r}"),
        Check("label", module.ENDPOINT_LABEL == LABELS[stage], f"label={module.ENDPOINT_LABEL!r}"),
        Check("secret name", module.SECRET_NAME == SECRET_NAMES[stage], f"SECRET_NAME={module.SECRET_NAME!r}"),
        Check("proxy auth", module.REQUIRES_PROXY_AUTH is True, f"requires_proxy_auth={module.REQUIRES_PROXY_AUTH!r}"),
        Check("gpu", module.WORKER_GPU == EXPECTED_GPU, f"gpu={module.WORKER_GPU!r}"),
        Check("cpu/memory", (module.WORKER_RESOURCES.get("cpu"), module.WORKER_RESOURCES.get("memory")) == (4, 8192),
              f"cpu={module.WORKER_RESOURCES.get('cpu')!r} memory={module.WORKER_RESOURCES.get('memory')!r}"),
        Check("cpu memory snapshot", module.WORKER_RESOURCES.get("enable_memory_snapshot") is True,
              f"enable_memory_snapshot={module.WORKER_RESOURCES.get('enable_memory_snapshot')!r}"),
        Check("min containers", module.WORKER_RESOURCES.get("min_containers") == 0,
              f"min_containers={module.WORKER_RESOURCES.get('min_containers')!r}"),
        Check("gpu snapshot mode", module.GPU_SNAPSHOT is expect_gpu, f"GPU_SNAPSHOT={module.GPU_SNAPSHOT!r}"),
    ]
    names = [name for name, _ in calls]
    checks.append(Check("code layer last", bool(names) and names[-1] == "add_local_dir" and names.count("add_local_dir") == 1,
                        f"image build order ends with {names[-3:]}"))
    baked = [env for name, env in calls if name == "env" and env and module.IMAGE_STAGE_VAR in env]
    function_env = module.STAGE_ENV.get("env", {})
    if expect_gpu:
        checks += [
            Check("gpu snapshot", module.STAGE_SNAPSHOT == {"experimental_options": {"enable_gpu_snapshot": True}},
                  f"STAGE_SNAPSHOT={module.STAGE_SNAPSHOT!r}"),
            Check("function stage env", function_env.get("BODY_IMAGE_WORKER_STAGE") == stage
                  and function_env.get(module.GPU_SNAPSHOT_VAR) == "true",
                  f"env stage={function_env.get('BODY_IMAGE_WORKER_STAGE')!r} "
                  f"gpu_snapshot={function_env.get(module.GPU_SNAPSHOT_VAR)!r}"),
            Check("image stage env",
                  baked == [{module.IMAGE_STAGE_VAR: stage, module.IMAGE_GPU_SNAPSHOT_VAR: "true"}],
                  f"image env={baked!r}"),
        ]
    else:
        checks += [
            Check("gpu snapshot off", module.STAGE_SNAPSHOT == {}, f"STAGE_SNAPSHOT={module.STAGE_SNAPSHOT!r}"),
            Check("no stage env", module.STAGE_ENV == {}, f"STAGE_ENV={module.STAGE_ENV!r}"),
            Check("no image stage env", baked == [], f"image env={baked!r}"),
        ]
    if stage == "staging":
        checks.append(Check("max containers", module.STAGE_SCALING.get("max_containers") == 1, f"{module.STAGE_SCALING!r}"))
    else:
        registered = module.STAGE_SCALING.get("max_containers")
        if gpu_snapshot:
            ok = isinstance(registered, int) and registered >= 1 and registered == max_containers
            detail = f"max_containers={registered!r} (expected {max_containers!r}; GPU snapshot 에는 필수)"
        else:
            ok = registered == max_containers  # 주지 않으면 None(상한 없음, 기존 그대로)
            detail = f"max_containers={registered!r} (expected {max_containers!r})"
        checks += [
            Check("production max containers", ok, detail),
            Check("no diagnostics", module.DIAGNOSTICS is False, f"DIAGNOSTICS={module.DIAGNOSTICS!r}"),
        ]
    return checks


def check_lifecycle_decorators(path: Path = WORKER_FILE) -> Check:
    """초기화 순서: snap=True·snap=False·exit hook 과 ASGI endpoint 가 하나씩 있고 @modal.method 는 없다."""
    counts = {"enter(snap=True)": 0, "enter(snap=False)": 0, "exit": 0, "asgi_app": 0, "method": 0}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            target = call.func if call else decorator
            name = target.attr if isinstance(target, ast.Attribute) else None
            if name == "enter" and call:
                snap = next((k.value.value for k in call.keywords if k.arg == "snap"), None)
                counts[f"enter(snap={snap})"] = counts.get(f"enter(snap={snap})", 0) + 1
            elif name in counts:
                counts[name] += 1
    expected = {"enter(snap=True)": 1, "enter(snap=False)": 1, "exit": 1, "asgi_app": 1, "method": 0}
    return Check("lifecycle hooks", counts == expected, f"{counts}")


def check_no_private_modal_api(path: Path = WORKER_FILE) -> Check:
    """배포 파일이 Modal 비공개 API(modal._*, _experimental_*·_singleton 속성)를 쓰지 않는다."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("modal._"):
            found.append(node.module)
        elif isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if alias.name.startswith("modal._")]
        elif isinstance(node, ast.Attribute) and node.attr.startswith(("_experimental", "_singleton")):
            found.append(node.attr)
    return Check("no private modal api", not found, ", ".join(found) or "none")


def resolve_profile(expected_workspace: str, profiles: list[dict], env_profile: str | None) -> tuple[str | None, Check]:
    """사용할 profile(MODAL_PROFILE 또는 active)의 Workspace 가 기대한 Workspace 인지."""
    if env_profile:
        chosen = next((p for p in profiles if p.get("name") == env_profile), None)
    else:
        chosen = next((p for p in profiles if p.get("active")), None)
    if chosen is None:
        return None, Check("workspace", False, f"profile 을 찾지 못했습니다(MODAL_PROFILE={env_profile!r})")
    ok = chosen.get("workspace") == expected_workspace
    return chosen["name"], Check("workspace", ok, f"profile={chosen['name']} workspace={chosen.get('workspace')}")


def check_secret(secret_name: str) -> Check:
    """Secret 이 있고 필수 key 가 모두 있는지. 값은 가져오지 않는다(required_keys 는 서버가 확인한다)."""
    import modal

    try:
        modal.Secret.from_name(secret_name, required_keys=list(REQUIRED_SECRET_KEYS)).hydrate()
    except Exception as error:  # noqa: BLE001 - NotFound·필수 key 누락·인증 오류를 모두 실패로 보고한다
        return Check("secret", False, f"{secret_name}: {type(error).__name__}: {str(error)[:200]}")
    return Check("secret", True, f"{secret_name}: required keys {', '.join(REQUIRED_SECRET_KEYS)} present")


def check_modal_target_env(environ) -> Check:
    """MODAL_PROFILE 이 아닌 경로로 Workspace·environment 를 바꾸는 변수가 없는지(있으면 확인할 수 없어 실패)."""
    found = [name for name in MODAL_TARGET_OVERRIDES if environ.get(name)]
    return Check("modal target env", not found,
                 f"셸에 {found} 가 있습니다. unset 한 뒤 MODAL_PROFILE 로만 선택하세요" if found else "none")


def read_dotenv_deploy_vars(path: Path) -> dict[str, str]:
    """.env 에서 BODY_IMAGE_WORKER_* 줄만 읽는다(다른 값은 읽지도 출력하지도 않는다)."""
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if not line.startswith(DEPLOY_VAR_PREFIX) or "=" not in line:
            continue
        name, value = line.split("=", 1)
        found[name.strip()] = value.strip().strip("'\"")
    return found


def check_env_sources(wanted: dict[str, str], environ, dotenv: dict[str, str]) -> Check:
    """셸·.env 의 BODY_IMAGE_WORKER_* 가 검사할 값과 다르면 실패한다(조용히 하나를 고르지 않는다)."""
    conflicts = [
        f"{source}:{name}={value!r}(검사값 {wanted.get(name)!r})"
        for source, values in (("shell", environ), (".env", dotenv))
        for name, value in values.items()
        if name.startswith(DEPLOY_VAR_PREFIX) and value != wanted.get(name)
    ]
    return Check("deploy env sources", not conflicts, "; ".join(conflicts) or "shell·.env 에 다른 값 없음")


def modal_cli(*args: str, profile: str | None = None, timeout: float = 120) -> str:
    env = dict(os.environ)
    if profile:
        env["MODAL_PROFILE"] = profile
    result = subprocess.run(
        ["modal", *args], capture_output=True, text=True, env=env, timeout=timeout, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"modal {' '.join(args[:2])} failed: {result.stderr.strip()[-300:]}")
    return result.stdout


def preflight(stage: str, workspace: str, diagnostics: bool = False, gpu_snapshot: bool = False,
              max_containers: int | None = None) -> tuple[str | None, list[Check]]:
    """읽기 전용 배포 전 검사. production 은 Workspace 가 PRODUCTION_WORKSPACE 여야 한다.

    배포 대상(Workspace·셸/.env 값)이 확인되지 않으면 Secret 을 조회하지 않는다.
    """
    wanted = deploy_env(stage, diagnostics, gpu_snapshot, max_containers)
    profiles = json.loads(modal_cli("profile", "list", "--json"))
    profile, workspace_check = resolve_profile(workspace, profiles, os.environ.get("MODAL_PROFILE"))
    checks = [workspace_check, check_modal_target_env(os.environ)]
    if stage == "production":
        checks.append(Check("production workspace", workspace == PRODUCTION_WORKSPACE,
                            f"--workspace={workspace!r} (production 은 {PRODUCTION_WORKSPACE!r})"))
    checks.append(check_env_sources(wanted, os.environ, read_dotenv_deploy_vars(DOTENV_FILE)))
    target_ok = all(check.ok for check in checks)
    try:
        module, calls = load_worker(stage, diagnostics, gpu_snapshot, max_containers)
    except Exception as error:  # noqa: BLE001 - 배포 파일 자체의 설정 오류(staging 이름 충돌 등)도 검사 실패다
        return profile, checks + [Check("load deploy file", False, f"{type(error).__name__}: {error}")]
    checks += check_deploy_config(module, calls, stage, gpu_snapshot, max_containers)
    checks.append(check_lifecycle_decorators())
    checks.append(check_no_private_modal_api())
    if profile and target_ok:
        os.environ["MODAL_PROFILE"] = profile  # SDK 호출(Secret 확인)도 같은 profile 로
        checks.append(check_secret(module.SECRET_NAME))
    return profile, checks


def print_checks(checks: list[Check]) -> bool:
    for check in checks:
        print(f"[{'PASS' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    return all(check.ok for check in checks)


# ---------------------------------------------------------------- smoke: 로그 판정 (순수 함수)

LOG_LINE = re.compile(r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d[+-]\d\d:\d\d) (?P<container>ta-[A-Z0-9]+) (?P<msg>.*)$")
INIT_FAILURE_MARKER = "Runner failed with exception"  # enter hook 등 사용자 코드 예외
# Modal 이 snapshot 을 만들다 실패했다(2026-10-25 L4 실측). 이어서 Runner killed 가 오기도, 로그가 끊기기도 한다.
SNAPSHOT_FAILURE_MARKER = "Failed creating Function memory snapshot"
# Modal 이 snapshot 복원에 실패했다(2026-10-30 T4 실측: "Transient snapshot error: failed to restore container from
# snapshot with exit code 137. Will retry with no snapshots."). 같은 컨테이너가 snapshot 없이 다시 초기화한다.
_RESTORE_FAILURE = re.compile(r"Transient snapshot error|failed to restore container from snapshot")
# 프로세스가 신호로 죽었다. 원인(메모리·snapshot·기타)은 이 줄만으로 알 수 없다.
_RUNNER_KILLED = re.compile(r"Runner killed \((?P<signal>\w+)\), exit code: (?P<code>\d+)")
_GPU_NAME = re.compile(r"snap=True done .* gpu=(?P<gpu>.+?) warmup_ms=")
_PROFILE = re.compile(r"rembg onnxruntime profile: runs=(?P<runs>\[.*\]) parse_ms")
_REQUEST = re.compile(r"(?P<method>GET|POST) (?P<path>/\S*) -> (?P<status>\d{3})")
# Modal 접근 로그의 시간: duration 은 요청 도착부터 응답까지(컨테이너 배정·복원 대기 포함), execution 은 실행 시간.
_REQUEST_TIMES = re.compile(r"duration: (?P<duration>[\d.]+) (?P<du>m?s), execution: (?P<execution>[\d.]+) (?P<eu>m?s)")


@dataclass
class ContainerLog:
    container: str
    first_seen: datetime
    creating_gpu: bool = False
    creating_cpu: bool = False
    restoring: bool = False
    restore_error: str | None = None  # 복원 실패 줄(이 줄 뒤의 snap=True·runtime=ready 는 재초기화 결과다)
    init_failures: list[str] = field(default_factory=list)
    snapshot_failed: bool = False
    runner_killed: str | None = None  # 예: "SIGKILL exit 137"
    stage_ok: bool = False
    gpu_name: str | None = None
    snap_true_done: bool = False
    runtime_ready: bool = False
    requests: list[tuple[str, str, int]] = field(default_factory=list)
    request_times: dict[str, dict[str, float]] = field(default_factory=dict)
    profile_runs: list[dict] | None = None


def summarize_logs(lines: list[str], since: datetime) -> dict[str, ContainerLog]:
    """`modal app logs --timestamps --show-container-id` 출력을 컨테이너별로 요약한다.

    since 이후 줄만 본다. since 전에 이미 로그를 남긴 컨테이너는 이전 배포·이전 단계의 것이므로, since 뒤에 그
    컨테이너가 남긴 줄(종료 시 Runner killed 등)도 판정에 넣지 않는다(smoke 는 컨테이너 0 에서 시작한다).
    """
    parsed = [(m, datetime.fromisoformat(m["ts"])) for m in map(LOG_LINE.match, (line.rstrip("\n") for line in lines)) if m]
    earlier = {m["container"] for m, ts in parsed if ts < since}
    containers: dict[str, ContainerLog] = {}
    for match, ts in parsed:
        if ts < since or match["container"] in earlier:
            continue
        msg = match["msg"]
        log = containers.setdefault(match["container"], ContainerLog(match["container"], ts))
        if "Creating GPU memory snapshot" in msg:
            log.creating_gpu = True
        elif "Creating CPU memory snapshot" in msg:
            log.creating_cpu = True
        elif "Restoring Function from memory snapshot" in msg:
            log.restoring = True
        if _RESTORE_FAILURE.search(msg):
            log.restore_error = msg[:200]
            # 실패 앞의 준비 완료 표시는 버린다. 재초기화가 끝났는지는 이 줄 뒤의 로그로만 본다.
            log.snap_true_done = log.runtime_ready = False
        if INIT_FAILURE_MARKER in msg:
            log.init_failures.append(msg.split(INIT_FAILURE_MARKER, 1)[1].strip(": ")[:200])
        if SNAPSHOT_FAILURE_MARKER in msg:
            log.snapshot_failed = True
        if killed := _RUNNER_KILLED.search(msg):
            log.runner_killed = f"{killed['signal']} exit {killed['code']}"
        if "lifecycle: stage=staging gpu_snapshot=True" in msg:
            log.stage_ok = True
        if "snap=True done" in msg:
            log.snap_true_done = True
        if (gpu := _GPU_NAME.search(msg)) and gpu["gpu"] != "None":
            log.gpu_name = gpu["gpu"]
        if "snap=False done" in msg and "runtime=ready" in msg:
            log.runtime_ready = True
        if profile := _PROFILE.search(msg):
            try:
                log.profile_runs = json.loads(profile["runs"])
            except ValueError:
                log.profile_runs = None
        if request := _REQUEST.search(msg):
            log.requests.append((request["method"], request["path"], int(request["status"])))
            if times := _REQUEST_TIMES.search(msg):
                seconds = {name: float(times[name]) / (1000 if times[unit] == "ms" else 1)
                           for name, unit in (("duration", "du"), ("execution", "eu"))}
                log.request_times[f"{request['method']} {request['path']}"] = seconds
    return containers


def startup_failures(containers: dict[str, ContainerLog]) -> list[dict]:
    """컨테이너별 시작 실패를 분류한다. 한 컨테이너는 하나로만 센다.

    - snapshot_creation_failed: `Failed creating Function memory snapshot` 이 있다. 뒤따르는 Runner killed 는 그 결과로
      본다(같은 컨테이너). snapshot 단계에서 멈춘 것이 확실하므로 치명적이다.
    - runner_failed: `Runner failed with exception` (enter hook 의 예외, fail-fast 포함).
    - runner_killed: 위 둘 없이 Runner killed 만 있다. 원인은 확정하지 않는다(메모리·기타). snapshot 실패로 보지 않는다.
    snapshot 생성 중(`Creating GPU memory snapshot`)인 것만으로는 실패가 아니다(정상 MISS).
    """
    failures = []
    for c in containers.values():
        if c.snapshot_failed:
            kind, detail = "snapshot_creation_failed", f"Failed creating Function memory snapshot (gpu={c.gpu_name})"
            if c.runner_killed:
                detail += f", then runner killed ({c.runner_killed})"
        elif c.init_failures:
            kind, detail = "runner_failed", c.init_failures[0]
        elif c.runner_killed:
            kind, detail = "runner_killed", f"runner killed ({c.runner_killed}), cause not confirmed by other logs"
        else:
            continue
        if c.restore_error:
            detail += " (snapshot 복원 실패 뒤 재초기화 중)"
        failures.append({"kind": kind, "container": c.container, "detail": detail})
    return failures


def init_failures(containers: dict[str, ContainerLog]) -> list[str]:
    return [f"{f['container']}: {f['kind']}: {f['detail']}" for f in startup_failures(containers)]


def _failure_status(containers: dict[str, ContainerLog]) -> tuple[str, str] | None:
    failures = startup_failures(containers)
    if not failures:
        return None
    snapshot = [f for f in failures if f["kind"] == "snapshot_creation_failed"]
    first = snapshot[0] if snapshot else failures[0]
    status = "snapshot_failed" if snapshot else "init_failure"
    return status, f"{len(failures)} startup failure(s): {first['container']}: {first['kind']}: {first['detail']}"


def _served(containers, method: str, path: str) -> ContainerLog | None:
    for log in containers.values():
        if any(m == method and p == path and s == 200 for m, p, s in log.requests):
            return log
    return None


def snapshot_state(log: ContainerLog) -> tuple[str, str]:
    """한 컨테이너의 snapshot 상태와 이유. 그 컨테이너의 줄만 본다(다른 컨테이너 줄은 summarize_logs 가 나눈다).

    - snapshot_creation_failed: `Failed creating Function memory snapshot` 이 있다.
    - restore_failed_fallback: 복원 시도 → 복원 오류(Transient snapshot error) → 그 뒤 runtime=ready.
      Modal 이 snapshot 없이 다시 초기화해 성공한 경우다. 복원 오류가 있으면 HIT 보다 이 판정이 먼저다.
    - snapshot_hit: 복원 시도, 복원 오류 없음, snap=True 재실행 없음, runtime=ready.
    - snapshot_miss: GPU snapshot 을 새로 만들고(처음이든 다시든) snap=True(GPU 이름)·runtime=ready 까지 끝났다.
    그 밖에 init_failure(Runner failed/killed), snapshot_disabled(복원·GPU snapshot 생성 표시 없음),
    incomplete(판정에 필요한 줄이 없음). 줄이 모자라면 정상 상태로 추측하지 않는다.
    """
    name = f"container={log.container}"
    if log.snapshot_failed:
        return "snapshot_creation_failed", f"{name} Failed creating Function memory snapshot (gpu={log.gpu_name})"
    if log.init_failures or log.runner_killed:
        cause = log.init_failures[0] if log.init_failures else f"runner killed ({log.runner_killed})"
        after = " (snapshot 복원 실패 뒤 재초기화 중)" if log.restore_error else ""
        return "init_failure", f"{name} {cause}{after}"
    if log.restore_error:
        if not log.restoring:
            return "incomplete", f"{name} 복원 오류가 있지만 'Restoring Function from memory snapshot' 이 없습니다"
        if not log.runtime_ready:
            return "incomplete", f"{name} 복원 실패 뒤 재초기화 완료(runtime=ready) 로그가 없습니다"
        return "restore_failed_fallback", f"{name} snapshot 복원 실패 후 snapshot 없이 재초기화: {log.restore_error}"
    if log.restoring:
        if log.snap_true_done:
            return "incomplete", f"{name} 복원 표시 뒤 snap=True 가 다시 실행됐지만 복원 오류 로그가 없습니다"
        if not log.runtime_ready:
            return "incomplete", f"{name} 복원 뒤 runtime=ready 로그가 없습니다"
        return "snapshot_hit", name
    if log.creating_cpu or not log.creating_gpu:
        return "snapshot_disabled", f"{name} 에 Restoring/Creating GPU snapshot 이 없습니다"
    missing = [n for n, ok in (("gpu name", log.gpu_name), ("runtime ready", log.runtime_ready)) if not ok]
    if missing:
        return "incomplete", f"{name} snapshot 생성 lifecycle 로그 누락: {missing}"
    return "snapshot_miss", f"{name} gpu={log.gpu_name} (새 snapshot 을 만들었다)"


# judge_* 가 돌려주던 이름은 그대로 둔다(smoke 보고서·호출처 호환).
_JUDGE_STATUS = {"snapshot_hit": "hit", "snapshot_miss": "miss", "snapshot_creation_failed": "snapshot_failed"}


def _note_other_restore_failures(containers, served: ContainerLog, reason: str) -> str:
    """응답한 컨테이너만 판정하되, 같은 구간의 다른 컨테이너 복원 실패는 이유에 남긴다(숨기지 않는다)."""
    others = [c.container for c in containers.values() if c is not served and c.restore_error]
    return f"{reason}; 다른 컨테이너의 snapshot 복원 실패: {others}" if others else reason


def judge_warmup(containers: dict[str, ContainerLog]) -> tuple[str, str]:
    """/health 로 snapshot 이 만들어졌는지.

    ("ok"|"restore_failed_fallback"|"snapshot_failed"|"init_failure"|"snapshot_disabled"|"incomplete", 이유).
    """
    if failed := _failure_status(containers):
        # 실패 뒤 다른 컨테이너가 snapshot 없이 200 을 내도(Modal 재시도) 성공으로 보지 않는다.
        return failed
    served = _served(containers, "GET", "/health")
    if served is None:
        return "incomplete", "GET /health 200 로그가 없습니다"
    state, reason = snapshot_state(served)
    if state in ("snapshot_hit", "snapshot_miss", "restore_failed_fallback") and not served.stage_ok:
        return "incomplete", f"container={served.container} lifecycle 로그 누락: ['stage']"
    if state == "snapshot_hit":
        # 이미 이 배포의 snapshot 이 있어 복원됐다(다시 실행한 경우). snap=True 로그(GPU 이름)는 없다.
        reason, state = f"container={served.container} snapshot already prepared (restored)", "ok"
    elif state == "snapshot_miss":
        reason, state = f"container={served.container} gpu={served.gpu_name}", "ok"
    return _JUDGE_STATUS.get(state, state), _note_other_restore_failures(containers, served, reason)


def judge_validation(containers: dict[str, ContainerLog]) -> tuple[str, str]:
    """검증 요청을 처리한 컨테이너가 snapshot 에서 복원됐는지(snapshot_state 를 기존 이름으로 돌려준다).

    ("hit"|"miss"|"restore_failed_fallback"|"snapshot_failed"|"init_failure"|"snapshot_disabled"|"incomplete", 이유).
    miss 는 snapshot 을 새로 만들었지만 실패하지 않은 경우다(정상 MISS). restore_failed_fallback 은 HIT 가 아니다.
    """
    if failed := _failure_status(containers):
        return failed
    served = _served(containers, "POST", "/api/v1/validation/body-image")
    if served is None:
        return "incomplete", "POST 200 로그가 없습니다"
    state, reason = snapshot_state(served)
    return _JUDGE_STATUS.get(state, state), _note_other_restore_failures(containers, served, reason)


def cuda_check(containers: dict[str, ContainerLog]) -> tuple[bool | None, str]:
    """진단 모드의 ORT profile 이 있으면 마지막 실행(복원 뒤 요청)이 모두 CUDA 노드인지. 없으면 None."""
    served = _served(containers, "POST", "/api/v1/validation/body-image")
    runs = served.profile_runs if served else None
    if not runs:
        return None, "profile 없음(진단 모드가 아니면 측정하지 않는다)"
    last = runs[-1]
    nodes = last.get("nodes", {})
    ok = bool(nodes) and set(nodes) == {"CUDAExecutionProvider"}
    return ok, f"run {last.get('run')} nodes={nodes} model_run_ms={last.get('model_run_ms')}"


# ---------------------------------------------------------------- smoke: Modal·HTTP 호출

CREDENTIAL_NAMES = {
    # 첫 번째로 찾은 이름을 쓴다. 값은 출력하지 않는다.
    "modal_key": ("BODY_IMAGE_SMOKE_MODAL_KEY", "BODY_IMAGE_VALIDATION_PROXY_KEY"),
    "modal_secret": ("BODY_IMAGE_SMOKE_MODAL_SECRET", "BODY_IMAGE_VALIDATION_PROXY_SECRET"),
    "api_key": ("BODY_IMAGE_SMOKE_API_KEY", "BODY_IMAGE_STAGING_API_KEY"),
}


def load_credentials(files: list[str]) -> dict[str, str]:
    values = {}
    for path in files:
        for line in Path(path).expanduser().read_text().splitlines():
            if match := re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip()):
                values[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if k.startswith("BODY_IMAGE_")})
    credentials = {}
    for key, names in CREDENTIAL_NAMES.items():
        found = next((values[name] for name in names if values.get(name)), None)
        if not found:
            raise SystemExit(f"인증 값이 없습니다: {' 또는 '.join(names)}")
        credentials[key] = found
    return credentials


class ModalOps:
    """smoke 가 쓰는 Modal 공개 CLI 호출. 테스트에서는 fake 로 바꾼다."""

    def __init__(self, profile: str):
        self.profile = profile

    def cli(self, *args: str, timeout: float = 120) -> str:
        return modal_cli(*args, profile=self.profile, timeout=timeout)

    def app_info(self, app_name: str) -> dict:
        return json.loads(self.cli("app", "info", app_name, "--json"))

    def function_info(self, function_id: str) -> dict:
        return json.loads(self.cli("function", "info", function_id, "--json"))

    def live_containers(self, app_id: str) -> list[dict]:
        return json.loads(self.cli("container", "list", "--app-id", app_id, "--json"))

    def logs(self, app_id: str) -> list[str]:
        out = self.cli("app", "logs", app_id, "--since", "30m", "--tail", "3000", "--timestamps", "--show-container-id")
        return out.splitlines()

    def rollback(self, app_id: str) -> str:
        return self.cli("app", "rollback", app_id)

    def stop(self, app_id: str) -> str:
        return self.cli("app", "stop", "-y", app_id)


@dataclass
class SmokeLimits:
    health_timeout_s: float = 150
    max_init_failures: int = 2  # 시작 실패(종류 무관)가 이만큼 보이면 Modal 의 재시작 반복으로 보고 멈춘다
    max_snapshot_failures: int = 1  # snapshot 생성 실패는 치명적이라 이만큼(기본 1번) 보이면 바로 멈춘다
    scaledown_timeout_s: float = 180
    validation_timeout_s: float = 120
    max_miss_retries: int = 2  # 일시적 MISS·복원 실패 fallback 으로 보고 다시 확인할 횟수(이후에도 HIT 가 없으면 실패)
    poll_interval_s: float = 5
    log_settle_s: float = 6  # 응답 직후 로그가 모일 때까지 기다리는 시간


class SmokeFailure(RuntimeError):
    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason = stage, reason


def wait_for_zero(ops, app_id: str, limits: SmokeLimits, sleep=time.sleep, clock=time.monotonic) -> float:
    started = clock()
    while True:
        if not ops.live_containers(app_id):
            return clock() - started
        if clock() - started > limits.scaledown_timeout_s:
            raise SmokeFailure("scaledown", f"{limits.scaledown_timeout_s}s 안에 컨테이너가 0 이 되지 않았습니다")
        sleep(limits.poll_interval_s)


def request_with_failure_watch(ops, app_id, send, since, timeout_s, limits, sleep=time.sleep, clock=time.monotonic):
    """요청을 보내는 동안 로그를 보며 시작 실패가 상한에 닿으면 timeout 까지 기다리지 않고 멈춘다.

    snapshot 생성 실패는 max_snapshot_failures(기본 1), 그 밖의 시작 실패(Runner failed·원인 미확정 Runner killed)는
    종류와 상관없이 max_init_failures(기본 2)에서 멈춘다.

    (응답, HTTP 요청 시간 초) 를 돌려준다. 시간은 요청을 보낸 스레드에서 잰 값이라 로그 polling 대기를 포함하지 않는다.
    """
    result: dict = {}

    def run():
        sent = time.monotonic()
        try:
            result["response"] = send(timeout_s)
        except Exception as error:  # noqa: BLE001 - timeout·연결 오류를 결과로 넘긴다
            result["error"] = error
        finally:
            result["seconds"] = time.monotonic() - sent

    thread = threading.Thread(target=run, daemon=True)
    started = clock()
    thread.start()
    while thread.is_alive():
        thread.join(limits.poll_interval_s)
        failures = startup_failures(summarize_logs(ops.logs(app_id), since))
        snapshot = [f for f in failures if f["kind"] == "snapshot_creation_failed"]
        if len(snapshot) >= limits.max_snapshot_failures:
            first = snapshot[0]
            raise SmokeFailure("snapshot", f"GPU snapshot 생성 실패({len(snapshot)}회): {first['container']}: "
                                           f"{first['detail']}")
        if len(failures) >= limits.max_init_failures:
            first = failures[0]
            raise SmokeFailure("init", f"시작 실패가 반복됩니다({len(failures)}회): {first['container']}: "
                                       f"{first['kind']}: {first['detail']}")
        if clock() - started > timeout_s + limits.poll_interval_s:
            raise SmokeFailure("timeout", f"{timeout_s}s 안에 응답이 없습니다")
    if "error" in result:
        raise SmokeFailure("request", f"{type(result['error']).__name__}: {str(result['error'])[:200]}")
    sleep(limits.log_settle_s)
    return result["response"], round(result["seconds"], 2)


def s3_head(secret_name: str, key: str) -> dict:
    """staging Secret 으로 업로드 객체를 HEAD/GET 한다(임시 Modal 함수). dev 버킷이 아니면 조회하지 않는다."""
    import modal

    app = modal.App("lookddak-smoke-s3-check")
    image = modal.Image.debian_slim().pip_install("boto3", "pillow")

    @app.function(image=image, secrets=[modal.Secret.from_name(secret_name)], serialized=True, timeout=120)
    def check(object_key: str) -> dict:
        import os as _os
        from io import BytesIO

        import boto3
        from PIL import Image

        bucket = _os.environ["S3_BUCKET"]
        if not bucket.startswith(STAGING_BUCKET_PREFIX):
            return {"refused": "not a dev bucket"}
        s3 = boto3.client("s3", region_name=_os.environ.get("AWS_REGION"))
        head = s3.head_object(Bucket=bucket, Key=object_key)
        data = s3.get_object(Bucket=bucket, Key=object_key)["Body"].read()
        with Image.open(BytesIO(data)) as png:
            png.load()
            return {"bucket": bucket, "content_type": head["ContentType"], "bytes": len(data),
                    "format": png.format, "mode": png.mode, "size": list(png.size)}

    with app.run():
        return check.remote(key)


def smoke(ops, app_name: str, credentials: dict, image_path: Path, limits: SmokeLimits,
          http=None, s3_checker=s3_head, secret_name: str | None = None, clock=time.time) -> dict:
    """/health 로 snapshot 생성 → 컨테이너 0 → 실제 검증 요청 → HIT·CUDA·S3 확인. 실패하면 SmokeFailure."""
    import httpx

    report: dict = {"app": app_name}
    info = ops.app_info(app_name)
    app_id = info["app_id"]
    [function] = info["functions"].values()
    function_info = ops.function_info(function["id"])
    url = function_info["method_details"]["web"]["web_url"]
    report["deployed"] = {"version": info["lifecycle"].get("version"), "gpus": function_info.get("gpus"),
                          "max_containers": function_info.get("max_containers"), "url": url}
    if [gpu for gpu, _ in function_info.get("gpus") or []] != [EXPECTED_GPU]:
        raise SmokeFailure("config", f"등록된 GPU 가 {EXPECTED_GPU} 가 아닙니다: {function_info.get('gpus')}")
    if function_info["method_details"]["web"].get("unauthenticated"):
        raise SmokeFailure("config", "proxy auth 가 꺼져 있습니다")
    proxy = {"Modal-Key": credentials["modal_key"], "Modal-Secret": credentials["modal_secret"]}
    client = http or httpx.Client(headers=proxy, follow_redirects=False)

    report["pre_zero_wait_s"] = round(wait_for_zero(ops, app_id, limits), 1)
    since = datetime.fromtimestamp(clock() - 2, UTC)
    response, seconds = request_with_failure_watch(
        ops, app_id, lambda t: client.get(f"{url}/health", timeout=httpx.Timeout(t, connect=10)), since,
        limits.health_timeout_s, limits)
    report["health"] = {"status": response.status_code, "seconds": seconds}
    status, reason = judge_warmup(summarize_logs(ops.logs(app_id), since))
    report["snapshot_creation"] = {"status": status, "reason": reason}
    if status == "snapshot_failed":
        raise SmokeFailure("snapshot", reason)
    # restore_failed_fallback: 복원할 snapshot 은 있었다. 쓸 수 있는지는 아래 검증 요청의 HIT 로 확인한다.
    if response.status_code != 200 or status not in ("ok", "restore_failed_fallback"):
        raise SmokeFailure("warmup", f"health={response.status_code} snapshot={status}: {reason}")

    body = image_path.read_bytes()
    attempts = []
    for attempt in range(1, limits.max_miss_retries + 2):
        wait_for_zero(ops, app_id, limits)
        since = datetime.fromtimestamp(clock() - 2, UTC)
        response, seconds = request_with_failure_watch(
            ops, app_id,
            lambda t: client.post(f"{url}/api/v1/validation/body-image",
                                  files={"image": (image_path.name, body, "image/jpeg")},
                                  headers={"Authorization": f"Bearer {credentials['api_key']}"},
                                  timeout=httpx.Timeout(t, connect=10)),
            since, limits.validation_timeout_s, limits)
        containers = summarize_logs(ops.logs(app_id), since)
        status, reason = judge_validation(containers)
        cuda_ok, cuda_detail = cuda_check(containers)
        served = _served(containers, "POST", "/api/v1/validation/body-image")
        modal_times = served.request_times.get("POST /api/v1/validation/body-image") if served else None
        attempts.append({"attempt": attempt, "http": response.status_code, "seconds": seconds, "snapshot": status,
                         "reason": reason, "cuda": cuda_ok, "cuda_detail": cuda_detail, "modal": modal_times})
        if response.status_code != 200:
            raise SmokeFailure("validation", f"HTTP {response.status_code}")
        if status == "snapshot_failed":
            report["validation"] = attempts
            raise SmokeFailure("snapshot", reason)
        if status in ("init_failure", "snapshot_disabled", "incomplete"):
            report["validation"] = attempts
            raise SmokeFailure("restore", f"{status}: {reason}")
        if cuda_ok is False:
            report["validation"] = attempts
            raise SmokeFailure("cuda", cuda_detail)
        if status == "hit":
            break
        # miss·restore_failed_fallback 은 일시적일 수 있어 다시 확인한다.
    report["validation"] = attempts
    if attempts[-1]["snapshot"] != "hit":
        statuses = [a["snapshot"] for a in attempts]
        raise SmokeFailure("restore", f"{len(attempts)}번 모두 HIT 가 아닙니다 {statuses}(장기 HIT 율 문제인지 확인 필요)")

    key = (response.json().get("data") or {}).get("s3_key")
    if not key:
        raise SmokeFailure("validation", "응답에 s3_key 가 없습니다")
    report["s3"] = {"key": key, **(s3_checker(secret_name, key) if secret_name else {"skipped": True})}
    if "refused" in report["s3"] or report["s3"].get("format") not in (None, "PNG"):
        raise SmokeFailure("s3", json.dumps(report["s3"]))
    report["result"] = "PASS"
    return report


def on_failure(ops, action: str, app_id: str, workspace: str, stage: str) -> str:
    """실패한 개인 staging App 만 rollback/stop 한다. 팀 Workspace·production 은 사람이 판단하도록 안내만 한다."""
    if action == "none":
        return f"자동 조치 없음. 필요하면: modal app rollback {app_id}  또는  modal app stop {app_id}"
    if workspace in PROTECTED_WORKSPACES or stage != "staging":
        return f"보호된 대상이라 자동 {action} 을 하지 않습니다(workspace={workspace}, stage={stage})"
    if action == "rollback":
        ops.rollback(app_id)
        return f"이전 버전으로 rollback 했습니다: {app_id}"
    ops.stop(app_id)
    return f"App 을 멈췄습니다: {app_id}"


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "deploy"):
        command = sub.add_parser(name)
        command.add_argument("--stage", choices=sorted(APP_NAMES), required=True)
        command.add_argument("--workspace", required=True, help="배포할 Modal Workspace(현재 profile 과 같아야 한다)")
        command.add_argument("--diagnostics", action="store_true", help="staging 진단 모드")
        command.add_argument("--gpu-snapshot", action="store_true",
                             help="production 을 GPU snapshot 으로 검사(배포 시 BODY_IMAGE_WORKER_GPU_SNAPSHOT=true)")
        command.add_argument("--max-containers", type=int,
                             help="production max_containers(배포 시 BODY_IMAGE_WORKER_MAX_CONTAINERS). GPU snapshot 에 필수")
    smoke_cmd = sub.add_parser("smoke")
    smoke_cmd.add_argument("--workspace", required=True)
    smoke_cmd.add_argument("--image", required=True, type=Path, help="정상 전신사진(JPEG)")
    smoke_cmd.add_argument("--credentials-file", action="append", default=[])
    smoke_cmd.add_argument("--on-failure", choices=("none", "rollback", "stop"), default="none")
    smoke_cmd.add_argument("--max-miss-retries", type=int, default=SmokeLimits.max_miss_retries)
    smoke_cmd.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    if args.command in ("preflight", "deploy"):
        wanted = deploy_env(args.stage, args.diagnostics, args.gpu_snapshot, args.max_containers)
        if args.command == "deploy" and args.stage == "production":
            # 공유 production App 은 이 도구로 배포하지 않는다(자동 배포·중단·롤백 없음). 검사만 한다.
            print("production 은 이 도구로 배포하지 않습니다. preflight 로 검사한 뒤 승인을 받아 직접 실행하세요:\n"
                  f"  {manual_deploy_command(wanted)}")
            return 2
        profile, checks = preflight(args.stage, args.workspace, args.diagnostics, args.gpu_snapshot,
                                    args.max_containers)
        if not print_checks(checks):
            print("preflight FAILED: 배포하지 않습니다.")
            return 1
        print(f"preflight PASSED (profile={profile})")
        if args.command == "preflight":
            if args.stage == "production":
                print(f"검사한 설정의 수동 배포 명령(승인 후, 같은 값으로만):\n  {manual_deploy_command(wanted)}")
            return 0
        # 검사한 것과 같은 값(wanted)으로만 배포한다. 셸의 BODY_IMAGE_WORKER_* 는 물려주지 않는다.
        env = deploy_process_env(wanted, profile)
        return subprocess.run(["modal", "deploy", str(WORKER_FILE)], env=env, check=False).returncode

    profiles = json.loads(modal_cli("profile", "list", "--json"))
    profile, workspace_check = resolve_profile(args.workspace, profiles, os.environ.get("MODAL_PROFILE"))
    if not workspace_check.ok:
        print_checks([workspace_check])
        return 1
    os.environ["MODAL_PROFILE"] = profile
    ops = ModalOps(profile)
    module, _ = load_worker("staging")
    app_name = module.APP_NAME
    report: dict = {"workspace": args.workspace, "profile": profile}
    try:
        report.update(smoke(ops, app_name, load_credentials(args.credentials_file), args.image,
                            SmokeLimits(max_miss_retries=args.max_miss_retries), secret_name=module.SECRET_NAME))
        code = 0
    except SmokeFailure as failure:
        report.update(result="FAIL", failed_stage=failure.stage, reason=failure.reason)
        app_id = ops.app_info(app_name)["app_id"]
        report["on_failure"] = on_failure(ops, args.on_failure, app_id, args.workspace, "staging")
        code = 1
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.report:
        args.report.write_text(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
