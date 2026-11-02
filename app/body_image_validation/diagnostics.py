import ctypes
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Self

logger = logging.getLogger(__name__)

MIB = 2**20
METRICS = ("process_rss_mib", "container_mib", "gpu_device_used_mib")
REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID = re.compile(r"[0-9a-f]{16}")

_current: ContextVar["RequestRecord | None"] = ContextVar("body_image_request_record", default=None)


@dataclass(eq=False)
class RequestRecord:
    """한 요청의 Lock·실행 구간 측정값. 시각은 perf_counter 값이라 같은 process_id 끼리만 비교할 수 있다."""

    seq: int
    process_id: str | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    locks: dict[str, dict[str, float]] = field(default_factory=dict)
    received: float = field(default_factory=time.perf_counter)
    slot_acquired: float | None = None
    execution_start: float | None = None
    execution_end: float | None = None
    active_at_start: int = 0
    active_max: int = 0
    spans: dict[str, tuple[float, float]] = field(default_factory=dict)
    ai_request_id: str | None = None

    def add_lock(self, name: str, wait_seconds: float, hold_seconds: float) -> None:
        entry = self.locks.setdefault(name, {"count": 0, "wait_ms": 0.0, "hold_ms": 0.0})
        entry["count"] += 1
        entry["wait_ms"] += wait_seconds * 1000
        entry["hold_ms"] += hold_seconds * 1000

    def to_json(self, outcome: str, timings: dict[str, float], resources: dict | None) -> str:
        """이미지·키·예외 메시지는 담지 않는다. 숫자, ID, outcome(오류 코드나 예외 클래스 이름)뿐이다."""
        return json.dumps({
            "request_id": self.request_id,
            "ai_request_id": self.ai_request_id,
            "seq": self.seq,
            "outcome": outcome,
            "ms": timings,
            "locks": {
                name: {"count": lock["count"], "wait_ms": round(lock["wait_ms"], 1), "hold_ms": round(lock["hold_ms"], 1)}
                for name, lock in self.locks.items()
            },
            "execution": {
                "process_id": self.process_id,
                "active_at_start": self.active_at_start,
                "active_max": self.active_max,
                "received_ms": _clock_ms(self.received),
                "slot_acquired_ms": _clock_ms(self.slot_acquired),
                "start_ms": _clock_ms(self.execution_start),
                "end_ms": _clock_ms(self.execution_end),
                "spans_ms": {stage: [_clock_ms(start), _clock_ms(end)] for stage, (start, end) in self.spans.items()},
            },
            "resources": resources,
        })


class ExecutionTracker:
    """limiter 슬롯을 쥐고 실행 중인 요청. active_at_start 는 슬롯을 얻을 때 이미 실행 중이던 다른 요청 수,
    active_max 는 이 요청의 실행 구간에서 실행 중이던 요청 수의 최대값(자기 포함)이다."""

    def __init__(self) -> None:
        self.process_id = uuid.uuid4().hex[:16]
        self._lock = threading.Lock()
        self._running: list[RequestRecord] = []

    def enter(self, record: RequestRecord) -> None:
        with self._lock:
            record.execution_start = time.perf_counter()
            record.active_at_start = len(self._running)
            self._running.append(record)
            for running in self._running:
                running.active_max = max(running.active_max, len(self._running))

    def leave(self, record: RequestRecord) -> None:
        with self._lock:
            self._running.remove(record)
            record.execution_end = time.perf_counter()


def _clock_ms(value: float | None) -> float | None:
    return None if value is None else round(value * 1000, 1)


def parse_request_id(value: str | None) -> str | None:
    """형식이 다른 값은 버린다. 로그를 잇는 데만 쓰는 값이라 요청을 거절하지 않는다."""
    return value if value and _REQUEST_ID.fullmatch(value) else None


def begin_request(
    seq: int, process_id: str | None = None, ai_request_id: str | None = None
) -> tuple[RequestRecord, Token]:
    record = RequestRecord(seq, process_id, ai_request_id=ai_request_id)
    return record, _current.set(record)


def current_request() -> RequestRecord | None:
    return _current.get()


def end_request(token: Token) -> None:
    _current.reset(token)


class TimedLock:
    """기존 Lock 을 그대로 쓰면서 대기·보유 시간을 지금 요청의 기록에 더한다. 요청 밖(warm-up, close)은 재지 않는다."""

    def __init__(self, name: str, lock: threading.Lock) -> None:
        self.name = name
        self._lock = lock
        self._wait = 0.0
        self._acquired_at = 0.0

    def __enter__(self) -> Self:
        started = time.perf_counter()
        self._lock.acquire()
        # 아래 두 값은 Lock 을 쥔 thread 만 쓴다.
        self._acquired_at = time.perf_counter()
        self._wait = self._acquired_at - started
        return self

    def __exit__(self, *exc_info) -> None:
        hold = time.perf_counter() - self._acquired_at
        wait = self._wait
        self._lock.release()
        record = _current.get()
        if record is not None:
            record.add_lock(self.name, wait, hold)


def install_lock_timers(models) -> None:
    """모델의 Lock 을 TimedLock 으로 감싼다. 모델 클래스는 바꾸지 않고, 진단이 꺼져 있으면 부르지 않는다.

    snapshot 복원 뒤에 만들어지는 BodyImageRuntime 만 부른다(요청이 없을 때). 그래서 snapshot 의 모델에는
    원래 Lock 만 담긴다.
    """
    preprocessor = models.gpu_preprocessor
    owners = {
        "person": models.person_detector,
        "pose": models.pose_detector,
        "rembg": models.background_remover,
        "nvjpeg": getattr(preprocessor, "_decoder", None),
    }
    for name, owner in owners.items():
        lock = getattr(owner, "_lock", None)
        if lock is not None:
            owner._lock = TimedLock(name, lock)


@dataclass(frozen=True)
class ResourcePaths:
    process_status: str = "/proc/self/status"
    container_usage: tuple[str, ...] = ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
    container_high_water: tuple[str, ...] = (
        "/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
    )


def read_process_memory(path: str) -> tuple[float | None, float | None]:
    """(VmRSS, VmHWM) MiB. /proc 의 kB 는 1024 byte 다. 없는 값은 None 이다(Linux 가 아니거나 커널이 주지 않을 때)."""
    rss = high_water = None
    try:
        with open(path) as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024
                elif line.startswith("VmHWM:"):
                    high_water = int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        return None, None
    return rss, high_water


def read_cgroup_mib(paths: tuple[str, ...]) -> float | None:
    """byte 단위 cgroup 값을 MiB 로. 읽지 못하는 값은 None 이다."""
    for path in paths:
        try:
            with open(path) as file:
                return int(file.read().strip()) / MIB
        except (OSError, ValueError):
            continue
    return None


class NvmlError(RuntimeError):
    pass


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class NvmlDevice:
    """GPU 0 의 장치 전체 메모리. PyTorch 통계와 달리 ONNX Runtime·CUDA context 까지 포함한다.

    nvidia-ml-py 가 부르는 것과 같은 함수를 ctypes 로 직접 부른다. 운영 이미지에 패키지를 더하지 않기 위해서다.
    만들면 장치 파일을 여는 것이라 snapshot 전에 만들면 안 된다.
    """

    def __init__(self, library=None) -> None:
        self._library = ctypes.CDLL("libnvidia-ml.so.1") if library is None else library
        self._call("nvmlInitWithFlags", ctypes.c_uint(0))
        self._handle = ctypes.c_void_p()
        self._call("nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint(0), ctypes.byref(self._handle))

    def _call(self, name: str, *args) -> None:
        status = getattr(self._library, name)(*args)
        if status != 0:
            raise NvmlError(f"{name} returned {status}")

    def memory_mib(self) -> tuple[float, float]:
        """(사용량, 전체) MiB."""
        memory = _NvmlMemory()
        self._call("nvmlDeviceGetMemoryInfo", self._handle, ctypes.byref(memory))
        return memory.used / MIB, memory.total / MIB


@dataclass
class _Sample:
    values: tuple[float | None, ...]
    at: float
    cost_ms: float


class _Window:
    """요청 하나가 열려 있던 구간. 값은 그 구간에 관측한 프로세스·컨테이너·장치 전체 값이다."""

    def __init__(self, first: _Sample, windows_at_start: int) -> None:
        self.first = first
        self.peak = list(first.values)
        self.windows_at_start = windows_at_start
        self.windows_max = windows_at_start + 1
        self.samples = 1
        self.last_at = first.at
        self.gap_ms_max = 0.0
        self.cost_ms_max = first.cost_ms
        self.cost_ms_total = first.cost_ms

    def add(self, sample: _Sample) -> None:
        self.peak = [
            None if value is None and peak is None else max(v for v in (value, peak) if v is not None)
            for value, peak in zip(sample.values, self.peak, strict=True)
        ]
        self.samples += 1
        self.gap_ms_max = max(self.gap_ms_max, (sample.at - self.last_at) * 1000)
        self.last_at = sample.at
        self.cost_ms_max = max(self.cost_ms_max, sample.cost_ms)
        self.cost_ms_total += sample.cost_ms


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


class ResourceSampler:
    """요청이 처리되는 동안만 프로세스 RSS·컨테이너 메모리·GPU 장치 메모리를 주기적으로 읽는다.

    - 모두 프로세스·컨테이너·장치 단위 값이다. 요청별 점유량이 아니다. windows_max 가 2 이상이면 그 구간의 값은
      겹친 요청들의 합이고(슬롯을 기다리는 요청도 센다), 1 이어도 peak−start 는 다른 thread·allocator 가 잡아 둔 메모리·복원 뒤 처음 읽는
      페이지를 포함한다.
    - 공유 모델과 요청별 버퍼를 직접 나누지 못한다. baseline(복원 뒤 첫 요청 전)은 아직 올라오지 않은 snapshot
      페이지가 빠져 있어 공유 메모리 전체도 아니다.
    - peak 는 샘플의 최대값이라 간격 사이의 순간 값을 놓칠 수 있다. 커널의 high water(프로세스·컨테이너 수명 전체)를
      같이 남겨 놓친 크기를 확인한다. thread 가 GIL 을 못 잡으면 샘플이 밀리므로 gap_ms_max 도 남긴다. cost_ms 도 그
      대기를 포함하므로 읽기 자체의 비용은 값이 작은 쪽을 본다.

    만들 때는 파일·장치·thread 를 건드리지 않는다. 첫 open() 에서 NVML 을 열고 thread 를 시작한다.
    """

    def __init__(
        self,
        interval: float,
        paths: ResourcePaths | None = None,
        open_gpu: Callable[[], NvmlDevice] = NvmlDevice,
    ) -> None:
        self._interval = interval
        self._paths = paths or ResourcePaths()
        self._open_gpu = open_gpu
        self._gpu: NvmlDevice | None = None
        self._gpu_total_mib: float | None = None
        self._gpu_error: str | None = None
        self._gpu_read_failed = False
        self._gpu_lock = threading.Lock()
        self._condition = threading.Condition()
        self._windows: list[_Window] = []
        self._thread: threading.Thread | None = None
        self._closed = False
        self._baseline_logged = False

    def open(self) -> _Window:
        """요청 시작. 이 값이 window 의 start 다. 앞선 요청이 없으면(windows_at_start=0) 유휴 상태 값이다."""
        sample = self._read()
        with self._condition:
            baseline_due, self._baseline_logged = not self._baseline_logged, True
            window = _Window(sample, len(self._windows))
            self._windows.append(window)
            for active in self._windows:
                active.windows_max = max(active.windows_max, len(self._windows))
            if self._thread is None and not self._closed:
                self._thread = threading.Thread(target=self._run, name="body-image-resource-sampler", daemon=True)
                self._thread.start()
            self._condition.notify_all()
        if baseline_due:
            self._log_baseline(sample)
        return window

    def finish(self, window: _Window) -> dict:
        """요청 끝. 로그에 넣을 값을 돌려준다."""
        sample = self._read()
        with self._condition:
            self._windows.remove(window)
            window.add(sample)
        return {
            "windows_at_start": window.windows_at_start,
            "windows_max": window.windows_max,
            **{
                metric: {"start": _rounded(start), "peak": _rounded(peak), "end": _rounded(end)}
                for metric, start, peak, end in zip(METRICS, window.first.values, window.peak, sample.values, strict=True)
            },
            "high_water_mib": {
                "process_rss": _rounded(read_process_memory(self._paths.process_status)[1]),
                "container": _rounded(read_cgroup_mib(self._paths.container_high_water)),
            },
            "sampling": {
                "samples": window.samples,
                "gap_ms_max": round(window.gap_ms_max, 1),
                "cost_ms_max": round(window.cost_ms_max, 2),
                "cost_ms_total": round(window.cost_ms_total, 2),
            },
        }

    def close(self) -> None:
        """thread 를 멈춘다. 진행 중인 요청의 finish() 는 그대로 동작한다. 여러 번 불려도 된다."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._windows and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
            sample = self._read()
            with self._condition:
                for window in self._windows:
                    window.add(sample)
                if not self._closed:
                    self._condition.wait(self._interval)

    def _read(self) -> _Sample:
        started = time.perf_counter()
        rss, _ = read_process_memory(self._paths.process_status)
        container = read_cgroup_mib(self._paths.container_usage)
        gpu = self._gpu_used_mib()
        ended = time.perf_counter()
        return _Sample((rss, container, gpu), ended, (ended - started) * 1000)

    def _device(self) -> NvmlDevice | None:
        with self._gpu_lock:
            if self._gpu is None and self._gpu_error is None:
                try:
                    self._gpu = self._open_gpu()
                    self._gpu_total_mib = self._gpu.memory_mib()[1]
                except Exception as error:  # noqa: BLE001 - GPU 지표 없이도 나머지는 잰다. 없는 라이브러리는 OSError 다.
                    self._gpu = None
                    self._gpu_error = f"{type(error).__name__}: {error}"
                    logger.warning("body image GPU memory is not measured: %s", self._gpu_error)
            return self._gpu

    def _gpu_used_mib(self) -> float | None:
        device = self._device()
        if device is None:
            return None
        try:
            return device.memory_mib()[0]
        except Exception as error:  # noqa: BLE001 - 진단 실패가 요청을 실패시키지 않게 한다.
            if not self._gpu_read_failed:
                self._gpu_read_failed = True
                logger.warning("body image GPU memory read failed: %s: %s", type(error).__name__, error)
            return None

    def _log_baseline(self, sample: _Sample) -> None:
        # 첫 요청에서 남긴다. 복원 단계의 INFO 는 configure_logging 전이라 출력되지 않는다.
        logger.info("body image resource baseline: %s", json.dumps({
            "phase": "idle_before_first_request",
            **{metric: _rounded(value) for metric, value in zip(METRICS, sample.values, strict=True)},
            "gpu_device_total_mib": _rounded(self._gpu_total_mib),
            "gpu_unavailable": self._gpu_error,
            "first_read_ms": round(sample.cost_ms, 2),
            "sample_interval_ms": round(self._interval * 1000, 1),
        }))
