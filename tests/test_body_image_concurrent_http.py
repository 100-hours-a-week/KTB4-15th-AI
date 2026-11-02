import asyncio
import contextvars
import json
import logging
import re
import socket
import sys
import threading
import time
from collections import Counter, defaultdict
from io import BytesIO
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.body_image_validation import deadline, worker
from app.body_image_validation import router as validation_router
from app.body_image_validation import service as service_module
from app.body_image_validation.detectors import DetectionImage
from app.body_image_validation.diagnostics import REQUEST_ID_HEADER, current_request
from app.body_image_validation.exceptions import user_error
from app.body_image_validation.gpu_preprocessing import GpuImagePreprocessor
from app.body_image_validation.inference import BodyImageModels
from app.body_image_validation.runtime import BodyImageRuntime, BodyImageRuntimeManager
from app.body_image_validation.warmup import ModalWarmupClient, WarmupOutcome
from app.clients.s3 import ImageStorageError
from app.config import body_image_validation as config
from app.config import settings
from tests.test_body_image_c3_http import live, wait_until  # noqa: F401 (pytest fixture)
from tests.test_body_image_diagnostics import (
    PersonTask,
    PoseTask,
    interval,
    overlaps,
    real_person,
    real_pose,
    real_remover,
    span,
)
from tests.test_body_image_gpu_preprocessing import fake_torch_decoder
from tests.test_body_image_queue_deadline import BUSY
from tests.test_body_image_warmup_api import (  # noqa: F401 (pytest fixture)
    AUTH,
    KEY,
    PROXY_SECRET,
    VALIDATE_URL,
    WORKER,
    remote,
    runtime,
)

WAIT = 10.0
TAGS = (25, 50, 75, 100, 125, 150, 175, 200, 225)
A, B, C, D = TAGS[:4]
SIZE = (600, 800)
RESOLUTION = 0.01  # 구간이 로그 해상도(0.1ms)보다 길게 겹치도록 머문다
TIMEOUT_CODE = "BODY_IMAGE_WORKER_TIMEOUT"
S3_KEY = re.compile(r"body-images/[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}\.png")

on_worker: contextvars.ContextVar[bool] = contextvars.ContextVar("test_on_worker", default=False)


def pixels(tag):
    array = np.full((SIZE[1], SIZE[0], 3), (tag, 255 - tag, 128), np.uint8)
    array[::7, ::5, 2] = tag // 2  # 단색이 아니어서 PNG 를 비교할 때 픽셀이 섞이면 드러난다
    return array


def picture(tag, image_format="PNG"):
    output = BytesIO()
    Image.fromarray(pixels(tag)).save(output, format=image_format)
    return output.getvalue()


def rgb_of(body):
    return np.asarray(Image.open(BytesIO(body)).convert("RGB"))


def tag_on_the_wire(request):
    return tag_of(rgb_of(request.content.split(b"\r\n\r\n", 1)[1]))


def tag_of(array):
    """이미지가 어느 요청의 것인지. 빨강 평균이 tag 다(JPEG 손실은 tag 간격보다 훨씬 작다)."""
    return round(float(np.asarray(array)[..., 0].mean()) / 25) * 25


class Pipeline:
    """모델 본체와 저장소 자리. 단계마다 어느 요청이 어느 thread 로 들어왔는지 남기고 테스트가 건 동작을 실행한다."""

    def __init__(self):
        self._lock = threading.Lock()
        self.hooks = {}
        self.gates = []
        self.order = defaultdict(list)
        self.arrays = defaultdict(dict)
        self.threads = defaultdict(set)
        self.inside = Counter()
        self.most_inside = Counter()
        self.request_ids = {}
        self.ai_request_ids = defaultdict(set)
        self.uploads = {}

    def on(self, stage, hook):
        self.hooks[stage] = hook
        return hook

    def gate(self, stage, *tags):
        gate = Gate(*tags)
        self.gates.append(gate)
        return self.on(stage, gate)

    def run(self, stage, array):
        tag = tag_of(array)
        with self._lock:
            self.order[stage].append((tag, "in"))
            self.arrays[stage][tag] = array
            self.threads[stage].add(threading.get_ident())
            self.inside[stage] += 1
            self.most_inside[stage] = max(self.most_inside[stage], self.inside[stage])
            self.request_ids[tag] = current_request().request_id
            self.ai_request_ids[tag].add(current_request().ai_request_id)
        try:
            if stage in self.hooks:
                self.hooks[stage](tag)
        finally:
            with self._lock:
                self.inside[stage] -= 1
                self.order[stage].append((tag, "out"))
        return tag

    def models(self):
        person = real_person(Task(self, "person", PersonTask().detect(None)))
        pose = real_pose(Task(self, "pose", PoseTask().detect(None)))
        decoder = fake_torch_decoder()
        decoder._decode_jpeg = self._decode_jpeg
        return BodyImageModels(person, pose, real_remover(), GpuImagePreprocessor(decoder))

    def _decode_jpeg(self, data, **kwargs):
        array = rgb_of(bytes(data))
        self.run("nvjpeg", array)
        return DecodedJpeg(array)

    def remove(self, image, session):
        tag = self.run("rembg", np.asarray(image))
        output = image.convert("RGBA")
        output.putalpha(tag)
        return output

    def detection_input(self, image):
        """각 요청이 슬롯 안에서 Lock 없이 한 번 지나는 곳."""
        if image._mediapipe_image is None:
            self.run("detection_input", image.array)
            image._mediapipe_image = image
        return image

    def upload_bytes(self, body, key, content_type):
        self.run("s3_upload", np.asarray(Image.open(BytesIO(body))))
        self.uploads[key] = (body, content_type)
        return key


class Task:
    """MediaPipe task 자리. 실제 detector 가 Lock 안에서 부른다."""

    def __init__(self, pipeline, stage, answer):
        self.pipeline, self.stage, self.answer = pipeline, stage, answer

    def detect(self, image):
        self.pipeline.run(self.stage, image.array)
        return self.answer

    def close(self):
        pass


class DecodedJpeg:
    """torch tensor 자리. HxWx3 배열을 그대로 돌려준다."""

    def __init__(self, array):
        self.array = array
        self.shape = (3, *array.shape[:2])

    def permute(self, *dims):
        return self

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class Gate:
    """들어온 요청을 open 될 때까지 붙잡는다. tag 를 주면 그 요청만 붙잡는다."""

    def __init__(self, *tags):
        self.tags = set(tags) or set(TAGS)
        self.arrived = []
        self._released = {tag: threading.Event() for tag in TAGS}

    def __call__(self, tag):
        if tag in self.tags:
            self.arrived.append(tag)
            assert self._released[tag].wait(WAIT), "테스트가 요청을 풀어 주지 않았다"

    def wait_for(self, count):
        wait_until(lambda: len(self.arrived) >= count)

    def open(self, *tags):
        for tag in tags or TAGS:
            self._released[tag].set()


def meet(count=2):
    """count 건이 모두 들어와야 풀린다. 겹치지 못하면 BrokenBarrierError 로 요청이 실패한다."""
    barrier = threading.Barrier(count, timeout=WAIT)

    def hook(tag):
        barrier.wait()
        time.sleep(RESOLUTION)

    return hook


class Modal(httpx.AsyncBaseTransport):
    """Modal 자리. proxy 인증을 확인하고 worker 앱을 호출자와 다른 Task·빈 context 에서 돌린다.

    실제 네트워크처럼 AI Server 의 취소와 ContextVar 가 worker 로 넘어가지 않는다. inputs 는 Modal 의 입력 슬롯을
    흉내 낸 것으로 앱 동작을 확인하는 용도이며 플랫폼의 스케줄링을 재현하지 않는다.
    """

    def __init__(self, app, inputs=None):
        self.requests = []
        self.tasks = []
        self.inputs = None if inputs is None else asyncio.Semaphore(inputs)
        self._asgi = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async def handle_async_request(self, request):
        await request.aread()
        self.requests.append(request)
        if (request.headers.get("modal-key"), request.headers.get("modal-secret")) != ("wk-test", PROXY_SECRET):
            return httpx.Response(401)
        task = asyncio.get_running_loop().create_task(self._input(request), context=contextvars.Context())
        self.tasks.append(task)
        return await asyncio.shield(task)

    async def _input(self, request):
        on_worker.set(True)
        if self.inputs is None:
            return await self._asgi.handle_async_request(request)
        async with self.inputs:
            return await self._asgi.handle_async_request(request)


class RuntimeRouter:
    """한 프로세스라 router 의 runtime_manager 를 나눠 쓴다. worker 앱으로 온 요청은 worker runtime 을 쓴다."""

    def __init__(self, ai_manager, worker_runtime):
        self.ai_manager = ai_manager
        self.worker_runtime = worker_runtime

    def get_service(self):
        return self.worker_runtime.service if on_worker.get() else self.ai_manager.get_service()


class Log(logging.Handler):
    PREFIX = "body image validation diagnostics: "

    def __init__(self):
        super().__init__(logging.INFO)
        self.messages = []
        self.logger = logging.getLogger("app.body_image_validation")
        self.level_before = self.logger.level
        self.logger.addHandler(self)
        self.logger.setLevel(logging.INFO)

    def emit(self, record):
        self.messages.append(record.getMessage())

    def detach(self):
        self.logger.setLevel(self.level_before)
        self.logger.removeHandler(self)

    @property
    def diagnostics(self):
        return [json.loads(m[len(self.PREFIX):]) for m in self.messages if m.startswith(self.PREFIX)]

    def requests(self):
        lines = [m for m in self.messages if m.startswith("body image request:")]
        return [dict(re.findall(r"(\w+)=(\S+)", line)) for line in lines]

    def request_fields(self, outcome):
        return next(fields for fields in self.requests() if fields["outcome"] == outcome)


class Clock:
    """AI Server 의 monotonic 시계. 실제로 기다리지 않고 대기시간을 앞당긴다."""

    def __init__(self):
        self.offset = 0.0

    def __call__(self):
        return time.monotonic() + self.offset

    def advance(self, seconds):
        self.offset += seconds


class Pending:
    """서버 event loop 에서 도는 요청 하나. cancel 은 실제 asyncio Task 취소다."""

    def __init__(self, portal, call):
        self.task = portal.call(self._start, call)
        self.portal = portal

    @staticmethod
    async def _start(call):
        return asyncio.ensure_future(call())

    def cancel(self):
        self.portal.call(self.task.cancel)

    def result(self):
        wait_until(self.task.done)
        return self.task.result()


class Stack:
    def __init__(self, client, ai_manager, worker_runtime, modal, pipeline, log):
        self.client = client
        self.portal = client.portal
        self.ai_manager = ai_manager
        self.runtime = worker_runtime
        self.modal = modal
        self.pipeline = pipeline
        self.log = log
        self.pending = []
        self.ai_http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app, raise_app_exceptions=False), base_url="http://ai"
        )
        self.worker_http = httpx.AsyncClient(transport=modal, base_url=WORKER)

    @property
    def ai(self):
        return self.ai_manager.runtime.service

    @property
    def worker(self):
        return self.runtime.service

    @property
    def ai_slots(self):
        return self.ai.limiter.running, self.ai.limiter.waiting

    @property
    def worker_slots(self):
        return self.worker.limiter.running, self.worker.limiter.waiting

    def _send(self, call):
        pending = Pending(self.portal, call)
        self.pending.append(pending)
        return pending

    def post(self, body, headers=AUTH):
        files = {"image": ("body.png", body, "image/png")}
        return self._send(lambda: self.ai_http.post(VALIDATE_URL, files=files, headers=headers))

    def post_to_worker(self, body, limit=None, bearer=KEY, trace=None):
        headers = {"Modal-Key": "wk-test", "Modal-Secret": PROXY_SECRET, "Authorization": f"Bearer {bearer}"}
        if limit is not None:
            headers[deadline.HEADER] = f"{limit:.3f}"
        if trace is not None:
            headers[REQUEST_ID_HEADER] = trace
        files = {"image": ("body.png", body, "image/png")}
        return self._send(lambda: self.worker_http.post(VALIDATE_URL, files=files, headers=headers))

    def health(self, proxy=True):
        headers = {"Modal-Key": "wk-test", "Modal-Secret": PROXY_SECRET} if proxy else {}
        return self._send(lambda: self.worker_http.get("/health", headers=headers))

    def cancel_input(self, index):
        """worker ASGI Task 자체를 취소한다. 호출자 쪽 취소는 Pending.cancel 이다."""
        task = self.modal.tasks[index]
        self.portal.call(task.cancel)
        return task

    def run_all(self, *tags, image_format="PNG"):
        return [self.post(picture(tag, image_format)) for tag in tags]

    def wait_for_queue(self, size=1):
        wait_until(lambda: self.ai.limiter.waiting == size)

    def use_clock(self):
        clock = Clock()
        self.ai._clock = self.ai.limiter._clock = clock
        return clock

    def wake_the_queue(self):
        async def notify():
            async with self.ai.limiter._condition:
                self.ai.limiter._condition.notify_all()

        self.portal.call(notify)

    def sent_ids(self):
        """worker 로 나간 요청에 실제로 실린 이미지와 추적 헤더."""
        return {tag_on_the_wire(request): request.headers[REQUEST_ID_HEADER] for request in self.modal.requests}

    def line(self, tag):
        request_id = self.pipeline.request_ids[tag]
        wait_until(lambda: any(line["request_id"] == request_id for line in self.log.diagnostics))
        return next(line for line in self.log.diagnostics if line["request_id"] == request_id)

    def settle(self):
        """모든 슬롯·실행 카운터·샘플러 구간이 돌아올 때까지 기다린다."""
        wait_until(
            lambda: self.ai_slots == self.worker_slots == (0, 0)
            and self.worker._execution._running == []
            and self.runtime.resources._windows == []
        )

    def drain(self):
        for gate in self.pipeline.gates:
            gate.open()
        tasks = [pending.task for pending in self.pending] + self.modal.tasks

        async def wait():
            if tasks:
                await asyncio.wait(tasks, timeout=WAIT)
            await self.ai_http.aclose()
            await self.worker_http.aclose()

        self.portal.call(wait)


@pytest.fixture
def stack(remote, monkeypatch):  # noqa: F811
    built = []

    def start(ai_running=2, worker_running=2, inputs=None, total=120.0):
        pipeline = Pipeline()
        # 한 프로세스라 AI Server 의 Bearer 와 worker 의 INTERNAL_API_KEY 가 같은 값이다.
        monkeypatch.setattr(settings, "BODY_IMAGE_REMOTE_API_KEY", KEY)
        monkeypatch.setattr(config, "DIAGNOSTICS", True)
        monkeypatch.setattr(config, "MAX_WAITING_BODY_VALIDATIONS", 1)
        monkeypatch.setattr(config, "MAX_QUEUE_WAIT_SECONDS", 10.0)
        monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", total)
        monkeypatch.setattr(DetectionImage, "mediapipe_image", lambda image: pipeline.detection_input(image))
        monkeypatch.setattr(service_module, "validate_brightness", lambda array, box: pipeline.run("brightness", array))
        monkeypatch.setitem(sys.modules, "rembg", SimpleNamespace(remove=pipeline.remove))
        monkeypatch.setattr(config, "MAX_CONCURRENT_BODY_VALIDATIONS", worker_running)
        worker_runtime = BodyImageRuntime(pipeline.models(), pipeline)
        monkeypatch.setattr(config, "MAX_CONCURRENT_BODY_VALIDATIONS", ai_running)
        modal = Modal(worker.create_app(), inputs)
        ai_manager = BodyImageRuntimeManager(remote_transport=modal)
        monkeypatch.setattr(main, "runtime_manager", ai_manager)
        monkeypatch.setattr(validation_router, "runtime_manager", RuntimeRouter(ai_manager, worker_runtime))
        log = Log()
        client = TestClient(main.app, raise_server_exceptions=False)
        client.__enter__()
        built.append(Stack(client, ai_manager, worker_runtime, modal, pipeline, log))
        return built[-1]

    yield start
    for server in built:
        try:
            server.drain()
        finally:
            server.client.__exit__(None, None, None)
            server.runtime.close()
            server.log.detach()


@pytest.fixture
def wall(monkeypatch):
    """worker 의 deadline 판단에 쓰는 wall clock. 실제로 기다리지 않고 deadline 을 지나게 한다."""
    clock = SimpleNamespace(offset=0.0, time=None)
    clock.time = lambda: time.time() + clock.offset
    monkeypatch.setattr(deadline, "time", clock)
    return clock


def statuses(*pending):
    return [request.result().status_code for request in pending]


def assert_timeline(line):
    execution = line["execution"]
    assert execution["received_ms"] <= execution["slot_acquired_ms"] <= execution["start_ms"] <= execution["end_ms"]
    for start, end in execution["spans_ms"].values():
        assert execution["start_ms"] <= start <= end <= execution["end_ms"]


# --- A. HTTP 동시 실행과 대기열 ---


def test_two_requests_run_at_once_through_both_apps(stack):
    s = stack()
    seen = []
    both = threading.Barrier(2, action=lambda: seen.append((s.ai_slots, s.worker_slots)), timeout=WAIT)
    s.pipeline.on("s3_upload", lambda tag: both.wait())

    assert statuses(*s.run_all(A, B)) == [200, 200]

    assert seen == [((2, 0), (2, 0))]
    assert len(s.pipeline.threads["s3_upload"]) == 2
    assert s.portal.call(threading.get_ident) not in s.pipeline.threads["s3_upload"]  # run_in_threadpool 안이다
    a, b = s.line(A), s.line(B)
    assert (a["execution"]["active_max"], b["execution"]["active_max"]) == (2, 2)
    assert a["execution"]["process_id"] == b["execution"]["process_id"] == s.worker._execution.process_id
    assert overlaps(interval(a), interval(b))
    s.settle()


def test_two_run_one_waits_and_the_fourth_is_429_and_the_waiter_runs_after_one_finishes(stack):
    s = stack()
    upload = s.pipeline.gate("s3_upload")
    a, b = s.run_all(A, B)
    upload.wait_for(2)

    c = s.post(picture(C))
    s.wait_for_queue()
    d = s.post(picture(D))

    assert d.result().status_code == 429
    assert d.result().json() == BUSY
    assert len(s.modal.requests) == 2  # 대기·초과 요청은 worker 를 부르지 않았다
    upload.open(A)
    assert a.result().status_code == 200
    upload.wait_for(3)  # B 가 아직 실행 중일 때 C 가 들어왔다
    assert upload.arrived[2] == C and not b.task.done()
    assert s.ai_slots == (2, 0)
    upload.open(B, C)
    assert statuses(b, c) == [200, 200]
    assert s.line(C)["execution"]["active_max"] == 2
    s.settle()


def test_waiting_past_ten_seconds_is_429_at_exactly_the_queue_limit(stack):
    s = stack()
    clock = s.use_clock()
    upload = s.pipeline.gate("s3_upload")
    a, b = s.run_all(A, B)
    upload.wait_for(2)
    c = s.post(picture(C))
    s.wait_for_queue()
    started = time.monotonic()

    clock.advance(9.9)
    s.wake_the_queue()
    clock.advance(0.2)
    s.wake_the_queue()

    assert c.result().status_code == 429
    assert c.result().json() == BUSY
    assert time.monotonic() - started < 5  # 실제로 10초를 기다리지 않았다
    assert s.ai.limiter.waiting == 0
    fields = s.log.request_fields("queue_timeout")
    assert 10.0 <= float(fields["queue_wait"].rstrip("s")) < 10.5  # 9.9초에는 포기하지 않았다
    assert len(s.modal.requests) == 2
    upload.open()
    assert statuses(a, b) == [200, 200]
    assert statuses(s.post(picture(D))) == [200]  # 대기열 자리가 돌아왔다
    s.settle()


def test_time_spent_in_the_queue_comes_out_of_the_remote_deadline(stack):
    s = stack()
    clock = s.use_clock()
    upload = s.pipeline.gate("s3_upload", A, B)
    seen = {}
    s.pipeline.on("detection_input", lambda tag: seen.update({tag: deadline._deadline.get()}))
    a, b = s.run_all(A, B)
    upload.wait_for(2)
    c = s.post(picture(C))
    s.wait_for_queue()

    clock.advance(8.0)
    upload.open(A)

    assert statuses(a, c) == [200, 200]
    upload.open(B)
    assert b.result().status_code == 200
    first, _, late = s.modal.requests
    first_limit, late_limit = float(first.headers[deadline.HEADER]), float(late.headers[deadline.HEADER])
    assert first_limit - late_limit == pytest.approx(8, abs=2)
    assert seen[A] == first_limit and seen[C] == late_limit  # worker 의 ContextVar 까지 같은 값이 갔다
    timeout = late.extensions["timeout"]
    assert 100 < timeout["read"] <= 112
    assert (timeout["connect"], timeout["write"], timeout["pool"]) == (10.0, 60.0, 5.0)
    waited = max(s.log.requests(), key=lambda fields: float(fields["queue_wait"].rstrip("s")))
    assert float(waited["queue_wait"].rstrip("s")) >= 8
    assert 100 < float(waited["budget"].rstrip("s")) <= 112
    s.settle()


def test_short_remaining_time_caps_every_http_timeout_phase(stack):
    s = stack(total=12.0)
    clock = s.use_clock()
    upload = s.pipeline.gate("s3_upload", A, B)
    a, b = s.run_all(A, B)
    upload.wait_for(2)
    c = s.post(picture(C))
    s.wait_for_queue()

    clock.advance(9.0)
    upload.open()

    assert statuses(a, b, c) == [200, 200, 200]
    timeout = s.modal.requests[2].extensions["timeout"]
    assert all(0 < value <= 3 for value in timeout.values())


def test_deadline_ending_first_in_the_queue_is_504_without_calling_the_worker(stack):
    s = stack(total=8.0)
    clock = s.use_clock()
    upload = s.pipeline.gate("s3_upload")
    a, b = s.run_all(A, B)
    upload.wait_for(2)
    c = s.post(picture(C))
    s.wait_for_queue()

    clock.advance(8.1)
    s.wake_the_queue()

    assert c.result().status_code == 504
    assert c.result().json()["code"] == TIMEOUT_CODE
    fields = s.log.request_fields("deadline_exceeded")
    assert fields["timeout_stage"] == "queue"
    assert len(s.modal.requests) == 2
    upload.open()
    assert statuses(a, b) == [200, 200]
    s.settle()


def test_slot_freed_after_the_deadline_is_504_and_the_worker_is_not_called(stack):
    s = stack(total=8.0)
    clock = s.use_clock()
    upload = s.pipeline.gate("s3_upload")
    a, b = s.run_all(A, B)
    upload.wait_for(2)
    c = s.post(picture(C))
    s.wait_for_queue()

    clock.advance(8.1)
    upload.open()

    assert statuses(a, b) == [200, 200]
    assert c.result().status_code == 504
    assert s.log.request_fields("deadline_exceeded")["timeout_stage"] == "queue"
    assert len(s.modal.requests) == 2
    s.settle()


def test_deadline_during_the_remote_call_is_504_and_the_worker_stops_at_its_next_check(stack):
    s = stack(total=0.5)
    held = s.pipeline.gate("detection_input")
    a, b = s.run_all(A, B)
    held.wait_for(2)

    assert statuses(a, b) == [504, 504]
    assert a.result().json()["code"] == TIMEOUT_CODE
    assert s.ai_slots == (0, 0)  # AI Server 는 돌아왔고
    assert s.worker_slots == (2, 0)  # worker thread 는 아직 돈다
    held.open()
    s.settle()
    assert {s.line(A)["outcome"], s.line(B)["outcome"]} == {TIMEOUT_CODE}
    assert s.pipeline.uploads == {}  # 아무도 받지 않을 결과는 올리지 않았다


def test_worker_filled_by_orphaned_work_answers_429_through_the_ai_server(stack, monkeypatch):
    s = stack(total=0.5)
    held = s.pipeline.gate("detection_input", A, B)
    a, b = s.run_all(A, B)
    held.wait_for(2)
    assert statuses(a, b) == [504, 504]
    monkeypatch.setattr(config, "REMOTE_TOTAL_DEADLINE_SECONDS", 120.0)

    c = s.post(picture(C))
    wait_until(lambda: s.worker.limiter.waiting == 1)
    d = s.post(picture(D))

    assert d.result().status_code == 429
    assert d.result().json() == BUSY
    assert s.ai_slots == (1, 0) and s.worker_slots == (2, 1)  # AI Server 슬롯은 worker 쪽 고아 작업을 모른다
    held.open()
    assert c.result().status_code == 200
    s.settle()


def test_two_requests_share_a_real_connection_pool_over_tcp(live, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "MAX_CONCURRENT_BODY_VALIDATIONS", 2)
    server = live()
    first, second = server.send().read(), server.send().read()
    wait_until(lambda: server.modal.started == 2)
    third = server.send().read()
    server.wait_for_queue(1)

    status, body = server.send().read().response()

    assert (status, body["code"]) == (429, "SERVER_BUSY")
    assert server.slots == (2, 1)
    server.modal.release.set()
    assert [first.response()[0], second.response()[0], third.response()[0]] == [200, 200, 200]
    assert server.modal.max_active == 2
    assert server.modal.connections == 2  # 대기하던 요청은 끝난 요청의 연결을 다시 썼다
    wait_until(lambda: server.slots == (0, 0))


def test_backend_disconnect_over_tcp_leaves_the_other_running_request_alone(live, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "MAX_CONCURRENT_BODY_VALIDATIONS", 2)
    server = live()
    dropped, kept = server.send(), server.send().read()
    wait_until(lambda: server.modal.started == 2)

    dropped.disconnect()
    health = socket.create_connection(("127.0.0.1", server.port), timeout=WAIT)
    health.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
    assert health.recv(4096).startswith(b"HTTP/1.1 200")
    health.close()

    assert server.slots == (2, 0)  # 끊겨도 호출은 끝까지 가고 슬롯도 그때 돌아온다
    server.modal.release.set()
    assert kept.response()[0] == 200
    wait_until(lambda: server.slots == (0, 0))
    assert server.modal.finished == 2


# --- B. 실제 실행 구간과 계측 ---


def test_limiter_1_never_reports_two_running_even_when_both_requests_are_inside_the_worker(stack):
    s = stack(worker_running=1)
    held = s.pipeline.gate("s3_upload", A)
    a = s.post(picture(A))
    held.wait_for(1)
    b = s.post(picture(B))
    wait_until(lambda: s.worker.limiter.waiting == 1)
    time.sleep(0.05)  # 대기 요청의 slot_wait 가 이만큼은 쌓인다
    held.open()

    assert statuses(a, b) == [200, 200]

    first, second = s.line(A), s.line(B)
    assert (first["execution"]["active_max"], second["execution"]["active_max"]) == (1, 1)
    assert first["resources"]["windows_max"] == second["resources"]["windows_max"] == 2  # 이 값으로 판단하면 오탐이다
    assert not overlaps(interval(first), interval(second))
    assert second["ms"]["slot_wait"] >= 40
    assert first["ms"]["slot_wait"] < second["ms"]["slot_wait"]
    assert first["execution"]["end_ms"] <= second["execution"]["slot_acquired_ms"]
    s.settle()


def test_limiter_2_reports_both_requests_in_one_process_with_their_own_ids(stack):
    s = stack()
    s.pipeline.on("detection_input", meet())

    assert statuses(*s.run_all(A, B)) == [200, 200]

    a, b = s.line(A), s.line(B)
    assert a["request_id"] != b["request_id"] and a["seq"] != b["seq"]
    assert a["execution"]["process_id"] == b["execution"]["process_id"]
    assert (a["execution"]["active_max"], b["execution"]["active_max"]) == (2, 2)
    assert sorted((a["execution"]["active_at_start"], b["execution"]["active_at_start"])) == [0, 1]
    assert overlaps(interval(a), interval(b))
    assert {"preprocess", "person", "pose", "brightness", "rembg", "png", "s3_upload"} <= set(a["execution"]["spans_ms"])
    assert set(a["execution"]["spans_ms"]) == set(b["execution"]["spans_ms"])


def test_timeline_of_every_request_stays_ordered_and_inside_its_execution(stack):
    s = stack()
    s.pipeline.on("s3_upload", meet())

    for first, second in ((A, B), (C, D)):
        assert statuses(*s.run_all(first, second)) == [200, 200]
        assert_timeline(s.line(first))
        assert_timeline(s.line(second))
    a, c = s.line(A), s.line(C)
    assert a["execution"]["end_ms"] <= c["execution"]["received_ms"]  # 같은 process 라 시각을 비교할 수 있다


def fail_with(stage, error):
    def install(s, wall):
        def hook(tag):
            if tag == A:
                raise error

        s.pipeline.on(stage, hook)

    return install


def pass_the_deadline(s, wall):
    def hook(tag):
        if tag == A:
            wall.offset = 118.5  # A 는 upload 여유(2초)를 남기지 못하고 worker 가 스스로 멈춘다

    s.pipeline.on("rembg", hook)


@pytest.mark.parametrize(
    ("install", "status", "code", "outcome"),
    [
        (None, 200, "BODY_IMAGE_UPLOAD_SUCCESS", "success"),
        (fail_with("person", RuntimeError("person")), 500, "PERSON_DETECTION_FAILED", "PERSON_DETECTION_FAILED"),
        (fail_with("pose", RuntimeError("pose")), 500, "POSE_ESTIMATION_FAILED", "POSE_ESTIMATION_FAILED"),
        (fail_with("rembg", RuntimeError("rembg")), 500, "BACKGROUND_REMOVAL_FAILED", "BACKGROUND_REMOVAL_FAILED"),
        (fail_with("s3_upload", ImageStorageError("s3")), 500, "IMAGE_UPLOAD_FAILED", "IMAGE_UPLOAD_FAILED"),
        (pass_the_deadline, 504, TIMEOUT_CODE, TIMEOUT_CODE),
    ],
    ids=["success", "person", "pose", "rembg", "s3", "deadline"],
)
def test_every_ending_closes_the_interval_and_returns_the_slot_while_the_other_request_runs(
    stack, wall, install, status, code, outcome
):
    s = stack()
    if install is not None:
        install(s, wall)
    s.pipeline.on("detection_input", meet())
    other = s.pipeline.gate("brightness", B)
    a, b = s.run_all(A, B)

    assert a.result().status_code == status
    assert a.result().json()["code"] == code
    wall.offset = 0.0
    other.open()
    assert b.result().status_code == 200
    failed = s.line(A)
    assert failed["outcome"] == outcome
    assert failed["execution"]["start_ms"] is not None and failed["execution"]["end_ms"] is not None
    assert_timeline(failed)
    s.settle()
    s.pipeline.on("detection_input", lambda tag: None)
    assert statuses(s.post(picture(C))) == [200]  # 다음 요청이 실행된다
    assert s.line(C)["execution"]["active_max"] == 1
    s.settle()


# --- C. 모델별 Lock ---


class WatchedLock:
    """실제 Lock 에 도착 수만 더한다. 두 번째 요청이 Lock 앞까지 온 것을 sleep 없이 확인한다."""

    def __init__(self, lock):
        self._lock = lock
        self.arrivals = 0

    def acquire(self):
        self.arrivals += 1
        return self._lock.acquire()

    def release(self):
        self._lock.release()


def watch(s, stage):
    models = s.runtime.models
    owner = {
        "person": models.person_detector,
        "pose": models.pose_detector,
        "rembg": models.background_remover,
        "nvjpeg": getattr(models.gpu_preprocessor, "_decoder", None),
    }[stage]
    owner._lock._lock = WatchedLock(owner._lock._lock)
    return owner._lock._lock


LOCK_STAGES = [("person", "PNG"), ("pose", "PNG"), ("rembg", "PNG"), ("nvjpeg", "JPEG")]


@pytest.mark.parametrize(("stage", "image_format"), LOCK_STAGES, ids=[stage for stage, _ in LOCK_STAGES])
def test_second_request_waits_at_the_model_lock_until_the_first_returns_it(stack, stage, image_format):
    s = stack()
    lock = watch(s, stage)
    held = s.pipeline.gate(stage, A)

    a = s.post(picture(A, image_format))
    held.wait_for(1)
    b = s.post(picture(B, image_format))
    wait_until(lambda: lock.arrivals == 2)
    time.sleep(0.02)  # B 의 wait_ms 가 이만큼은 쌓인다
    assert s.pipeline.order[stage] == [(A, "in")]  # B 는 Lock 안에 들어오지 못했다
    held.open()

    assert statuses(a, b) == [200, 200]
    assert s.pipeline.order[stage] == [(A, "in"), (A, "out"), (B, "in"), (B, "out")]
    assert s.pipeline.most_inside[stage] == 1
    holder, waiter = s.line(A)["locks"][stage], s.line(B)["locks"][stage]
    assert holder["count"] == waiter["count"] == 1
    assert waiter["wait_ms"] >= 15 and holder["wait_ms"] < waiter["wait_ms"]
    assert holder["hold_ms"] >= 15
    assert s.line(A)["execution"]["active_max"] == s.line(B)["execution"]["active_max"] == 2
    s.settle()


@pytest.mark.parametrize(
    ("stage", "image_format", "error", "status", "code"),
    [
        ("person", "PNG", RuntimeError("model"), 500, "PERSON_DETECTION_FAILED"),
        ("pose", "PNG", RuntimeError("model"), 500, "POSE_ESTIMATION_FAILED"),
        ("rembg", "PNG", RuntimeError("model"), 500, "BACKGROUND_REMOVAL_FAILED"),
        ("nvjpeg", "JPEG", RuntimeError("cuda"), 200, "BODY_IMAGE_UPLOAD_SUCCESS"),
        ("nvjpeg", "JPEG", NotImplementedError("code"), 500, "INTERNAL_SERVER_ERROR"),
    ],
    ids=["person", "pose", "rembg", "nvjpeg-falls-back-to-pillow", "nvjpeg-code-error"],
)
def test_model_error_inside_the_lock_hands_it_to_the_waiting_request(stack, stage, image_format, error, status, code):
    s = stack()
    lock = watch(s, stage)
    held, release = threading.Event(), threading.Event()

    def fail_first(tag):
        if tag == A:
            held.set()
            assert release.wait(WAIT)
            raise error

    s.pipeline.on(stage, fail_first)
    a = s.post(picture(A, image_format))
    assert held.wait(WAIT)
    b = s.post(picture(B, image_format))
    wait_until(lambda: lock.arrivals == 2)
    release.set()

    assert a.result().status_code == status
    assert a.result().json()["code"] == code
    assert b.result().status_code == 200
    assert s.line(A)["locks"][stage]["count"] == 1
    assert s.line(B)["locks"][stage]["count"] == 1
    s.settle()
    s.pipeline.on(stage, lambda tag: None)
    assert statuses(s.post(picture(C, image_format))) == [200]
    s.settle()


# --- D. 이미지 격리 ---


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_each_response_points_at_the_png_of_its_own_image_while_the_paths_overlap(stack, image_format):
    s = stack()
    s.pipeline.on("detection_input", meet())
    s.pipeline.on("brightness", meet())
    s.pipeline.on("s3_upload", meet())
    keys = []

    for first, second in zip(TAGS[::2], TAGS[1::2], strict=False):
        bodies = {tag: picture(tag, image_format) for tag in (first, second)}
        requests = {tag: s.post(body) for tag, body in bodies.items()}

        for tag, request in requests.items():
            assert request.result().status_code == 200
            key = request.result().json()["data"]["s3_key"]
            keys.append(key)
            assert S3_KEY.fullmatch(key)
            png, content_type = s.pipeline.uploads[key]
            assert content_type == "image/png"
            stored = Image.open(BytesIO(png))
            assert stored.mode == "RGBA"
            result = np.asarray(stored)
            assert np.array_equal(result[..., :3], rgb_of(bodies[tag]))  # 다시 decode 해 픽셀 전체를 비교한다
            assert (result[..., 3] == tag).all()  # rembg 도 이 요청의 이미지를 받았다

    assert len(set(keys)) == len(keys)
    assert len(s.pipeline.uploads) == len(keys)


def test_each_stage_sees_its_own_arrays(stack):
    s = stack()
    s.pipeline.on("detection_input", meet())

    assert statuses(*s.run_all(A, B)) == [200, 200]

    for stage in ("person", "pose", "brightness", "rembg"):
        first, second = s.pipeline.arrays[stage][A], s.pipeline.arrays[stage][B]
        assert np.array_equal(first, pixels(A)) and np.array_equal(second, pixels(B))
        assert not np.shares_memory(first, second)
    assert s.pipeline.arrays["person"][A] is s.pipeline.arrays["pose"][A]  # 한 요청 안에서는 같은 변환 결과를 쓴다


# --- E. 저장소 동시성(FakeStorage. 실제 boto3·S3 는 3-1C-4 에서 본다) ---


def test_two_uploads_overlap_and_keep_their_own_png_and_key(stack):
    s = stack()
    s.pipeline.on("s3_upload", meet())
    a, b = s.run_all(A, B)

    assert statuses(a, b) == [200, 200]

    first, second = s.line(A), s.line(B)
    assert overlaps(span(first, "s3_upload"), span(second, "s3_upload"))
    key_a, key_b = a.result().json()["data"]["s3_key"], b.result().json()["data"]["s3_key"]
    assert key_a != key_b
    assert tag_of(np.asarray(Image.open(BytesIO(s.pipeline.uploads[key_a][0])))) == A
    assert tag_of(np.asarray(Image.open(BytesIO(s.pipeline.uploads[key_b][0])))) == B


def test_upload_of_one_request_overlaps_inference_of_the_other(stack):
    s = stack()
    uploading, inferring = threading.Event(), threading.Event()
    rembg_order = []

    def rembg(tag):
        rembg_order.append(tag)
        if len(rembg_order) == 2:
            assert uploading.wait(WAIT)
            inferring.set()
            time.sleep(RESOLUTION)

    def upload(tag):
        if not uploading.is_set():
            uploading.set()
            assert inferring.wait(WAIT)
            time.sleep(RESOLUTION)

    s.pipeline.on("rembg", rembg)
    s.pipeline.on("s3_upload", upload)

    assert statuses(*s.run_all(A, B)) == [200, 200]

    uploader, inferrer = (s.line(tag) for tag in rembg_order)
    assert overlaps(span(uploader, "s3_upload"), span(inferrer, "rembg"))
    assert overlaps(interval(uploader), interval(inferrer))


def test_failed_upload_leaves_the_other_upload_and_the_next_request_alone(stack):
    s = stack()
    both = threading.Barrier(2, timeout=WAIT)

    def upload(tag):
        both.wait()
        if tag == A:
            raise ImageStorageError("upload failed")

    s.pipeline.on("s3_upload", upload)
    a, b = s.run_all(A, B)

    assert a.result().status_code == 500
    assert a.result().json()["code"] == "IMAGE_UPLOAD_FAILED"
    assert b.result().status_code == 200
    assert b.result().json()["data"]["s3_key"] in s.pipeline.uploads
    assert len(s.pipeline.uploads) == 1
    s.settle()
    s.pipeline.on("s3_upload", lambda tag: None)
    assert statuses(s.post(picture(C))) == [200]
    s.settle()


# --- F. Deadline·요청 context 격리 ---


def test_timeout_of_one_request_leaves_the_deadline_of_the_other_alone(stack, wall):
    s = stack()
    seen = defaultdict(list)
    near, far = time.time() + 60, time.time() + 600
    both = meet()
    passed, a_ended = threading.Event(), threading.Event()

    def entered(tag):
        seen[tag].append(deadline._deadline.get())
        both(tag)
        if tag == A:
            assert passed.wait(WAIT)
        else:
            wall.offset = 100.0
            passed.set()
            assert a_ended.wait(WAIT)
            seen[tag].append(deadline._deadline.get())

    s.pipeline.on("detection_input", entered)
    a = s.post_to_worker(picture(A), near)
    b = s.post_to_worker(picture(B), far)

    assert a.result().status_code == 504
    a_ended.set()
    assert b.result().status_code == 200

    assert seen[A] == [pytest.approx(near, abs=0.01)]
    assert seen[B] == [pytest.approx(far, abs=0.01)] * 2
    assert s.line(A)["outcome"] == TIMEOUT_CODE and s.line(B)["outcome"] == "success"


def test_request_records_stay_separate_while_two_requests_interleave(stack):
    s = stack()
    records = {}

    def entered(tag):
        records[tag] = current_request()
        both(tag)

    both = meet()
    s.pipeline.on("detection_input", entered)

    assert statuses(*s.run_all(A, B)) == [200, 200]

    assert records[A] is not records[B]
    assert records[A].request_id == s.line(A)["request_id"]
    assert records[B].request_id == s.line(B)["request_id"]
    assert records[A].locks is not records[B].locks
    assert all(s.line(tag)["locks"]["rembg"]["count"] == 1 for tag in (A, B))
    assert len({fields["request_id"] for fields in s.log.requests()}) == 2


def test_bad_key_during_two_running_requests_is_rejected_on_both_sides_without_touching_them(stack):
    s = stack()
    held = s.pipeline.gate("detection_input")
    a, b = s.run_all(A, B)
    held.wait_for(2)

    no_bearer = s.post(picture(C), headers={})
    worker_bad_bearer = s.post_to_worker(picture(C), bearer="wrong")

    assert statuses(no_bearer, worker_bad_bearer) == [401, 401]
    assert (s.ai_slots, s.worker_slots) == ((2, 0), (2, 0))
    assert len(held.arrived) == 2
    held.open()
    assert statuses(a, b) == [200, 200]
    for request in s.modal.requests[:2]:
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert request.headers["modal-key"] == "wk-test"
        assert "cookie" not in request.headers


def test_consecutive_requests_on_the_same_thread_start_clean(stack, wall):
    s = stack()
    seen = {}

    def entered(tag):
        seen[tag] = (threading.get_ident(), deadline._deadline.get(), dict(current_request().spans))
        if tag == A:
            wall.offset = 100.0

    s.pipeline.on("detection_input", entered)
    s.pipeline.on("rembg", lambda tag: None)
    limit = time.time() + 60

    failed = s.post_to_worker(picture(A), limit)
    assert failed.result().status_code == 504
    wall.offset = 0.0
    clean = s.post_to_worker(picture(B))
    assert clean.result().status_code == 200

    assert seen[A][0] == seen[B][0]
    assert seen[B][1] is None  # 앞 요청의 deadline 이 남지 않았다
    assert set(seen[B][2]) == {"preprocess"}  # 앞 요청의 구간이 남지 않았다
    first, second = s.line(A), s.line(B)
    assert first["request_id"] != second["request_id"]
    assert second["execution"]["active_max"] == 1
    assert second["locks"]["person"]["count"] == 1


# --- 취소: 요청이 취소돼도 이미 도는 thread 는 멈추지 않는다 ---


def test_cancel_while_waiting_in_the_ai_server_queue_returns_the_place_and_calls_no_worker(stack):
    s = stack()
    upload = s.pipeline.gate("s3_upload")
    a, b = s.run_all(A, B)
    upload.wait_for(2)
    c = s.post(picture(C))
    s.wait_for_queue()

    c.cancel()

    wait_until(c.task.done)
    assert c.task.cancelled()
    assert s.ai_slots == (2, 0)
    assert len(s.modal.requests) == 2
    d = s.post(picture(D))  # 비었던 대기열 자리를 쓴다
    s.wait_for_queue()
    upload.open()
    assert statuses(a, b, d) == [200, 200, 200]
    assert s.log.request_fields("cancelled")["queue_wait"] != "-"
    s.settle()


def test_cancel_during_the_remote_call_frees_the_ai_slot_and_the_worker_finishes_on_its_own(stack):
    s = stack()
    held = s.pipeline.gate("detection_input", A)
    a = s.post(picture(A))
    held.wait_for(1)

    a.cancel()

    wait_until(lambda: s.ai_slots == (0, 0))
    assert s.worker_slots == (1, 0)  # worker 쪽 작업은 AI Server 의 취소를 모른다
    s.pipeline.on("s3_upload", meet())
    b = s.post(picture(B))
    wait_until(lambda: s.worker.limiter.running == 2)  # AI Server 슬롯은 1건인데 worker 에서는 2건이 겹친다
    held.open()

    assert b.result().status_code == 200
    s.settle()
    assert len(s.pipeline.uploads) == 2  # 취소된 요청의 결과도 올라간다(이미 시작한 Modal 작업은 멈추지 않는다)
    assert s.line(A)["outcome"] == "success"
    assert s.line(A)["execution"]["active_max"] == 2


def test_cancel_before_the_worker_slot_does_not_leave_the_queue_and_runs_after_the_slot_frees(stack):
    s = stack()
    held = s.pipeline.gate("s3_upload", A, B)
    a, b = s.post_to_worker(picture(A)), s.post_to_worker(picture(B))
    held.wait_for(2)
    s.post_to_worker(picture(C))
    wait_until(lambda: s.worker.limiter.waiting == 1)

    input_task = s.cancel_input(2)
    time.sleep(0.05)

    assert not input_task.done()  # threadpool 의 thread 가 끝날 때까지 ASGI Task 도 끝나지 않는다
    assert s.worker_slots == (2, 1)  # 슬롯을 기다리는 thread 는 취소를 받지 못한다
    held.open()
    assert statuses(a, b) == [200, 200]
    s.settle()
    wait_until(input_task.done)
    assert input_task.cancelled()
    assert len(s.pipeline.uploads) == 3  # 취소됐어도 슬롯을 얻으면 끝까지 돈다
    assert s.line(C)["outcome"] == "success"


def test_cancelled_request_waiting_for_the_worker_slot_is_stopped_only_by_its_deadline(stack, wall):
    s = stack()
    held = s.pipeline.gate("s3_upload", A, B)
    a, b = s.post_to_worker(picture(A)), s.post_to_worker(picture(B))
    held.wait_for(2)
    s.post_to_worker(picture(C), time.time() + 60)
    wait_until(lambda: s.worker.limiter.waiting == 1)
    s.cancel_input(2)

    wall.offset = 100.0
    held.open()

    assert statuses(a, b) == [200, 200]
    s.settle()
    assert len(s.pipeline.uploads) == 2
    outcome = next(line for line in s.log.diagnostics if line["seq"] == 3)
    assert outcome["outcome"] == TIMEOUT_CODE
    assert outcome["execution"]["spans_ms"] == {}  # 슬롯을 얻은 직후 멈췄다


def test_cancel_while_the_worker_thread_runs_keeps_the_slot_until_the_thread_ends(stack):
    s = stack()
    held = s.pipeline.gate("detection_input", A)
    s.post_to_worker(picture(A))
    held.wait_for(1)

    input_task = s.cancel_input(0)
    time.sleep(0.05)

    assert not input_task.done()
    assert s.worker_slots == (1, 0)  # thread 는 취소되지 않고 슬롯을 쥐고 있다
    assert len(s.worker._execution._running) == 1
    held.open()
    s.settle()
    wait_until(input_task.done)
    assert input_task.cancelled()
    assert s.line(A)["outcome"] == "success"  # 취소 뒤에도 thread 는 끝까지 갔다
    assert s.line(A)["execution"]["end_ms"] is not None
    assert len(s.pipeline.uploads) == 1
    assert statuses(s.post_to_worker(picture(B))) == [200]


# --- G. Health ---


def test_health_answers_while_the_worker_limiter_is_full_because_it_never_touches_it(stack):
    s = stack()
    assert s.health().result().json() == {"status": "ok"}
    held = s.pipeline.gate("s3_upload", A, B)
    a, b = s.post_to_worker(picture(A)), s.post_to_worker(picture(B))
    held.wait_for(2)
    c = s.post_to_worker(picture(C))
    wait_until(lambda: s.worker.limiter.waiting == 1)

    assert s.health().result().json() == {"status": "ok"}

    assert s.worker_slots == (2, 1)
    held.open()
    assert statuses(a, b, c) == [200, 200, 200]


def test_health_needs_the_modal_proxy_keys(stack):
    assert stack().health(proxy=False).result().status_code == 401


def test_health_waits_for_an_input_slot_and_recovers(stack):
    s = stack(inputs=2)
    held = s.pipeline.gate("detection_input")
    a, b = s.run_all(A, B)
    held.wait_for(2)

    health = s.health()
    time.sleep(0.05)

    assert not health.task.done()  # 흉내 낸 입력 슬롯이 찼다. Worker Limiter 를 기다리는 것이 아니다
    assert s.ai_slots == (2, 0) and s.worker_slots == (2, 0)
    held.open()
    assert statuses(a, b, health) == [200, 200, 200]
    s.settle()


def test_warmup_timeout_ends_the_health_call_while_inputs_are_full_and_the_next_one_succeeds(stack):
    s = stack(inputs=2)
    held = s.pipeline.gate("detection_input")
    a, b = s.run_all(A, B)
    held.wait_for(2)
    client = ModalWarmupClient(
        WORKER, proxy_key="wk-test", proxy_secret=PROXY_SECRET, timeout=0.1, transport=s.modal
    )

    saturated = s.portal.call(client.warm_up)

    assert saturated.outcome is WarmupOutcome.DEADLINE_EXCEEDED
    assert s.worker_slots == (2, 0)  # 실패한 warm-up 이 실행 중인 요청을 건드리지 않는다
    held.open()
    assert statuses(a, b) == [200, 200]
    wait_until(lambda: s.modal.inputs.locked() is False and all(task.done() for task in s.modal.tasks))
    assert s.portal.call(client.warm_up).outcome is WarmupOutcome.SUCCESS


# --- 요청 ID 전파: 시간이나 완료 순서가 아니라 실제로 전달된 값으로 잇는다 ---


TRACE_ID = "85c390e8344842ad"


def test_worker_line_names_the_ai_request_and_keeps_its_own_id(stack):
    s = stack()

    [request] = s.run_all(A)

    response = request.result()
    assert response.status_code == 200
    [fields] = s.log.requests()
    line = s.line(A)
    assert re.fullmatch(r"[0-9a-f]{16}", fields["request_id"])
    assert line["ai_request_id"] == fields["request_id"] == s.sent_ids()[A]
    assert line["request_id"] == s.pipeline.request_ids[A] != fields["request_id"]
    assert set(response.json()) == {"code", "data", "message"}
    assert fields["request_id"] not in response.text
    assert REQUEST_ID_HEADER not in response.headers


def test_two_overlapping_requests_keep_their_own_ids_on_both_sides(stack):
    s = stack()
    s.pipeline.on("s3_upload", meet())

    assert statuses(*s.run_all(A, B)) == [200, 200]

    sent = s.sent_ids()
    a, b = s.line(A), s.line(B)
    assert sent[A] != sent[B] and a["request_id"] != b["request_id"]
    assert (a["ai_request_id"], b["ai_request_id"]) == (sent[A], sent[B])
    assert {fields["request_id"] for fields in s.log.requests()} == {sent[A], sent[B]}
    assert s.pipeline.ai_request_ids == {A: {sent[A]}, B: {sent[B]}}  # threadpool 의 모든 단계에서 자기 ID 다
    assert (a["execution"]["active_max"], b["execution"]["active_max"]) == (2, 2)
    assert a["execution"]["process_id"] == b["execution"]["process_id"]
    assert overlaps(interval(a), interval(b))
    s.settle()


def test_ids_stay_matched_when_the_second_request_finishes_first(stack):
    s = stack()
    upload = s.pipeline.gate("s3_upload", A)
    a = s.post(picture(A))
    upload.wait_for(1)

    b = s.post(picture(B))
    assert b.result().status_code == 200
    assert not a.task.done()
    upload.open(A)
    assert a.result().status_code == 200

    sent = s.sent_ids()
    assert [fields["request_id"] for fields in s.log.requests()] == [sent[B], sent[A]]
    assert [line["ai_request_id"] for line in s.log.diagnostics] == [sent[B], sent[A]]
    assert overlaps(interval(s.line(A)), interval(s.line(B)))
    s.settle()


def test_id_survives_the_wait_for_the_worker_slot(stack):
    s = stack(worker_running=1)
    upload = s.pipeline.gate("s3_upload", A)
    a = s.post(picture(A))
    upload.wait_for(1)
    b = s.post(picture(B))
    wait_until(lambda: s.worker.limiter.waiting == 1)

    upload.open(A)

    assert statuses(a, b) == [200, 200]
    sent = s.sent_ids()
    assert (s.line(A)["ai_request_id"], s.line(B)["ai_request_id"]) == (sent[A], sent[B])
    assert s.line(B)["ms"]["slot_wait"] > 0
    s.settle()


@pytest.mark.parametrize(
    ("install", "status", "code"),
    [
        (fail_with("brightness", user_error("FULL_BODY_NOT_VISIBLE")), 422, "FULL_BODY_NOT_VISIBLE"),
        (fail_with("rembg", RuntimeError("rembg")), 500, "BACKGROUND_REMOVAL_FAILED"),
        (fail_with("s3_upload", ImageStorageError("s3")), 500, "IMAGE_UPLOAD_FAILED"),
        (pass_the_deadline, 504, TIMEOUT_CODE),
    ],
    ids=["validation", "rembg", "s3", "deadline"],
)
def test_failed_request_and_its_worker_line_share_one_id_while_the_other_succeeds(stack, wall, install, status, code):
    s = stack()
    install(s, wall)
    failing = s.pipeline.hooks.get("brightness", lambda tag: None)
    other = Gate(B)
    s.pipeline.gates.append(other)

    def brightness(tag):
        failing(tag)
        other(tag)

    s.pipeline.on("brightness", brightness)
    s.pipeline.on("detection_input", meet())
    a, b = s.run_all(A, B)

    assert a.result().status_code == status
    assert a.result().json()["code"] == code
    wall.offset = 0.0
    other.open()
    assert b.result().status_code == 200
    sent = s.sent_ids()
    fields = next(fields for fields in s.log.requests() if fields["code"] == code)
    failed, passed = s.line(A), s.line(B)
    assert fields["request_id"] == sent[A] == failed["ai_request_id"]
    assert failed["outcome"] == code
    assert passed["ai_request_id"] == sent[B] != sent[A]
    assert passed["outcome"] == "success"
    s.settle()


def test_worker_line_of_an_abandoned_request_still_names_it(stack):
    s = stack(total=0.5)
    held = s.pipeline.gate("detection_input")
    a, b = s.run_all(A, B)
    held.wait_for(2)

    assert statuses(a, b) == [504, 504]
    assert s.worker_slots == (2, 0)  # worker 는 아직 돌고 있다
    held.open()
    s.settle()

    sent = s.sent_ids()
    assert {fields["request_id"] for fields in s.log.requests()} == {sent[A], sent[B]}
    assert (s.line(A)["ai_request_id"], s.line(B)["ai_request_id"]) == (sent[A], sent[B])
    assert {s.line(A)["outcome"], s.line(B)["outcome"]} == {TIMEOUT_CODE}


def test_requests_that_never_reach_the_worker_are_logged_only_by_the_ai_server(stack):
    s = stack(total=8.0)
    clock = s.use_clock()
    upload = s.pipeline.gate("s3_upload")
    a, b = s.run_all(A, B)
    upload.wait_for(2)
    c = s.post(picture(C))
    s.wait_for_queue()
    d = s.post(picture(D))
    assert d.result().status_code == 429

    clock.advance(8.1)
    s.wake_the_queue()

    assert c.result().status_code == 504
    upload.open()
    assert statuses(a, b) == [200, 200]
    s.settle()
    sent = s.sent_ids()
    stopped = {s.log.request_fields(outcome)["request_id"] for outcome in ("queue_full", "deadline_exceeded")}
    assert len(stopped) == 2 and set(sent) == {A, B}
    assert not stopped & {line["ai_request_id"] for line in s.log.diagnostics}
    assert not stopped & set(sent.values())


@pytest.mark.parametrize(
    ("header", "value"), [("Authorization", "Bearer wrong"), ("Modal-Key", "wrong")], ids=["worker_bearer", "modal_proxy"]
)
def test_rejected_call_carries_the_id_and_leaves_no_worker_line(stack, header, value):
    s = stack()
    s.ai._headers[header] = value

    [request] = s.run_all(A)

    assert request.result().status_code == 502
    [fields] = s.log.requests()
    assert fields["status"] == "401"
    assert s.sent_ids()[A] == fields["request_id"]
    assert s.log.diagnostics == []
    assert not s.pipeline.order


def test_shared_client_keeps_no_request_id_between_requests(stack):
    s = stack()
    s.pipeline.on("s3_upload", meet())
    before = dict(s.ai._shared_client.headers)

    assert statuses(*s.run_all(A, B)) == [200, 200]

    assert dict(s.ai._shared_client.headers) == before
    assert REQUEST_ID_HEADER not in s.ai._shared_client.headers
    s.pipeline.on("s3_upload", lambda tag: None)
    assert statuses(*s.run_all(C)) == [200]
    assert len(set(s.sent_ids().values())) == 3


def test_id_sent_by_a_client_of_the_ai_server_goes_nowhere(stack):
    s = stack()

    request = s.post(picture(A), headers={**AUTH, REQUEST_ID_HEADER: TRACE_ID})

    assert request.result().status_code == 200
    [fields] = s.log.requests()
    assert fields["request_id"] != TRACE_ID
    assert s.sent_ids()[A] == s.line(A)["ai_request_id"] == fields["request_id"]
    assert TRACE_ID not in "\n".join(s.log.messages)


def test_worker_records_a_valid_id_from_a_direct_call(stack):
    s = stack()

    request = s.post_to_worker(picture(A), trace=TRACE_ID)

    assert request.result().status_code == 200
    assert s.line(A)["ai_request_id"] == TRACE_ID
    assert s.line(A)["request_id"] != TRACE_ID


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        TRACE_ID[:-1],
        TRACE_ID + "0",
        "a" * 4096,
        TRACE_ID.upper(),
        "85c390e8344842a-",
        TRACE_ID + "\n",
        f"{TRACE_ID}\r\nX-Injected: 1",
        f"{TRACE_ID},{TRACE_ID}",
        "../../etc/passwd",
    ],
    ids=["missing", "empty", "short", "long", "huge", "upper", "symbol", "newline", "crlf", "list", "path"],
)
def test_worker_runs_normally_and_records_nothing_for_an_unusable_id(stack, value):
    s = stack()

    request = s.post_to_worker(picture(A), trace=value)

    assert request.result().status_code == 200
    line = s.line(A)
    assert line["ai_request_id"] is None
    assert re.fullmatch(r"[0-9a-f]{16}", line["request_id"])
    assert line["outcome"] == "success"
    if value:
        assert value not in "\n".join(s.log.messages)


@pytest.mark.parametrize("trace", [TRACE_ID, "not valid"], ids=["valid", "invalid"])
def test_trace_header_never_changes_the_worker_auth_decision(stack, trace):
    s = stack()

    rejected = s.post_to_worker(picture(A), bearer="wrong", trace=trace)
    assert rejected.result().status_code == 401
    assert s.log.diagnostics == [] and not s.pipeline.order

    accepted = s.post_to_worker(picture(B), trace=trace)
    assert accepted.result().status_code == 200


def test_id_of_one_worker_call_does_not_reach_the_next_one(stack):
    s = stack()

    assert statuses(s.post_to_worker(picture(A), trace=TRACE_ID)) == [200]
    assert statuses(s.post_to_worker(picture(B))) == [200]

    assert s.line(A)["ai_request_id"] == TRACE_ID
    assert s.line(B)["ai_request_id"] is None


def test_each_request_carries_its_own_id_over_a_real_connection(live, monkeypatch, caplog):  # noqa: F811
    monkeypatch.setattr(config, "MAX_CONCURRENT_BODY_VALIDATIONS", 2)
    server = live()
    first, second = server.send().read(), server.send().read()
    wait_until(lambda: server.modal.started == 2)

    with caplog.at_level(logging.INFO, logger="app.body_image_validation.remote"):
        server.modal.release.set()
        assert [first.response()[0], second.response()[0]] == [200, 200]
        wait_until(lambda: server.slots == (0, 0))

    assert server.modal.connections == 2
    sent = {headers[REQUEST_ID_HEADER] for headers in server.modal.headers}
    logged = set(re.findall(r"request_id=(\w+)", caplog.text))
    assert len(sent) == 2 and sent == logged
