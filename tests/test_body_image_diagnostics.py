import ctypes
import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from app.body_image_validation import deadline, diagnostics, inference, runtime
from app.body_image_validation import service as service_module
from app.body_image_validation.background import RembgBackgroundRemover
from app.body_image_validation.concurrency import BodyValidationLimiter
from app.body_image_validation.detectors import (
    LANDMARK_NAMES,
    DetectionImage,
    MediaPipePersonDetector,
    MediaPipePoseDetector,
)
from app.body_image_validation.diagnostics import (
    ExecutionTracker,
    NvmlDevice,
    RequestRecord,
    ResourcePaths,
    ResourceSampler,
    TimedLock,
    begin_request,
    end_request,
    install_lock_timers,
    parse_request_id,
    read_cgroup_mib,
    read_process_memory,
)
from app.body_image_validation.exceptions import (
    BodyImageSystemError,
    ServerBusyError,
    UserImageValidationError,
)
from app.body_image_validation.gpu_preprocessing import GpuImagePreprocessor, GpuPreprocessingError
from app.body_image_validation.image_io import open_image
from app.body_image_validation.inference import BodyImageModels
from app.body_image_validation.service import BodyImageValidationService
from app.clients.s3 import ImageStorageError
from app.config import body_image_validation as config
from tests.test_body_image_gpu_preprocessing import fake_torch_decoder
from tests.test_body_image_validation_service import FakeStorage, valid_landmarks

SERVICE_LOGGER = "app.body_image_validation.service"
DIAGNOSTICS_LOGGER = "app.body_image_validation.diagnostics"
SAMPLER_THREAD = "body-image-resource-sampler"
LOCK_NAMES = ("person", "pose", "rembg", "nvjpeg")
LANDMARKS = valid_landmarks()


def jpeg(size=(600, 800)):
    output = BytesIO()
    Image.new("RGB", size, (180, 170, 160)).save(output, format="JPEG")
    return output.getvalue()


def png(size=(600, 800)):
    output = BytesIO()
    Image.new("RGB", size, (180, 170, 160)).save(output, format="PNG")
    return output.getvalue()


def wait_until(condition, timeout=5.0):
    limit = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < limit, "시간 안에 조건이 만족되지 않았다"
        time.sleep(0.001)


def diagnostic_lines(caplog):
    prefix = "body image validation diagnostics: "
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]
    return [json.loads(line[len(prefix):]) for line in lines]


class PersonTask:
    """MediaPipe ObjectDetector 자리. 실제 detect 의 결과 변환까지 탄다."""

    def __init__(self, error=None):
        self.error = error
        self.closed = 0

    def detect(self, image):
        if self.error:
            raise self.error
        box = SimpleNamespace(origin_x=50, origin_y=50, width=500, height=700)
        return SimpleNamespace(detections=[SimpleNamespace(categories=[SimpleNamespace(score=0.9)], bounding_box=box)])

    def close(self):
        self.closed += 1


class PoseTask(PersonTask):
    def detect(self, image):
        if self.error:
            raise self.error
        missing = SimpleNamespace(x=0.5, y=0.5, visibility=1.0, presence=1.0)
        return SimpleNamespace(pose_landmarks=[[LANDMARKS.get(name, missing) for name in LANDMARK_NAMES]])


def real_person(task=None):
    detector = MediaPipePersonDetector.__new__(MediaPipePersonDetector)
    detector._detector, detector._lock, detector._closed = task or PersonTask(), threading.Lock(), False
    return detector


def real_pose(task=None):
    detector = MediaPipePoseDetector.__new__(MediaPipePoseDetector)
    detector._detector, detector._lock, detector._closed = task or PoseTask(), threading.Lock(), False
    return detector


def real_remover():
    remover = RembgBackgroundRemover.__new__(RembgBackgroundRemover)
    remover._session, remover._lock, remover._profile_runs, remover._runs = object(), threading.Lock(), 0, 0
    return remover


@pytest.fixture
def real_models(monkeypatch):
    """모델 본체만 fake 다. Lock 을 쥐는 코드는 실제 detector·rembg adapter·nvJPEG decoder 의 것이다."""
    monkeypatch.setattr(DetectionImage, "mediapipe_image", lambda self: object())
    monkeypatch.setattr(service_module, "validate_brightness", lambda *args: None)
    monkeypatch.setitem(sys.modules, "rembg", SimpleNamespace(remove=lambda image, session: image.convert("RGBA")))
    models = BodyImageModels(
        real_person(), real_pose(), real_remover(), GpuImagePreprocessor(fake_torch_decoder())
    )
    install_lock_timers(models)
    return models


def timed_locks(models):
    return (
        models.person_detector._lock, models.pose_detector._lock,
        models.background_remover._lock, models.gpu_preprocessor._decoder._lock,
    )


def service_for(models, limiter=None, resources=None, storage=None, diagnostic=True):
    return BodyImageValidationService(
        models.person_detector, models.pose_detector, models.background_remover, storage or FakeStorage(),
        limiter=limiter, gpu_preprocessor=models.gpu_preprocessor, diagnostics=diagnostic, resources=resources,
    )


class Gpu:
    def __init__(self, used=900.0, total=15360.0, error=None):
        self.used, self.total, self.error, self.reads = used, total, error, 0

    def memory_mib(self):
        self.reads += 1
        if self.error:
            raise self.error
        return self.used, self.total


class Files:
    def __init__(self, root):
        self.status = root / "status"
        self.usage = root / "memory.current"
        self.peak = root / "memory.peak"
        self.paths = ResourcePaths(str(self.status), (str(self.usage), str(root / "missing")), (str(self.peak),))

    def write(self, rss_mib=2048, high_water_mib=4096, usage_mib=1800, peak_mib=3900):
        self.status.write_text(
            f"Name:\tpython\nVmPeak:\t 9999999 kB\nVmHWM:\t {high_water_mib * 1024} kB\n"
            f"VmRSS:\t {rss_mib * 1024} kB\nThreads:\t12\n"
        )
        self.usage.write_text(f"{usage_mib * 2**20}\n")
        self.peak.write_text(f"{peak_mib * 2**20}\n")


@pytest.fixture
def files(tmp_path):
    files = Files(tmp_path)
    files.write()
    return files


@pytest.fixture
def gpu():
    return Gpu()


@pytest.fixture
def sampler(files, gpu):
    sampler = ResourceSampler(0.001, files.paths, lambda: gpu)
    yield sampler
    sampler.close()


# --- Lock 대기·보유 ---


def test_second_request_waits_for_the_rembg_lock_and_each_request_gets_its_own_times(real_models, sampler, monkeypatch, caplog):
    held, release = threading.Event(), threading.Event()
    calls = []

    def remove(image, session):
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            held.set()
            assert release.wait(5)
        return image.convert("RGBA")

    monkeypatch.setitem(sys.modules, "rembg", SimpleNamespace(remove=remove))
    service = service_for(real_models, BodyValidationLimiter(2, 1), sampler)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        first = threading.Thread(target=service.validate, args=(png(),), name="first")
        second = threading.Thread(target=service.validate, args=(png(),), name="second")
        first.start()
        assert held.wait(5)
        second.start()
        wait_until(lambda: service.limiter.running == 2)
        time.sleep(0.1)  # second 가 rembg Lock 앞까지 가서 기다리는 시간
        release.set()
        first.join(5)
        second.join(5)

    assert calls == ["first", "second"]  # Lock 이 그대로 순서를 지켰다
    holder, waiter = sorted(diagnostic_lines(caplog), key=lambda line: line["locks"]["rembg"]["wait_ms"])
    assert holder["request_id"] != waiter["request_id"]
    assert waiter["locks"]["rembg"]["wait_ms"] > 50
    assert holder["locks"]["rembg"]["wait_ms"] < waiter["locks"]["rembg"]["wait_ms"]
    assert holder["locks"]["rembg"]["hold_ms"] > 50  # second 가 기다린 시간은 first 가 쥐고 있던 시간이다
    assert holder["locks"]["rembg"]["count"] == waiter["locks"]["rembg"]["count"] == 1
    assert {holder["execution"]["active_max"], waiter["execution"]["active_max"]} == {2}


def test_every_lock_of_the_real_pipeline_is_measured_under_the_stage_names(real_models, caplog):
    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        service_for(real_models).validate(jpeg())

    [line] = diagnostic_lines(caplog)
    assert set(line["locks"]) == set(LOCK_NAMES)
    assert all(lock["count"] == 1 and lock["wait_ms"] >= 0 and lock["hold_ms"] >= 0 for lock in line["locks"].values())
    assert {"person", "pose", "rembg"} <= set(line["ms"])  # 같은 이름의 단계 시간과 비교할 수 있다


def test_png_never_touches_the_nvjpeg_lock(real_models, caplog):
    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        service_for(real_models).validate(png())

    assert set(diagnostic_lines(caplog)[0]["locks"]) == {"person", "pose", "rembg"}


def test_timer_wraps_the_same_lock_and_keeps_mutual_exclusion():
    original = threading.Lock()
    timer = TimedLock("person", original)
    inside, overlaps = [0], []

    def work():
        record, token = begin_request(1)
        for _ in range(200):
            with timer:
                inside[0] += 1
                overlaps.append(inside[0])
                inside[0] -= 1
        end_request(token)
        assert record.locks["person"]["count"] == 200

    threads = [threading.Thread(target=work) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert timer._lock is original
    assert set(overlaps) == {1}
    assert original.locked() is False


def test_each_real_model_lock_is_wrapped_once_without_replacing_it():
    decoder = fake_torch_decoder()
    models = BodyImageModels(real_person(), real_pose(), real_remover(), GpuImagePreprocessor(decoder))
    originals = (
        models.person_detector._lock, models.pose_detector._lock, models.background_remover._lock, decoder._lock,
    )

    install_lock_timers(models)

    assert [lock.name for lock in timed_locks(models)] == list(LOCK_NAMES)
    assert all(timer._lock is original for timer, original in zip(timed_locks(models), originals, strict=True))


def test_pillow_only_models_have_no_decoder_to_wrap():
    models = BodyImageModels(real_person(), real_pose(), real_remover(), None)

    install_lock_timers(models)

    assert isinstance(models.person_detector._lock, TimedLock)


@pytest.mark.parametrize(
    ("build", "call"),
    [
        (lambda error: real_person(PersonTask(error)), lambda owner: owner.detect(SimpleNamespace(mediapipe_image=object))),
        (lambda error: real_pose(PoseTask(error)), lambda owner: owner.detect(SimpleNamespace(mediapipe_image=object))),
        (lambda error: fake_torch_decoder(error), lambda owner: owner.decode(b"jpeg", (600, 800), 1)),
    ],
    ids=["person", "pose", "nvjpeg"],
)
def test_failing_work_inside_the_lock_releases_it_and_is_still_measured(build, call):
    owner = build(RuntimeError("model failed"))
    original = owner._lock
    owner._lock = TimedLock("model", original)
    record, token = begin_request(1)

    with pytest.raises((RuntimeError, GpuPreprocessingError)):
        call(owner)
    end_request(token)

    assert original.locked() is False
    assert record.locks["model"]["count"] == 1
    with pytest.raises((RuntimeError, GpuPreprocessingError)):
        call(owner)  # 다음 요청이 Lock 에 걸리지 않는다


def test_rembg_failure_releases_the_lock_and_is_measured(monkeypatch):
    def remove(image, session):
        raise RuntimeError("cuda failed")

    monkeypatch.setitem(sys.modules, "rembg", SimpleNamespace(remove=remove))
    remover = real_remover()
    original = remover._lock
    remover._lock = TimedLock("rembg", original)
    record, token = begin_request(1)

    with pytest.raises(RuntimeError, match="cuda failed"):
        remover.remove(Image.new("RGB", (4, 4)))
    end_request(token)

    assert original.locked() is False
    assert record.locks["rembg"]["count"] == 1


def test_lock_use_outside_a_request_records_nothing_and_close_still_works():
    task = PersonTask()
    detector = real_person(task)
    original = detector._lock
    detector._lock = timer = TimedLock("person", original)
    record, token = begin_request(1)
    end_request(token)

    detector.detect(SimpleNamespace(mediapipe_image=object))  # warm-up 은 service 를 지나지 않는다
    detector.close()
    detector.close()

    assert record.locks == {}
    assert task.closed == 1
    assert timer._lock.locked() is False


def test_several_uses_in_one_request_add_up():
    timer = TimedLock("rembg", threading.Lock())
    record, token = begin_request(1)
    for _ in range(3):
        with timer:
            time.sleep(0.01)
    end_request(token)

    assert record.locks["rembg"]["count"] == 3
    assert record.locks["rembg"]["hold_ms"] >= 25  # 합계이고 마지막 값이 아니다


def test_wait_and_hold_of_each_use_are_summed_separately():
    record, token = begin_request(1)
    end_request(token)

    record.add_lock("rembg", 0.010, 0.100)
    record.add_lock("rembg", 0.030, 0.200)

    assert record.locks["rembg"]["count"] == 2
    assert record.locks["rembg"]["wait_ms"] == pytest.approx(40.0)
    assert record.locks["rembg"]["hold_ms"] == pytest.approx(300.0)


@pytest.mark.parametrize("body", [png(), b"not an image"], ids=["success", "failure"])
def test_validate_clears_the_record_of_its_thread(real_models, body):
    service = service_for(real_models)

    try:
        service.validate(body)
    except UserImageValidationError:
        pass

    assert diagnostics._current.get() is None  # threadpool thread 가 다음 요청에 재사용돼도 섞이지 않는다


def test_request_record_does_not_leak_into_the_next_request():
    timer = TimedLock("rembg", threading.Lock())
    first, token = begin_request(1)
    with timer:
        pass
    end_request(token)
    second, token = begin_request(2)
    end_request(token)

    assert first.request_id != second.request_id and second.seq == 2
    assert second.locks == {}


# --- service 연결: 결과·로그·실패 경로 ---


def test_diagnostics_do_not_change_the_result(real_models, sampler):
    instrumented, plain = FakeStorage(), FakeStorage()

    service_for(real_models, resources=sampler, storage=instrumented).validate(jpeg())
    models = BodyImageModels(real_person(), real_pose(), real_remover(), GpuImagePreprocessor(fake_torch_decoder()))
    service_for(models, storage=plain, diagnostic=False).validate(jpeg())

    assert instrumented.uploads[0][0] == plain.uploads[0][0]  # WebP bytes 가 같다
    assert instrumented.uploads[0][2] == plain.uploads[0][2]


def test_off_means_plain_locks_no_sampler_and_no_record(monkeypatch):
    calls = []
    monkeypatch.setattr(service_module, "begin_request", lambda seq: calls.append(seq))
    models = BodyImageModels(real_person(), real_pose(), real_remover(), None)
    monkeypatch.setattr(config, "DIAGNOSTICS", False)
    monkeypatch.setattr(runtime, "ResourceSampler", lambda *args: calls.append("sampler"))

    built = runtime.BodyImageRuntime(models, FakeStorage())

    assert built.resources is None and built.service._resources is None
    assert built.service._execution is None
    assert type(models.person_detector._lock) is type(threading.Lock())
    assert type(models.background_remover._lock) is type(threading.Lock())
    assert calls == []
    assert not any(thread.name == SAMPLER_THREAD for thread in threading.enumerate())


def test_off_validate_never_enters_the_diagnostic_path(real_models, monkeypatch, caplog):
    monkeypatch.setattr(service_module, "begin_request", lambda seq: pytest.fail("진단 경로를 탔다"))

    with caplog.at_level(logging.INFO):
        service_for(real_models, diagnostic=False).validate(png())

    assert caplog.text == ""


def test_timing_line_keeps_its_format_and_the_json_line_repeats_it_with_the_id(real_models, sampler, caplog):
    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        service = service_for(real_models, resources=sampler)
        service.validate(jpeg())
        service.validate(jpeg())

    records = [r for r in caplog.records if r.msg == "body image validation timings: outcome=%s ms=%s"]
    assert [len(record.args) for record in records] == [2, 2]  # 이 줄을 읽는 실험 코드가 args 두 개를 전제한다
    lines = diagnostic_lines(caplog)
    assert [line["ms"] for line in lines] == [record.args[1] for record in records]
    assert [line["seq"] for line in lines] == [1, 2]
    assert lines[0]["request_id"] != lines[1]["request_id"]
    assert all(len(line["request_id"]) == 16 and line["outcome"] == "success" for line in lines)


def test_diagnostics_without_a_sampler_still_log_the_locks(real_models, caplog):
    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        service_for(real_models).validate(png())

    [line] = diagnostic_lines(caplog)
    assert line["resources"] is None and line["locks"]


def fail_person(models, monkeypatch):
    models.person_detector._detector.error = RuntimeError("secret detector message")


def fail_pose(models, monkeypatch):
    models.pose_detector._detector.error = RuntimeError("secret detector message")


def fail_rembg(models, monkeypatch):
    def remove(image, session):
        raise RuntimeError("secret detector message")

    monkeypatch.setitem(sys.modules, "rembg", SimpleNamespace(remove=remove))


@pytest.mark.parametrize(
    ("fail", "code"),
    [
        (fail_person, "PERSON_DETECTION_FAILED"),
        (fail_pose, "POSE_ESTIMATION_FAILED"),
        (fail_rembg, "BACKGROUND_REMOVAL_FAILED"),
    ],
    ids=["person", "pose", "rembg"],
)
def test_system_failure_returns_every_lock_slot_and_window(real_models, sampler, monkeypatch, caplog, fail, code):
    fail(real_models, monkeypatch)
    service = service_for(real_models, resources=sampler)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER), pytest.raises(BodyImageSystemError):
        service.validate(jpeg())

    [line] = diagnostic_lines(caplog)
    assert line["outcome"] == code
    assert not any(timer._lock.locked() for timer in timed_locks(real_models))
    assert service.limiter.running == 0
    assert service._execution._running == []
    assert line["execution"]["end_ms"] is not None
    assert sampler._windows == []
    assert "secret detector message" not in caplog.text


def test_upload_failure_is_logged_with_the_locks_already_released(real_models, sampler, caplog):
    service = service_for(real_models, resources=sampler, storage=FakeStorage(ImageStorageError("s3 failed")))

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER), pytest.raises(BodyImageSystemError):
        service.validate(jpeg())

    [line] = diagnostic_lines(caplog)
    assert line["outcome"] == "IMAGE_UPLOAD_FAILED"
    assert not any(timer._lock.locked() for timer in timed_locks(real_models))
    assert (service.limiter.running, sampler._windows) == (0, [])
    assert service._execution._running == []
    assert len(line["execution"]["spans_ms"]["s3_upload"]) == 2  # 실패한 업로드의 구간도 남는다


def test_rejected_image_is_logged_and_closes_its_window(real_models, sampler, caplog):
    real_models.pose_detector._detector.detect = lambda image: SimpleNamespace(pose_landmarks=[])
    service = service_for(real_models, resources=sampler)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER), pytest.raises(UserImageValidationError):
        service.validate(jpeg())

    assert diagnostic_lines(caplog)[0]["outcome"] == "FULL_BODY_NOT_VISIBLE"
    assert "rembg" not in diagnostic_lines(caplog)[0]["locks"]
    assert sampler._windows == []


def test_undecodable_image_stops_before_any_lock(real_models, sampler, caplog):
    service = service_for(real_models, resources=sampler)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER), pytest.raises(UserImageValidationError):
        service.validate(b"not an image")

    assert diagnostic_lines(caplog)[0]["locks"] == {}
    assert sampler._windows == []


def test_second_request_is_not_blocked_by_a_failed_one(real_models, caplog):
    service = service_for(real_models)
    real_models.person_detector._detector.error = RuntimeError("failed")
    with pytest.raises(BodyImageSystemError):
        service.validate(jpeg())

    real_models.person_detector._detector.error = None
    assert service.validate(jpeg()).s3_key.startswith("body-images/")


def test_broken_measurements_never_fail_a_request(real_models, tmp_path, caplog):
    (tmp_path / "bad").write_bytes(b"\xff\xfe\x00garbage")
    bad = str(tmp_path / "bad")
    gpu = Gpu(error=RuntimeError("gpu lost"))
    sampler = ResourceSampler(0.001, ResourcePaths(bad, (bad,), (bad,)), lambda: gpu)
    service = service_for(real_models, resources=sampler)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        assert service.validate(jpeg()).s3_key.startswith("body-images/")
    sampler.close()

    resources = diagnostic_lines(caplog)[0]["resources"]
    assert resources["process_rss_mib"] == {"start": None, "peak": None, "end": None}
    assert resources["gpu_device_used_mib"]["peak"] is None


def test_log_holds_only_numbers_ids_and_codes(real_models, sampler, caplog):
    body = jpeg()
    service = service_for(real_models, resources=sampler)

    with caplog.at_level(logging.INFO):
        result = service.validate(body)

    [line] = diagnostic_lines(caplog)
    assert set(line) == {"request_id", "ai_request_id", "seq", "outcome", "ms", "locks", "execution", "resources"}
    assert result.s3_key not in caplog.text  # 결과 객체 키도 남기지 않는다
    assert not any(isinstance(value, str) for value in line["ms"].values())


def test_ai_request_id_is_recorded_as_given_and_is_null_without_one(real_models, sampler, caplog):
    service = service_for(real_models, resources=sampler)

    with caplog.at_level(logging.INFO):
        service.validate(jpeg(), "85c390e8344842ad")
        service.validate(jpeg())

    traced, plain = diagnostic_lines(caplog)
    assert traced["ai_request_id"] == "85c390e8344842ad"
    assert plain["ai_request_id"] is None
    assert traced["request_id"] != plain["request_id"]
    assert "85c390e8344842ad" not in {traced["request_id"], plain["request_id"]}


def test_ai_request_id_changes_nothing_when_diagnostics_are_off(real_models, caplog):
    service = service_for(real_models, diagnostic=False)

    with caplog.at_level(logging.INFO):
        result = service.validate(jpeg(), "85c390e8344842ad")

    assert result.s3_key.startswith("body-images/")
    assert "85c390e8344842ad" not in caplog.text


@pytest.mark.parametrize("value", ["85c390e8344842ad", "0" * 16])
def test_parse_request_id_keeps_the_ai_server_format(value):
    assert parse_request_id(value) == value


@pytest.mark.parametrize(
    "value", [None, "", "0" * 15, "0" * 17, "A" * 16, "g" * 16, "0" * 16 + "\n", " " + "0" * 15, "0" * 8 + "\r\n" + "0" * 6]
)
def test_parse_request_id_drops_everything_else(value):
    assert parse_request_id(value) is None


# --- 실제 실행 구간: 슬롯을 쥔 구간만 동시 실행으로 센다 ---


def by_seq(caplog):
    return sorted(diagnostic_lines(caplog), key=lambda line: line["seq"])


def interval(line):
    return line["execution"]["start_ms"], line["execution"]["end_ms"]


def span(line, stage):
    return tuple(line["execution"]["spans_ms"][stage])


def overlaps(first, second):
    return first[0] < second[1] and second[0] < first[1]


def run_together(service, count=2):
    errors = []

    def call():
        try:
            service.validate(png())
        except Exception as error:  # noqa: BLE001 - 실패를 assert 로 드러낸다
            errors.append(error)

    threads = [threading.Thread(target=call) for _ in range(count)]
    for thread in threads:
        thread.start()
    return threads, errors


def join_all(threads, errors):
    for thread in threads:
        thread.join(5)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []


class HeldStorage(FakeStorage):
    def __init__(self, wait):
        super().__init__()
        self.wait = wait

    def upload_bytes(self, body, key, content_type):
        self.wait()
        return super().upload_bytes(body, key, content_type)


def test_limiter_1_reports_one_running_even_while_two_requests_are_inside_the_service(
    real_models, sampler, monkeypatch, caplog
):
    held, release = threading.Event(), threading.Event()

    def remove(image, session):
        held.set()
        assert release.wait(5)
        return image.convert("RGBA")

    monkeypatch.setitem(sys.modules, "rembg", SimpleNamespace(remove=remove))
    service = service_for(real_models, BodyValidationLimiter(1, 1), sampler)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        first, errors = run_together(service, 1)
        assert held.wait(5)
        second, more_errors = run_together(service, 1)
        wait_until(lambda: service.limiter.waiting == 1)
        time.sleep(0.05)  # 두 번째 요청이 슬롯을 기다린 시간이 이만큼은 된다
        release.set()
        join_all(first + second, errors + more_errors)

    a, b = by_seq(caplog)
    assert (a["execution"]["active_max"], b["execution"]["active_max"]) == (1, 1)
    assert (a["execution"]["active_at_start"], b["execution"]["active_at_start"]) == (0, 0)
    assert a["resources"]["windows_max"] == b["resources"]["windows_max"] == 2  # 이 값으로 판단하면 오탐이다
    assert not overlaps(interval(a), interval(b))
    assert b["ms"]["slot_wait"] > 40 > a["ms"]["slot_wait"]


def test_limiter_2_reports_two_running_only_when_both_are_inside_at_once(real_models, sampler, caplog):
    barrier = threading.Barrier(2, timeout=5)  # 둘 다 업로드에 들어오지 않으면 BrokenBarrierError 로 실패한다

    def upload():
        barrier.wait()
        time.sleep(0.01)  # 구간이 로그 해상도(0.1ms)보다 길도록 머문다

    service = service_for(real_models, BodyValidationLimiter(2, 1), sampler, HeldStorage(upload))

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        join_all(*run_together(service))

    a, b = by_seq(caplog)
    assert a["request_id"] != b["request_id"]
    assert a["execution"]["process_id"] == b["execution"]["process_id"] == service._execution.process_id
    assert (a["execution"]["active_max"], b["execution"]["active_max"]) == (2, 2)
    assert sorted((a["execution"]["active_at_start"], b["execution"]["active_at_start"])) == [0, 1]
    assert overlaps(interval(a), interval(b))
    assert overlaps(span(a, "s3_upload"), span(b, "s3_upload"))


def test_upload_overlapping_another_requests_inference_can_be_read_from_the_spans(real_models, sampler, monkeypatch, caplog):
    uploading, inferring = threading.Event(), threading.Event()
    calls = []

    def remove(image, session):  # rembg Lock 안이다. 앞 요청은 Lock 을 놓은 뒤 업로드로 간다
        calls.append(1)
        if len(calls) == 2:
            assert uploading.wait(5)
            inferring.set()
            time.sleep(0.01)
        return image.convert("RGBA")

    def upload():
        if not uploading.is_set():
            uploading.set()
            assert inferring.wait(5)
            time.sleep(0.01)

    monkeypatch.setitem(sys.modules, "rembg", SimpleNamespace(remove=remove))
    service = service_for(real_models, BodyValidationLimiter(2, 1), sampler, HeldStorage(upload))

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        join_all(*run_together(service))

    lines = by_seq(caplog)
    assert any(overlaps(span(x, "s3_upload"), span(y, "rembg")) for x, y in (lines, lines[::-1]))


def test_slot_wait_covers_only_the_limiter(real_models, monkeypatch, caplog):
    class SlowLimiter(BodyValidationLimiter):
        @contextmanager
        def slot(self, max_wait=None):
            time.sleep(0.06)
            with super().slot(max_wait):
                yield

    def slow_open(body):
        time.sleep(0.5)
        return open_image(body)

    monkeypatch.setattr(service_module, "open_image", slow_open)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        service_for(real_models, SlowLimiter(1, 1)).validate(png())

    [line] = diagnostic_lines(caplog)
    assert 50 <= line["ms"]["slot_wait"] < 400  # 슬롯 전 입력 검사(0.5초)는 들어가지 않는다
    assert line["execution"]["slot_acquired_ms"] - line["execution"]["received_ms"] >= 550


def test_timeline_points_follow_the_request_order(real_models, sampler, caplog):
    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        service_for(real_models, resources=sampler).validate(jpeg())

    [line] = diagnostic_lines(caplog)
    execution = line["execution"]
    assert list(execution["spans_ms"]) == ["preprocess", "person", "pose", "brightness", "rembg", "webp", "s3_upload"]
    points = [execution["received_ms"], execution["slot_acquired_ms"], execution["start_ms"]]
    points += [point for stage in execution["spans_ms"].values() for point in stage]
    points += [execution["end_ms"]]
    assert points == sorted(points)
    assert span(line, "s3_upload")[1] - span(line, "s3_upload")[0] == pytest.approx(line["ms"]["s3_upload"], abs=0.2)


def test_rejected_request_never_counts_as_running(real_models, monkeypatch, caplog):
    held, release = threading.Event(), threading.Event()

    def remove(image, session):
        held.set()
        assert release.wait(5)
        return image.convert("RGBA")

    monkeypatch.setitem(sys.modules, "rembg", SimpleNamespace(remove=remove))
    service = service_for(real_models, BodyValidationLimiter(1, 0))

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER):
        first, errors = run_together(service, 1)
        assert held.wait(5)
        with pytest.raises(ServerBusyError):
            service.validate(png())
        release.set()
        join_all(first, errors)

    rejected = next(line for line in diagnostic_lines(caplog) if line["outcome"] == "SERVER_BUSY")
    assert rejected["execution"]["active_max"] == 0
    assert rejected["execution"]["start_ms"] is None and rejected["execution"]["end_ms"] is None
    assert "slot_wait" not in rejected["ms"]


def test_cancel_while_running_returns_the_counter_and_the_slot(real_models, sampler, caplog):
    class Cancelled(BaseException):
        pass

    def cancel():
        raise Cancelled

    service = service_for(real_models, BodyValidationLimiter(1, 1), sampler, HeldStorage(cancel))

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER), pytest.raises(Cancelled):
        service.validate(png())

    assert service._execution._running == []
    assert (service.limiter.running, sampler._windows) == (0, [])
    assert diagnostic_lines(caplog)[0]["execution"]["end_ms"] is not None


def test_deadline_stop_inside_the_slot_returns_the_counter(real_models, caplog):
    service = service_for(real_models)
    token = deadline.set_deadline(time.time() - 1)

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER), pytest.raises(deadline.RequestDeadlineExceeded):
        service.validate(png())
    deadline.reset(token)

    assert service._execution._running == []
    assert service.limiter.running == 0


def test_tracker_counts_every_request_inside_at_once_without_races():
    tracker = ExecutionTracker()
    records = [RequestRecord(number) for number in range(8)]
    barrier = threading.Barrier(len(records), timeout=5)

    def work(record):
        tracker.enter(record)
        barrier.wait()
        tracker.leave(record)

    threads = [threading.Thread(target=work, args=(record,)) for record in records]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert [record.active_max for record in records] == [8] * 8
    assert sorted(record.active_at_start for record in records) == list(range(8))
    assert tracker._running == []


def test_process_id_tells_apart_clocks_that_must_not_be_compared(real_models):
    assert ExecutionTracker().process_id != ExecutionTracker().process_id
    service = service_for(real_models)
    assert service_for(real_models)._execution.process_id != service._execution.process_id


# --- 메모리 값의 단위와 의미 ---


def test_proc_kb_are_read_as_kib(files):
    files.write(rss_mib=2048, high_water_mib=4096)

    assert read_process_memory(str(files.status)) == (2048.0, 4096.0)


def test_process_memory_without_high_water_keeps_the_rss(tmp_path):
    (tmp_path / "status").write_text("VmRSS:\t 1024 kB\n")  # gVisor 처럼 VmHWM 을 주지 않는 커널

    assert read_process_memory(str(tmp_path / "status")) == (1.0, None)


@pytest.mark.parametrize("content", ["VmRSS:\n", "VmRSS:\t abc kB\n"], ids=["no-value", "not-a-number"])
def test_malformed_proc_status_is_unknown(tmp_path, content):
    (tmp_path / "status").write_text(content)

    assert read_process_memory(str(tmp_path / "status")) == (None, None)


def test_cgroup_bytes_are_read_as_mib_and_v1_is_the_fallback(tmp_path):
    (tmp_path / "v1").write_text(f"{3 * 2**20}\n")

    assert read_cgroup_mib((str(tmp_path / "v2-missing"), str(tmp_path / "v1"))) == 3.0


def test_non_numeric_cgroup_value_is_unknown(tmp_path):
    (tmp_path / "memory.current").write_text("max\n")

    assert read_cgroup_mib((str(tmp_path / "memory.current"),)) is None


def fake_nvml(used, total, failing=None):
    """nvidia-ml-py 13.590.48 의 함수 이름·구조체 순서(total, free, used)를 그대로 흉내 낸 C 호출 경계."""
    device = ctypes.c_void_p(0x7fff_dead_beef)
    seen = []

    def init(flags):
        seen.append(("init", flags))
        return 2 if failing == "init" else 0

    def handle(index, out):
        seen.append(("handle", index))
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = device.value
        return 0

    def memory(handle_value, out):
        seen.append(("memory", handle_value))
        if failing == "memory":
            return 999
        values = ctypes.cast(out, ctypes.POINTER(ctypes.c_ulonglong))
        values[0], values[1], values[2] = total, total - used, used
        return 0

    return SimpleNamespace(
        seen=seen,
        expected_handle=device.value,
        nvmlInitWithFlags=ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint)(init),
        nvmlDeviceGetHandleByIndex_v2=ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint, ctypes.c_void_p)(handle),
        nvmlDeviceGetMemoryInfo=ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(memory),
    )


def test_nvml_struct_fields_and_units_cross_the_c_boundary():
    library = fake_nvml(used=1258 * 2**20, total=15360 * 2**20)

    used, total = NvmlDevice(library).memory_mib()

    assert (used, total) == (1258.0, 15360.0)  # 필드 순서가 어긋나면 free 나 total 이 나온다
    assert library.seen == [("init", 0), ("handle", 0), ("memory", library.expected_handle)]


@pytest.mark.parametrize("failing", ["init", "memory"])
def test_nvml_status_codes_become_errors(failing):
    library = fake_nvml(1, 2, failing)

    with pytest.raises(diagnostics.NvmlError, match="returned"):
        NvmlDevice(library).memory_mib()


def test_window_separates_start_peak_and_end(files, gpu, sampler):
    window = sampler.open()

    files.write(rss_mib=4300, usage_mib=3900)
    gpu.used = 1250.0
    wait_until(lambda: window.peak == [4300.0, 3900.0, 1250.0])
    files.write(rss_mib=4200, usage_mib=3850)
    gpu.used = 1200.0
    summary = sampler.finish(window)

    assert summary["process_rss_mib"] == {"start": 2048.0, "peak": 4300.0, "end": 4200.0}
    assert summary["container_mib"] == {"start": 1800.0, "peak": 3900.0, "end": 3850.0}
    assert summary["gpu_device_used_mib"] == {"start": 900.0, "peak": 1250.0, "end": 1200.0}


def test_peak_on_the_last_read_is_kept(files, sampler):
    window = sampler.open()

    files.write(rss_mib=5000)
    summary = sampler.finish(window)

    assert summary["process_rss_mib"]["peak"] == summary["process_rss_mib"]["end"] == 5000.0


def test_high_water_comes_from_the_kernel_counters(files, sampler):
    files.write(high_water_mib=6000, peak_mib=5900)

    summary = sampler.finish(sampler.open())

    assert summary["high_water_mib"] == {"process_rss": 6000.0, "container": 5900.0}


def test_values_keep_process_container_and_device_scope_in_their_names(sampler):
    summary = sampler.finish(sampler.open())

    assert {"process_rss_mib", "container_mib", "gpu_device_used_mib"} <= set(summary)
    assert not any("request" in key for key in summary)  # 요청별 점유량처럼 보이는 이름이 없다


def test_overlap_is_visible_on_both_windows(sampler):
    first = sampler.open()
    second = sampler.open()

    second_summary = sampler.finish(second)
    first_summary = sampler.finish(first)

    assert (first_summary["windows_at_start"], first_summary["windows_max"]) == (0, 2)
    assert (second_summary["windows_at_start"], second_summary["windows_max"]) == (1, 2)


def test_single_request_is_marked_as_not_overlapping(sampler):
    summary = sampler.finish(sampler.open())

    assert (summary["windows_at_start"], summary["windows_max"]) == (0, 1)


def test_sampling_reports_its_own_cost_and_gaps(sampler):
    window = sampler.open()
    wait_until(lambda: window.samples >= 5)

    sampling = sampler.finish(window)["sampling"]

    assert sampling["samples"] >= 6
    assert sampling["cost_ms_max"] > 0
    assert sampling["cost_ms_total"] >= sampling["cost_ms_max"]
    assert sampling["gap_ms_max"] > 0


def test_nothing_is_read_while_idle(gpu, sampler):
    sampler.finish(sampler.open())
    reads = gpu.reads

    time.sleep(0.05)  # 1ms 간격이라 유휴 때 읽었다면 수십 번이다

    assert gpu.reads <= reads + 1  # 요청이 끝날 때 진행 중이던 읽기 하나만 더 끝날 수 있다


def test_sampling_stops_between_requests_and_resumes(gpu, sampler):
    sampler.finish(sampler.open())

    window = sampler.open()
    wait_until(lambda: window.samples >= 3)  # 같은 thread 가 다시 깨어난다

    sampler.finish(window)
    assert sum(thread.name == SAMPLER_THREAD for thread in threading.enumerate()) == 1


def test_baseline_is_logged_once_before_the_first_request(files, gpu, sampler, caplog):
    with caplog.at_level(logging.INFO, logger=DIAGNOSTICS_LOGGER):
        first = sampler.open()
        sampler.finish(first)
        sampler.finish(sampler.open())

    baselines = [r.getMessage() for r in caplog.records if "resource baseline" in r.getMessage()]
    assert len(baselines) == 1
    baseline = json.loads(baselines[0].split(": ", 1)[1])
    assert baseline["phase"] == "idle_before_first_request"
    assert baseline["process_rss_mib"] == 2048.0
    assert baseline["container_mib"] == 1800.0
    assert baseline["gpu_device_used_mib"] == 900.0
    assert baseline["gpu_device_total_mib"] == 15360.0
    assert "container_limit_mib" not in baseline
    assert baseline["gpu_unavailable"] is None


def test_without_nvml_the_rest_is_still_measured_and_the_reason_is_logged_once(files, caplog):
    opens = []

    def no_library():
        opens.append(1)
        raise OSError("libnvidia-ml.so.1: cannot open shared object file")

    sampler = ResourceSampler(0.001, files.paths, no_library)

    with caplog.at_level(logging.INFO, logger=DIAGNOSTICS_LOGGER):
        summary = sampler.finish(sampler.open())
        sampler.finish(sampler.open())
    sampler.close()

    assert summary["process_rss_mib"]["start"] == 2048.0
    assert summary["gpu_device_used_mib"] == {"start": None, "peak": None, "end": None}
    assert len(opens) == 1  # 매 샘플 다시 열지 않는다
    assert '"gpu_unavailable": "OSError: libnvidia-ml.so.1' in caplog.text
    assert caplog.text.count("GPU memory is not measured") == 1


def test_gpu_read_failure_keeps_the_last_peak_and_warns_once(gpu, sampler, caplog):
    window = sampler.open()
    gpu.used = 1250.0
    wait_until(lambda: window.peak[2] == 1250.0)

    with caplog.at_level(logging.WARNING, logger=DIAGNOSTICS_LOGGER):
        gpu.error = RuntimeError("gpu lost")
        reads = gpu.reads
        wait_until(lambda: gpu.reads > reads + 3)
        summary = sampler.finish(window)

    assert summary["gpu_device_used_mib"] == {"start": 900.0, "peak": 1250.0, "end": None}
    assert caplog.text.count("GPU memory read failed") == 1


def test_real_device_library_missing_here_is_handled(files):
    sampler = ResourceSampler(0.001, files.paths)  # 이 머신에 없으면 OSError, 있으면 실제 값이다

    summary = sampler.finish(sampler.open())
    sampler.close()

    assert summary["process_rss_mib"]["start"] == 2048.0


# --- 수명: 시작과 종료 ---


def test_creating_a_sampler_touches_nothing(files):
    opened = []

    sampler = ResourceSampler(0.001, files.paths, lambda: opened.append(1))

    assert opened == []
    assert sampler._thread is None
    assert not any(thread.name == SAMPLER_THREAD for thread in threading.enumerate())


def test_close_stops_the_thread_and_can_be_repeated(sampler):
    sampler.finish(sampler.open())
    thread = sampler._thread

    sampler.close()
    sampler.close()

    assert not thread.is_alive()


def test_close_while_sampling_stops_the_thread_and_the_request_still_finishes(files, sampler):
    window = sampler.open()
    thread = sampler._thread

    sampler.close()
    files.write(rss_mib=3000)
    summary = sampler.finish(window)

    assert not thread.is_alive()
    assert summary["process_rss_mib"]["end"] == 3000.0


def test_open_after_close_starts_no_thread_and_still_gives_start_and_end(sampler):
    sampler.close()

    summary = sampler.finish(sampler.open())

    assert sampler._thread is None
    assert summary["sampling"]["samples"] == 2


def test_close_without_any_request_is_fine(files):
    ResourceSampler(0.001, files.paths, Gpu).close()


def test_runtime_installs_timers_and_owns_the_sampler_only_when_on(monkeypatch):
    monkeypatch.setattr(config, "DIAGNOSTICS", True)
    models = BodyImageModels(real_person(), real_pose(), real_remover(), None)

    built = runtime.BodyImageRuntime(models, FakeStorage())

    assert isinstance(models.person_detector._lock, TimedLock)
    assert built.service._resources is built.resources
    assert built.resources._interval == config.DIAGNOSTICS_SAMPLE_INTERVAL_SECONDS


def test_runtime_close_stops_the_sampler_and_closes_the_models(monkeypatch, files):
    monkeypatch.setattr(config, "DIAGNOSTICS", True)
    person_task = PersonTask()
    models = BodyImageModels(real_person(person_task), real_pose(), real_remover(), None)
    built = runtime.BodyImageRuntime(models, FakeStorage())
    built.resources._paths = files.paths
    built.resources._open_gpu = Gpu
    built.resources.finish(built.resources.open())
    thread = built.resources._thread

    built.close()

    assert not thread.is_alive()
    assert person_task.closed == 1


def test_inference_module_stays_free_of_measurement_state():
    assert not hasattr(inference, "ResourceSampler")  # snapshot 에 담기는 쪽은 계측을 모른다
