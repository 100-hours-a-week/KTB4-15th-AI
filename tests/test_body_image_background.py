"""RembgBackgroundRemover 의 session 설정과 PIL 입출력, encode_png 의 PNG 저장을 확인한다.

실제 모델은 session 설정 확인 한 곳에서만 쓰고, 모델 파일이 없으면 그 테스트는 건너뛴다.
"""

import importlib
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from app.body_image_validation.background import RembgBackgroundRemover, encode_png
from app.config import body_image_validation as config

ROOT = Path(__file__).resolve().parents[1]


def is_closed(image):
    try:
        image.getpixel((0, 0))
    except ValueError:
        return True
    return False


@pytest.fixture
def fake_rembg(monkeypatch):
    """rembg 대신 쓰는 모듈. remove() 에 들어온 인자를 기록하고 정해 둔 결과를 돌려준다."""
    module = ModuleType("rembg")
    module.calls = []
    module.output_mode = "RGBA"
    module.sess_opts = None

    def new_session(name, model_path, sess_opts):
        module.sess_opts = sess_opts
        return "session"

    def remove(data, session):
        module.calls.append((data, session))
        module.output = Image.new(module.output_mode, data.size)
        return module.output

    module.new_session = new_session
    module.remove = remove
    monkeypatch.setitem(sys.modules, "rembg", module)
    return module


def test_pil_image_is_passed_to_rembg_without_png_encoding(fake_rembg, monkeypatch):
    image = Image.new("RGB", (40, 60), "white")
    saves = []
    original_save = Image.Image.save
    monkeypatch.setattr(
        Image.Image, "save", lambda self, *a, **k: saves.append(a) or original_save(self, *a, **k)
    )

    result = RembgBackgroundRemover("model.onnx").remove(image)

    [(data, session)] = fake_rembg.calls
    assert data is image  # bytes 가 아니라 받은 PIL 이미지를 그대로 넘긴다
    assert session == "session"
    assert saves == []  # 입력을 PNG 로 인코딩하지 않는다
    assert (result.mode, result.size) == ("RGBA", (40, 60))


@pytest.mark.parametrize("enabled", [True, False])
def test_only_cpu_mem_arena_differs_from_default_session_options(fake_rembg, enabled):
    default = ort.SessionOptions()

    RembgBackgroundRemover("model.onnx", enable_cpu_mem_arena=enabled)

    options = fake_rembg.sess_opts
    assert options.enable_cpu_mem_arena is enabled
    assert options.enable_mem_pattern == default.enable_mem_pattern
    assert options.intra_op_num_threads == default.intra_op_num_threads
    assert options.inter_op_num_threads == default.inter_op_num_threads


def test_cpu_mem_arena_is_off_by_default(fake_rembg):
    RembgBackgroundRemover("model.onnx")

    assert fake_rembg.sess_opts.enable_cpu_mem_arena is False


@pytest.fixture
def reload_config(monkeypatch):
    """환경변수를 바꿔 config 모듈을 다시 읽는다. 끝나면 원래 환경으로 다시 읽는다."""

    def load(value):
        if value is None:
            monkeypatch.delenv("REMBG_ENABLE_CPU_MEM_ARENA", raising=False)
        else:
            monkeypatch.setenv("REMBG_ENABLE_CPU_MEM_ARENA", value)
        return importlib.reload(config)

    yield load
    monkeypatch.undo()
    importlib.reload(config)


@pytest.mark.parametrize(
    ("value", "expected"), [(None, False), ("false", False), ("true", True), ("TRUE", True)]
)
def test_arena_config_defaults_to_false_and_true_rolls_back(reload_config, value, expected):
    assert reload_config(value).REMBG_ENABLE_CPU_MEM_ARENA is expected


def _rembg_model():
    candidates = (config.REMBG_MODEL, ROOT / "models/body_image_validation/u2netp.onnx")
    return next((Path(path) for path in candidates if Path(path).is_file()), None)


@pytest.mark.skipif(_rembg_model() is None, reason="u2netp.onnx 모델 파일이 없다")
@pytest.mark.parametrize("enabled", [False, True])
def test_real_session_differs_from_rembg_default_only_in_cpu_mem_arena(monkeypatch, enabled):
    from rembg import new_session

    model = _rembg_model()
    monkeypatch.setenv("REMBG_HOME", str(model.parent))  # rembg 는 이 폴더 밖 모델을 거절한다
    default = new_session("u2net_custom", model_path=str(model)).inner_session

    session = RembgBackgroundRemover(
        str(model), enable_cpu_mem_arena=enabled
    )._session.inner_session

    options, base = session.get_session_options(), default.get_session_options()
    assert base.enable_cpu_mem_arena is True  # ONNX Runtime 기본값
    assert options.enable_cpu_mem_arena is enabled
    assert options.enable_mem_pattern == base.enable_mem_pattern
    assert options.intra_op_num_threads == base.intra_op_num_threads
    assert options.inter_op_num_threads == base.inter_op_num_threads
    assert session.get_providers() == default.get_providers()


def test_rgba_result_is_returned_as_is_and_left_open_for_the_caller(fake_rembg):
    image = Image.new("RGB", (40, 60), "white")

    result = RembgBackgroundRemover("model.onnx").remove(image)

    assert result is fake_rembg.output
    assert not is_closed(result)  # 호출자가 인코딩 뒤에 닫는다
    assert not is_closed(image)  # 입력은 호출자 소유라 닫지 않는다


def test_non_rgba_result_is_converted_and_the_intermediate_is_closed(fake_rembg):
    fake_rembg.output_mode = "RGB"
    image = Image.new("RGB", (40, 60), "white")

    result = RembgBackgroundRemover("model.onnx").remove(image)

    assert result.mode == "RGBA"
    assert result is not fake_rembg.output
    assert is_closed(fake_rembg.output)
    assert not is_closed(result)
    assert not is_closed(image)


# --- encode_png: 압축 수준 4 로 저장하고 픽셀은 바꾸지 않는다 ---


def rgba_with_varied_alpha(size=(64, 48)):
    """색은 무작위, alpha 는 완전 투명·반투명·완전 불투명이 섞인 RGBA 이미지."""
    width, height = size
    rgb = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    alpha = np.tile(np.linspace(0, 255, width).astype(np.uint8), (height, 1))
    alpha[: height // 3] = 0
    alpha[-height // 3 :] = 255
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def test_encode_png_saves_as_png_with_compress_level_4(monkeypatch):
    calls = []
    original_save = Image.Image.save

    def spy(self, fp, *args, **kwargs):
        calls.append(kwargs)
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", spy)

    encode_png(rgba_with_varied_alpha())

    [kwargs] = calls
    assert kwargs["format"] == "PNG"
    assert kwargs.get("compress_level") == 4  # Pillow 기본값 6 으로 되돌아가면 실패한다
    assert kwargs.get("optimize", False) is False


def test_encode_png_output_is_what_pillow_writes_at_level_4_not_the_default():
    image = rgba_with_varied_alpha()

    def pillow_png(**options):
        output = BytesIO()
        image.save(output, format="PNG", **options)
        return output.getvalue()

    level_4, default = pillow_png(compress_level=4), pillow_png()
    assert level_4 != default  # 이 이미지에서 두 수준의 결과가 달라야 비교가 의미 있다

    assert encode_png(image) == level_4


def test_encode_png_round_trip_keeps_every_rgba_pixel():
    image = rgba_with_varied_alpha()
    original = np.asarray(image)
    assert {0, 255} <= set(np.unique(original[:, :, 3]))
    assert ((original[:, :, 3] > 0) & (original[:, :, 3] < 255)).any()  # 반투명 픽셀

    with Image.open(BytesIO(encode_png(image))) as decoded:
        assert decoded.format == "PNG"
        assert decoded.mode == "RGBA"
        assert decoded.size == image.size
        restored = np.asarray(decoded)

    assert np.array_equal(restored, original)
    assert np.array_equal(restored[:, :, 3], original[:, :, 3])


def test_providers_are_left_to_rembg_unless_configured(fake_rembg):
    # fake new_session 은 providers 인자를 받지 않는다. 넘기면 TypeError 로 실패한다.
    RembgBackgroundRemover("model.onnx")


def test_configured_providers_are_passed_in_order(fake_rembg):
    received = {}

    def new_session(name, model_path, sess_opts, **kwargs):
        received.update(kwargs, sess_opts=sess_opts)
        return "session"

    fake_rembg.new_session = new_session

    RembgBackgroundRemover("model.onnx", providers=("CUDAExecutionProvider", "CPUExecutionProvider"))

    assert received["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert received["sess_opts"].enable_cpu_mem_arena is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ()),
        ("", ()),
        ("CUDAExecutionProvider,CPUExecutionProvider", ("CUDAExecutionProvider", "CPUExecutionProvider")),
        (" CUDAExecutionProvider , CPUExecutionProvider ", ("CUDAExecutionProvider", "CPUExecutionProvider")),
    ],
)
def test_rembg_providers_config(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("BODY_IMAGE_REMBG_PROVIDERS", raising=False)
    else:
        monkeypatch.setenv("BODY_IMAGE_REMBG_PROVIDERS", value)
    try:
        assert importlib.reload(config).REMBG_PROVIDERS == expected
    finally:
        monkeypatch.undo()
        importlib.reload(config)


# --- 진단: rembg ONNX Runtime profiling ---


def write_profile(tmp_path, events):
    path = tmp_path / "profile.json"
    path.write_text(__import__("json").dumps(events))
    return str(path)


def node(provider, dur):
    return {"cat": "Node", "name": "Conv_kernel_time", "dur": dur, "args": {"provider": provider}}


def model_run(dur):
    return {"cat": "Session", "name": "model_run", "dur": dur}


def test_profile_summary_splits_events_by_run_in_recorded_order(tmp_path):
    from app.body_image_validation.background import summarize_ort_profile

    events = [
        {"cat": "Session", "name": "session_initialization", "dur": 5000},
        node("CUDAExecutionProvider", 100), node("CUDAExecutionProvider", 200),
        {"cat": "Node", "name": "Conv_fence_before", "dur": 1, "args": {}},  # provider 없는 이벤트는 세지 않는다
        model_run(1500),
        node("CUDAExecutionProvider", 50), node("CPUExecutionProvider", 30),
        model_run(900),
    ]

    runs = summarize_ort_profile(write_profile(tmp_path, events))

    assert runs == [
        {"run": 1, "model_run_ms": 1.5, "nodes": {"CUDAExecutionProvider": 2},
         "node_ms": {"CUDAExecutionProvider": 0.3}},
        {"run": 2, "model_run_ms": 0.9, "nodes": {"CUDAExecutionProvider": 1, "CPUExecutionProvider": 1},
         "node_ms": {"CUDAExecutionProvider": 0.05, "CPUExecutionProvider": 0.03}},
    ]


class _ProfilingSession:
    def __init__(self, path):
        self.path = path
        self.ended = 0
        self.inner_session = self

    def end_profiling(self):
        self.ended += 1
        return self.path


def test_profiling_ends_after_the_configured_runs_and_logs_the_summary(fake_rembg, tmp_path, caplog):
    import logging

    path = write_profile(tmp_path, [node("CUDAExecutionProvider", 10), model_run(20)])
    session = _ProfilingSession(path)
    fake_rembg.new_session = lambda name, model_path, sess_opts: session
    remover = RembgBackgroundRemover("model.onnx", profile_runs=2)
    image = Image.new("RGB", (8, 8))

    with caplog.at_level(logging.INFO, logger="app.body_image_validation.background"):
        remover.remove(image)
        assert session.ended == 0
        remover.remove(image)
        remover.remove(image)  # 끝난 뒤에는 다시 끝내지 않는다

    assert session.ended == 1
    assert "rembg onnxruntime profile" in caplog.text and "CUDAExecutionProvider" in caplog.text
    assert not Path(path).exists()  # profile 파일은 정리한다


def test_profiling_is_off_by_default(fake_rembg):
    RembgBackgroundRemover("model.onnx")

    assert fake_rembg.sess_opts.enable_profiling is False


def test_profiling_failure_does_not_fail_the_request(fake_rembg, caplog):
    import logging

    class Broken:
        inner_session = None

        def __init__(self):
            self.inner_session = self

        def end_profiling(self):
            raise RuntimeError("profiler broke")

    fake_rembg.new_session = lambda name, model_path, sess_opts: Broken()
    remover = RembgBackgroundRemover("model.onnx", profile_runs=1)

    with caplog.at_level(logging.WARNING, logger="app.body_image_validation.background"):
        result = remover.remove(Image.new("RGB", (8, 8)))

    assert result.mode == "RGBA"
    assert "profiling summary failed" in caplog.text


@pytest.mark.skipif(_rembg_model() is None, reason="u2netp.onnx 모델 파일이 없다")
def test_real_session_profile_counts_nodes_per_run(monkeypatch, caplog):
    import json
    import logging
    import re

    model = _rembg_model()
    monkeypatch.setenv("REMBG_HOME", str(model.parent))
    remover = RembgBackgroundRemover(str(model), profile_runs=2)
    image = Image.new("RGB", (64, 96), "white")

    with caplog.at_level(logging.INFO, logger="app.body_image_validation.background"):
        remover.remove(image)
        remover.remove(image)

    [line] = [r.getMessage() for r in caplog.records if "rembg onnxruntime profile" in r.getMessage()]
    runs = json.loads(re.search(r"runs=(\[.*\]) parse_ms", line).group(1))
    assert [run["run"] for run in runs] == [1, 2]
    for run in runs:  # 로컬은 CPU provider 만 있다(Modal T4 에서는 CUDA 로 나와야 한다)
        assert sum(run["nodes"].values()) > 0 and run["model_run_ms"] > 0
    remover.remove(image)  # profiling 이 끝난 뒤에도 같은 session 으로 계속 동작한다
