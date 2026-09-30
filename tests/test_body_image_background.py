"""RembgBackgroundRemover 의 session 설정과 PIL 입출력을 확인한다.

실제 모델은 session 설정 확인 한 곳에서만 쓰고, 모델 파일이 없으면 그 테스트는 건너뛴다.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType

import onnxruntime as ort
import pytest
from PIL import Image

from app.body_image_validation.background import RembgBackgroundRemover
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
