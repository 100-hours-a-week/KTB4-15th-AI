"""RembgBackgroundRemover 가 PIL 이미지를 그대로 넘기고 받는지 확인한다. 실제 모델은 쓰지 않는다."""

import sys
from types import ModuleType

import pytest
from PIL import Image

from app.body_image_validation.background import RembgBackgroundRemover


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

    def new_session(name, model_path):
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
