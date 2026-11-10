"""로컬 검증 CLI: 모델 경계만 대체하고 실제 서비스·인코딩·파일 저장을 확인한다."""

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from app.body_image_validation import background, detectors
from tests.test_body_image_validation_service import FakePersonDetector, FakePoseDetector

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_body_image_validation.py"
if not SCRIPT.is_file():
    pytest.skip("scripts/ 가 없는 환경(Docker test stage)", allow_module_level=True)


@pytest.fixture
def local_check(monkeypatch):
    class Person(FakePersonDetector):
        def __init__(self, *args):
            super().__init__()

        def close(self):
            pass

    class Pose(FakePoseDetector):
        def __init__(self, *args):
            super().__init__()

        def close(self):
            pass

    class Remover:
        def __init__(self, *args):
            pass

        def remove(self, image):
            rgba = image.convert("RGBA")
            rgba.putalpha(128)
            return rgba

    monkeypatch.setattr(detectors, "MediaPipePersonDetector", Person)
    monkeypatch.setattr(detectors, "MediaPipePoseDetector", Pose)
    monkeypatch.setattr(background, "RembgBackgroundRemover", Remover)
    monkeypatch.delenv("REMBG_HOME", raising=False)
    spec = importlib.util.spec_from_file_location("local_body_image_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("output_name", [None, "nested/result.webp", "nested/result.WEBP"])
@pytest.mark.parametrize("input_format", ["PNG", "JPEG"])
def test_local_check_writes_actual_webp_to_default_or_explicit_path(
    local_check, monkeypatch, tmp_path, output_name, input_format
):
    import sys

    source = tmp_path / ("body.png" if input_format == "PNG" else "body.jpg")
    Image.new("RGB", (600, 800), (180, 170, 160)).save(source)
    original = source.read_bytes()
    args = [str(SCRIPT), str(source), "--model-dir", str(tmp_path)]
    destination = tmp_path / output_name if output_name else tmp_path / "body-validated.webp"
    if output_name:
        args.extend(["--output", str(destination)])
    monkeypatch.setattr(sys, "argv", args)

    assert local_check.main() == 0

    with Image.open(destination) as decoded:
        decoded.load()
        assert decoded.format == "WEBP"
        assert decoded.mode == "RGBA" and decoded.size == (600, 800)
        assert decoded.getchannel("A").getextrema() == (128, 128)
    assert source.read_bytes() == original
    assert not (tmp_path / "body-validated.png").exists()
    # 명시한 경로의 대소문자를 유지하며 자동 개명하지 않는다.
    if output_name:
        assert destination.name == Path(output_name).name


@pytest.mark.parametrize("extension", [".png", ".jpg", ""])
@pytest.mark.parametrize("existing_output", [False, True])
def test_invalid_output_is_rejected_before_models_io_or_directory_creation(
    local_check, monkeypatch, tmp_path, capsys, extension, existing_output
):
    import sys
    from unittest.mock import Mock

    from app.body_image_validation import service

    source = tmp_path / "body.png"
    Image.new("RGB", (600, 800), (180, 170, 160)).save(source)
    original = source.read_bytes()
    destination = tmp_path / "output" / f"result{extension}"
    sentinel = b"existing output must stay unchanged"
    if existing_output:
        destination.parent.mkdir()
        destination.write_bytes(sentinel)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(source), "--output", str(destination)])

    # Fail immediately if any processing dependency or image read is reached.
    forbidden = Mock(side_effect=AssertionError("invalid output must stop before processing"))
    monkeypatch.setattr(detectors, "MediaPipePersonDetector", forbidden)
    monkeypatch.setattr(detectors, "MediaPipePoseDetector", forbidden)
    monkeypatch.setattr(background, "RembgBackgroundRemover", forbidden)
    monkeypatch.setattr(service, "BodyImageValidationService", forbidden)
    monkeypatch.setattr(local_check, "LocalImageStorage", forbidden)
    with monkeypatch.context() as io_guard:
        io_guard.setattr(Path, "read_bytes", forbidden)
        io_guard.setattr(Path, "mkdir", forbidden)
        with pytest.raises(SystemExit) as error:
            local_check.main()

    assert error.value.code == 2
    assert "출력 파일 확장자는 .webp를 사용해주세요." in capsys.readouterr().err
    forbidden.assert_not_called()
    assert source.read_bytes() == original
    if existing_output:
        assert destination.read_bytes() == sentinel
    else:
        assert not destination.exists()
        assert not destination.parent.exists()
