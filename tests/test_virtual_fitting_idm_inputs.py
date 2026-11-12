"""IDM-VTON 입력 검증(idm_image_validator)과 Modal 호출 결과 분류(idm_modal_client).

MediaPipe 는 가짜 Pose 탐지기로 바꾼다(실제 모델 측정은 docs/idm_vton_fallback/02_image_validation.md, Linux 컨테이너).
"""

import io
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from PIL import Image, ImageDraw

from app.virtual_fitting.idm_image_validator import (
    IdmGarmentImageClassifier,
    IdmImageClass,
    IdmImageUnavailable,
    check_idm_products,
    download_image,
    garment_description,
)
from app.virtual_fitting.idm_modal_client import (
    IdmCallKind,
    IdmModalUnavailable,
    IdmVtonModalClient,
    validate_output,
)
from tests.virtual_fitting_fixtures import make_bottom, make_top

TOP = make_top("1", sub_category="후디", image_url="https://img/top.jpg")
BOTTOM = make_bottom("2", image_url="https://img/bottom.jpg")


# --- 조합 ---


def test_only_a_single_mapped_top_is_supported():
    assert check_idm_products([TOP]).supported
    assert check_idm_products([TOP]).garment_name == "hoodie"
    assert check_idm_products([BOTTOM]).reason == "not_top"
    assert check_idm_products([TOP, BOTTOM]).reason == "not_single_product"
    unmapped = make_top("3", sub_category="없는 분류", image_url="https://img/x.jpg")
    assert check_idm_products([unmapped]).reason == "sub_category_unmapped"
    blank = make_top("4", image_url=" ")
    assert check_idm_products([blank]).reason == "image_missing"


@pytest.mark.parametrize(
    "url", ["http://img/top.jpg", "ftp://img/top.jpg", "https://", "//img/top.jpg", "img/top.jpg"]
)
def test_product_image_url_must_be_https_like_the_modal_service(url):
    # Modal 서비스(_require_https_url)가 거절할 URL 은 호출 기회를 쓰기 전에 미지원으로 끝낸다.
    product = make_top("5", sub_category="후디", image_url=url)
    assert check_idm_products([product]).reason == "image_url_not_https"


def test_garment_description_uses_known_colors_only():
    assert garment_description("hoodie", "블랙") == "black hoodie"
    assert garment_description("hoodie", "알수없는색") == "hoodie"
    assert garment_description("hoodie", None) == "hoodie"


# --- 이미지 분류 ---


def image_bytes(fmt="JPEG", size=(600, 800), background=(240, 240, 240), box=True, mode="RGB"):
    image = Image.new(mode, size, background if mode == "RGB" else (*background, 255))
    if box:
        draw = ImageDraw.Draw(image)
        w, h = size
        draw.rectangle(
            (w * 0.2, h * 0.2, w * 0.8, h * 0.75),
            fill=(30, 50, 80) if mode == "RGB" else (30, 50, 80, 255),
        )
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def transparent_png():
    image = Image.new("RGBA", (600, 800), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((100, 100, 500, 700), fill=(30, 50, 80, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def busy_background():
    image = Image.new("RGB", (600, 800))
    pixels = image.load()
    for x in range(600):
        for y in range(800):
            pixels[x, y] = ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakePose:
    def __init__(self, landmarks=None):
        self.landmarks, self.calls = landmarks, 0

    def detect(self, image):
        self.calls += 1
        return self.landmarks


def landmarks(visible_names, visibility=0.95):
    names = ("NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", "LEFT_ELBOW")
    return {
        name: SimpleNamespace(visibility=visibility if name in visible_names else 0.1)
        for name in names
    }


def classify(body, pose=None):
    return IdmGarmentImageClassifier(pose_detector=pose or FakePose()).classify_bytes(body)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_plain_background_garment_without_a_person_is_garment_only(fmt):
    result = classify(image_bytes(fmt))
    assert result.classification is IdmImageClass.GARMENT_ONLY and result.allowed


def test_pose_landmarks_make_it_model_worn():
    pose = FakePose(landmarks({"NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP"}))
    assert classify(image_bytes(), pose).classification is IdmImageClass.MODEL_WORN


def test_weak_pose_is_ambiguous():
    pose = FakePose(landmarks({"LEFT_SHOULDER"}))
    result = classify(image_bytes(), pose)
    assert (result.classification, result.reason) == (IdmImageClass.AMBIGUOUS, "weak_pose")


@pytest.mark.parametrize(
    "body, reason",
    [
        (transparent_png(), "transparent_background"),  # 배경 제거 결과를 의류 단독으로 보지 않는다
        (busy_background(), "busy_background"),
        (image_bytes(size=(600, 800), box=False, background=(10, 10, 10)), "no_foreground"),
    ],
)
def test_uncertain_or_empty_images_are_not_allowed(body, reason):
    result = classify(body)
    assert result.reason == reason and not result.allowed


def test_tiny_foreground_is_ambiguous():
    image = Image.new("RGB", (600, 800), (240, 240, 240))
    ImageDraw.Draw(image).rectangle((250, 350, 350, 450), fill=(30, 50, 80))  # 약 2%
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    assert classify(buffer.getvalue()).reason == "foreground_size"


@pytest.mark.parametrize(
    "body",
    [b"not an image", image_bytes(size=(200, 200)), image_bytes("GIF")],
    ids=["garbage", "too_small", "gif"],
)
def test_invalid_images(body):
    pose = FakePose()
    result = classify(body, pose)
    assert result.classification is IdmImageClass.INVALID and pose.calls == 0


class Opener:
    def __init__(self, result):
        self.result = result

    def __call__(self, request, timeout):
        if isinstance(self.result, BaseException):
            raise self.result
        body = self.result

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        return Response(body)


def test_download_classifies_http_errors():
    url = "https://img/top.jpg"
    with pytest.raises(ValueError):
        download_image(url, opener=Opener(HTTPError(url, 404, "x", {}, None)))
    with pytest.raises(IdmImageUnavailable):
        download_image(url, opener=Opener(HTTPError(url, 503, "x", {}, None)))
    with pytest.raises(IdmImageUnavailable):
        download_image(url, opener=Opener(URLError("timeout")))
    with pytest.raises(ValueError):
        download_image("ftp://img/top.jpg", opener=Opener(b"x"))
    with pytest.raises(ValueError):
        download_image("http://img/top.jpg", opener=Opener(b"x"))  # Modal 서비스와 같은 https 조건
    assert download_image(url, opener=Opener(b"abc")) == b"abc"


def test_classify_url_maps_permanent_download_failures_to_invalid():
    def downloader(url):
        raise ValueError("상품 이미지 HTTP 404")

    classifier = IdmGarmentImageClassifier(pose_detector=FakePose(), downloader=downloader)
    assert classifier.classify_url("https://img/x.jpg").classification is IdmImageClass.INVALID


# --- Modal 결과 분류 ---


def ok_output(**overrides):
    output = {
        "status": "succeeded",
        "fitting_job_id": 7,
        "attempt_id": "a",
        "content_type": "image/webp",
        "bytes": 10,
        "sha256": "0" * 64,
        "width": 768,
        "height": 1024,
    }
    output.update(overrides)
    return output


def test_validate_output():
    assert validate_output(ok_output(), 7, "a").kind is IdmCallKind.SUCCEEDED
    assert validate_output(ok_output(fitting_job_id=8), 7, "a").reason == "invalid_response_fields"
    assert validate_output(ok_output(attempt_id="b"), 7, "a").kind is IdmCallKind.FAILED
    assert validate_output(ok_output(sha256="xyz"), 7, "a").kind is IdmCallKind.FAILED
    failed = validate_output({"status": "failed", "error_kind": "preprocess_rejected"}, 7, "a")
    assert (failed.kind, failed.reason) == (IdmCallKind.FAILED, "preprocess_rejected")
    assert validate_output("x", 7, "a").kind is IdmCallKind.FAILED


class FakeFunction:
    def __init__(self, spawn_error=None, get_error=None, output=None):
        self.spawn_error, self.get_error, self.output = spawn_error, get_error, output
        self.spawned = []

    def spawn(self, request):
        self.spawned.append(request)
        if self.spawn_error:
            raise self.spawn_error
        return self

    def get(self, timeout=None):
        if self.get_error:
            raise self.get_error
        return self.output


def test_spawn_error_is_unknown_and_not_retried():
    function = FakeFunction(spawn_error=ConnectionResetError())
    client = IdmVtonModalClient("app", "fn", function=function)
    call, unknown = client.spawn({"x": 1})
    assert call is None and unknown.kind is IdmCallKind.UNKNOWN and len(function.spawned) == 1


@pytest.mark.parametrize(
    "error, expected",
    [
        ("TimeoutError", None),
        ("FunctionTimeoutError", IdmCallKind.FAILED),
        ("OutputExpiredError", IdmCallKind.UNKNOWN),
        ("RemoteError", IdmCallKind.FAILED),
        ("ConnectionError", IdmCallKind.UNKNOWN),
        ("InternalError", IdmCallKind.UNKNOWN),
    ],
)
def test_wait_classifies_modal_errors(error, expected):
    from modal import exception

    cls = getattr(exception, error)
    instance = cls.__new__(cls)
    Exception.__init__(instance, "x")
    client = IdmVtonModalClient("app", "fn", function=FakeFunction())
    result = client.wait(FakeFunction(get_error=instance), 1, 7, "a")
    assert (result.kind if result is not None else None) is expected


def test_wait_validates_the_output():
    client = IdmVtonModalClient("app", "fn", function=FakeFunction())
    result = client.wait(FakeFunction(output=ok_output()), 1, 7, "a")
    assert result.kind is IdmCallKind.SUCCEEDED


def test_resolve_failure_is_unavailable(monkeypatch):
    import modal

    def from_name(*args, **kwargs):
        raise RuntimeError("no token")

    monkeypatch.setattr(modal.Function, "from_name", from_name)
    with pytest.raises(IdmModalUnavailable):
        IdmVtonModalClient("app", "fn").resolve()
