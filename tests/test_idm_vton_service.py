"""deploy/modal/idm_vton/service.py 의 GPU 밖 동작(요청 검증·내려받기·입력 정규화·복원 합성·WebP·업로드 1회).

추론(inference_check.run_tryon)은 가짜로 바꾸고, 복원 합성은 실제 restore_composite 를 쓴다. Docker test stage 에는 deploy/
가 없어 건너뛴다(로컬·리뷰용). 실제 L40S 실행은 docs/idm_vton_fallback/01_modal_service.md 에 기록한다.
"""

import io
import sys
import time
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pytest
from PIL import Image

SERVICE_DIR = Path(__file__).resolve().parents[1] / "deploy" / "modal" / "idm_vton"
if not (SERVICE_DIR / "service.py").is_file():
    pytest.skip("deploy/modal/idm_vton 이 없다(Docker test stage)", allow_module_level=True)
pytest.importorskip("modal")
sys.path.insert(0, str(SERVICE_DIR))

import service

SIZE = (768, 1024)


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg(image, size=None):
    buffer = io.BytesIO()
    (image.resize(size) if size else image).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def webp(image):
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP")
    return buffer.getvalue()


PERSON = jpeg(Image.new("RGB", (600, 1000), (200, 180, 160)))
GARMENT = webp(Image.new("RGB", (800, 800), (240, 240, 240)))


def request(**overrides):
    value = {
        "fitting_job_id": 7,
        "attempt_id": "attempt-1",
        "person_image_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/users/7.jpg?X-Amz-Signature=a",
        "garment_image_url": "https://img.example/top.jpg",
        "garment_description": "black hoodie",
        "upload_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/r/idm-vton.webp?X-Amz-Signature=b",
        "upload_headers": {
            "Content-Type": "image/webp",
            "x-amz-meta-fitting-job-id": "7",
            "x-amz-meta-attempt-id": "attempt-1",
        },
        "not_after_epoch": time.time() + 600,
    }
    value.update(overrides)
    return value


class Response(io.BytesIO):
    status = 200
    headers = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def getcode(self):
        return self.status


class Opener:
    """GET 은 URL 로 응답을 고르고, PUT 은 기록만 한다(실제 네트워크 없음)."""

    def __init__(self, put_error=None, get=None):
        self.puts, self.gets = [], []
        self.put_error = put_error
        self.get = get or {"users/7.jpg": PERSON, "img.example": GARMENT}

    def __call__(self, request, timeout):
        if request.get_method() == "PUT":
            self.puts.append((request.full_url, dict(request.header_items()), request.data))
            if self.put_error:
                raise self.put_error
            return Response(b"")
        self.gets.append(request.full_url)
        for marker, body in self.get.items():
            if marker in request.full_url:
                if isinstance(body, BaseException):
                    raise body
                return Response(body)
        raise AssertionError(request.full_url)


def fake_tryon(calls):
    """run_tryon 이 돌려주는 형태: pad 여백이 있는 768×1024 사람 입력·결과·이진 마스크와 person_fit."""

    def tryon(description, person, garment, person_mode, garment_mode, steps):
        calls.append((description, person_mode, garment_mode, steps))
        assert Image.open(io.BytesIO(garment)).format in (
            "JPEG",
            "PNG",
        )  # WebP 는 PNG 로 바뀌어 온다
        person_img = Image.new("RGB", SIZE, (128, 128, 128))
        person_img.paste(Image.new("RGB", (614, 1024), (200, 180, 160)), (77, 0))
        result_img = Image.new("RGB", SIZE, (20, 30, 200))
        mask = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
        mask[300:600, 250:520] = 255
        return {
            "images": {
                "result.jpg": jpeg(result_img),
                "inpaint_mask.png": png(Image.fromarray(mask)),
                "person_input.png": png(person_img),
            },
            "person_fit": {"content_box": [77, 0, 691, 1024]},
            "settings": {"steps": steps, "seed": 42},
            "timings": {"inference": 1.0},
            "gpu_name": "fake",
        }

    return tryon


def test_success_uploads_one_webp_with_the_signed_headers():
    calls, opener = [], Opener()
    output = service.run_service(request(), tryon=fake_tryon(calls), opener=opener)
    assert output["status"] == "succeeded" and calls == [("black hoodie", "pad", "pad", 30)]
    ((url, headers, body),) = opener.puts  # 업로드 1회
    assert url.endswith("X-Amz-Signature=b")
    assert {k.lower(): v for k, v in headers.items()} == {
        "content-type": "image/webp",
        "x-amz-meta-fitting-job-id": "7",
        "x-amz-meta-attempt-id": "attempt-1",
    }
    decoded = Image.open(io.BytesIO(body))
    assert decoded.format == "WEBP" and decoded.size == (614, 1024)  # pad 여백을 잘라 원래 비율
    assert output["bytes"] == len(body) and output["width"] == 614
    import hashlib

    assert output["sha256"] == hashlib.sha256(body).hexdigest()
    # 마스크 밖은 원본(사람 입력) 픽셀이 복원된다
    assert decoded.convert("RGB").getpixel((5, 5)) == pytest.approx((200, 180, 160), abs=6)
    assert "X-Amz-Signature" not in repr(output)  # URL 을 돌려주지 않는다


@pytest.mark.parametrize(
    "overrides, kind",
    [
        ({"person_image_url": "http://plain/x.jpg"}, "invalid_request"),
        ({"upload_headers": {"Content-Type": "image/jpeg"}}, "invalid_request"),
        (
            {"upload_headers": {"Content-Type": "image/webp", "Authorization": "x"}},
            "invalid_request",
        ),
        ({"garment_description": " "}, "invalid_request"),
        ({"not_after_epoch": time.time() - 1}, "deadline_exceeded"),
    ],
)
def test_bad_requests_fail_without_inference_or_upload(overrides, kind):
    calls, opener = [], Opener()
    output = service.run_service(request(**overrides), tryon=fake_tryon(calls), opener=opener)
    assert output["error_kind"] == kind and calls == [] and opener.puts == []


@pytest.mark.parametrize(
    "get, kind",
    [
        (
            {"users/7.jpg": HTTPError("u", 403, "x", {}, None), "img.example": GARMENT},
            "person_download_failed",
        ),
        ({"users/7.jpg": PERSON, "img.example": TimeoutError()}, "garment_download_failed"),
        ({"users/7.jpg": b"garbage", "img.example": GARMENT}, "person_image_invalid"),
        (
            {"users/7.jpg": PERSON, "img.example": jpeg(Image.new("RGB", (100, 100)))},
            "garment_image_invalid",
        ),
    ],
)
def test_input_failures_do_not_run_inference(get, kind):
    calls, opener = [], Opener(get=get)
    output = service.run_service(request(), tryon=fake_tryon(calls), opener=opener)
    assert output["error_kind"] == kind and calls == [] and opener.puts == []


def test_preprocess_rejection_is_reported_without_upload():
    def rejecting(*args):
        raise service.InferenceCheckError("OpenPose 가 사람을 찾지 못했다")

    opener = Opener()
    output = service.run_service(request(), tryon=rejecting, opener=opener)
    assert output["error_kind"] == "preprocess_rejected" and opener.puts == []


def test_inference_crash_is_classified_and_not_retried():
    attempts = []

    def crashing(*args):
        attempts.append(1)
        raise RuntimeError("CUDA error")

    output = service.run_service(request(), tryon=crashing, opener=Opener())
    assert output["error_kind"] == "inference_failed" and attempts == [1]


def test_upload_failure_is_reported_once_with_the_s3_error_code():
    body = io.BytesIO(b"<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>")
    opener = Opener(put_error=HTTPError("u", 403, "x", {}, body))
    output = service.run_service(request(), tryon=fake_tryon([]), opener=opener)
    assert output["error_kind"] == "upload_failed" and len(opener.puts) == 1
    assert output["detail"] == "PUT HTTP 403 AccessDenied"


def test_transparent_inputs_are_flattened_to_png():
    rgba = Image.new("RGBA", (600, 800), (0, 0, 0, 0))
    data = service.normalize_input_image(png(rgba), "garment")
    flat = Image.open(io.BytesIO(data))
    assert flat.format == "PNG" and flat.mode == "RGB" and flat.getpixel((1, 1)) == (255, 255, 255)
    assert service.normalize_input_image(PERSON, "person") is PERSON  # JPEG 는 그대로


def test_function_settings_match_the_verified_experiment():
    import inference_check
    import model_loading_check

    assert service.APP_NAME == "lookddak-idm-vton-fallback" and service.FUNCTION_NAME == "generate"
    assert model_loading_check.GPU == "L40S" and model_loading_check.RETRIES == 0
    assert inference_check.INFERENCE_MEMORY_MIB == 32768 and service.SERVICE_TIMEOUT_SECONDS == 600
    assert inference_check.DENOISE_STEPS == 30 and inference_check.GUIDANCE_SCALE == 2.0
    assert inference_check.IMAGE_SIZE == (768, 1024)
    assert (service.PERSON_RESIZE_MODE, service.GARMENT_RESIZE_MODE) == ("pad", "pad")
    assert service.APP_NAME != "lookddak-body-image-worker"  # T4 전신사진 검증 App 과 분리
    assert set(service.ERROR_KINDS) == set(
        __import__("app.virtual_fitting.idm_modal_client", fromlist=["x"]).SERVICE_ERROR_KINDS
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://img.example/top.jpg",
        "http://img.example/top.jpg",
        "ftp://img.example/top.jpg",
        "https://",
        "//img.example/top.jpg",
        "",
    ],
)
def test_ai_server_and_modal_service_agree_on_image_urls(url):
    # AI Server 가 통과시킨 상품 URL 은 Modal 이 받아야 하고, Modal 이 거절할 URL 은 AI Server 가 먼저 거른다(A-4).
    from app.virtual_fitting.idm_image_validator import is_https_url

    try:
        service._require_https_url(url, "garment_image_url")
        accepted = True
    except service.ServiceError:
        accepted = False
    assert is_https_url(url) is accepted
