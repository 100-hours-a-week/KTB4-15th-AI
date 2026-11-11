"""LOOKDDAK 가상피팅 V2 fallback 용 IDM-VTON 서비스 함수(Modal L40S).

AI Server 후처리 Worker(app/virtual_fitting/idm_fallback_handler.py)가 작업당 최대 1회 spawn 한다. 이 함수는
사용자 이미지 URL(S3 Presigned GET)·상품 이미지 URL(상품 DB 의 외부 URL)·상품 설명·결과 업로드 URL(S3 Presigned PUT)을
받아 내려받기 → 검증 → 추론 → 원본 영역 복원 → 여백 제거 → WebP 인코딩 → PUT 한 번을 하고 결과 요약만 돌려준다.

- 추론은 inference_check.run_tryon 을 그대로 부른다(.local, 같은 컨테이너). 모델 로딩·OpenPose·Human Parsing·DensePose·
  마스크·추론 설정(768×1024, 30 steps, guidance 2.0, seed 42)이 실험과 같다. 사람·상품 모두 pad.
- 원본 영역 복원은 restore_composite.composite_restored(dilate 3, blur 3)를 그대로 쓴다. 손상된 의류를 고치지 않는다.
- 재시도는 없다(retries=0, single_use_containers). 실패해도 다시 추론하지 않고 실패 요약을 돌려준다.
- 업로드는 마지막 단계 하나뿐이다. 실패를 돌려줬다면 업로드 PUT 은 성공하지 않았다(PUT 응답을 받지 못한 경우 제외).
- URL·서명·이미지 바이트는 로그·반환값에 남기지 않는다. 오류는 정해진 분류(error_kind)와 짧은 사유만 돌려준다.

배포(팀 workspace, 기존 Volume lookddak-idm-vton-models 를 읽기 전용으로 붙인다):

    cd deploy/modal/idm_vton
    MODAL_PROFILE=lookddak ../../../.venv/bin/modal deploy service.py

독립 테스트(배포된 함수를 이름으로 찾아 1회 부른다. URL 은 실행 시 인자로만 받는다):

    ../../../.venv/bin/python service_smoke.py --help
"""

import time

import modal
from inference_check import (
    DENOISE_STEPS,
    GUIDANCE_SCALE,
    IMAGE_SIZE,
    INFERENCE_CPU,
    INFERENCE_MEMORY_MIB,
    INPAINT_MASK_FILE,
    INPUT_FORMATS,
    MAX_INPUT_BYTES,
    MIN_INPUT_SIDE,
    PERSON_INPUT_FILE,
    RESULT_FILE,
    InferenceCheckError,
    run_tryon,
)
from inference_check import image as inference_image
from model_loading_check import GPU, RETRIES
from model_loading_check import volume as read_only_volume
from model_volume import VOLUME_MOUNT_PATH

APP_NAME = "lookddak-idm-vton-fallback"
FUNCTION_NAME = "generate"
# inference_check 와 같은 실행 시간 상한. 컨테이너 시작·모델 로딩·추론·업로드가 모두 이 안에 들어간다.
SERVICE_TIMEOUT_SECONDS = 600
# GPU 를 기다리는 시간 상한. 넘으면 Modal 이 입력을 실패로 끝낸다(추론·업로드 없음).
STARTUP_TIMEOUT_SECONDS = 600
# fallback 은 드물다. 동시에 L40S 2대까지만 쓴다(넘치면 Modal 대기열).
MAX_CONTAINERS = 2
PERSON_RESIZE_MODE = "pad"
GARMENT_RESIZE_MODE = "pad"

DOWNLOAD_TIMEOUT_SECONDS = 20
UPLOAD_TIMEOUT_SECONDS = 60
USER_AGENT = "LookDDak-IDM-VTON/1.0"
# 상품 이미지는 외부 CDN 이 WebP 로 줄 수 있다. 모델 입력 전에 PNG 로 바꿔 inference_check 의 JPEG·PNG 검증을 통과시킨다.
DOWNLOAD_FORMATS = INPUT_FORMATS | {"WEBP"}
MAX_DESCRIPTION_LENGTH = 300
WEBP_QUALITY = 90
RESULT_CONTENT_TYPE = "image/webp"
# 결과 업로드 PUT 에 함께 보낼 수 있는 헤더. Presigned PUT 서명에 들어간 헤더와 정확히 같아야 한다.
ALLOWED_UPLOAD_HEADER_PREFIXES = ("x-amz-meta-",)
ALLOWED_UPLOAD_HEADERS = {"content-type"}
# 실패 분류. AI Server 의 idm_modal_client 가 같은 목록을 쓴다.
ERROR_KINDS = (
    "invalid_request",  # 요청 형식 오류(호출 쪽 버그)
    "deadline_exceeded",  # 업로드 URL 만료 전에 끝낼 수 없어 시작하지 않음
    "person_download_failed",
    "garment_download_failed",
    "person_image_invalid",
    "garment_image_invalid",
    "preprocess_rejected",  # OpenPose·Parsing·DensePose·마스크 검증 실패(입력 문제)
    "inference_failed",
    "postprocess_failed",  # 복원·인코딩 실패
    "upload_failed",
    "internal_error",  # 위 분류에 없는 예외(서비스 코드 결함). 원격 예외 직렬화에 기대지 않고 요약으로 돌려준다.
)

image = inference_image.add_local_python_source("inference_check", "restore_composite")

app = modal.App(APP_NAME)


class ServiceError(RuntimeError):
    """정해진 실패 분류. 메시지에는 URL·서명을 넣지 않는다."""

    def __init__(self, kind: str, detail: str) -> None:
        if kind not in ERROR_KINDS:
            raise ValueError(f"알 수 없는 error_kind: {kind}")
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


# --- 요청 검증 ---


def _require_https_url(value, name: str) -> str:
    from urllib.parse import urlsplit

    if not isinstance(value, str) or not value:
        raise ServiceError("invalid_request", f"{name} 가 비어 있다")
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.netloc:
        raise ServiceError("invalid_request", f"{name} 는 https URL 이어야 한다")
    return value


def validate_request(request) -> dict:
    """AI Server 가 보낸 요청을 확인한다. 반환값의 URL 은 쓰고 버린다(로그 금지)."""
    if not isinstance(request, dict):
        raise ServiceError("invalid_request", "요청은 dict 여야 한다")
    job_id = request.get("fitting_job_id")
    if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id < 1:
        raise ServiceError("invalid_request", "fitting_job_id 는 양의 정수여야 한다")
    attempt_id = request.get("attempt_id")
    if not isinstance(attempt_id, str) or not 1 <= len(attempt_id) <= 64:
        raise ServiceError("invalid_request", "attempt_id 가 올바르지 않다")
    description = request.get("garment_description")
    if not isinstance(description, str) or not description.strip():
        raise ServiceError("invalid_request", "garment_description 이 비어 있다")
    description = " ".join(description.split())[:MAX_DESCRIPTION_LENGTH]
    headers = request.get("upload_headers") or {}
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise ServiceError("invalid_request", "upload_headers 는 문자열 dict 여야 한다")
    for name in headers:
        lowered = name.lower()
        if lowered not in ALLOWED_UPLOAD_HEADERS and not lowered.startswith(
            ALLOWED_UPLOAD_HEADER_PREFIXES
        ):
            raise ServiceError("invalid_request", f"허용하지 않는 업로드 헤더: {name}")
    if {k.lower(): v for k, v in headers.items()}.get("content-type") != RESULT_CONTENT_TYPE:
        raise ServiceError(
            "invalid_request", "upload_headers 의 Content-Type 은 image/webp 여야 한다"
        )
    not_after = request.get("not_after_epoch")
    if isinstance(not_after, bool) or not isinstance(not_after, int | float):
        raise ServiceError("invalid_request", "not_after_epoch 가 없다")
    return {
        "fitting_job_id": job_id,
        "attempt_id": attempt_id,
        "person_image_url": _require_https_url(request.get("person_image_url"), "person_image_url"),
        "garment_image_url": _require_https_url(
            request.get("garment_image_url"), "garment_image_url"
        ),
        "garment_description": description,
        "upload_url": _require_https_url(request.get("upload_url"), "upload_url"),
        "upload_headers": dict(headers),
        "not_after_epoch": float(not_after),
    }


def ensure_before_deadline(not_after_epoch: float, now: float | None = None) -> None:
    """업로드 URL 이 만료되기 전에 끝낼 수 없으면 시작하지 않는다(GPU 대기열에서 오래 기다린 경우)."""
    if (time.time() if now is None else now) >= not_after_epoch:
        raise ServiceError("deadline_exceeded", "업로드 가능 시각이 지나 시작하지 않는다")


# --- 내려받기·이미지 ---


def download_image(url: str, role: str, *, opener=None) -> bytes:
    """https URL 을 시간·크기 제한 안에서 내려받는다. 실패 사유에 URL 을 넣지 않는다."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with (opener or urlopen)(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            declared = response.headers.get("Content-Length") if response.headers else None
            if declared and declared.isdigit() and int(declared) > MAX_INPUT_BYTES:
                raise ServiceError(f"{role}_download_failed", f"{role} 이미지가 너무 크다")
            body = response.read(MAX_INPUT_BYTES + 1)
    except HTTPError as error:
        raise ServiceError(f"{role}_download_failed", f"{role} HTTP {error.code}") from None
    except (URLError, TimeoutError, OSError) as error:
        raise ServiceError(f"{role}_download_failed", f"{role} {type(error).__name__}") from None
    if not body:
        raise ServiceError(f"{role}_download_failed", f"{role} 응답이 비어 있다")
    if len(body) > MAX_INPUT_BYTES:
        raise ServiceError(f"{role}_download_failed", f"{role} 이미지가 너무 크다")
    return body


def normalize_input_image(data: bytes, role: str) -> bytes:
    """디코딩·형식·크기를 확인하고 inference_check 가 받는 JPEG·PNG 바이트로 맞춘다.

    JPEG·PNG 는 원본 바이트를 그대로 넘긴다(EXIF 방향은 inference_check 가 반영). WebP·투명 PNG 는 EXIF 방향을
    반영하고 RGB 로 바꾼 PNG 로 다시 인코딩한다(투명 영역은 흰색). 그 밖의 형식은 거절한다.
    """
    import io

    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as opened:
            image_format = opened.format
            if image_format not in DOWNLOAD_FORMATS:
                raise ServiceError(f"{role}_image_invalid", f"{role} 형식 {image_format}")
            opened.load()
            if min(opened.size) < MIN_INPUT_SIDE:
                raise ServiceError(f"{role}_image_invalid", f"{role} 크기 {opened.size}")
            has_alpha = opened.mode in ("RGBA", "LA", "PA") or "transparency" in opened.info
            if image_format in INPUT_FORMATS and not has_alpha:
                return data
            upright = ImageOps.exif_transpose(opened)
            if has_alpha:
                rgba = upright.convert("RGBA")
                flat = Image.new("RGB", rgba.size, (255, 255, 255))
                flat.paste(rgba, mask=rgba.getchannel("A"))
            else:
                flat = upright.convert("RGB")
    except ServiceError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise ServiceError(f"{role}_image_invalid", f"{role} 디코딩 실패 {type(error).__name__}")
    buffer = io.BytesIO()
    flat.save(buffer, format="PNG")
    return buffer.getvalue()


def compose_result(outputs: dict, person_fit: dict):
    """추론 결과를 원본 사람 입력과 이진 마스크로 복원 합성하고, pad 여백을 잘라 원래 비율로 돌린다."""
    import restore_composite

    try:
        person = restore_composite.open_image(
            outputs[PERSON_INPUT_FILE], PERSON_INPUT_FILE, "PNG", "RGB"
        )
        result = restore_composite.open_image(outputs[RESULT_FILE], RESULT_FILE, "JPEG", "RGB")
        mask = restore_composite.open_binary_mask(outputs[INPAINT_MASK_FILE])
        restored, stats = restore_composite.composite_restored(person, result, mask)
    except (KeyError, restore_composite.RestoreError) as error:
        raise ServiceError("postprocess_failed", f"복원 합성 실패: {error}") from None
    box = person_fit.get("content_box")
    if (
        not isinstance(box, list | tuple)
        or len(box) != 4
        or not 0 <= box[0] < box[2] <= IMAGE_SIZE[0]
        or not 0 <= box[1] < box[3] <= IMAGE_SIZE[1]
    ):
        raise ServiceError("postprocess_failed", f"사람 입력 영역이 올바르지 않다: {box}")
    return restored.crop(tuple(int(value) for value in box)), stats


def encode_webp(pil_image) -> bytes:
    import io

    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="WEBP", quality=WEBP_QUALITY, method=6)
    return buffer.getvalue()


def inspect_webp(data: bytes) -> dict:
    """업로드 직전 확인: 디코딩되는 RGB WebP 인지."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as decoded:
        decoded.load()
        info = {"format": decoded.format, "size": list(decoded.size), "mode": decoded.mode}
    if info["format"] != "WEBP":
        raise ServiceError("postprocess_failed", f"WebP 가 아니다: {info}")
    return info


def s3_error_code(error) -> str:
    """S3 오류 응답 본문의 <Code>(AccessDenied·SignatureDoesNotMatch 등)만 꺼낸다. 본문의 다른 값은 남기지 않는다."""
    import re

    try:
        text = error.read(4096).decode("utf-8", "replace") if error.fp else ""
    except Exception:  # noqa: BLE001 (진단용 부가 정보)
        return "unknown"
    match = re.search(r"<Code>([A-Za-z0-9]{1,64})</Code>", text)
    return match.group(1) if match else "unknown"


def upload_result(url: str, headers: dict, body: bytes, *, opener=None) -> int:
    """Presigned PUT 한 번. 재시도하지 않는다. 2xx 가 아니면 upload_failed."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    request = Request(url, data=body, method="PUT", headers=dict(headers))
    try:
        with (opener or urlopen)(request, timeout=UPLOAD_TIMEOUT_SECONDS) as response:
            status = int(getattr(response, "status", 0) or response.getcode())
    except HTTPError as error:
        raise ServiceError(
            "upload_failed", f"PUT HTTP {error.code} {s3_error_code(error)}"
        ) from None
    except (URLError, TimeoutError, OSError) as error:
        raise ServiceError("upload_failed", f"PUT {type(error).__name__}") from None
    if not 200 <= status < 300:
        raise ServiceError("upload_failed", f"PUT HTTP {status}")
    return status


def failure(kind: str, detail: str, timings: dict) -> dict:
    return {"status": "failed", "error_kind": kind, "detail": detail[:300], "timings": timings}


def run_service(request, *, tryon=None, opener=None, clock=time.perf_counter) -> dict:
    """서비스 본체. 정해진 실패는 예외 대신 실패 요약으로 돌려준다(원격 예외 직렬화에 의존하지 않음)."""
    import hashlib

    timings: dict[str, float] = {}
    started = clock()

    def mark(name: str, since: float) -> float:
        now = clock()
        timings[name] = round(now - since, 2)
        return now

    try:
        req = validate_request(request)
        ensure_before_deadline(req["not_after_epoch"])
        step = clock()
        person = normalize_input_image(
            download_image(req["person_image_url"], "person", opener=opener), "person"
        )
        garment = normalize_input_image(
            download_image(req["garment_image_url"], "garment", opener=opener), "garment"
        )
        step = mark("download", step)
        try:
            output = (tryon or run_tryon.local)(
                req["garment_description"],
                person,
                garment,
                PERSON_RESIZE_MODE,
                GARMENT_RESIZE_MODE,
                DENOISE_STEPS,
            )
        except InferenceCheckError as error:
            raise ServiceError("preprocess_rejected", str(error)) from None
        except ServiceError:
            raise
        except Exception as error:  # noqa: BLE001 (모델·CUDA 오류도 분류해 돌려준다, 재시도 없음)
            raise ServiceError("inference_failed", type(error).__name__) from None
        step = mark("tryon", step)
        final, restore_stats = compose_result(output["images"], output["person_fit"])
        body = encode_webp(final)
        info = inspect_webp(body)
        step = mark("encode", step)
        # 업로드 직전에도 만료를 본다. 만료된 URL 로 보내 봐야 403 이다.
        ensure_before_deadline(req["not_after_epoch"])
        upload_result(req["upload_url"], req["upload_headers"], body, opener=opener)
        mark("upload", step)
    except ServiceError as error:
        timings["total"] = round(clock() - started, 2)
        print(f"[FAIL] error_kind={error.kind} detail={error.detail}", flush=True)
        return failure(error.kind, error.detail, timings)
    timings["total"] = round(clock() - started, 2)
    model_timings = output.get("timings", {})
    print(
        f"[DONE] fitting_job_id={req['fitting_job_id']} size={info['size']} bytes={len(body)} "
        f"timings={timings} model={model_timings}",
        flush=True,
    )
    return {
        "status": "succeeded",
        "fitting_job_id": req["fitting_job_id"],
        "attempt_id": req["attempt_id"],
        "content_type": RESULT_CONTENT_TYPE,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "width": info["size"][0],
        "height": info["size"][1],
        "settings": {
            **output.get("settings", {}),
            "guidance_scale": GUIDANCE_SCALE,
            "restore": {k: restore_stats[k] for k in ("dilate_px", "blur_radius")},
        },
        "timings": {**timings, "model": model_timings},
        "gpu_name": output.get("gpu_name"),
    }


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_MOUNT_PATH: read_only_volume},
    cpu=INFERENCE_CPU,
    memory=INFERENCE_MEMORY_MIB,
    timeout=SERVICE_TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    retries=RETRIES,
    max_containers=MAX_CONTAINERS,
    single_use_containers=True,
)
def generate(request: dict) -> dict:
    """작업 하나를 처리한다. 반환값은 성공·실패 요약 dict 다(URL·이미지 바이트 없음)."""
    try:
        return run_service(request)
    except Exception as error:  # noqa: BLE001 (예외로 끝내지 않고 분류된 실패로 돌려준다, 재시도 없음)
        print(f"[FAIL] error_kind=internal_error error={type(error).__name__}", flush=True)
        return failure("internal_error", type(error).__name__, {})
