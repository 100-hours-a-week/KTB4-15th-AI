import re

import modal
from detectron2_build_check import GPU, GRADIO_DEMO_DIR, REPO_DIR
from integrated_environment_check import HF_OFFLINE_ENV
from model_loading_check import (
    DENSEPOSE_ARGS,
    DEVICE,
    MAX_CONTAINERS,
    RETRIES,
    verify_official_ckpt_files,
    verify_volume_files,
)
from model_loading_check import image as model_loading_image
from model_loading_check import volume as read_only_volume
from model_volume import MODEL_DIR, VOLUME_MOUNT_PATH, link_official_ckpt_dirs

APP_NAME = "lookddak-idm-vton-inference-check"
# 실행 시간 상한이지 금액 상한이 아니다. 컨테이너 시작·모델 로딩·추론이 모두 이 안에 들어간다.
INFERENCE_TIMEOUT_SECONDS = 600
INFERENCE_CPU = 4.0
INFERENCE_MEMORY_MIB = 32768

# 이미지에 clone 한 공식 저장소(고정 커밋)의 예제. 같은 파일인지 해시로도 확인한다.
HUMAN_IMAGE_PATH = f"{GRADIO_DEMO_DIR}/example/human/00034_00.jpg"
CLOTH_IMAGE_PATH = f"{GRADIO_DEMO_DIR}/example/cloth/04469_00.jpg"
EXPECTED_INPUT_SHA256 = {
    HUMAN_IMAGE_PATH: "9b948383f55e572f4a6cfe3ef3a2d2f228f519560613c631237abb5c66a2f725",
    CLOTH_IMAGE_PATH: "eb462f22f2bf55ea434e8a9593b3a8205c6261cf6b0c6c5cba84e62290124d27",
}
# 04469_00.jpg 를 보고 적었다. 청록색 벨벳, 반팔, V넥, 옆으로 묶는 랩 탑. --garment-description 으로 바꿀 수 있다.
DEFAULT_GARMENT_DESCRIPTION = "teal blue velvet short sleeve v-neck wrap top"

# gradio_demo/app.py start_tryon 의 값. width·height 순서는 PIL 의 (width, height) 와 같다.
IMAGE_SIZE = (768, 1024)
PREPROCESS_SIZE = (384, 512)
MASK_MODEL_TYPE = "hd"
MASK_CATEGORY = "upper_body"
DENOISE_STEPS = 30
# --inference-steps 로 고를 수 있는 값. 생략하면 DENOISE_STEPS(공식 기본값 30)를 쓴다.
ALLOWED_INFERENCE_STEPS = (30, 40)
SEED = 42
GUIDANCE_SCALE = 2.0
STRENGTH = 1.0
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality"

# 전처리 결과가 비정상인지 가르는 기준. 품질 판정이 아니라 명백히 잘못된 입력만 거른다.
# OpenPose 18점 중 상의 마스크가 기대는 어깨(2: 오른쪽, 5: 왼쪽). 못 찾으면 (0, 0) 으로 채워진다.
REQUIRED_KEYPOINTS = {2: "right_shoulder", 5: "left_shoulder"}
# Human Parsing(ATR) 라벨. upper_body 마스크는 4(upper_clothes)·7(dress) 영역을 중심으로 만든다.
UPPER_CLOTH_LABELS = {4: "upper_clothes", 7: "dress"}
MIN_MASK_RATIO = 0.01
MAX_MASK_RATIO = 0.9
# 결과 픽셀 표준편차가 이보다 작으면 한 가지 색(검은 화면·NaN 등)으로 본다.
MIN_IMAGE_STDDEV = 2.0
JPEG_QUALITY = 95
RESULT_FILE = "result.jpg"
MASK_FILE = "mask.jpg"
# 합성 실험용 손실 없는 PNG. 이름은 restore_composite.py 와 같아야 한다.
INPAINT_MASK_FILE = "inpaint_mask.png"
PERSON_INPUT_FILE = "person_input.png"
# pipe 의 mask_processor(diffusers 0.25.0 VaeImageProcessor, do_binarize=True)는 255 로 나눈 값이 0.5 이상이면 1 로 본다.
INPAINT_MASK_THRESHOLD = 128
OUTPUT_DIR_NAME = "test_outputs"
# --case-id 를 생략했을 때 공식 예제 결과를 둘 디렉터리. 사용자 입력은 실행 시각으로 이름을 만든다.
OFFICIAL_CASE_ID = "official-example"
# 디렉터리 이름 하나만 허용한다. 경로 구분자·상위 경로(..)·숨김 이름(. 시작)은 맞지 않는다.
CASE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

# 사용자 입력(--person-image·--garment-image) 조건. 로컬에서 먼저 확인해 GPU 를 띄우기 전에 거른다.
INPUT_FORMATS = {"JPEG", "PNG"}
MAX_INPUT_BYTES = 20 * 1024 * 1024
MIN_INPUT_SIDE = 256
EXIF_ORIENTATION_TAG = 0x0112
# 768×1024(3:4) 와 비율이 이만큼 넘게 다르면 공식 방식(자동 크롭 없이 리사이즈)대로 늘어난다는 경고만 남긴다.
TARGET_ASPECT_RATIO = IMAGE_SIZE[0] / IMAGE_SIZE[1]
ASPECT_RATIO_TOLERANCE = 0.05
# 사람·상품 이미지에 따로 고른다. stretch 는 기존(공식) 강제 리사이즈, pad 는 비율을 유지해 축소한 뒤 가운데 두고 여백을 채운다.
RESIZE_MODES = ("stretch", "pad")
PERSON_RESIZE_MODES = RESIZE_MODES
DEFAULT_PERSON_RESIZE_MODE = "stretch"
GARMENT_RESIZE_MODES = RESIZE_MODES
DEFAULT_GARMENT_RESIZE_MODE = "stretch"
# pad 여백 색은 축소한 이미지 네 모서리(짧은 변의 5%, 최소 4px 정사각형) 픽셀의 채널별 중앙값으로 정한다.
PAD_CORNER_RATIO = 0.05
PAD_CORNER_MIN_PX = 4

image = (
    model_loading_image
    # 모델 로딩 검증과 같은 이미지에 이 파일이 import 하는 model_loading_check 만 더 올린다. 빌드 단계를 더하지 않는다.
    .add_local_python_source("model_loading_check")
)

app = modal.App(APP_NAME)


class InferenceCheckError(RuntimeError):
    """입력 이미지, 전처리 결과, 추론 결과 중 하나 이상이 기대와 다르다."""


def require_inference_steps(steps) -> int:
    # bool 은 int 의 하위 형이라 따로 막는다.
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps not in ALLOWED_INFERENCE_STEPS
    ):
        raise InferenceCheckError(
            f"inference-steps 는 {list(ALLOWED_INFERENCE_STEPS)} 중 하나여야 한다: {steps!r}"
        )
    return steps


def require_keypoints(keypoints: dict) -> dict[str, list[float]]:
    """OpenPose 결과에 18점이 있고 상의 마스크에 필요한 어깨를 찾았는지 본다."""
    points = keypoints.get("pose_keypoints_2d")
    if not isinstance(points, list) or len(points) != 18:
        raise InferenceCheckError(f"OpenPose 관절이 18개가 아니다: {points!r:.200}")
    found = {
        name: [float(v) for v in points[index][:2]] for index, name in REQUIRED_KEYPOINTS.items()
    }
    missing = [name for name, (x, y) in found.items() if x <= 1.0 and y <= 1.0]
    if missing:
        raise InferenceCheckError(f"OpenPose 가 {missing} 를 찾지 못했다: {found}")
    return found


def require_upper_cloth(label_counts: dict[int, int]) -> dict[str, int]:
    """Human Parsing 결과에 상의 영역이 있는지 본다. 없으면 upper_body 마스크가 팔·목 근처만 남는다."""
    found = {name: int(label_counts.get(label, 0)) for label, name in UPPER_CLOTH_LABELS.items()}
    if not any(found.values()):
        raise InferenceCheckError(f"Human Parsing 결과에 상의 영역이 없다: {label_counts}")
    return found


def require_mask_ratio(ratio: float) -> float:
    if not MIN_MASK_RATIO <= ratio <= MAX_MASK_RATIO:
        raise InferenceCheckError(
            f"상의 마스크 면적 비율 {ratio:.4f} 가 [{MIN_MASK_RATIO}, {MAX_MASK_RATIO}] 밖이다"
        )
    return ratio


def require_varied_image(name: str, stddev: float) -> float:
    if stddev < MIN_IMAGE_STDDEV:
        raise InferenceCheckError(f"{name}: 한 가지 색에 가깝다(픽셀 표준편차 {stddev:.2f})")
    return stddev


def encode_jpeg(pil_image) -> bytes:
    import io

    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def encode_png(pil_image) -> bytes:
    import io

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def inspect_jpeg(data: bytes, expected_size: tuple[int, int] = IMAGE_SIZE) -> dict:
    """바이트가 비어 있지 않고 JPEG 이며 기대한 해상도인지 본다. 품질은 판단하지 않는다."""
    import io

    from PIL import Image

    if not data or data[:3] != b"\xff\xd8\xff":
        raise InferenceCheckError(f"JPEG 바이트가 아니다(길이 {len(data)}, 앞부분 {data[:4]!r})")
    with Image.open(io.BytesIO(data)) as decoded:
        decoded.load()
        info = {"format": decoded.format, "size": list(decoded.size), "mode": decoded.mode}
    if info["format"] != "JPEG" or tuple(info["size"]) != expected_size:
        raise InferenceCheckError(f"JPEG {expected_size} 가 아니다: {info}")
    return {**info, "bytes": len(data)}


def decode_input_image(data: bytes, role: str):
    """JPEG·PNG 바이트를 EXIF 방향을 반영한 RGB PIL 이미지와 메타데이터로 바꾼다. 픽셀·바이트는 로그에 남기지 않는다."""
    import io

    from PIL import Image, ImageOps, UnidentifiedImageError

    if not data:
        raise InferenceCheckError(f"{role} 이미지가 비어 있다")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image_format = opened.format
            if image_format not in INPUT_FORMATS:
                raise InferenceCheckError(
                    f"{role} 이미지 형식 {image_format} 은 지원하지 않는다({sorted(INPUT_FORMATS)})"
                )
            original_size = opened.size
            orientation = int(opened.getexif().get(EXIF_ORIENTATION_TAG, 1))
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InferenceCheckError(f"{role} 이미지를 열 수 없다: {type(error).__name__}") from error
    width, height = rgb.size
    if min(width, height) < MIN_INPUT_SIDE:
        raise InferenceCheckError(f"{role} 이미지가 너무 작다: {width}x{height}")
    aspect_ratio = width / height
    info = {
        "format": str(image_format),
        "bytes": len(data),
        "original_size": list(original_size),
        "exif_orientation": orientation,
        "size": [width, height],
        "aspect_ratio": round(aspect_ratio, 4),
    }
    if abs(aspect_ratio - TARGET_ASPECT_RATIO) > ASPECT_RATIO_TOLERANCE:
        info["aspect_warning"] = f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} 로 리사이즈하면서 비율이 바뀐다"
    return rgb, info


def corner_background_color(rgb) -> tuple[int, int, int]:
    """네 모서리 픽셀의 채널별 중앙값. 한 모서리에 사람·물체가 걸쳐도 나머지 배경색을 따른다."""
    from PIL import Image, ImageStat

    width, height = rgb.size
    patch = max(PAD_CORNER_MIN_PX, round(min(width, height) * PAD_CORNER_RATIO))
    patch = min(patch, width, height)
    corners = [
        (0, 0),
        (width - patch, 0),
        (0, height - patch),
        (width - patch, height - patch),
    ]
    strip = Image.new("RGB", (patch * len(corners), patch))
    for index, (left, top) in enumerate(corners):
        strip.paste(rgb.crop((left, top, left + patch, top + patch)), (index * patch, 0))
    red, green, blue = (int(value) for value in ImageStat.Stat(strip).median)
    return red, green, blue


def fit_model_input(rgb, mode: str, option: str):
    """EXIF 보정·RGB 변환을 마친 이미지를 768×1024 모델 입력으로 만든다. 사람·상품이 같은 규칙을 쓴다.

    사람: OpenPose·Parsing·DensePose·마스크·추론·person_input.png·복원 합성이 모두 이 결과 하나만 쓴다.
    상품: cloth Tensor·ip_adapter_image 가 모두 이 결과 하나만 쓴다.
    """
    from PIL import Image

    if mode not in RESIZE_MODES:
        raise InferenceCheckError(f"{option} 는 {list(RESIZE_MODES)} 중 하나여야 한다: {mode!r}")
    width, height = rgb.size
    target_width, target_height = IMAGE_SIZE
    info = {
        "mode": mode,
        "original_size": [width, height],
        "original_aspect": round(width / height, 4),
    }
    if mode == "stretch":
        # 기존 방식 그대로. 비율이 3:4 가 아니면 사람·상품이 늘어나거나 눌린다.
        fitted = rgb.resize(IMAGE_SIZE)
        padding = {"left": 0, "right": 0, "top": 0, "bottom": 0}
        resized_size = list(IMAGE_SIZE)
    else:
        scale = min(target_width / width, target_height / height)
        resized_width = min(target_width, max(1, round(width * scale)))
        resized_height = min(target_height, max(1, round(height * scale)))
        resized = rgb.resize((resized_width, resized_height))
        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        padding = {
            "left": left,
            "right": target_width - resized_width - left,
            "top": top,
            "bottom": target_height - resized_height - top,
        }
        color = corner_background_color(resized)
        fitted = Image.new("RGB", IMAGE_SIZE, color)
        fitted.paste(resized, (left, top))
        resized_size = [resized_width, resized_height]
        info["pad_color"] = list(color)
    info.update(
        {
            "resized_size": resized_size,
            "resized_aspect": round(resized_size[0] / resized_size[1], 4),
            "padding": padding,
            # 768×1024 좌표계에서 실제 이미지가 놓인 영역 [left, top, right, bottom]. 나머지는 여백이다.
            "content_box": [
                padding["left"],
                padding["top"],
                target_width - padding["right"],
                target_height - padding["bottom"],
            ],
            "output_size": list(fitted.size),
        }
    )
    if fitted.size != IMAGE_SIZE or fitted.mode != "RGB":
        raise InferenceCheckError(
            f"모델 입력이 RGB {IMAGE_SIZE} 가 아니다: {fitted.mode} {fitted.size}"
        )
    return fitted, info


def fit_person_image(rgb, mode: str):
    return fit_model_input(rgb, mode, "person-resize-mode")


def fit_garment_image(rgb, mode: str):
    return fit_model_input(rgb, mode, "garment-resize-mode")


def read_local_image(path: str, role: str) -> tuple[bytes, dict]:
    """Mac 의 이미지 파일을 확인한 뒤 bytes 로 읽는다. Modal 컨테이너는 이 경로를 볼 수 없어 bytes 로 넘긴다."""
    from pathlib import Path

    file = Path(path).expanduser()
    if not file.is_file():
        raise InferenceCheckError(f"{role} 이미지 파일이 없다: {file}")
    size = file.stat().st_size
    if not 0 < size <= MAX_INPUT_BYTES:
        raise InferenceCheckError(
            f"{role} 이미지 크기 {size} bytes 가 (0, {MAX_INPUT_BYTES}] 밖이다"
        )
    data = file.read_bytes()
    _, info = decode_input_image(data, role)
    return data, info


def case_output_dir(output_root, case_id: str):
    """--case-id 를 output_root 바로 아래 디렉터리 하나로만 바꾼다. 다른 위치로 빠져나가는 값은 거부한다."""
    from pathlib import Path

    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise InferenceCheckError(
            f"--case-id 는 영문·숫자로 시작하고 영문·숫자·.·_·- 만 쓸 수 있다(64자 이하): {case_id!r}"
        )
    root = Path(output_root).resolve()
    target = (root / case_id).resolve()
    if target.parent != root:
        raise InferenceCheckError(f"--case-id 가 {root} 밖을 가리킨다: {case_id!r}")
    return target


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_MOUNT_PATH: read_only_volume},
    cpu=INFERENCE_CPU,
    memory=INFERENCE_MEMORY_MIB,
    timeout=INFERENCE_TIMEOUT_SECONDS,
    retries=RETRIES,
    max_containers=MAX_CONTAINERS,
    # scaledown_window 는 Modal 1.6.1 이 0 을 거부한다. 입력 하나를 처리하면 컨테이너를 바로 내린다.
    single_use_containers=True,
)
def run_tryon(
    garment_description: str = DEFAULT_GARMENT_DESCRIPTION,
    person_image: bytes | None = None,
    garment_image: bytes | None = None,
    person_resize_mode: str = DEFAULT_PERSON_RESIZE_MODE,
    garment_resize_mode: str = DEFAULT_GARMENT_RESIZE_MODE,
    inference_steps: int = DENOISE_STEPS,
) -> dict:
    # 두 이미지를 모두 생략하면 이미지에 든 공식 예제를, 모두 넘기면 그 bytes 를 쓴다. 사용자 이미지는 이미지 빌드에 넣지 않는다.
    import os

    # HF 라이브러리를 import 하기 전에 둬야 적용된다. Volume 의 로컬 경로만 읽는다.
    os.environ.update(HF_OFFLINE_ENV)

    import contextlib
    import hashlib
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image

    run_started = time.perf_counter()
    timings: dict[str, float] = {}
    vram: dict[str, dict] = {}

    def mib(num_bytes: int) -> float:
        return round(num_bytes / 1024**2, 1)

    @contextlib.contextmanager
    def measured(name: str):
        # 구간마다 최고치를 새로 잰다. 비동기 CUDA 작업이 끝난 뒤의 시간을 재도록 앞뒤로 동기화한다.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        print(f"[START] {name}", flush=True)
        yield
        torch.cuda.synchronize()
        timings[name] = round(time.perf_counter() - started, 2)
        free, total = torch.cuda.mem_get_info()
        vram[name] = {
            "allocated_mib": mib(torch.cuda.memory_allocated()),
            "reserved_mib": mib(torch.cuda.memory_reserved()),
            "max_allocated_mib": mib(torch.cuda.max_memory_allocated()),
            "max_reserved_mib": mib(torch.cuda.max_memory_reserved()),
            "device_free_mib": mib(free),
            "device_total_mib": mib(total),
        }
        print(f"[DONE] {name}: {timings[name]}s {vram[name]}", flush=True)

    garment_description = garment_description.strip()
    if not garment_description:
        raise InferenceCheckError("garment description 이 비어 있다")
    if (person_image is None) != (garment_image is None):
        raise InferenceCheckError("사람 이미지와 상품 이미지는 함께 넘기거나 둘 다 생략해야 한다")
    if person_resize_mode not in PERSON_RESIZE_MODES:
        raise InferenceCheckError(
            f"person-resize-mode 는 {list(PERSON_RESIZE_MODES)} 중 하나여야 한다: {person_resize_mode!r}"
        )
    if garment_resize_mode not in GARMENT_RESIZE_MODES:
        raise InferenceCheckError(
            f"garment-resize-mode 는 {list(GARMENT_RESIZE_MODES)} 중 하나여야 한다: {garment_resize_mode!r}"
        )
    inference_steps = require_inference_steps(inference_steps)
    mode = "official" if person_image is None else "custom"
    print(
        f"[INFO] 입력 모드: {mode}, person-resize-mode: {person_resize_mode}, "
        f"garment-resize-mode: {garment_resize_mode}, inference-steps: {inference_steps}",
        flush=True,
    )
    if not torch.cuda.is_available() or GPU not in torch.cuda.get_device_name(0):
        raise InferenceCheckError(f"{GPU} GPU 가 아니다: {torch.cuda.is_available()}")

    with measured("checkpoint_check"):
        # model_loading_check 와 같은 사전 점검. Volume 은 읽기만 하고, ckpt/ 연결은 컨테이너 안 저장소에만 만든다.
        sizes = verify_volume_files()
        link_official_ckpt_dirs(REPO_DIR)
        verify_official_ckpt_files(REPO_DIR, sizes)

    # gradio_demo/app.py 를 저장소 루트에서 실행할 때와 같다. apply_net 인자의 ./configs·./ckpt 가 cwd 기준이다.
    os.chdir(REPO_DIR)
    sys.path[:0] = [GRADIO_DEMO_DIR, REPO_DIR]

    sources: dict[str, tuple[str, bytes]] = {}
    if mode == "official":
        # 고정 커밋의 예제인지 해시로 확인하는 것은 공식 예제 모드에서만 한다.
        for role, path in (("human", HUMAN_IMAGE_PATH), ("cloth", CLOTH_IMAGE_PATH)):
            file = Path(path)
            if not file.is_file():
                raise InferenceCheckError(f"{role} 예제 이미지가 없다: {path}")
            data = file.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != EXPECTED_INPUT_SHA256[path]:
                raise InferenceCheckError(f"{path} 가 고정 커밋의 예제와 다르다: sha256 {digest}")
            sources[role] = (path, data)
    else:
        sources = {
            "human": ("uploaded bytes", person_image),
            "cloth": ("uploaded bytes", garment_image),
        }

    inputs = {}
    for role, (source, data) in sources.items():
        # 1. EXIF 방향 보정 뒤 RGB 변환 (gradio 가 넘기는 PIL 이미지와 같게). 공식 예제에는 EXIF 방향 태그가 없다.
        rgb, info = decode_input_image(data, role)
        inputs[role] = {"source": source, **info, "image": rgb}
        # 원본 바이트·픽셀·URL 은 출력하지 않는다. 형식·크기·방향만 남긴다.
        print(f"[INFO] {role}: {source} {info}", flush=True)
        # pad 모드는 비율을 유지하므로 늘어난다는 경고가 해당하지 않는다.
        role_mode = person_resize_mode if role == "human" else garment_resize_mode
        if "aspect_warning" in info and role_mode == "stretch":
            print(
                f"[WARN] {role}: {info['aspect_warning']} (비율 {info['aspect_ratio']})", flush=True
            )

    from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image

    with measured("preprocess_model_loading"):
        import apply_net
        from detectron2.engine.defaults import DefaultPredictor
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose

        openpose_model = OpenPose(0)
        parsing_model = Parsing(0)
        # 공식 start_tryon 은 요청마다 args.func(args, img) 로 DensePose 모델을 새로 만든다. 같은 설정으로 한 번만 만든다.
        densepose_args = apply_net.create_argument_parser().parse_args(DENSEPOSE_ARGS)
        densepose_action = densepose_args.func.__self__
        densepose_cfg = densepose_action.setup_config(
            densepose_args.cfg, densepose_args.model, densepose_args, []
        )
        densepose_predictor = DefaultPredictor(densepose_cfg)
        densepose_context = densepose_action.create_context(densepose_args, densepose_cfg)

    with measured("idm_vton_model_loading"):
        from diffusers import AutoencoderKL, DDPMScheduler
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
        from src.unet_hacked_tryon import UNet2DConditionModel
        from transformers import (
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPVisionModelWithProjection,
        )

        def gpu_module(model_class, subfolder: str):
            # gradio_demo/app.py 와 같은 클래스·subfolder·fp16. model_loading_check 처럼 하나씩 GPU 로 옮겨
            # 모든 가중치가 CPU 에 동시에 올라가지 않게 한다.
            module = model_class.from_pretrained(
                MODEL_DIR, subfolder=subfolder, torch_dtype=torch.float16
            )
            module.requires_grad_(False)
            return module.to(DEVICE)

        unet = gpu_module(UNet2DConditionModel, "unet")
        tokenizer_one = AutoTokenizer.from_pretrained(
            MODEL_DIR, subfolder="tokenizer", revision=None, use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            MODEL_DIR, subfolder="tokenizer_2", revision=None, use_fast=False
        )
        noise_scheduler = DDPMScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
        text_encoder_one = gpu_module(CLIPTextModel, "text_encoder")
        text_encoder_two = gpu_module(CLIPTextModelWithProjection, "text_encoder_2")
        image_encoder = gpu_module(CLIPVisionModelWithProjection, "image_encoder")
        vae = gpu_module(AutoencoderKL, "vae")
        unet_encoder = gpu_module(UNet2DConditionModel_ref, "unet_encoder")
        pipe = TryonPipeline.from_pretrained(
            MODEL_DIR,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        )
        pipe.unet_encoder = unet_encoder
        # start_tryon 첫 줄과 같다. 이미 GPU 에 있어 옮길 것은 없지만 공식 호출을 그대로 둔다.
        openpose_model.preprocessor.body_estimation.model.to(DEVICE)
        pipe.to(DEVICE)
        pipe.unet_encoder.to(DEVICE)

    tensor_transfrom = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    preprocess_report: dict = {}
    person_fit: dict = {}
    garment_fit: dict = {}

    @contextlib.contextmanager
    def explain_failure():
        # 검증 기준은 그대로 두고, 어느 모드·어느 단계에서 왜 멈췄는지와 그때까지의 결과를 남긴다.
        try:
            yield
        except InferenceCheckError as error:
            print(
                f"[FAIL] person-resize-mode={person_resize_mode}, "
                f"garment-resize-mode={garment_resize_mode}: {error}",
                flush=True,
            )
            print(f"[FAIL] 사람 입력 맞춤: {person_fit}", flush=True)
            print(f"[FAIL] 상품 입력 맞춤: {garment_fit}", flush=True)
            print(f"[FAIL] 실패 전까지의 전처리 결과: {preprocess_report}", flush=True)
            raise

    with measured("preprocess"), explain_failure():
        # 2. 768×1024 맞춤. 사람·상품 각각 stretch(기존)·pad 중 고른 방식으로 한 번만 만든다.
        # 상품: 아래 cloth Tensor·ip_adapter_image 가 모두 이 garm_img 에서 나온다.
        garm_img, garment_fit_info = fit_garment_image(
            inputs["cloth"]["image"], garment_resize_mode
        )
        garment_fit.update(garment_fit_info)
        print(f"[INFO] 상품 입력 맞춤: {garment_fit}", flush=True)
        # 사람: 아래 OpenPose·Parsing·DensePose·마스크·추론·person_input.png 는 모두 이 human_img 에서 나온다.
        human_img, fit_info = fit_person_image(inputs["human"]["image"], person_resize_mode)
        person_fit.update(fit_info)
        print(f"[INFO] 사람 입력 맞춤: {person_fit}", flush=True)

        # 3. OpenPose (384×512). 사람을 못 찾으면 공식 코드는 IndexError 로 끝난다.
        try:
            keypoints = openpose_model(human_img.resize(PREPROCESS_SIZE))
        except IndexError as error:
            raise InferenceCheckError("OpenPose 가 사람을 찾지 못했다") from error
        points = keypoints.get("pose_keypoints_2d")
        if isinstance(points, list):
            # 384×512 좌표. 못 찾은 관절은 (0, 0) 이다. 검증 전에 남겨 실패해도 볼 수 있게 한다.
            preprocess_report["pose_keypoints"] = [
                [round(float(value), 1) for value in point[:2]] for point in points
            ]
            preprocess_report["pose_keypoints_found"] = sum(
                1 for x, y in preprocess_report["pose_keypoints"] if x > 1.0 or y > 1.0
            )
        preprocess_report["keypoints"] = require_keypoints(keypoints)

        # 4. Human Parsing (384×512)
        model_parse, _ = parsing_model(human_img.resize(PREPROCESS_SIZE))
        labels, counts = np.unique(np.array(model_parse), return_counts=True)
        label_counts = {int(label): int(count) for label, count in zip(labels, counts, strict=True)}
        preprocess_report["parsing_size"] = list(model_parse.size)
        preprocess_report["upper_cloth_pixels"] = require_upper_cloth(label_counts)

        # 5. 상의 마스크. 공식과 같이 768×1024 로 키운 뒤 회색 확인용 이미지를 만든다.
        from utils_mask import get_mask_location

        mask, mask_gray = get_mask_location(MASK_MODEL_TYPE, MASK_CATEGORY, model_parse, keypoints)
        mask = mask.resize(IMAGE_SIZE)
        mask_ratio = float((np.array(mask) > 0).mean())
        preprocess_report["mask_ratio"] = round(require_mask_ratio(mask_ratio), 4)
        # pipe 가 실제로 인페인팅하는 영역. 리사이즈로 경계에 생긴 중간값을 pipe 와 같은 기준으로 나눈다.
        # pipe 에는 기존대로 mask 를 넘기고, 이 이진 마스크는 로컬 합성 실험에만 쓴다.
        inpaint_mask = Image.fromarray(
            np.where(np.array(mask) >= INPAINT_MASK_THRESHOLD, 255, 0).astype(np.uint8)
        )
        preprocess_report["inpaint_mask_ratio"] = round(
            float((np.array(inpaint_mask) == 255).mean()), 4
        )
        # 마스크 흰색 중 여백에 걸친 비율. pad 모드에서 마스크가 사진 영역과 맞는지 보는 진단값이다(stretch 는 0).
        left, top, right, bottom = person_fit["content_box"]
        in_padding = np.ones((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=bool)
        in_padding[top:bottom, left:right] = False
        white = np.array(inpaint_mask) == 255
        preprocess_report["inpaint_mask_padding_ratio"] = round(
            float((white & in_padding).sum() / max(int(white.sum()), 1)), 4
        )
        mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        # 6. DensePose. 공식 InferenceAction.execute 에서 모델 생성만 빼고 같은 호출 순서다.
        human_img_arg = _apply_exif_orientation(human_img.resize(PREPROCESS_SIZE))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        with torch.no_grad():
            instances = densepose_predictor(human_img_arg)["instances"]
            if len(instances) == 0:
                raise InferenceCheckError("DensePose 가 사람을 찾지 못했다(점수 0.8 이상 없음)")
            pose_img = densepose_action.execute_on_outputs(
                densepose_context, {"image": human_img_arg}, instances
            )
        densepose_action.postexecute(densepose_context)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize(IMAGE_SIZE)
        preprocess_report["densepose_instances"] = len(instances)
        # 384×512 기준 점수와 사람 상자. 여러 명이 잡히면 결과가 섞일 수 있어 함께 남긴다.
        preprocess_report["densepose_scores"] = [
            round(float(score), 3) for score in instances.scores.tolist()
        ]
        preprocess_report["densepose_boxes"] = [
            [round(float(value), 1) for value in box]
            for box in instances.pred_boxes.tensor.tolist()
        ]
        preprocess_report["densepose_stddev"] = round(
            require_varied_image("DensePose 결과", float(np.array(pose_img).std())), 2
        )
    print(f"[INFO] 전처리 결과: {preprocess_report}", flush=True)

    # 7. start_tryon 과 같은 문맥·인자. 프롬프트 두 종류(사람용·의류용)를 만들어 pipe 에 넘긴다.
    with measured("inference"), torch.no_grad(), torch.cuda.amp.autocast(), torch.inference_mode():
        prompt = "model is wearing " + garment_description
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=NEGATIVE_PROMPT,
        )
        prompt = ["a photo of " + garment_description]
        negative_prompt = [NEGATIVE_PROMPT]
        (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=negative_prompt,
        )
        pose_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(DEVICE, torch.float16)
        garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(DEVICE, torch.float16)
        generator = torch.Generator(DEVICE).manual_seed(SEED)
        images = pipe(
            prompt_embeds=prompt_embeds.to(DEVICE, torch.float16),
            negative_prompt_embeds=negative_prompt_embeds.to(DEVICE, torch.float16),
            pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE, torch.float16),
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(DEVICE, torch.float16),
            num_inference_steps=inference_steps,
            generator=generator,
            strength=STRENGTH,
            pose_img=pose_tensor,
            text_embeds_cloth=prompt_embeds_c.to(DEVICE, torch.float16),
            cloth=garm_tensor,
            mask_image=mask,
            image=human_img,
            height=IMAGE_SIZE[1],
            width=IMAGE_SIZE[0],
            ip_adapter_image=garm_img.resize(IMAGE_SIZE),
            guidance_scale=GUIDANCE_SCALE,
        )[0]

    with measured("encode_outputs"):
        if len(images) != 1:
            raise InferenceCheckError(f"결과 이미지가 1장이 아니다: {len(images)}")
        result_img = images[0]
        if tuple(result_img.size) != IMAGE_SIZE:
            raise InferenceCheckError(f"결과 해상도가 {IMAGE_SIZE} 가 아니다: {result_img.size}")
        result_stddev = require_varied_image("결과 이미지", float(np.array(result_img).std()))
        # Modal 반환값은 bytes 만 넘긴다. PIL·NumPy·Tensor 는 컨테이너 밖으로 보내지 않는다.
        outputs = {RESULT_FILE: encode_jpeg(result_img), MASK_FILE: encode_jpeg(mask_gray)}
        output_info = {name: inspect_jpeg(data) for name, data in outputs.items()}
        # 합성 실험용. 모델이 실제로 받은 사람 이미지(EXIF 보정·RGB·768×1024 리사이즈 뒤)와 이진 마스크를
        # 같은 좌표계 그대로 손실 없이 넘긴다.
        if inpaint_mask.size != human_img.size:
            raise InferenceCheckError(
                f"이진 마스크 {inpaint_mask.size} 와 사람 이미지 {human_img.size} 의 크기가 다르다"
            )
        outputs[INPAINT_MASK_FILE] = encode_png(inpaint_mask)
        outputs[PERSON_INPUT_FILE] = encode_png(human_img)

    total_seconds = round(time.perf_counter() - run_started, 2)
    peak_allocated = max(stats["max_allocated_mib"] for stats in vram.values())
    peak_reserved = max(stats["max_reserved_mib"] for stats in vram.values())
    print(
        f"[INFO] 전처리 {timings['preprocess']}s, 전처리 모델 로딩 {timings['preprocess_model_loading']}s, "
        f"IDM-VTON 모델 로딩 {timings['idm_vton_model_loading']}s, 추론({inference_steps} steps) {timings['inference']}s, "
        f"전체 {total_seconds}s, 최대 VRAM allocated {peak_allocated} MiB / reserved {peak_reserved} MiB",
        flush=True,
    )
    print(
        "=== IDM-VTON Inference Check Passed (파일 생성 확인, 품질은 육안 확인 필요) ===",
        flush=True,
    )
    return {
        "mode": mode,
        "garment_description": str(garment_description),
        "settings": {
            "size": list(IMAGE_SIZE),
            "steps": inference_steps,
            "seed": SEED,
            "guidance_scale": GUIDANCE_SCALE,
            "strength": STRENGTH,
            "dtype": "float16",
            "person_resize_mode": person_resize_mode,
            "garment_resize_mode": garment_resize_mode,
        },
        "inputs": {
            role: {key: value for key, value in info.items() if key != "image"}
            for role, info in inputs.items()
        },
        "person_fit": person_fit,
        "garment_fit": garment_fit,
        "preprocess": preprocess_report,
        "result_stddev": round(result_stddev, 2),
        "timings": {**timings, "total": total_seconds},
        "vram": vram,
        "peak_vram_allocated_mib": float(peak_allocated),
        "peak_vram_reserved_mib": float(peak_reserved),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "images": outputs,
        "image_info": output_info,
    }


@app.local_entrypoint()
def main(
    person_image: str = "",
    garment_image: str = "",
    garment_description: str = "",
    case_id: str = "",
    person_resize_mode: str = DEFAULT_PERSON_RESIZE_MODE,
    garment_resize_mode: str = DEFAULT_GARMENT_RESIZE_MODE,
    inference_steps: int = DENOISE_STEPS,
) -> None:
    # 두 이미지를 모두 생략하면 공식 예제, 모두 주면 사용자 입력으로 실행한다. 실패하면 이미지는 저장하지 않는다.
    import time
    from pathlib import Path

    # 로컬 후처리 모듈. Modal 컨테이너에는 올리지 않으므로 여기서만 import 한다.
    import restore_composite

    if bool(person_image) != bool(garment_image):
        raise SystemExit(
            "--person-image 와 --garment-image 는 함께 지정해야 한다. 둘 다 생략하면 공식 예제로 실행한다."
        )
    if person_resize_mode not in PERSON_RESIZE_MODES:
        raise SystemExit(
            f"--person-resize-mode 는 {list(PERSON_RESIZE_MODES)} 중 하나여야 한다: {person_resize_mode!r}"
        )
    if garment_resize_mode not in GARMENT_RESIZE_MODES:
        raise SystemExit(
            f"--garment-resize-mode 는 {list(GARMENT_RESIZE_MODES)} 중 하나여야 한다: {garment_resize_mode!r}"
        )
    try:
        require_inference_steps(inference_steps)
    except InferenceCheckError as error:
        raise SystemExit(f"--{error}") from error
    custom = bool(person_image)
    description = garment_description.strip()
    if custom and not description:
        raise SystemExit(
            "사용자 입력 모드에서는 --garment-description 으로 상품 설명을 지정해야 한다."
        )
    description = description or DEFAULT_GARMENT_DESCRIPTION
    case_id = case_id or (time.strftime("custom-%Y%m%d-%H%M%S") if custom else OFFICIAL_CASE_ID)

    # GPU 를 띄우기 전에 로컬에서 끝낼 수 있는 검증을 먼저 한다.
    try:
        output_dir = case_output_dir(Path(__file__).resolve().parent / OUTPUT_DIR_NAME, case_id)
        # 이전 결과(예: real-01)를 덮어쓰지 않는다. 합성만 다시 할 때는 restore_composite.py 를 쓴다.
        if output_dir.exists() and any(output_dir.iterdir()):
            raise InferenceCheckError(
                f"{output_dir} 에 이전 결과가 있다. 덮어쓰지 않도록 다른 --case-id 를 지정한다."
            )
        payload = {}
        if custom:
            for role, path in (("person", person_image), ("garment", garment_image)):
                data, info = read_local_image(path, role)
                payload[role] = data
                role_mode = person_resize_mode if role == "person" else garment_resize_mode
                if role_mode == "pad":
                    info.pop("aspect_warning", None)
                print(f"input {role}: {Path(path).expanduser().resolve()} {info}")
            # 원격과 같은 함수로 여백을 미리 계산해 보여준다. 모델 입력은 원격에서 다시 만든다.
            _, planned = fit_person_image(
                decode_input_image(payload["person"], "person")[0], person_resize_mode
            )
            print(f"person fit (planned): {planned}")
            _, planned = fit_garment_image(
                decode_input_image(payload["garment"], "garment")[0], garment_resize_mode
            )
            print(f"garment fit (planned): {planned}")
    except InferenceCheckError as error:
        raise SystemExit(str(error)) from error
    print(
        f"case: {case_id} ({'custom' if custom else 'official example'}), "
        f"person-resize-mode: {person_resize_mode}, garment-resize-mode: {garment_resize_mode}, "
        f"inference-steps: {inference_steps}, garment: {description!r}"
    )

    started = time.perf_counter()
    try:
        result = run_tryon.remote(
            description,
            payload.get("person"),
            payload.get("garment"),
            person_resize_mode,
            garment_resize_mode,
            inference_steps,
        )
    except InferenceCheckError as error:
        # 전처리 검증 실패. 기준은 완화하지 않는다. 단계별 값은 원격 로그의 [FAIL] 줄에 있다.
        raise SystemExit(
            f"원격 검증 실패 (person-resize-mode={person_resize_mode}, "
            f"garment-resize-mode={garment_resize_mode}): {error}"
        ) from error
    finally:
        # 컨테이너 시작·이미지 준비까지 포함한 호출 전체 시간.
        print(f"IDM-VTON inference check wall time: {time.perf_counter() - started:.1f}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    images = result.pop("images")
    expected = {
        RESULT_FILE: ("JPEG", "RGB"),
        MASK_FILE: ("JPEG", "RGB"),
        INPAINT_MASK_FILE: ("PNG", "L"),
        PERSON_INPUT_FILE: ("PNG", "RGB"),
    }
    if set(images) != set(expected):
        raise SystemExit(f"반환된 이미지가 기대와 다르다: {sorted(images)}")
    for name, data in images.items():
        path = output_dir / name
        path.write_bytes(data)
        # 저장된 파일을 다시 읽어 확인한다. 파일이 정상이라는 뜻이지 가상피팅 품질이 좋다는 뜻은 아니다.
        info = restore_composite.inspect_file(path, *expected[name])
        print(
            f"saved {path} {info['format']} {info['mode']} {info['size'][0]}x{info['size'][1]} "
            f"{info['bytes']:,} bytes"
        )
    # 저장한 파일만으로 합성한다. 블렌딩 값을 바꿔 다시 합성할 때(restore_composite.py)와 같은 경로다.
    try:
        restore_report = restore_composite.restore_case_dir(output_dir)
    except restore_composite.RestoreError as error:
        raise SystemExit(f"추론 결과는 저장했지만 합성에 실패했다: {error}") from error
    restore_composite.print_report(restore_report)

    for role, info in result["inputs"].items():
        summary = {
            key: info[key] for key in ("format", "original_size", "exif_orientation", "size")
        }
        role_mode = person_resize_mode if role == "human" else garment_resize_mode
        warn = "aspect_warning" in info and role_mode == "stretch"
        print(f"input {role}: {summary}" + (f" WARN {info['aspect_warning']}" if warn else ""))
    for label, key in (("person", "person_fit"), ("garment", "garment_fit")):
        fit = result[key]
        print(
            f"{label} fit: mode {fit['mode']}, original {fit['original_size']} (aspect {fit['original_aspect']}), "
            f"resized {fit['resized_size']} (aspect {fit['resized_aspect']}), padding {fit['padding']}, "
            f"pad color {fit.get('pad_color')}, model input {fit['output_size']}"
        )
    preprocess = result["preprocess"]
    print(
        f"mask ratio: {preprocess['mask_ratio']}, inpaint mask ratio: {preprocess['inpaint_mask_ratio']}, "
        f"inpaint mask in padding: {preprocess['inpaint_mask_padding_ratio']}"
    )
    print(
        f"openpose (384x512): found {preprocess.get('pose_keypoints_found')}/18, "
        f"shoulders {preprocess['keypoints']}, all {preprocess.get('pose_keypoints')}"
    )
    print(
        f"densepose: instances {preprocess['densepose_instances']}, "
        f"scores {preprocess['densepose_scores']}, boxes {preprocess['densepose_boxes']}"
    )
    settings = result["settings"]
    print(
        f"settings: steps {settings['steps']}, seed {settings['seed']}, "
        f"guidance {settings['guidance_scale']}, strength {settings['strength']}"
    )
    timings = result["timings"]
    print(
        f"timings (s): preprocess {timings['preprocess']}, "
        f"model loading {timings['preprocess_model_loading'] + timings['idm_vton_model_loading']:.2f} "
        f"(preprocess models {timings['preprocess_model_loading']}, IDM-VTON {timings['idm_vton_model_loading']}), "
        f"inference {timings['inference']}, total {timings['total']}"
    )
    print(
        f"peak VRAM: allocated {result['peak_vram_allocated_mib']} MiB, "
        f"reserved {result['peak_vram_reserved_mib']} MiB on {result['gpu_name']}"
    )
    print(
        "파일 형식·해상도·합성 영역 검증만 통과했다. 품질은 result.jpg·restored_result.jpg·comparison.jpg 를 직접 열어 확인한다."
    )
