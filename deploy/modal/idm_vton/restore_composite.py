import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, UnidentifiedImageError

# IDM-VTON 은 사람 이미지 전체를 VAE 로 인코딩·디코딩해서 마스크 밖(얼굴·머리카락·하의·배경)도 원본보다 흐려진다.
# inpaint_mask.png 의 흰색(255)은 결과를, 검은색(0)은 원본을 쓰고 경계만 섞는다. Modal·GPU 없이 Mac 에서만 돈다.
#
#   .venv/bin/python deploy/modal/idm_vton/restore_composite.py deploy/modal/idm_vton/test_outputs/<case-id>

# inference_check.py 의 IMAGE_SIZE·파일 이름과 같아야 한다. 이 파일은 Modal 컨테이너에 올리지 않아 import 하지 않는다.
IMAGE_SIZE = (768, 1024)
PERSON_INPUT_FILE = "person_input.png"
RESULT_FILE = "result.jpg"
INPAINT_MASK_FILE = "inpaint_mask.png"
RESTORED_FILE = "restored_result.jpg"
COMPARISON_FILE = "comparison.jpg"
JPEG_QUALITY = 95
COMPARISON_LABEL_HEIGHT = 48

# 팽창은 의류 경계가 원본 쪽으로 잘리지 않게 마스크를 바깥으로 넓히는 픽셀 수, 블러는 Pillow GaussianBlur 의 radius 다.
# 상한은 원본 신체를 결과 픽셀로 과도하게 덮지 않게 둔 값이다.
DEFAULT_DILATE_PX = 3
DEFAULT_BLUR_RADIUS = 3.0
MAX_DILATE_PX = 15
MAX_BLUR_RADIUS = 10.0


class RestoreError(RuntimeError):
    """합성 입력 파일이 없거나 형식·해상도·마스크 값이 기대와 다르다."""


def encode_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def open_image(data: bytes, name: str, image_format: str, mode: str) -> Image.Image:
    """형식·모드·768×1024 를 확인한 뒤 연다. 리사이즈·변환은 하지 않는다(좌표계가 이미 같아야 한다)."""
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            info = (opened.format, opened.mode, opened.size)
            if info != (image_format, mode, IMAGE_SIZE):
                raise RestoreError(
                    f"{name}: {image_format} {mode} {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} 가 아니다: {info}"
                )
            return opened.copy()
    except (UnidentifiedImageError, OSError) as error:
        raise RestoreError(f"{name}: 이미지를 열 수 없다({type(error).__name__})") from error


def open_binary_mask(data: bytes) -> Image.Image:
    mask = open_image(data, INPAINT_MASK_FILE, "PNG", "L")
    values = {int(value) for value in np.unique(np.asarray(mask))}
    if values != {0, 255}:
        raise RestoreError(f"{INPAINT_MASK_FILE}: 0·255 두 값만 있어야 한다: {sorted(values)[:10]}")
    return mask


def inspect_file(path, image_format: str, mode: str, size: tuple[int, int] = IMAGE_SIZE) -> dict:
    """저장한 파일을 다시 읽어 형식·모드·해상도를 확인한다. 품질은 판단하지 않는다."""
    data = Path(path).read_bytes()
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            info = {"format": opened.format, "mode": opened.mode, "size": list(opened.size)}
    except (UnidentifiedImageError, OSError) as error:
        raise RestoreError(f"{path}: 이미지를 열 수 없다({type(error).__name__})") from error
    if (info["format"], info["mode"], tuple(info["size"])) != (image_format, mode, size):
        raise RestoreError(f"{path}: {image_format} {mode} {size} 가 아니다: {info}")
    return {**info, "bytes": len(data)}


def soften_mask(mask: Image.Image, dilate_px: int, blur_radius: float) -> Image.Image:
    """이진 마스크를 조금 넓힌 뒤 경계를 흐려 0~255 가중치로 바꾼다. 255 는 결과, 0 은 원본이다."""
    if not 0 <= dilate_px <= MAX_DILATE_PX:
        raise RestoreError(f"dilate_px {dilate_px} 가 [0, {MAX_DILATE_PX}] 밖이다")
    if not 0 <= blur_radius <= MAX_BLUR_RADIUS:
        raise RestoreError(f"blur_radius {blur_radius} 가 [0, {MAX_BLUR_RADIUS}] 밖이다")
    soft = mask
    if dilate_px:
        # MaxFilter 는 홀수 크기 정사각형이다. 흰색이 사방으로 dilate_px 만큼 넓어진다.
        soft = soft.filter(ImageFilter.MaxFilter(2 * dilate_px + 1))
    if blur_radius:
        soft = soft.filter(ImageFilter.GaussianBlur(blur_radius))
    return soft


def laplacian_variance(image: Image.Image, region: np.ndarray) -> float:
    """흑백 Laplacian 분산. 같은 영역에서 값이 클수록 경계가 또렷하다. 영역 비교용이지 절대 기준은 아니다."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    )
    inner = region[1:-1, 1:-1]
    return round(float(laplacian[inner].var()), 1) if inner.any() else 0.0


def composite_restored(
    person: Image.Image,
    result: Image.Image,
    mask: Image.Image,
    dilate_px: int = DEFAULT_DILATE_PX,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
) -> tuple[Image.Image, dict]:
    """흰색은 결과, 검은색은 원본 픽셀을 쓰고 경계는 섞는다. 영역별로 실제 픽셀을 대조해 확인한다."""
    sizes = {"person": person.size, "result": result.size, "mask": mask.size}
    if set(sizes.values()) != {IMAGE_SIZE}:
        raise RestoreError(f"세 이미지가 모두 {IMAGE_SIZE} 여야 한다: {sizes}")
    soft = soften_mask(mask, dilate_px, blur_radius)
    restored = Image.composite(result, person, soft)

    original, generated, output = (
        np.asarray(image, dtype=np.int16) for image in (person.convert("RGB"), result, restored)
    )
    white = np.asarray(mask) == 255
    weight = np.asarray(soft)
    keep = weight == 0
    replace = weight == 255
    blend = ~(keep | replace)
    keep_diff = int(np.abs(output - original)[keep].max()) if keep.any() else 0
    replace_diff = int(np.abs(output - generated)[replace].max()) if replace.any() else 0
    if keep_diff or replace_diff:
        raise RestoreError(
            f"원본 영역 차이 {keep_diff}, 결과 영역 차이 {replace_diff} 가 0 이 아니다"
        )
    # 섞인 픽셀은 원본과 결과 값 사이에 있어야 한다. 반올림 오차 1 을 허용한다.
    low = np.minimum(original, generated) - 1
    high = np.maximum(original, generated) + 1
    if blend.any() and ((output < low) | (output > high))[blend].any():
        raise RestoreError("경계 픽셀이 원본과 결과 사이 값이 아니다")
    if not white.any():
        raise RestoreError("마스크에 흰색(결과) 영역이 없다")

    black = ~white
    stats = {
        "dilate_px": dilate_px,
        "blur_radius": blur_radius,
        "mask_white_ratio": round(float(white.mean()), 4),
        "result_only_ratio": round(float(replace.mean()), 4),
        "original_only_ratio": round(float(keep.mean()), 4),
        "blend_ratio": round(float(blend.mean()), 4),
        # 흰색 영역의 결과 가중치. 최솟값이 1 에 가까울수록 의류 경계가 원본에 덜 섞인다.
        "white_mean_result_weight": round(float(weight[white].mean()) / 255, 4),
        "white_min_result_weight": round(float(weight[white].min()) / 255, 3),
        # 검은색(원본) 영역 중 결과 픽셀이 조금이라도 섞인 비율. 팽창·블러가 원본을 덮은 정도다.
        "black_spill_ratio": round(float((weight[black] > 0).mean()), 4) if black.any() else 0.0,
        "keep_max_diff": keep_diff,
        "replace_max_diff": replace_diff,
        # 원본만 쓰는 영역의 선명도. restored 는 original 과 같아야 하고, result 와의 차이가 개선 폭이다.
        "keep_sharpness": {
            "original": laplacian_variance(person, keep),
            "result": laplacian_variance(result, keep),
            "restored": laplacian_variance(restored, keep),
        },
    }
    return restored, stats


def make_comparison(panels: list[tuple[str, Image.Image]]) -> bytes:
    """768×1024 이미지들을 이름표와 함께 가로로 붙인다."""
    width, height = IMAGE_SIZE
    canvas = Image.new("RGB", (width * len(panels), height + COMPARISON_LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=28)
    for index, (label, panel) in enumerate(panels):
        canvas.paste(
            panel.convert("RGB").resize(IMAGE_SIZE), (index * width, COMPARISON_LABEL_HEIGHT)
        )
        draw.text((index * width + 16, 10), label, fill="black", font=font)
    return encode_jpeg(canvas)


def restore_case_dir(
    case_dir, dilate_px: int = DEFAULT_DILATE_PX, blur_radius: float = DEFAULT_BLUR_RADIUS
) -> dict:
    """case 디렉터리의 person_input.png·result.jpg·inpaint_mask.png 로 restored_result.jpg·comparison.jpg 를 만든다."""
    case_dir = Path(case_dir)
    required = (PERSON_INPUT_FILE, RESULT_FILE, INPAINT_MASK_FILE)
    missing = [name for name in required if not (case_dir / name).is_file()]
    if missing:
        raise RestoreError(
            f"{case_dir} 에 {missing} 가 없다. inference_check.py 를 새 --case-id 로 실행해 만든다."
        )
    person = open_image(
        (case_dir / PERSON_INPUT_FILE).read_bytes(), PERSON_INPUT_FILE, "PNG", "RGB"
    )
    result = open_image((case_dir / RESULT_FILE).read_bytes(), RESULT_FILE, "JPEG", "RGB")
    mask = open_binary_mask((case_dir / INPAINT_MASK_FILE).read_bytes())
    restored, stats = composite_restored(person, result, mask, dilate_px, blur_radius)

    comparison_size = (IMAGE_SIZE[0] * 3, IMAGE_SIZE[1] + COMPARISON_LABEL_HEIGHT)
    outputs = {
        RESTORED_FILE: (encode_jpeg(restored), IMAGE_SIZE),
        COMPARISON_FILE: (
            make_comparison(
                [
                    ("original (model input)", person),
                    ("IDM-VTON result", result),
                    (f"restored (dilate {dilate_px}, blur {blur_radius:g})", restored),
                ]
            ),
            comparison_size,
        ),
    }
    saved = {}
    for name, (data, size) in outputs.items():
        path = case_dir / name
        path.write_bytes(data)
        saved[str(path)] = inspect_file(path, "JPEG", "RGB", size)
    return {"stats": stats, "saved": saved}


def print_report(report: dict) -> None:
    for path, info in report["saved"].items():
        print(
            f"saved {path} {info['format']} {info['size'][0]}x{info['size'][1]} {info['bytes']:,} bytes"
        )
    stats = dict(report["stats"])
    sharpness = stats.pop("keep_sharpness")
    print(f"restore: {stats}")
    print(
        "keep 영역 선명도(Laplacian 분산): "
        f"original {sharpness['original']}, result {sharpness['result']}, restored {sharpness['restored']}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="GPU 추론 없이 test_outputs/<case-id> 의 결과와 원본을 이진 마스크로 다시 합성한다."
    )
    parser.add_argument(
        "case_dir", help="person_input.png·result.jpg·inpaint_mask.png 가 있는 디렉터리"
    )
    parser.add_argument("--dilate-px", type=int, default=DEFAULT_DILATE_PX)
    parser.add_argument("--blur-radius", type=float, default=DEFAULT_BLUR_RADIUS)
    args = parser.parse_args(argv)
    try:
        report = restore_case_dir(args.case_dir, args.dilate_px, args.blur_radius)
    except RestoreError as error:
        raise SystemExit(str(error)) from error
    print_report(report)


if __name__ == "__main__":
    main()
