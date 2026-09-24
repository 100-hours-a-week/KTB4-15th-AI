"""Docker build에서 전신 이미지 모델을 checksum 검증하며 미리 내려받는다."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

MODELS = {
    "efficientdet_lite0.tflite": (
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        + "efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
        "0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb",
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        + "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
    "u2netp.onnx": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
    ),
}


def download(url: str, destination: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"모델 checksum 불일치: {destination.name}")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for filename, (url, checksum) in MODELS.items():
        destination = args.destination / filename
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == checksum:
            continue
        download(url, destination, checksum)


if __name__ == "__main__":
    main()
