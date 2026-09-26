"""실제 전신 이미지 모델을 로컬 이미지에 실행한다."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LocalImageStorage:
    def __init__(self, destination: Path) -> None:
        self.destination = destination

    def upload_bytes(self, body: bytes, key: str, content_type: str) -> str:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination.write_bytes(body)
        return key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/body_image_validation")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("REMBG_HOME", str(args.model_dir.resolve()))

    from app.body_image_validation.background import RembgBackgroundRemover
    from app.body_image_validation.detectors import MediaPipePersonDetector, MediaPipePoseDetector
    from app.body_image_validation.exceptions import BodyImageValidationError
    from app.body_image_validation.service import BodyImageValidationService

    output = args.output or args.image.with_name(f"{args.image.stem}-validated.png")
    person_detector = MediaPipePersonDetector(
        str(args.model_dir / "efficientdet_lite0.tflite")
    )
    pose_detector = MediaPipePoseDetector(str(args.model_dir / "pose_landmarker_lite.task"))
    background_remover = RembgBackgroundRemover(str(args.model_dir / "u2netp.onnx"))

    try:
        service = BodyImageValidationService(
            person_detector,
            pose_detector,
            background_remover,
            LocalImageStorage(output),
        )
        service.validate(args.image.read_bytes())
    except BodyImageValidationError as error:
        print(f"검증 실패: {error.reason_code} - {error.reason or error.message}")
        return 1
    finally:
        person_detector.close()
        pose_detector.close()

    print("검증 성공")
    print(f"배경 제거 결과: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
