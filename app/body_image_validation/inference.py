"""요청 사이에 재사용하는 전신 이미지 모델 묶음(MediaPipe Person/Pose·rembg session·GPU 전처리).

S3 client·secret 설정(app.config.settings)·limiter 와 분리한 모듈이다. 이 모듈과 이 모듈이 import 하는
모듈은 settings 를 읽지 않으므로, Modal worker 가 snapshot 단계에서 모델만 만들어도 secret 이나 연결 객체가
함께 담기지 않는다. S3 client·service(limiter)·runtime 등록은 runtime.BodyImageRuntime 이 맡는다.

모델 객체 안의 threading.Lock(detector·rembg·nvJPEG decoder)은 모델과 함께 snapshot 에 담겨도 된다.
요청이 없을 때(잠기지 않은 상태) 만들어지고, Lock 은 OS 자원을 쥐지 않는 프로세스 메모리 상태라
복원된 프로세스에서도 그대로 쓸 수 있다. 요청 중에 snapshot 을 만들지 않는 것이 전제다.
"""

import logging
from pathlib import Path

from app.body_image_validation.background import RembgBackgroundRemover
from app.body_image_validation.detectors import (
    MediaPipePersonDetector,
    MediaPipePoseDetector,
    PersonDetector,
    PoseDetector,
)
from app.body_image_validation.gpu_preprocessing import (
    GpuImagePreprocessor,
    GpuJpegDecoder,
    GpuPreprocessingUnavailableError,
)
from app.config import body_image_validation as config

logger = logging.getLogger(__name__)


class BodyImageRuntimeError(RuntimeError):
    """모델 파일 또는 runtime 초기화 실패."""


def check_model_files() -> None:
    paths = (
        config.PERSON_DETECTOR_MODEL,
        config.POSE_LANDMARKER_MODEL,
        config.REMBG_MODEL,
    )
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise BodyImageRuntimeError(f"전신 이미지 모델 파일이 없습니다: {missing}")


def build_background_remover() -> RembgBackgroundRemover:
    """rembg session. REMBG_PROVIDERS 가 CUDA 면 여기서 CUDA session(과 CUDA context)이 만들어진다."""
    options = {"enable_cpu_mem_arena": config.REMBG_ENABLE_CPU_MEM_ARENA}
    if config.REMBG_PROVIDERS:
        options["providers"] = list(config.REMBG_PROVIDERS)
    if config.REMBG_PROFILE_RUNS > 0:  # 진단용(기본 꺼짐)
        options["profile_runs"] = config.REMBG_PROFILE_RUNS
    return RembgBackgroundRemover(str(config.REMBG_MODEL), **options)


def build_gpu_preprocessor() -> GpuImagePreprocessor | None:
    """GPU 경로는 선택 사항이다. 켰는데 쓸 수 없으면 경고만 남기고 Pillow 로만 동작한다."""
    if not config.GPU_PREPROCESSING:
        return None
    try:
        return GpuImagePreprocessor(GpuJpegDecoder())
    except GpuPreprocessingUnavailableError as error:
        logger.warning("body image GPU preprocessing is unavailable, using Pillow only: %s", error)
        return None


class BodyImageModels:
    """모델만 담는다. 연결·secret·요청 상태(limiter)는 담지 않는다.

    detector 는 close() 가 닫는다. 여러 곳(초기화 실패 정리·runtime 종료)에서 불려도 한 번만 닫는다.
    """

    def __init__(
        self,
        person_detector: PersonDetector,
        pose_detector: PoseDetector,
        background_remover: RembgBackgroundRemover,
        gpu_preprocessor: GpuImagePreprocessor | None = None,
    ) -> None:
        self.person_detector = person_detector
        self.pose_detector = pose_detector
        self.background_remover = background_remover
        # None 이면 Pillow 로만 처리한다.
        self.gpu_preprocessor = gpu_preprocessor
        self._closed = False

    @classmethod
    def load(
        cls,
        person_detector: MediaPipePersonDetector | None = None,
        pose_detector: MediaPipePoseDetector | None = None,
    ) -> "BodyImageModels":
        """모델 파일을 확인하고 없는 모델을 만든다. Modal worker 는 MediaPipe detector 를 prepare_cpu 에서
        먼저 만들어 넘긴다.

        소유권: 넘긴 detector 는 호출한 순간부터 이 메서드가 소유한다. 성공하면 돌려준 BodyImageModels 가,
        실패하면(모델 파일 없음 포함) 이 메서드가 넘긴 것과 여기서 만든 것을 모두 닫는다. 호출자는 실패 뒤
        넘긴 detector 를 다시 쓰거나 닫지 않는다.

        모델 파일 없음·MediaPipe·rembg 생성 실패는 BodyImageRuntimeError 다. GPU 전처리는 쓸 수 없으면 None 이고,
        그 밖의 코드 오류는 감싸지 않고 그대로 낸다.
        """
        owned = [detector for detector in (person_detector, pose_detector) if detector is not None]
        try:
            check_model_files()
            try:
                if person_detector is None:
                    person_detector = MediaPipePersonDetector(str(config.PERSON_DETECTOR_MODEL))
                    owned.append(person_detector)
                if pose_detector is None:
                    pose_detector = MediaPipePoseDetector(str(config.POSE_LANDMARKER_MODEL))
                    owned.append(pose_detector)
                background_remover = build_background_remover()
            # MediaPipe, ONNX Runtime 이 서로 다른 예외 계층을 쓰므로 초기화 경계에서 하나로 묶는다.
            except Exception as error:
                raise BodyImageRuntimeError("전신 이미지 runtime 초기화에 실패했습니다.") from error
            gpu_preprocessor = build_gpu_preprocessor()
        except BaseException:
            _close_quietly(owned)
            raise
        return cls(person_detector, pose_detector, background_remover, gpu_preprocessor)

    def close(self) -> None:
        """detector 를 닫는다. 여러 번 불려도 한 번만 닫고, 하나가 실패해도 나머지는 닫는다."""
        if self._closed:
            return
        self._closed = True
        _close_quietly([self.person_detector, self.pose_detector])


def _close_quietly(resources) -> None:
    """정리 중 오류는 남기기만 한다. 원래 예외(초기화 실패)를 가리거나 다음 자원 정리를 막지 않는다."""
    for resource in resources:
        try:
            resource.close()
        except Exception:
            logger.warning("failed to close %s", type(resource).__name__, exc_info=True)
