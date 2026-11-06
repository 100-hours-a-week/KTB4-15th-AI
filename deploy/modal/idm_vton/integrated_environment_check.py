import modal
from detectron2_build_check import (
    CUDA_ARCH,
    DETECTRON2_DIR,
    EXPECTED_CUDA_RUNTIME,
    EXPECTED_DETECTRON2_VERSION,
    FVCORE_VERSION,
    GPU,
    GRADIO_DEMO_DIR,
    NUMPY_VERSION,
    PYTHON_VERSION,
    REPO_DIR,
    SETUPTOOLS_VERSION,
    TORCH_INDEX_URL,
    TORCH_VERSION,
    TORCHVISION_VERSION,
)
from detectron2_build_check import image as detectron2_image

APP_NAME = "lookddak-idm-vton-integrated-environment-check"
CHECK_TIMEOUT_SECONDS = 600
TORCHAUDIO_VERSION = "2.0.2+cu118"

# 빌드 이미지에 이미 깔린 버전을 같은 pip 명령에 다시 적어, 아래 패키지를 풀면서 해석기가 torch·numpy 를 바꾸지 못하게 한다.
PINNED_BASE_PACKAGES = [
    f"torch=={TORCH_VERSION}",
    f"torchvision=={TORCHVISION_VERSION}",
    f"torchaudio=={TORCHAUDIO_VERSION}",
    f"numpy=={NUMPY_VERSION}",
    f"setuptools=={SETUPTOOLS_VERSION}",
    f"fvcore=={FVCORE_VERSION}",
]
# dependency_check.py 와 같은 목록·버전. 버전이 없던 것 중 근거가 있는 것만 고정했다.
DEPENDENCY_CHECK_PACKAGES = [
    "accelerate==0.25.0",
    "torchmetrics==1.2.1",
    "tqdm==4.66.1",
    "transformers==4.36.2",
    "diffusers==0.25.0",
    "einops==0.7.0",
    "scipy==1.11.1",
    "onnxruntime==1.16.2",
    "opencv-python-headless==4.8.1.78",
    "huggingface_hub==0.25.2",
    "cloudpickle",
    # 그대로 두면 2.4.0 이 깔린다. upstream detectron2 는 omegaconf>=2.1,<2.4 로 묶는다.
    "omegaconf==2.3.0",
    "pycocotools",
    "basicsr==1.4.2",
    "av",
]
# 다섯 모듈을 import 할 때 모듈 레벨에서 필요하지만 dependency_check.py 에 없던 것.
IMPORT_TIME_PACKAGES = [
    # detectron2/utils/visualizer.py 가 import 시점에 쓴다. 공식 environment.yaml 에서는 gradio 가 끌어왔다.
    "matplotlib",
    # preprocess/openpose/annotator/openpose/hand.py 가 import 시점에 쓴다. basicsr 도 끌어오지만 직접 쓰므로 적는다.
    "scikit-image",
    # detectron2 v0.6 setup.py 의 iopath>=0.1.7,<0.1.10. fvcore 만 두면 0.1.10 이 깔린다.
    "iopath==0.1.9",
    # basicsr 가 opencv-python 을 버전 없이 요구해 headless 와 같은 cv2/ 를 덮어쓴다. 어느 쪽이 남든 같은 버전이 되게 맞춘다.
    "opencv-python==4.8.1.78",
]
EXPECTED_DISTRIBUTIONS = dict(
    spec.split("==", 1)
    for spec in [*PINNED_BASE_PACKAGES, *DEPENDENCY_CHECK_PACKAGES, *IMPORT_TIME_PACKAGES]
    if "==" in spec
)
EXPECTED_CV2_VERSION = "4.8.1"

image = (
    detectron2_image
    # dependency_check.py 와 같다. opencv-python(GUI 빌드)의 cv2 가 남으면 libGL·glib 이 필요하다.
    .apt_install("libglib2.0-0", "libgl1")
    .pip_install(
        *PINNED_BASE_PACKAGES,
        *DEPENDENCY_CHECK_PACKAGES,
        *IMPORT_TIME_PACKAGES,
        extra_index_url=TORCH_INDEX_URL,
    )
    # Modal 1.x 는 함수가 정의된 파일만 컨테이너에 올린다. 이 파일이 import 하는 detectron2_build_check 를 함께 올리지 않으면
    # 컨테이너가 이 모듈을 불러올 때 ModuleNotFoundError 가 난다. 빌드 단계 뒤에는 둘 수 없어 마지막에 둔다.
    .add_local_python_source("detectron2_build_check")
)

app = modal.App(APP_NAME)

# (모듈, 가져올 이름, 그 이름이 정의돼 있어야 하는 디렉터리). detectron2·densepose 는 gradio_demo 쪽을 써야 한다.
IMPORT_TARGETS = [
    ("preprocess.openpose.run_openpose", "OpenPose", f"{REPO_DIR}/preprocess/openpose"),
    ("preprocess.humanparsing.run_parsing", "Parsing", f"{REPO_DIR}/preprocess/humanparsing"),
    ("detectron2", "_C", DETECTRON2_DIR),
    ("densepose", "add_densepose_config", f"{GRADIO_DEMO_DIR}/densepose"),
    ("src.tryon_pipeline", "StableDiffusionXLInpaintPipeline", f"{REPO_DIR}/src"),
]
# import 만으로 가중치가 내려오지 않았는지 볼 위치. OpenPose 는 ckpt/openpose/ckpts 로, HF 는 ~/.cache/huggingface 로 받는다.
DOWNLOAD_WATCH_DIRS = [f"{REPO_DIR}/ckpt", "/root/.cache/huggingface", "/root/.cache/torch"]
# 새 파일이 생겼다는 것만으로는 다운로드로 보지 않는다. huggingface_hub·diffusers 는 import 만으로
# hub/version.txt·version_diffusers_cache.txt 같은 캐시 메타데이터를 쓴다. 아래에 해당할 때만 모델 아티팩트로 본다.
MODEL_ARTIFACT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".pkl",  # DensePose 체크포인트(model_final_*.pkl)
    ".h5",
    ".msgpack",
    ".gguf",
}
# HF 허브 캐시는 실제 파일을 blobs/ 에 해시 이름(받는 중이면 .incomplete)으로 두고 snapshots/ 에서 가리킨다. 확장자로는 못 거른다.
HF_MODEL_CACHE_DIRS = {"snapshots", "blobs"}
# huggingface_hub·transformers·datasets 가 import 시점에 읽는다. 어떤 경로로든 허브 다운로드를 시도하면 실패하게 한다.
HF_OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


class IntegratedEnvironmentCheckError(RuntimeError):
    """통합 이미지의 버전·Detectron2 확장·모듈 import·패키지 의존성 중 하나 이상이 기대와 다르다."""


@app.function(image=image, gpu=GPU, timeout=CHECK_TIMEOUT_SECONDS)
def check_integrated_environment() -> dict:
    import importlib
    import importlib.metadata
    import inspect
    import os
    import platform
    import re
    import subprocess
    import sys
    import sysconfig
    from pathlib import Path

    # HF 라이브러리를 import 하기 전에 둬야 적용된다.
    os.environ.update(HF_OFFLINE_ENV)

    import torch
    import torchvision

    results: list[dict] = []

    def record(item: int, name: str, check) -> None:
        try:
            detail = check()
        except Exception as error:  # noqa: BLE001 — 한 항목이 실패해도 나머지 결과를 모두 출력한 뒤 끝에서 실패시킨다
            passed, detail = False, f"{type(error).__name__}: {error}"
        else:
            passed = True
        print(f"[{'PASS' if passed else 'FAIL'}] {item}. {name}: {detail}", flush=True)
        results.append({"item": item, "name": name, "passed": passed, "detail": str(detail)})

    def check_versions() -> str:
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
        nvcc_release = re.search(r"release (\d+\.\d+)", nvcc.stdout)
        actual = {
            "Python": platform.python_version().rsplit(".", 1)[0],
            "PyTorch": torch.__version__,
            "TorchVision": torchvision.__version__,
            "PyTorch CUDA runtime": torch.version.cuda,
            "nvcc": nvcc_release.group(1) if nvcc_release else nvcc.stderr.strip(),
        }
        expected = {
            "Python": PYTHON_VERSION,
            "PyTorch": TORCH_VERSION,
            "TorchVision": TORCHVISION_VERSION,
            "PyTorch CUDA runtime": EXPECTED_CUDA_RUNTIME,
            "nvcc": EXPECTED_CUDA_RUNTIME,
        }
        mismatched = {k: (actual[k], expected[k]) for k in expected if actual[k] != expected[k]}
        if mismatched:
            raise IntegratedEnvironmentCheckError(f"(실제, 기대): {mismatched}")
        return str(actual)

    def check_gpu() -> str:
        if not torch.cuda.is_available():
            raise IntegratedEnvironmentCheckError("torch.cuda.is_available() 가 False 다")
        name = torch.cuda.get_device_name(0)
        capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
        if GPU not in name or capability != CUDA_ARCH:
            raise IntegratedEnvironmentCheckError(
                f"{GPU}(compute capability {CUDA_ARCH}) 가 아니다: {name} ({capability})"
            )
        return f"{name} (compute capability {capability})"

    def check_pinned_distributions() -> str:
        mismatched = {}
        for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                actual = None
            if actual != expected:
                mismatched[distribution] = (actual, expected)
        # opencv-python 과 headless 가 cv2/ 를 공유하므로 실제로 불리는 cv2 의 버전도 본다.
        import cv2

        if cv2.__version__ != EXPECTED_CV2_VERSION:
            mismatched["cv2"] = (cv2.__version__, EXPECTED_CV2_VERSION)
        if mismatched:
            raise IntegratedEnvironmentCheckError(f"(실제, 기대): {mismatched}")
        return f"{len(EXPECTED_DISTRIBUTIONS)}개 고정 버전 유지, cv2 {cv2.__version__}"

    # gradio_demo/app.py 를 저장소 루트에서 실행할 때와 같은 순서: 스크립트 디렉터리(gradio_demo) 다음 저장소 루트.
    # run_openpose·run_parsing 은 import 될 때 자기 디렉터리를 sys.path 맨 앞에 더 넣는다.
    sys.path[:0] = [GRADIO_DEMO_DIR, REPO_DIR]

    def check_detectron2_version() -> str:
        import detectron2

        package_dir = Path(detectron2.__file__).resolve().parent
        if package_dir != Path(DETECTRON2_DIR):
            raise IntegratedEnvironmentCheckError(
                f"gradio_demo 가 아닌 detectron2 가 불렸다: {package_dir}"
            )
        if detectron2.__version__ != EXPECTED_DETECTRON2_VERSION:
            raise IntegratedEnvironmentCheckError(
                f"detectron2 {EXPECTED_DETECTRON2_VERSION} 이 아니다: {detectron2.__version__}"
            )
        return f"{detectron2.__version__} ({package_dir})"

    def check_detectron2_extension() -> str:
        expected_binary = Path(DETECTRON2_DIR) / f"_C{sysconfig.get_config_var('EXT_SUFFIX')}"
        binaries = sorted(Path(DETECTRON2_DIR).glob("_C*.so"))
        if binaries != [expected_binary]:
            raise IntegratedEnvironmentCheckError(
                f"{expected_binary.name} 하나만 있어야 한다: {binaries}"
            )
        from detectron2 import _C

        extension_path = Path(_C.__file__).resolve()
        cuda_version = _C.get_cuda_version()
        if extension_path != expected_binary.resolve():
            raise IntegratedEnvironmentCheckError(f"다른 위치의 _C 가 불렸다: {extension_path}")
        if not _C.has_cuda() or cuda_version != f"CUDA {EXPECTED_CUDA_RUNTIME}":
            raise IntegratedEnvironmentCheckError(
                f"_C 가 CUDA {EXPECTED_CUDA_RUNTIME} 빌드가 아니다: {cuda_version}"
            )
        # 통합 레이어의 pip 설치 뒤에도 _C 의 CUDA 커널이 이 GPU 에서 도는지 연산 하나로 확인한다.
        boxes1 = torch.tensor([[10.0, 10.0, 8.0, 4.0, 0.0], [20.0, 20.0, 6.0, 6.0, 45.0]])
        boxes2 = torch.tensor([[10.0, 10.0, 8.0, 4.0, 30.0], [21.0, 20.0, 6.0, 6.0, 45.0]])
        iou_cpu = torch.ops.detectron2.box_iou_rotated(boxes1, boxes2)
        iou_gpu = torch.ops.detectron2.box_iou_rotated(boxes1.cuda(), boxes2.cuda()).cpu()
        if not torch.allclose(iou_gpu, iou_cpu, atol=1e-5):
            raise IntegratedEnvironmentCheckError(f"box_iou_rotated GPU {iou_gpu} != CPU {iou_cpu}")
        return f"{extension_path.name}, {cuda_version}, {_C.get_compiler_version()}"

    module_files: dict[str, str] = {}

    def import_target(module_name: str, attribute: str, expected_dir: str):
        def check() -> str:
            module = importlib.import_module(module_name)
            # `from detectron2 import _C` 처럼 하위 모듈이면 속성이 아직 없을 수 있다.
            target = getattr(module, attribute, None) or importlib.import_module(
                f"{module_name}.{attribute}"
            )
            target_file = Path(inspect.getfile(target)).resolve()
            if not target_file.is_relative_to(expected_dir):
                raise IntegratedEnvironmentCheckError(
                    f"{expected_dir} 밖에서 불렸다: {target_file}"
                )
            module_files[f"{module_name}.{attribute}"] = str(target_file)
            return str(target_file)

        return check

    def watched_files() -> set[str]:
        return {
            str(path)
            for directory in map(Path, DOWNLOAD_WATCH_DIRS)
            if directory.is_dir()
            for path in directory.rglob("*")
            if path.is_file()
        }

    def check_offline_mode() -> str:
        # 환경 변수를 넣기 전에 huggingface_hub 가 먼저 import 됐다면 상수에 반영되지 않는다.
        from huggingface_hub import constants

        if not constants.HF_HUB_OFFLINE:
            raise IntegratedEnvironmentCheckError(
                "huggingface_hub 에 HF_HUB_OFFLINE 이 적용되지 않았다"
            )
        return str({name: os.environ.get(name) for name in HF_OFFLINE_ENV})

    def is_model_artifact(path: Path) -> bool:
        return path.suffix.lower() in MODEL_ARTIFACT_SUFFIXES or not HF_MODEL_CACHE_DIRS.isdisjoint(
            path.parts
        )

    def check_no_model_downloads() -> str:
        created = sorted(watched_files() - files_before)
        artifacts = [path for path in created if is_model_artifact(Path(path))]
        if artifacts:
            raise IntegratedEnvironmentCheckError(f"import 중 모델 아티팩트가 생겼다: {artifacts}")
        return f"모델 아티팩트 없음. 허용한 캐시 메타데이터 {len(created)}개: {created}"

    def check_pip() -> str:
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
        )
        output = (pip_check.stdout + pip_check.stderr).strip()
        if pip_check.returncode != 0:
            raise IntegratedEnvironmentCheckError(output)
        return output

    record(1, "Python·PyTorch·CUDA 버전", check_versions)
    record(1, "GPU", check_gpu)
    record(1, "고정 패키지 버전 유지", check_pinned_distributions)
    record(2, "Detectron2 0.6", check_detectron2_version)
    record(3, "Detectron2 _C 확장", check_detectron2_extension)
    files_before = watched_files()
    for module_name, attribute, expected_dir in IMPORT_TARGETS:
        record(
            4,
            f"from {module_name} import {attribute}",
            import_target(module_name, attribute, expected_dir),
        )
    record(4, "Hugging Face 오프라인 모드", check_offline_mode)
    record(4, "import 중 모델 가중치 다운로드 없음", check_no_model_downloads)
    record(5, "pip check", check_pip)

    failed = [f"{r['item']}. {r['name']}" for r in results if not r["passed"]]
    if failed:
        raise IntegratedEnvironmentCheckError(f"실패한 항목: {failed}")

    print("=== Integrated Environment Check Passed ===", flush=True)
    return {
        "python": str(platform.python_version()),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "module_files": {str(k): str(v) for k, v in module_files.items()},
        "checks": [
            {"item": int(r["item"]), "name": str(r["name"]), "passed": bool(r["passed"])}
            for r in results
        ],
    }


@app.local_entrypoint()
def main() -> None:
    # 하나라도 실패하면 원격 예외가 그대로 올라와 modal run 이 0 이 아닌 코드로 끝난다.
    result = check_integrated_environment.remote()
    print(f"IDM-VTON integrated environment check result: {result}")
