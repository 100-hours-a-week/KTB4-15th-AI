import modal

APP_NAME = "lookddak-idm-vton-detectron2-build-check"
GPU = "L40S"
CHECK_TIMEOUT_SECONDS = 600

CUDA_BASE_IMAGE = "nvidia/cuda:11.8.0-devel-ubuntu22.04"
PYTHON_VERSION = "3.10"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu118"
TORCH_VERSION = "2.0.1+cu118"
TORCHVISION_VERSION = "0.15.2+cu118"
NUMPY_VERSION = "1.24.4"
EXPECTED_CUDA_RUNTIME = "11.8"
# torch 2.0.1 의 cpp_extension 은 `from pkg_resources import packaging` 을 쓰는데 setuptools 70 에서 사라졌다.
SETUPTOOLS_VERSION = "69.5.1"
NINJA_VERSION = "1.11.1.1"
# detectron2/__init__.py 가 import 시점에 fvcore 를 요구한다.
FVCORE_VERSION = "0.1.5.post20221221"

SOURCE_REPO = "https://github.com/yisol/IDM-VTON.git"
SOURCE_COMMIT = "06ad89ba2f78d77a5ba21d7f7fd392254971fda2"
REPO_DIR = "/opt/IDM-VTON"
# gradio_demo/apply_net.py 가 import 하는 v0.6 소스만 빌드한다.
# preprocess/humanparsing/mhp_extension/detectron2 는 v0.1.3 이라 csrc 연산 구성(ROIAlign 유무, cocoeval 없음)이 달라 섞지 않는다.
GRADIO_DEMO_DIR = f"{REPO_DIR}/gradio_demo"
DETECTRON2_DIR = f"{GRADIO_DEMO_DIR}/detectron2"
EXPECTED_DETECTRON2_VERSION = "0.6"

CUDA_HOME = "/usr/local/cuda"
CUDA_ARCH = "8.9"  # L40S (Ada, sm_89). CUDA 11.8 부터 지원한다.
# nvcc 프로세스 하나가 deform_conv 컴파일에 수 GiB 를 쓴다. memory 는 request 일 뿐이라 MAX_JOBS 로 동시 컴파일 수를 직접 묶는다.
BUILD_MAX_JOBS = 4
BUILD_CPU = 8
BUILD_MEMORY_MIB = 16384
BUILD_TIMEOUT_SECONDS = 3600
BUILD_ENV = {
    "FORCE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": CUDA_ARCH,
    "CUDA_HOME": CUDA_HOME,
    "MAX_JOBS": str(BUILD_MAX_JOBS),
    # add_python 이 설치하는 python-build-standalone 의 sysconfig 는 CC/LDSHARED 가 clang 인데 이 이미지에는 clang 이 없다.
    "CC": "gcc",
    "CXX": "g++",
    "LDSHARED": "g++ -pthread -shared",
}
# detectron2 v0.6 setup.py 와 같은 nvcc 플래그. CC 가 정해져 있으면 원본도 -ccbin 을 붙인다.
NVCC_FLAGS = [
    "-O3",
    "-DCUDA_HAS_FP16=1",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    f"-ccbin={BUILD_ENV['CC']}",
]


class Detectron2BuildCheckError(RuntimeError):
    """Detectron2 C++/CUDA 확장이 Python 3.10·PyTorch 2.0.1·sm_89 용으로 빌드되지 않았거나 불러올 수 없다."""


def extension_sources(csrc_dir: str) -> list[str]:
    """detectron2 v0.6 setup.py 의 get_extensions() 와 같은 규칙으로 컴파일할 소스를 모은다."""
    from pathlib import Path

    csrc = Path(csrc_dir)
    cpp_sources = sorted(csrc.glob("*/*.cpp"))
    cuda_sources = sorted(csrc.glob("*/*.cu")) + sorted(csrc.glob("*.cu"))
    return [str(path) for path in [csrc / "vision.cpp", *cpp_sources, *cuda_sources]]


def build_detectron2_extension() -> None:
    """이미지 빌드 단계에서 gradio_demo/detectron2 의 _C 확장을 현재 Python·PyTorch 로 컴파일해 제자리에 둔다."""
    import os
    import shutil
    import subprocess
    import sysconfig
    from pathlib import Path

    import torch
    from setuptools import setup
    from torch.utils import cpp_extension

    subprocess.run(["nvcc", "--version"], check=True)
    subprocess.run([BUILD_ENV["CXX"], "--version"], check=True)
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})", flush=True)
    print(f"torch CUDA_HOME: {cpp_extension.CUDA_HOME}", flush=True)
    print({name: os.environ.get(name) for name in BUILD_ENV}, flush=True)
    if cpp_extension.CUDA_HOME != CUDA_HOME:
        raise Detectron2BuildCheckError(
            f"torch 가 찾은 CUDA_HOME 이 다르다: {cpp_extension.CUDA_HOME}"
        )

    detectron2_dir = Path(DETECTRON2_DIR)
    # 저장소에 들어 있는 Python 3.9 용 바이너리는 쓰지 않는다. 남겨 두면 검증에서 어느 쪽이 불렸는지 흐려진다.
    for stale in detectron2_dir.glob("_C*.so"):
        print(f"remove prebuilt binary: {stale}", flush=True)
        stale.unlink()

    csrc_dir = detectron2_dir / "layers" / "csrc"
    sources = extension_sources(str(csrc_dir))
    print("sources:\n  " + "\n  ".join(sources), flush=True)
    extension = cpp_extension.CUDAExtension(
        "detectron2._C",
        sources,
        include_dirs=[str(csrc_dir)],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={"cxx": [], "nvcc": NVCC_FLAGS},
    )

    # build_ext --inplace 는 cwd 기준 detectron2/ 패키지 자리에 _C<EXT_SUFFIX> 를 쓴다.
    os.chdir(GRADIO_DEMO_DIR)
    try:
        setup(
            name="idm-vton-detectron2-ext",
            version=EXPECTED_DETECTRON2_VERSION,
            ext_modules=[extension],
            cmdclass={"build_ext": cpp_extension.BuildExtension},
            script_args=["build_ext", "--inplace"],
        )
    except SystemExit as error:
        # setuptools 는 컴파일 실패를 SystemExit("error: ...") 로 알린다.
        raise Detectron2BuildCheckError(f"Detectron2 확장 빌드 실패: {error}") from error
    shutil.rmtree(Path(GRADIO_DEMO_DIR) / "build", ignore_errors=True)

    expected = detectron2_dir / f"_C{sysconfig.get_config_var('EXT_SUFFIX')}"
    built = sorted(detectron2_dir.glob("_C*.so"))
    print(f"built: {built}", flush=True)
    if built != [expected]:
        raise Detectron2BuildCheckError(f"빌드 결과가 기대와 다르다(실제 {built}, 기대 {expected})")


image = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python=PYTHON_VERSION)
    .apt_install("git", "build-essential")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"numpy=={NUMPY_VERSION}",
        extra_index_url=TORCH_INDEX_URL,
    )
    .pip_install(
        f"setuptools=={SETUPTOOLS_VERSION}",
        f"ninja=={NINJA_VERSION}",
        f"fvcore=={FVCORE_VERSION}",
    )
    .run_commands(
        f"git clone {SOURCE_REPO} {REPO_DIR}",
        f"git -C {REPO_DIR} checkout {SOURCE_COMMIT}",
    )
    .run_function(
        build_detectron2_extension,
        env=BUILD_ENV,
        cpu=BUILD_CPU,
        memory=BUILD_MEMORY_MIB,
        timeout=BUILD_TIMEOUT_SECONDS,
    )
)

app = modal.App(APP_NAME)


@app.function(image=image, gpu=GPU, timeout=CHECK_TIMEOUT_SECONDS)
def check_detectron2_build() -> dict:
    import platform
    import subprocess
    import sys
    import sysconfig
    from pathlib import Path

    import torch

    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
    print(nvcc.stdout, flush=True)
    if nvcc.returncode != 0 or f"release {EXPECTED_CUDA_RUNTIME}" not in nvcc.stdout:
        raise Detectron2BuildCheckError(
            f"nvcc {EXPECTED_CUDA_RUNTIME} 가 아니다: {nvcc.stdout or nvcc.stderr}"
        )

    python_version = platform.python_version()
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    print(f"Python: {python_version} (EXT_SUFFIX {ext_suffix})", flush=True)
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})", flush=True)
    expected = {
        "Python": (python_version.rsplit(".", 1)[0], PYTHON_VERSION),
        "PyTorch": (torch.__version__, TORCH_VERSION),
        "PyTorch CUDA runtime": (torch.version.cuda, EXPECTED_CUDA_RUNTIME),
    }
    mismatched = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
    if mismatched:
        raise Detectron2BuildCheckError(f"버전이 기대와 다르다(실제, 기대): {mismatched}")

    binaries = sorted(Path(DETECTRON2_DIR).glob("_C*.so"))
    print(f"Detectron2 binaries: {[str(path) for path in binaries]}", flush=True)
    if [path.name for path in binaries] != [f"_C{ext_suffix}"]:
        raise Detectron2BuildCheckError(f"_C{ext_suffix} 하나만 있어야 한다: {binaries}")
    binary = binaries[0]

    # 빌드한 .so 에 sm_89 cubin 이 실제로 들어 있는지 본다. 없으면 L40S 에서 커널 실행 시점에야 실패한다.
    target_sm = f"sm_{CUDA_ARCH.replace('.', '')}"
    elf_list = subprocess.run(
        ["cuobjdump", "--list-elf", str(binary)], capture_output=True, text=True, check=False
    )
    print(elf_list.stdout, flush=True)
    if target_sm not in elf_list.stdout:
        raise Detectron2BuildCheckError(
            f"{binary.name} 에 {target_sm} cubin 이 없다: {elf_list.stdout or elf_list.stderr}"
        )

    sys.path.insert(0, GRADIO_DEMO_DIR)
    try:
        import detectron2
        from detectron2 import _C
    except ImportError as error:
        raise Detectron2BuildCheckError(
            f"from detectron2 import _C 실패: {type(error).__name__}: {error}"
        ) from error
    extension_path = Path(_C.__file__).resolve()
    cuda_version = _C.get_cuda_version()
    compiler_version = _C.get_compiler_version()
    print(f"detectron2 {detectron2.__version__}: {Path(detectron2.__file__).parent}", flush=True)
    print(f"_C: {extension_path}", flush=True)
    print(
        f"_C CUDA: {cuda_version}, compiler: {compiler_version}, has_cuda: {_C.has_cuda()}",
        flush=True,
    )
    if extension_path != binary.resolve():
        raise Detectron2BuildCheckError(f"다른 위치의 _C 가 불렸다: {extension_path}")
    if detectron2.__version__ != EXPECTED_DETECTRON2_VERSION:
        raise Detectron2BuildCheckError(
            f"gradio_demo 의 detectron2 {EXPECTED_DETECTRON2_VERSION} 이 아니다: {detectron2.__version__}"
        )
    if not _C.has_cuda() or cuda_version != f"CUDA {EXPECTED_CUDA_RUNTIME}":
        raise Detectron2BuildCheckError(
            f"_C 가 CUDA {EXPECTED_CUDA_RUNTIME} 로 빌드되지 않았다: {cuda_version}"
        )

    if not torch.cuda.is_available():
        raise Detectron2BuildCheckError("torch.cuda.is_available() 가 False 다. GPU 를 쓸 수 없다.")
    gpu_name = torch.cuda.get_device_name(0)
    capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
    print(f"GPU: {gpu_name} (compute capability {capability})", flush=True)
    if capability != CUDA_ARCH:
        raise Detectron2BuildCheckError(
            f"GPU compute capability 가 빌드 대상({CUDA_ARCH})과 다르다: {capability}"
        )

    # _C 가 TORCH_LIBRARY 로 등록한 연산 하나를 CPU·GPU 에서 돌려 CUDA 커널이 이 GPU 에서 실행되는지 확인한다.
    boxes1 = torch.tensor([[10.0, 10.0, 8.0, 4.0, 0.0], [20.0, 20.0, 6.0, 6.0, 45.0]])
    boxes2 = torch.tensor([[10.0, 10.0, 8.0, 4.0, 30.0], [21.0, 20.0, 6.0, 6.0, 45.0]])
    iou_cpu = torch.ops.detectron2.box_iou_rotated(boxes1, boxes2)
    try:
        iou_gpu = torch.ops.detectron2.box_iou_rotated(boxes1.cuda(), boxes2.cuda())
        torch.cuda.synchronize()
    except RuntimeError as error:
        raise Detectron2BuildCheckError(f"box_iou_rotated CUDA 커널 실행 실패: {error}") from error
    print(f"box_iou_rotated cpu={iou_cpu.tolist()} gpu={iou_gpu.cpu().tolist()}", flush=True)
    if not torch.allclose(iou_gpu.cpu(), iou_cpu, atol=1e-5):
        raise Detectron2BuildCheckError("box_iou_rotated GPU 결과가 CPU 결과와 다르다")

    print("=== Detectron2 Build Check Passed ===", flush=True)
    return {
        "python": str(python_version),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "detectron2": str(detectron2.__version__),
        "extension_path": str(extension_path),
        "extension_cuda": str(cuda_version),
        "extension_compiler": str(compiler_version),
        "gpu_name": str(gpu_name),
        "compute_capability": str(capability),
    }


@app.local_entrypoint()
def main() -> None:
    # 빌드나 검증이 실패하면 예외가 그대로 올라와 modal run 이 0 이 아닌 코드로 끝난다.
    result = check_detectron2_build.remote()
    print(f"IDM-VTON detectron2 build check result: {result}")
