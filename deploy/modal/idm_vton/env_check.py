import modal

APP_NAME = "lookddak-idm-vton-env-check"
PYTHON_VERSION = "3.10"
GPU = "L40S"
TIMEOUT_SECONDS = 600

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu118"
TORCH_VERSION = "2.0.1+cu118"
TORCHVISION_VERSION = "0.15.2+cu118"
TORCHAUDIO_VERSION = "2.0.2+cu118"
EXPECTED_CUDA_RUNTIME = "11.8"
NUMPY_VERSION = "1.24.4"

image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(
    f"torch=={TORCH_VERSION}",
    f"torchvision=={TORCHVISION_VERSION}",
    f"torchaudio=={TORCHAUDIO_VERSION}",
    f"numpy=={NUMPY_VERSION}",
    extra_index_url=TORCH_INDEX_URL,
)

app = modal.App(APP_NAME)

MATMUL_SIZE = 1024


class EnvironmentCheckError(RuntimeError):
    """기대한 Python·PyTorch·CUDA·GPU 환경이 아니거나 GPU 연산이 실패했다."""


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_SECONDS)
def check_environment() -> dict:
    import platform
    import time

    import torch
    import torchvision

    python_version = platform.python_version()
    cuda_runtime = torch.version.cuda
    cuda_available = torch.cuda.is_available()
    print(f"Python: {python_version}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"TorchVision: {torchvision.__version__}", flush=True)
    print(f"PyTorch CUDA runtime: {cuda_runtime}", flush=True)
    print(f"torch.cuda.is_available(): {cuda_available}", flush=True)

    expected = {
        "Python": (python_version.rsplit(".", 1)[0], PYTHON_VERSION),
        "PyTorch": (torch.__version__, TORCH_VERSION),
        "TorchVision": (torchvision.__version__, TORCHVISION_VERSION),
        "PyTorch CUDA runtime": (cuda_runtime, EXPECTED_CUDA_RUNTIME),
    }
    mismatched = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
    if mismatched:
        raise EnvironmentCheckError(f"버전이 기대와 다르다(실제, 기대): {mismatched}")
    if not cuda_available:
        raise EnvironmentCheckError("torch.cuda.is_available() 가 False 다. GPU 를 쓸 수 없다.")

    properties = torch.cuda.get_device_properties(0)
    memory_gib = properties.total_memory / 1024**3
    print(f"GPU: {properties.name}", flush=True)
    print(f"GPU memory: {memory_gib:.2f} GiB", flush=True)
    if GPU not in properties.name:
        raise EnvironmentCheckError(f"요청한 GPU({GPU})가 아니다: {properties.name}")

    # 같은 입력을 CPU 에서도 곱해 GPU 결과가 맞는지 확인한다. float32 합산 순서 차이만 허용한다.
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, generator=generator)
    b = torch.randn(MATMUL_SIZE, MATMUL_SIZE, generator=generator)
    try:
        a_gpu, b_gpu = a.to("cuda"), b.to("cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        c_gpu = a_gpu @ b_gpu
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        result = c_gpu.cpu()
    except RuntimeError as error:
        raise EnvironmentCheckError(f"GPU 행렬 곱셈 실패: {error}") from error
    expected_result = a @ b
    max_abs_diff = (result - expected_result).abs().max().item()
    if not torch.isfinite(result).all() or not torch.allclose(result, expected_result, rtol=1e-3, atol=1e-3):
        raise EnvironmentCheckError(f"GPU 행렬 곱셈 결과가 CPU 결과와 다르다(max abs diff {max_abs_diff})")
    print(
        f"GPU matmul {MATMUL_SIZE}x{MATMUL_SIZE}: device={c_gpu.device} dtype={c_gpu.dtype} "
        f"elapsed={elapsed_ms:.2f}ms sum={result.sum().item():.4f} max_abs_diff_vs_cpu={max_abs_diff:.3e}",
        flush=True,
    )

    print("=== Environment Check Passed ===", flush=True)
    return {
        "python": str(python_version),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_runtime": str(cuda_runtime),
        "cuda_available": bool(cuda_available),
        "gpu_name": str(properties.name),
        "gpu_memory_gib": float(round(memory_gib, 2)),
        "matmul_elapsed_ms": float(round(elapsed_ms, 2)),
        "matmul_max_abs_diff_vs_cpu": float(max_abs_diff),
    }


@app.local_entrypoint()
def main() -> None:
    # 실패하면 원격 예외가 그대로 올라와 modal run 이 0 이 아닌 코드로 끝난다.
    result = check_environment.remote()
    print(f"IDM-VTON env check result: {result}")
