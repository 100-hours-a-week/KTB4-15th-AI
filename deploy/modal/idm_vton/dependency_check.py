import modal

app = modal.App("lookddak-idm-vton-dependency-check")

SOURCE_COMMIT = "06ad89ba2f78d77a5ba21d7f7fd392254971fda2"
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libglib2.0-0", "libgl1")
    .pip_install(
        "torch==2.0.1+cu118",
        "torchvision==0.15.2+cu118",
        "torchaudio==2.0.2+cu118",
        "numpy==1.24.4",
        extra_index_url=TORCH_INDEX,
    )
    .pip_install(
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
        "fvcore",
        "cloudpickle",
        "omegaconf",
        "pycocotools",
        "basicsr",
        "av",
    )
    .run_commands(
        "git clone https://github.com/yisol/IDM-VTON.git /opt/IDM-VTON",
        f"git -C /opt/IDM-VTON checkout {SOURCE_COMMIT}",
    )
)


@app.function(image=image, timeout=300)
def check_dependencies() -> bool:
    import importlib
    from pathlib import Path

    packages = [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "einops",
        "scipy",
        "onnxruntime",
        "cv2",
        "fvcore",
        "omegaconf",
        "pycocotools",
    ]

    for package in packages:
        importlib.import_module(package)
        print(f"[OK] {package}", flush=True)

    repo = Path("/opt/IDM-VTON")
    required_paths = [
        "src/tryon_pipeline.py",
        "preprocess/openpose/run_openpose.py",
        "preprocess/humanparsing/run_parsing.py",
        "gradio_demo/apply_net.py",
    ]

    for relative_path in required_paths:
        path = repo / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"[OK] {relative_path}", flush=True)

    print("=== Dependency Check Passed ===", flush=True)
    return True


@app.local_entrypoint()
def main():
    result = check_dependencies.remote()
    print("Dependency check:", result)
