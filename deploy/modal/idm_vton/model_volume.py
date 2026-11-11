import modal

APP_NAME = "lookddak-idm-vton-model-volume"
PYTHON_VERSION = "3.10"
# integrated_environment_check.py 와 같은 버전. 내려받기만 하므로 CUDA·torch 는 필요 없다.
HUGGINGFACE_HUB_VERSION = "0.25.2"
DOWNLOAD_TIMEOUT_SECONDS = 3 * 60 * 60
INSPECT_TIMEOUT_SECONDS = 300

HF_REPO_ID = "yisol/IDM-VTON"
# gradio_demo/app.py 는 revision 없이 main 을 받는다. 같은 내용을 다시 받을 수 있게 2024-04-22 커밋으로 고정한다.
HF_REVISION = "585a32e74aee241cbc0d0cc3ab21392ca58c916a"

VOLUME_NAME = "lookddak-idm-vton-models"
VOLUME_MOUNT_PATH = "/models"
# HF 저장소 구조를 그대로 둔다. 파이프라인은 from_pretrained(MODEL_DIR, subfolder=...) 로 읽는다.
MODEL_DIR = f"{VOLUME_MOUNT_PATH}/idm-vton"
MANIFEST_NAME = "lookddak_manifest.json"

# gradio_demo/app.py 가 from_pretrained(base_path, subfolder=...) 로 읽는 구성요소.
# model_index.json 의 나머지 구성요소(feature_extractor)는 app.py 가 CLIPImageProcessor() 로 직접 만든다.
PIPELINE_FILES = (
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.bin",
    "unet_encoder/config.json",
    "unet_encoder/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "scheduler/scheduler_config.json",
    "image_encoder/config.json",
    "image_encoder/model.safetensors",
)
# 공식 코드가 저장소의 ckpt/ 아래에서 찾는 전처리 체크포인트. git 에는 24~31바이트 자리표시자만 있다.
PREPROCESS_FILES = (
    "densepose/model_final_162be9.pkl",  # apply_net 인자 ./ckpt/densepose/... (MODEL.WEIGHTS 를 덮어써 R-50 은 받지 않는다)
    "humanparsing/parsing_atr.onnx",  # run_parsing.py: parents[2]/ckpt/humanparsing
    "humanparsing/parsing_lip.onnx",
    "openpose/ckpts/body_pose_model.pth",  # annotator/util.py: parents[3]/ckpt/openpose/ckpts (hand·face 는 주석 처리됨)
)
REQUIRED_FILES = frozenset(PIPELINE_FILES + PREPROCESS_FILES)

# 공식 저장소의 ckpt/ 디렉터리 → Volume 안 디렉터리. HF 저장소에서도 같은 상대 경로라 디렉터리 단위로 이으면 된다.
# ckpt/image_encoder·ckpt/ip_adapter 는 inference.py·train_xl.py 용이라 app.py 경로에는 필요 없다.
OFFICIAL_CKPT_LINKS = {
    "ckpt/densepose": "densepose",
    "ckpt/humanparsing": "humanparsing",
    "ckpt/openpose/ckpts": "openpose/ckpts",
}
# 이보다 큰 파일이 들어 있으면 자리표시자가 아니라 실제 체크포인트로 보고 지우지 않는다.
PLACEHOLDER_MAX_BYTES = 1024 * 1024

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(f"huggingface_hub=={HUGGINGFACE_HUB_VERSION}")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


class ModelVolumeError(RuntimeError):
    """고정 리비전의 IDM-VTON 파일 목록이 기대와 다르거나, Volume 의 파일이 없거나 크기가 다르다."""


def fetch_repo_files() -> list[dict]:
    """고정 리비전의 파일 목록과 크기를 HF API 로 가져온다. 파일 내용은 받지 않는다."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(HF_REPO_ID, revision=HF_REVISION, files_metadata=True)
    if info.sha != HF_REVISION:
        raise ModelVolumeError(f"{HF_REPO_ID} 리비전이 고정값과 다르다: {info.sha}")
    files = [
        {
            "path": str(sibling.rfilename),
            "size": int(sibling.size or 0),
            "sha256": str(sibling.lfs.sha256) if sibling.lfs else "",
            "required": sibling.rfilename in REQUIRED_FILES,
        }
        for sibling in sorted(info.siblings, key=lambda sibling: sibling.rfilename)
    ]
    missing = sorted(REQUIRED_FILES - {file["path"] for file in files})
    if missing:
        raise ModelVolumeError(f"{HF_REPO_ID}@{HF_REVISION} 에 필수 파일이 없다: {missing}")
    return files


def summarize(files: list[dict]) -> dict:
    required = [file for file in files if file["required"]]
    return {
        "repo_id": HF_REPO_ID,
        "revision": HF_REVISION,
        "file_count": len(files),
        "total_bytes": sum(file["size"] for file in files),
        "required_file_count": len(required),
        "required_bytes": sum(file["size"] for file in required),
        "skipped_files": [file["path"] for file in files if not file["required"]],
    }


def local_file_status(model_dir: str, files: list[dict]) -> dict[str, str]:
    """필수 파일마다 ok(크기 일치)·missing·size_mismatch 중 하나를 돌려준다."""
    from pathlib import Path

    status = {}
    for file in files:
        if not file["required"]:
            continue
        path = Path(model_dir) / file["path"]
        if not path.is_file():
            status[file["path"]] = "missing"
        elif path.stat().st_size != file["size"]:
            status[file["path"]] = "size_mismatch"
        else:
            status[file["path"]] = "ok"
    return status


def print_files(files: list[dict]) -> None:
    for file in files:
        mark = "required" if file["required"] else "skip    "
        print(f"{file['size'] / 1024**2:12.2f} MiB  {mark}  {file['path']}", flush=True)
    summary = summarize(files)
    print(
        f"전체 {summary['file_count']}개 {summary['total_bytes'] / 1024**3:.2f} GiB, "
        f"필수 {summary['required_file_count']}개 {summary['required_bytes'] / 1024**3:.2f} GiB",
        flush=True,
    )


def link_official_ckpt_dirs(repo_dir: str, model_dir: str = MODEL_DIR) -> dict[str, str]:
    """공식 저장소 ckpt/ 의 자리표시자 디렉터리를 Volume 디렉터리를 가리키는 symlink 로 바꾼다.

    공식 코드는 __file__ 기준으로 ckpt/ 를 찾으므로 코드를 고치지 않고 경로만 잇는다. 이미지 빌드 단계에서
    부르면 Volume 이 붙기 전까지는 끊어진 링크이고, 런타임에 Volume 을 MODEL_DIR 상위에 붙이면 이어진다.
    """
    import shutil
    from pathlib import Path

    links = {}
    for official, relative in OFFICIAL_CKPT_LINKS.items():
        link = Path(repo_dir) / official
        target = Path(model_dir) / relative
        if link.is_symlink():
            if link.readlink() == target:
                links[str(link)] = str(target)
                continue
            link.unlink()
        elif link.exists():
            large = [
                p
                for p in link.rglob("*")
                if p.is_file() and p.stat().st_size > PLACEHOLDER_MAX_BYTES
            ]
            if large:
                raise ModelVolumeError(f"자리표시자가 아닌 파일이 있어 지우지 않는다: {large}")
            shutil.rmtree(link)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
        links[str(link)] = str(target)
    return links


@app.function(image=image, timeout=INSPECT_TIMEOUT_SECONDS)
def inspect_model_repo() -> dict:
    files = fetch_repo_files()
    print_files(files)
    return {**summarize(files), "files": files}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=INSPECT_TIMEOUT_SECONDS)
def verify_models() -> dict:
    """Volume 의 필수 파일이 고정 리비전의 크기와 같은지만 본다. 내려받지 않는다."""
    files = fetch_repo_files()
    status = local_file_status(MODEL_DIR, files)
    for path, state in status.items():
        print(f"[{state}] {path}", flush=True)
    problems = {path: state for path, state in status.items() if state != "ok"}
    if problems:
        raise ModelVolumeError(f"Volume {VOLUME_NAME} 의 필수 파일이 맞지 않다: {problems}")
    print(f"=== {len(status)} required files verified in {MODEL_DIR} ===", flush=True)
    return {**summarize(files), "verified_files": len(status)}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=DOWNLOAD_TIMEOUT_SECONDS)
def download_models() -> dict:
    """필수 파일 중 없거나 크기가 다른 것만 받는다. 크기가 같은 파일은 다시 받지 않는다."""
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    files = fetch_repo_files()
    print_files(files)
    sizes = {file["path"]: file["size"] for file in files}
    status = local_file_status(MODEL_DIR, files)
    pending = [path for path, state in status.items() if state != "ok"]
    print(
        f"이미 있는 파일 {len(status) - len(pending)}개는 건너뛰고 {len(pending)}개를 받는다",
        flush=True,
    )

    downloaded = []
    for path in pending:
        print(f"download {path} ({sizes[path] / 1024**2:.2f} MiB, {status[path]})", flush=True)
        local_path = hf_hub_download(
            HF_REPO_ID,
            path,
            revision=HF_REVISION,
            local_dir=MODEL_DIR,
            # 크기가 다른 파일은 hf 의 로컬 메타데이터가 최신이라고 판단해 건너뛸 수 있어 강제로 받는다.
            force_download=status[path] == "size_mismatch",
        )
        actual = Path(local_path).stat().st_size
        if actual != sizes[path]:
            raise ModelVolumeError(f"{path} 크기가 다르다(실제 {actual}, 기대 {sizes[path]})")
        downloaded.append(path)
        # 큰 파일마다 커밋해, 제한 시간에 걸려 끊겨도 다음 실행이 받은 파일부터 건너뛴다.
        volume.commit()

    problems = {
        path: state for path, state in local_file_status(MODEL_DIR, files).items() if state != "ok"
    }
    if problems:
        raise ModelVolumeError(f"다운로드 뒤에도 맞지 않는 파일이 있다: {problems}")

    summary = summarize(files)
    manifest = {
        **summary,
        "files": {path: sizes[path] for path in sorted(REQUIRED_FILES)},
        "official_ckpt_links": OFFICIAL_CKPT_LINKS,
    }
    (Path(MODEL_DIR) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    volume.commit()
    print(f"=== IDM-VTON models ready in {VOLUME_NAME}:{MODEL_DIR} ===", flush=True)
    return {**summary, "downloaded": downloaded, "skipped_existing": len(status) - len(pending)}


@app.local_entrypoint()
def main(action: str = "inspect") -> None:
    # inspect: 목록·크기만 조회. verify: Volume 검사만. download: 약 27GiB 를 Volume 에 받는다.
    actions = {"inspect": inspect_model_repo, "verify": verify_models, "download": download_models}
    if action not in actions:
        raise SystemExit(f"--action 은 {sorted(actions)} 중 하나여야 한다: {action}")
    result = actions[action].remote()
    result.pop("files", None)
    print(f"IDM-VTON model volume {action} result: {result}")
