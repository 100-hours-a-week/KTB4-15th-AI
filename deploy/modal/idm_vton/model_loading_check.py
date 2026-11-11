import modal
from detectron2_build_check import (
    CUDA_ARCH,
    EXPECTED_CUDA_RUNTIME,
    GPU,
    GRADIO_DEMO_DIR,
    REPO_DIR,
    TORCH_VERSION,
)
from integrated_environment_check import (
    DOWNLOAD_WATCH_DIRS,
    HF_MODEL_CACHE_DIRS,
    HF_OFFLINE_ENV,
    MODEL_ARTIFACT_SUFFIXES,
)
from integrated_environment_check import image as integrated_image
from model_volume import (
    HF_REVISION,
    MANIFEST_NAME,
    MODEL_DIR,
    PREPROCESS_FILES,
    REQUIRED_FILES,
    VOLUME_MOUNT_PATH,
    VOLUME_NAME,
    link_official_ckpt_dirs,
    local_file_status,
)

APP_NAME = "lookddak-idm-vton-model-loading-check"
# 실행 시간 상한이지 금액 상한이 아니다. 컨테이너 시작 시간도 이 안에 들어간다.
LOADING_TIMEOUT_SECONDS = 600
MAX_CONTAINERS = 1
RETRIES = 0
LOADING_CPU = 4.0
# request 일 뿐 limit 이 아니다. unet .bin(fp32 11.4GB)을 torch.load 한 뒤 fp16 으로 바꾸는 동안 호스트 메모리가 가장 많이 든다.
LOADING_MEMORY_MIB = 32768

DEVICE = "cuda:0"  # gradio_demo/app.py 와 같다
# GPU 로 옮기기 전에 이만큼은 남아 있어야 한다. 모자라면 옮기지 않고 OOM 예상으로 실패시킨다.
VRAM_HEADROOM_MIB = 512
# gradio_demo/app.py 의 start_tryon 이 apply_net 에 넘기는 인자 그대로. cwd 가 저장소 루트여야 상대 경로가 맞는다.
DENSEPOSE_ARGS = (
    "show",
    "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
    "./ckpt/densepose/model_final_162be9.pkl",
    "dp_segm",
    "-v",
    "--opts",
    "MODEL.DEVICE",
    "cuda",
)
# detectron2 DetectionCheckpointer 도 체크포인트에 없어도 넘어가는 버퍼. cfg 에서 채운다.
DENSEPOSE_IGNORED_MISSING_KEYS = {"pixel_mean", "pixel_std"}

# (단계 이름, 묶음). 사용자 지정 순서대로 전처리 셋을 먼저, IDM-VTON 본체는 gradio_demo/app.py 의 로딩 순서를 따른다.
LOADING_STEPS = (
    ("openpose", "preprocess"),
    ("parsing", "preprocess"),
    ("densepose", "preprocess"),
    ("unet", "idm_vton"),
    ("tokenizer", "idm_vton"),
    ("tokenizer_2", "idm_vton"),
    ("scheduler", "idm_vton"),
    ("text_encoder", "idm_vton"),
    ("text_encoder_2", "idm_vton"),
    ("image_encoder", "idm_vton"),
    ("vae", "idm_vton"),
    ("unet_encoder", "idm_vton"),
    ("pipeline", "idm_vton"),
)
STEP_NAMES = tuple(name for name, _ in LOADING_STEPS)
STEP_GROUPS = {group for _, group in LOADING_STEPS}
# 파이프라인 조립 단계가 받는 구성요소. 하나라도 없으면 조립하지 않는다.
PIPELINE_INPUTS = STEP_NAMES[3:-1]
# 동시 상주를 확인할 GPU 모델. Parsing(공식 설정이 onnxruntime CPU)·토크나이저·스케줄러는 GPU 에 올라가지 않는다.
GPU_MODEL_STEPS = (
    "openpose",
    "densepose",
    "unet",
    "text_encoder",
    "text_encoder_2",
    "image_encoder",
    "vae",
    "unet_encoder",
)

image = (
    integrated_image
    # 통합 이미지의 빌드 레이어를 그대로 쓰고, 이 파일이 import 하는 모듈만 더 올린다. 빌드 단계를 더하지 않는다.
    .add_local_python_source("integrated_environment_check", "model_volume")
)
# 읽기 전용으로 붙여 Volume 을 바꾸지 않는다. create_if_missing 을 쓰지 않아 Volume 이 없으면 빈 Volume 을 만들지 않고 실패한다.
volume = modal.Volume.from_name(VOLUME_NAME).read_only()

app = modal.App(APP_NAME)


class ModelLoadingCheckError(RuntimeError):
    """Volume 체크포인트, 공식 ckpt/ 경로 연결, 모델 로딩 중 하나 이상이 기대와 다르다."""


class PredictedOutOfMemoryError(ModelLoadingCheckError):
    """GPU 로 옮기기 전에 남은 VRAM 이 모델 크기보다 작아 옮기지 않았다."""


class StepSkippedError(ModelLoadingCheckError):
    """앞 단계가 실패했거나 선택되지 않아 이 단계를 실행할 수 없다."""


def parse_model_selection(models: str) -> list[str]:
    """--models 값(all, 묶음 이름, 단계 이름을 쉼표로 나열)을 LOADING_STEPS 순서의 단계 이름 목록으로 바꾼다."""
    requested = {item.strip() for item in models.split(",") if item.strip()}
    unknown = requested - set(STEP_NAMES) - STEP_GROUPS - {"all"}
    if not requested or unknown:
        raise ModelLoadingCheckError(
            f"--models 는 all, {sorted(STEP_GROUPS)}, {list(STEP_NAMES)} 중에서 고른다: {sorted(unknown)}"
        )
    return [
        name
        for name, group in LOADING_STEPS
        if "all" in requested or name in requested or group in requested
    ]


def verify_volume_files(model_dir: str = MODEL_DIR) -> dict[str, int]:
    """download_models 가 남긴 매니페스트로 필수 파일 크기를 확인한다. 네트워크를 쓰지 않는다."""
    import json
    from pathlib import Path

    manifest_path = Path(model_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ModelLoadingCheckError(
            f"{manifest_path} 가 없다. model_volume.py --action download 를 먼저 마쳐야 한다"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("revision") != HF_REVISION:
        raise ModelLoadingCheckError(
            f"Volume 의 리비전이 고정값과 다르다: {manifest.get('revision')}"
        )
    sizes = {str(path): int(size) for path, size in manifest["files"].items()}
    if set(sizes) != REQUIRED_FILES:
        raise ModelLoadingCheckError(
            f"매니페스트의 파일 목록이 필수 목록과 다르다: {sorted(set(sizes) ^ REQUIRED_FILES)}"
        )
    files = [{"path": path, "size": size, "required": True} for path, size in sizes.items()]
    problems = {
        path: state for path, state in local_file_status(model_dir, files).items() if state != "ok"
    }
    if problems:
        raise ModelLoadingCheckError(f"Volume {VOLUME_NAME} 의 필수 파일이 맞지 않다: {problems}")
    return sizes


def verify_official_ckpt_files(
    repo_dir: str, sizes: dict[str, int], model_dir: str = MODEL_DIR
) -> dict[str, str]:
    """공식 코드가 여는 ckpt/ 경로로 전처리 가중치를 열 수 있고, 그 실체가 Volume 의 파일인지 확인한다."""
    from pathlib import Path

    resolved = {}
    for relative in PREPROCESS_FILES:
        official = Path(repo_dir) / "ckpt" / relative
        if not official.is_file():
            raise ModelLoadingCheckError(f"공식 경로에 가중치가 없다: {official}")
        target = official.resolve(strict=True)
        if not target.is_relative_to(Path(model_dir).resolve()):
            raise ModelLoadingCheckError(f"{official} 이 Volume 밖을 가리킨다: {target}")
        if official.stat().st_size != sizes[relative]:
            raise ModelLoadingCheckError(
                f"{official} 크기가 다르다(실제 {official.stat().st_size}, 기대 {sizes[relative]})"
            )
        resolved[str(official)] = str(target)
    return resolved


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_MOUNT_PATH: volume},
    cpu=LOADING_CPU,
    memory=LOADING_MEMORY_MIB,
    timeout=LOADING_TIMEOUT_SECONDS,
    retries=RETRIES,
    max_containers=MAX_CONTAINERS,
    # scaledown_window 는 Modal 1.6.1 이 0 을 거부한다(> 0). 입력 하나를 처리하면 컨테이너를 바로 내리도록 이것으로 대신한다.
    single_use_containers=True,
)
def check_model_loading(models: str = "all") -> dict:
    import os

    # HF 라이브러리를 import 하기 전에 둬야 적용된다. 로컬 경로만 읽으므로 허브에 갈 일이 없어야 한다.
    os.environ.update(HF_OFFLINE_ENV)

    import gc
    import itertools
    import pickle
    import resource
    import sys
    import time
    from pathlib import Path

    import torch

    def utc_now() -> str:
        # datetime.UTC 는 3.11 부터라 컨테이너(3.10)에서 못 쓴다. 초 단위 시각만 남기고 소요 시간은 perf_counter 로 잰다.
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    run_started = time.perf_counter()
    selected = parse_model_selection(models)
    print(f"[INFO] 시작 {utc_now()} 단계: {selected}", flush=True)

    def mib(num_bytes: int) -> float:
        return round(num_bytes / 1024**2, 1)

    def gpu_memory() -> dict:
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_mib": mib(torch.cuda.memory_allocated()),
            "reserved_mib": mib(torch.cuda.memory_reserved()),
            "max_allocated_mib": mib(torch.cuda.max_memory_allocated()),
            # mem_get_info 는 장치 전체 기준이라 CUDA context 처럼 PyTorch 할당기 밖에서 쓰는 메모리도 들어간다.
            "device_free_mib": mib(free),
            "device_total_mib": mib(total),
            "device_used_mib": mib(total - free),
        }

    def host_peak_rss_mib() -> float:
        # Linux 의 ru_maxrss 는 KiB 단위다.
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

    def tensor_bytes(module: torch.nn.Module) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in itertools.chain(module.parameters(), module.buffers())
        )

    def module_detail(module: torch.nn.Module) -> dict:
        return {
            "parameters_m": round(sum(p.numel() for p in module.parameters()) / 1e6, 1),
            "dtypes": ",".join(
                sorted({str(p.dtype).removeprefix("torch.") for p in module.parameters()})
            ),
            "devices": ",".join(sorted({str(p.device) for p in module.parameters()})),
            "tensor_mib": mib(tensor_bytes(module)),
        }

    def is_oom(error: BaseException) -> bool:
        return isinstance(error, (torch.cuda.OutOfMemoryError, PredictedOutOfMemoryError)) or (
            "out of memory" in str(error).lower()
        )

    checks: list[dict] = []

    def preflight(name: str, check) -> None:
        # 사전 점검이 하나라도 실패하면 GPU 로딩을 시작하지 않는다(비용 절약).
        try:
            detail = check()
        except Exception as error:  # 실패 내용을 출력한 뒤 그대로 올려 중단한다
            print(f"[FAIL] {name}: {type(error).__name__}: {error}", flush=True)
            raise
        print(f"[PASS] {name}: {detail}", flush=True)
        checks.append({"name": name, "passed": True, "detail": str(detail)})

    def check_environment() -> str:
        if not torch.cuda.is_available():
            raise ModelLoadingCheckError("torch.cuda.is_available() 가 False 다")
        name = torch.cuda.get_device_name(0)
        capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
        actual = (torch.__version__, torch.version.cuda, capability)
        if GPU not in name or actual != (TORCH_VERSION, EXPECTED_CUDA_RUNTIME, CUDA_ARCH):
            raise ModelLoadingCheckError(f"{name} {actual} 가 기대와 다르다")
        return f"{name}, torch {torch.__version__}, CUDA {torch.version.cuda}, sm {capability}"

    sizes: dict[str, int] = {}

    def check_volume() -> str:
        sizes.update(verify_volume_files())
        return f"{len(sizes)}개 필수 파일 크기 일치 (revision {HF_REVISION[:12]})"

    def check_ckpt_links() -> str:
        links = link_official_ckpt_dirs(REPO_DIR)
        resolved = verify_official_ckpt_files(REPO_DIR, sizes)
        return f"links={links} files={resolved}"

    # gradio_demo/app.py 를 저장소 루트에서 실행할 때와 같게 맞춘다: cwd 는 저장소 루트, sys.path 는 gradio_demo 다음 루트.
    os.chdir(REPO_DIR)
    sys.path[:0] = [GRADIO_DEMO_DIR, REPO_DIR]

    def check_official_paths() -> str:
        # 공식 코드가 __file__·cwd 로 계산하는 경로가 실제로 Volume 파일에 닿는지 본다. OpenPose 는 파일이 없으면
        # basicsr 로 직접 내려받으므로, 로딩 전에 여기서 막는다.
        from preprocess.humanparsing import run_parsing
        from preprocess.openpose.annotator.util import annotator_ckpts_path

        expected = {
            Path(annotator_ckpts_path)
            / "body_pose_model.pth": "openpose/ckpts/body_pose_model.pth",
            Path(run_parsing.__file__).absolute().parents[2]
            / "ckpt/humanparsing/parsing_atr.onnx": ("humanparsing/parsing_atr.onnx"),
            Path(run_parsing.__file__).absolute().parents[2]
            / "ckpt/humanparsing/parsing_lip.onnx": ("humanparsing/parsing_lip.onnx"),
            Path.cwd() / DENSEPOSE_ARGS[2]: "densepose/model_final_162be9.pkl",
        }
        for official, relative in expected.items():
            target = Path(MODEL_DIR) / relative
            if not official.is_file() or official.resolve() != target.resolve():
                raise ModelLoadingCheckError(f"{official} 가 {target} 에 닿지 않는다")
        return f"{len(expected)}개 공식 경로가 Volume 파일로 이어짐"

    preflight("환경 (GPU·PyTorch·CUDA)", check_environment)
    preflight(f"Volume {VOLUME_NAME} 필수 파일", check_volume)
    preflight("공식 ckpt/ 경로 연결", check_ckpt_links)
    preflight("공식 코드가 계산하는 체크포인트 경로", check_official_paths)

    def watched_files() -> set[str]:
        return {
            str(path)
            for directory in map(Path, DOWNLOAD_WATCH_DIRS)
            if directory.is_dir()
            for path in directory.rglob("*")
            if path.is_file()
        }

    files_before = watched_files()
    loaded: dict = {}
    phases: dict[str, float] = {}

    def phase(label: str, load):
        started = time.perf_counter()
        result = load()
        torch.cuda.synchronize()
        phases[label] = round(time.perf_counter() - started, 2)
        return result

    def move_to_gpu(module: torch.nn.Module) -> torch.nn.Module:
        needed = tensor_bytes(module)
        free, _ = torch.cuda.mem_get_info()
        if needed > free - VRAM_HEADROOM_MIB * 1024**2:
            raise PredictedOutOfMemoryError(
                f"필요 {mib(needed)} MiB > 남은 {mib(free)} MiB - 여유 {VRAM_HEADROOM_MIB} MiB"
            )
        return module.to(DEVICE)

    def load_openpose() -> dict:
        from preprocess.openpose.run_openpose import OpenPose

        # 공식 Body.__init__ 은 모델을 먼저 GPU 에 올리고 가중치를 strict load_state_dict 로 넣는다.
        model = phase("load", lambda: OpenPose(0))
        loaded["openpose"] = model
        return module_detail(model.preprocessor.body_estimation.model)

    def load_parsing() -> dict:
        from preprocess.humanparsing.run_parsing import Parsing

        model = phase("load", lambda: Parsing(0))
        providers = {model.session.get_providers()[0], model.lip_session.get_providers()[0]}
        # 공식 코드가 CPUExecutionProvider 로 고정한다. GPU 메모리를 쓰지 않는 것이 정상이다.
        if providers != {"CPUExecutionProvider"}:
            raise ModelLoadingCheckError(f"공식 설정과 다른 실행 장치: {providers}")
        loaded["parsing"] = model
        return {"devices": "cpu (onnxruntime CPUExecutionProvider)", "sessions": 2}

    def load_densepose() -> dict:
        import apply_net
        from detectron2.engine.defaults import DefaultPredictor

        # start_tryon 의 args.func(args, img) 가 하는 일 중 이미지 입력 직전까지: ShowAction.setup_config → DefaultPredictor.
        args = apply_net.create_argument_parser().parse_args(DENSEPOSE_ARGS)
        action = args.func.__self__
        cfg = action.setup_config(args.cfg, args.model, args, [])
        predictor = phase("load", lambda: DefaultPredictor(cfg))
        loaded["densepose"] = predictor

        # 예외 없이 끝났다는 것만으로는 가중치가 들어갔는지 모른다. 체크포인트 값과 모델 값을 직접 비교한다.
        # DetectionCheckpointer._load_file 과 같은 방식으로 같은 파일을 읽는다. Detectron2 Model Zoo 형식이라 키 이름이 같다.
        with open(args.model, "rb") as file:
            data = pickle.load(file, encoding="latin1")
        if "model" not in data or "__author__" not in data:
            raise ModelLoadingCheckError(f"Detectron2 형식 체크포인트가 아니다: {sorted(data)[:5]}")
        checkpoint = data["model"]
        state = predictor.model.state_dict()
        shared = sorted(state.keys() & checkpoint.keys())
        mismatched = [
            key
            for key in shared
            if not torch.equal(
                state[key].cpu(), torch.as_tensor(checkpoint[key]).to(state[key].dtype)
            )
        ]
        missing = sorted(state.keys() - checkpoint.keys() - DENSEPOSE_IGNORED_MISSING_KEYS)
        unexpected = sorted(checkpoint.keys() - state.keys())
        if mismatched or missing or not shared:
            raise ModelLoadingCheckError(
                f"체크포인트가 모델에 들어가지 않았다: 값 불일치 {mismatched[:5]}, 누락 {missing[:5]}"
            )
        return {
            **module_detail(predictor.model),
            "weights": cfg.MODEL.WEIGHTS,
            "checkpoint_author": str(data["__author__"]),
            "matched_keys": len(shared),
            "unexpected_keys": len(unexpected),
        }

    def pretrained_module(model_class, subfolder: str):
        # gradio_demo/app.py 와 같게 fp16 으로 CPU 에 올린 뒤, 모델 하나씩 GPU 로 옮겨 단계마다 메모리를 잰다.
        def load() -> dict:
            module = phase(
                "load",
                lambda: model_class.from_pretrained(
                    MODEL_DIR, subfolder=subfolder, torch_dtype=torch.float16
                ),
            )
            module.requires_grad_(False)
            loaded[subfolder] = phase("to_gpu", lambda: move_to_gpu(module))
            return module_detail(loaded[subfolder])

        return load

    def tokenizer(subfolder: str):
        def load() -> dict:
            from transformers import AutoTokenizer

            loaded[subfolder] = phase(
                "load",
                lambda: AutoTokenizer.from_pretrained(
                    MODEL_DIR, subfolder=subfolder, revision=None, use_fast=False
                ),
            )
            return {"devices": "cpu", "class": type(loaded[subfolder]).__name__}

        return load

    def load_scheduler() -> dict:
        from diffusers import DDPMScheduler

        loaded["scheduler"] = phase(
            "load", lambda: DDPMScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
        )
        return {"devices": "cpu", "class": type(loaded["scheduler"]).__name__}

    def assemble_pipeline() -> dict:
        missing = [name for name in PIPELINE_INPUTS if name not in loaded]
        if missing:
            raise StepSkippedError(f"구성요소가 없어 조립하지 않는다: {missing}")
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from transformers import CLIPImageProcessor

        # model_index.json 에 image_encoder·unet_encoder 가 없고 나머지는 모두 넘기므로 디스크에서 더 읽는 가중치는 없다.
        pipe = phase(
            "load",
            lambda: TryonPipeline.from_pretrained(
                MODEL_DIR,
                unet=loaded["unet"],
                vae=loaded["vae"],
                feature_extractor=CLIPImageProcessor(),
                text_encoder=loaded["text_encoder"],
                text_encoder_2=loaded["text_encoder_2"],
                tokenizer=loaded["tokenizer"],
                tokenizer_2=loaded["tokenizer_2"],
                scheduler=loaded["scheduler"],
                image_encoder=loaded["image_encoder"],
                torch_dtype=torch.float16,
            ),
        )
        pipe.unet_encoder = loaded["unet_encoder"]
        if pipe.unet_encoder is not loaded["unet_encoder"] or pipe.device.type != "cuda":
            raise ModelLoadingCheckError(f"파이프라인 조립 결과가 다르다: device={pipe.device}")
        loaded["pipeline"] = pipe
        return {"devices": str(pipe.device), "class": type(pipe).__name__}

    from diffusers import AutoencoderKL
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel
    from transformers import (
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    )

    loaders = {
        "openpose": load_openpose,
        "parsing": load_parsing,
        "densepose": load_densepose,
        "unet": pretrained_module(UNet2DConditionModel, "unet"),
        "tokenizer": tokenizer("tokenizer"),
        "tokenizer_2": tokenizer("tokenizer_2"),
        "scheduler": load_scheduler,
        "text_encoder": pretrained_module(CLIPTextModel, "text_encoder"),
        "text_encoder_2": pretrained_module(CLIPTextModelWithProjection, "text_encoder_2"),
        "image_encoder": pretrained_module(CLIPVisionModelWithProjection, "image_encoder"),
        "vae": pretrained_module(AutoencoderKL, "vae"),
        "unet_encoder": pretrained_module(UNet2DConditionModel_ref, "unet_encoder"),
        "pipeline": assemble_pipeline,
    }

    def gpu_model(name: str) -> torch.nn.Module:
        if name == "openpose":
            return loaded[name].preprocessor.body_estimation.model
        if name == "densepose":
            return loaded[name].model
        return loaded[name]

    steps: list[dict] = []
    for name, group in LOADING_STEPS:
        if name not in selected:
            continue
        # 호스트 메모리 부족으로 프로세스가 죽으면 예외가 남지 않는다. 마지막 START 줄이 원인 모델이다.
        print(f"[START] {name}", flush=True)
        phases.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        started_at = utc_now()
        started = time.perf_counter()
        detail: dict = {}
        error_text = ""
        oom = False
        try:
            detail = loaders[name]()
        except StepSkippedError as error:
            status, error_text = "skipped", str(error)
        except Exception as error:  # noqa: BLE001 — 한 모델이 실패해도 나머지 모델을 따로 확인하고 끝에서 실패시킨다
            status, error_text, oom = "failed", f"{type(error).__name__}: {error}", is_oom(error)
        else:
            status = "passed"
        if status == "failed":
            # except 블록을 벗어나야 traceback 이 잡고 있던 텐서가 풀린다. 다음 모델이 쓸 수 있게 여기서 비운다.
            loaded.pop(name, None)
            gc.collect()
            torch.cuda.empty_cache()
        seconds = round(time.perf_counter() - started, 2)
        memory = gpu_memory()
        step = {
            "name": name,
            "group": group,
            "status": status,
            "oom": oom,
            "error": error_text,
            "started_at": started_at,
            "finished_at": utc_now(),
            "seconds": seconds,
            "load_seconds": phases.get("load", 0.0),
            "to_gpu_seconds": phases.get("to_gpu", 0.0),
            **memory,
            "allocated_delta_mib": mib(torch.cuda.memory_allocated() - allocated_before),
            # 이 단계 동안의 최고점에서 시작 시점 할당량을 뺀 값. 옮기는 도중의 일시 사용량까지 잡힌다.
            "step_peak_delta_mib": mib(torch.cuda.max_memory_allocated() - allocated_before),
            "host_peak_rss_mib": host_peak_rss_mib(),
            "detail": {str(k): str(v) for k, v in detail.items()},
        }
        steps.append(step)
        mark = {"passed": "PASS", "failed": "OOM" if oom else "FAIL", "skipped": "SKIP"}[status]
        print(
            f"[{mark}] {name}: {seconds}s (load {step['load_seconds']}s, to_gpu {step['to_gpu_seconds']}s) "
            f"allocated {memory['allocated_mib']} MiB (+{step['allocated_delta_mib']}), "
            f"reserved {memory['reserved_mib']} MiB, step peak +{step['step_peak_delta_mib']} MiB, "
            f"device free {memory['device_free_mib']}/{memory['device_total_mib']} MiB, "
            f"host peak RSS {step['host_peak_rss_mib']} MiB {step['detail']}"
            + (f" error={error_text}" if error_text else ""),
            flush=True,
        )

    # 모든 단계를 고른 실행에서만 동시 로딩을 판정한다. 일부만 고르면 개별 로딩 결과만 의미가 있다.
    all_selected = list(STEP_NAMES) == selected
    all_passed = all(step["status"] == "passed" for step in steps)
    resident = {
        name: sorted({str(p.device) for p in gpu_model(name).parameters()})
        for name in GPU_MODEL_STEPS
        if name in loaded
    }
    off_gpu = {name: devices for name, devices in resident.items() if devices != [DEVICE]}
    concurrent_memory = gpu_memory()
    concurrent = {
        "evaluated": all_selected,
        "passed": all_selected
        and all_passed
        and not off_gpu
        and len(resident) == len(GPU_MODEL_STEPS),
        "resident_gpu_models": sorted(resident),
        "off_gpu_models": {k: ",".join(v) for k, v in off_gpu.items()},
        **concurrent_memory,
    }
    print(f"[INFO] 동시 로딩: {concurrent}", flush=True)

    created = sorted(watched_files() - files_before)
    downloaded = [
        path
        for path in created
        if Path(path).suffix.lower() in MODEL_ARTIFACT_SUFFIXES
        or not HF_MODEL_CACHE_DIRS.isdisjoint(Path(path).parts)
    ]
    print(
        f"[{'FAIL' if downloaded else 'PASS'}] 로딩 중 모델 다운로드 없음: {downloaded or created}",
        flush=True,
    )

    # 컨테이너를 내리기 전에 GPU 메모리를 직접 비워, 해제까지 정상으로 끝났는지 로그로 남긴다.
    loaded.clear()
    gc.collect()
    torch.cuda.empty_cache()
    released = gpu_memory()
    total_seconds = round(time.perf_counter() - run_started, 2)
    print(
        f"[INFO] 해제 후 allocated {released['allocated_mib']} MiB, reserved {released['reserved_mib']} MiB, "
        f"전체 실행 {total_seconds}s",
        flush=True,
    )

    failed = [step["name"] for step in steps if step["status"] != "passed"]
    oom_steps = [step["name"] for step in steps if step["oom"]]
    if failed or downloaded or (all_selected and not concurrent["passed"]):
        raise ModelLoadingCheckError(
            f"실패·건너뜀 {failed}, OOM {oom_steps}, 다운로드 {downloaded}, "
            f"동시 로딩 {concurrent['passed'] if all_selected else '미판정'}"
        )

    print("=== IDM-VTON Model Loading Check Passed ===", flush=True)
    return {
        "selected": [str(name) for name in selected],
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "total_seconds": float(total_seconds),
        "checks": checks,
        "steps": steps,
        "individual_passed": [step["name"] for step in steps if step["status"] == "passed"],
        "concurrent": concurrent,
        "released": released,
    }


@app.local_entrypoint()
def main(models: str = "all") -> None:
    # models: all | preprocess | idm_vton | 단계 이름(쉼표로 여러 개). 실패하면 원격 예외가 올라와 0 이 아닌 코드로 끝난다.
    import time

    parse_model_selection(models)
    started = time.perf_counter()
    try:
        result = check_model_loading.remote(models)
    finally:
        # 컨테이너 시작·이미지 준비 시간을 포함한 호출 전체 시간. GPU 과금 시간의 상한 추정에 쓴다.
        print(f"IDM-VTON model loading check wall time: {time.perf_counter() - started:.1f}s")
    for step in result["steps"]:
        print(
            f"{step['name']:>15} {step['status']:>7} {step['seconds']:>7.2f}s "
            f"alloc {step['allocated_mib']:>9.1f} MiB (+{step['allocated_delta_mib']:.1f}) "
            f"peak +{step['step_peak_delta_mib']:.1f} MiB"
        )
    print(f"concurrent: {result['concurrent']}")
    print(f"IDM-VTON model loading check total: {result['total_seconds']}s on {result['gpu_name']}")
