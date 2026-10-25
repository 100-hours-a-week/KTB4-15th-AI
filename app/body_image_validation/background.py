"""재사용 rembg ONNX session을 사용하는 배경 제거."""

import json
import logging
import os
import tempfile
import threading
import time
from io import BytesIO

from PIL import Image

logger = logging.getLogger(__name__)


class RembgBackgroundRemover:
    def __init__(
        self,
        model_path: str,
        enable_cpu_mem_arena: bool = False,
        providers: list[str] | None = None,
        profile_runs: int = 0,
    ) -> None:
        import onnxruntime as ort
        from rembg import new_session

        # rembg 가 sess_opts 없이 만드는 것과 같은 기본 SessionOptions 에서 arena 값만 바꾼다.
        # mem_pattern·스레드 수는 건드리지 않는다. providers 를 넘기지 않으면 rembg 가 고른다
        # (onnxruntime CPU 패키지에서는 CPU). Modal T4 worker 는 CUDA 를 명시해 넘긴다.
        sess_opts = ort.SessionOptions()
        sess_opts.enable_cpu_mem_arena = enable_cpu_mem_arena
        # 진단용. profile_runs 번 실행한 뒤 profiling 을 끝내고 실행별 provider 노드 수를 로그로 남긴다.
        # session 은 그대로 재사용하고, profiling 이 끝난 뒤의 실행은 기록하지 않는다.
        self._profile_runs = profile_runs
        self._runs = 0
        if profile_runs > 0:
            sess_opts.enable_profiling = True
            sess_opts.profile_file_prefix = os.path.join(tempfile.gettempdir(), "rembg_ort_profile")
        options = {"providers": list(providers)} if providers else {}
        self._session = new_session(
            "u2net_custom", model_path=model_path, sess_opts=sess_opts, **options
        )
        self._lock = threading.Lock()

    @property
    def providers(self) -> list[str]:
        """실제로 붙은 ONNX Runtime execution provider. CUDA 요청이 CPU 로 떨어졌는지 확인용."""
        return list(self._session.inner_session.get_providers())

    def remove(self, image: Image.Image) -> Image.Image:
        """PIL 이미지를 그대로 넘기고 RGBA PIL 이미지를 돌려받는다. 결과는 호출자가 닫는다.

        예전에는 PNG bytes 로 인코딩해 넘기고 결과 PNG 를 다시 decode 했다. rembg 는 PIL 입력이면
        PIL 로 돌려주므로 그 왕복(인코딩·복사·재decode)이 필요 없다.
        """
        from rembg import remove

        with self._lock:
            output = remove(image, session=self._session)
            if self._profile_runs:
                self._runs += 1
                if self._runs == self._profile_runs:
                    self._finish_profiling()
        if output.mode == "RGBA":
            return output
        try:
            return output.convert("RGBA")
        finally:
            if output is not image:
                output.close()


    def _finish_profiling(self) -> None:
        """profiling 을 끝내고 요약을 로그로 남긴다. 진단 실패가 요청을 실패시키지 않게 경고만 남긴다."""
        path = None
        try:
            started = time.perf_counter()
            path = self._session.inner_session.end_profiling()
            runs = summarize_ort_profile(path)
            logger.info(
                "rembg onnxruntime profile: runs=%s parse_ms=%.1f",
                json.dumps(runs), (time.perf_counter() - started) * 1000,
            )
        except Exception:
            logger.warning("rembg onnxruntime profiling summary failed", exc_info=True)
        finally:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass


def summarize_ort_profile(path: str) -> list[dict]:
    """ONNX Runtime profile JSON 을 실행(model_run)별로 나눠 provider 별 노드 수·노드 시간을 센다.

    이벤트는 기록된 순서대로 읽는다. 각 실행의 노드 이벤트 다음에 그 실행의 model_run 이벤트가 온다.
    (시간값으로 나누지 않으므로 snapshot 복원으로 시계가 바뀌어도 실행이 섞이지 않는다.)
    노드 시간(dur)은 ONNX Runtime 이 잰 kernel 호출 시간이다. CUDA kernel 은 비동기라 GPU 실행 시간과 같지 않다.
    model_run 시간은 출력을 CPU 로 가져올 때까지라 GPU 실행을 포함한 추론 전체 시간이다.
    """
    with open(path) as file:
        events = json.load(file)
    runs, nodes, node_us = [], {}, {}
    for event in events:
        args = event.get("args") or {}
        if event.get("cat") == "Node" and args.get("provider"):
            provider = args["provider"]
            nodes[provider] = nodes.get(provider, 0) + 1
            node_us[provider] = node_us.get(provider, 0) + int(event.get("dur", 0))
        elif event.get("cat") == "Session" and event.get("name") == "model_run":
            runs.append({
                "run": len(runs) + 1,
                "model_run_ms": round(int(event.get("dur", 0)) / 1000, 2),
                "nodes": nodes,
                "node_ms": {name: round(us / 1000, 2) for name, us in node_us.items()},
            })
            nodes, node_us = {}, {}
    return runs


def encode_png(image: Image.Image) -> bytes:
    # 압축 수준만 Pillow 기본값(6) 대신 4 로 둔다. 픽셀은 그대로이고, 대표 이미지(RGBA 1069x1600)
    # 측정에서 encode 가 약 50ms(-36%) 빨라지고 파일은 4.4% 커졌다. optimize 는 기본값(False) 그대로다.
    output = BytesIO()
    image.save(output, format="PNG", compress_level=4)
    return output.getvalue()
