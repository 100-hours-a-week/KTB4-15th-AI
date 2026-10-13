"""재사용 rembg ONNX session을 사용하는 배경 제거."""

import threading
from io import BytesIO

from PIL import Image


class RembgBackgroundRemover:
    def __init__(
        self,
        model_path: str,
        enable_cpu_mem_arena: bool = False,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort
        from rembg import new_session

        # rembg 가 sess_opts 없이 만드는 것과 같은 기본 SessionOptions 에서 arena 값만 바꾼다.
        # mem_pattern·스레드 수는 건드리지 않는다. providers 를 넘기지 않으면 rembg 가 고른다
        # (onnxruntime CPU 패키지에서는 CPU). Modal T4 worker 는 CUDA 를 명시해 넘긴다.
        sess_opts = ort.SessionOptions()
        sess_opts.enable_cpu_mem_arena = enable_cpu_mem_arena
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
        if output.mode == "RGBA":
            return output
        try:
            return output.convert("RGBA")
        finally:
            if output is not image:
                output.close()


def encode_png(image: Image.Image) -> bytes:
    # 압축 수준만 Pillow 기본값(6) 대신 4 로 둔다. 픽셀은 그대로이고, 대표 이미지(RGBA 1069x1600)
    # 측정에서 encode 가 약 50ms(-36%) 빨라지고 파일은 4.4% 커졌다. optimize 는 기본값(False) 그대로다.
    output = BytesIO()
    image.save(output, format="PNG", compress_level=4)
    return output.getvalue()
