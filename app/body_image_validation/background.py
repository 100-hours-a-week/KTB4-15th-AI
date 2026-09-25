"""재사용 rembg ONNX session을 사용하는 배경 제거."""

import threading
from io import BytesIO

from PIL import Image


class RembgBackgroundRemover:
    def __init__(self, model_path: str) -> None:
        from rembg import new_session

        self._session = new_session("u2net_custom", model_path=model_path)
        self._lock = threading.Lock()

    def remove(self, image: Image.Image) -> Image.Image:
        from rembg import remove

        source = BytesIO()
        image.save(source, format="PNG")
        with self._lock:
            output = remove(source.getvalue(), session=self._session)
        result = Image.open(BytesIO(output)).convert("RGBA")
        result.load()
        return result


def encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
