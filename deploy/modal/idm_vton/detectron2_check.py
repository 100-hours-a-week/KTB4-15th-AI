import modal
from dependency_check import image

app = modal.App("lookddak-idm-vton-detectron2-check")

@app.function(image=image, timeout=300)
def check_detectron2() -> dict:
    import glob
    import sys

    sys.path.insert(0, "/opt/IDM-VTON/gradio_demo")

    binaries = glob.glob(
        "/opt/IDM-VTON/gradio_demo/detectron2/_C*.so"
    )
    print("Detectron2 binaries:", binaries)

    try:
        from detectron2 import _C

        print("Detectron2 C++ extension: SUCCESS")
        return {"success": True}

    except Exception as error:
        print(
            f"Detectron2 import failed: "
            f"{type(error).__name__}: {error}"
        )
        return {
            "success": False,
            "error": str(error),
        }


@app.local_entrypoint()
def main():
    print(check_detectron2.remote())
