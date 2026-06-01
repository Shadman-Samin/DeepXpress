from pathlib import Path
from typing import Optional


MODEL_DIR = Path("models/saved")

ONNX_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "facial_expression_recognition/"
    "facial_expression_recognition_mobilefacenet_2022july.onnx"
)

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

ONNX_MODEL_PATH = MODEL_DIR / "raf_mobilefacenet.onnx"
YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet.onnx"


def download_onnx_model() -> Optional[Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if ONNX_MODEL_PATH.exists() and ONNX_MODEL_PATH.stat().st_size > 1000:
        return ONNX_MODEL_PATH

    import urllib.request
    print("Downloading MobileFaceNet (88.27% on RAF-DB)...")
    try:
        urllib.request.urlretrieve(ONNX_MODEL_URL, ONNX_MODEL_PATH)
        size = ONNX_MODEL_PATH.stat().st_size
        print(f"  Saved: {ONNX_MODEL_PATH} ({size // 1024} KB)")
        return ONNX_MODEL_PATH
    except Exception as e:
        print(f"  Download failed: {e}")
        ONNX_MODEL_PATH.unlink(missing_ok=True)
        return None


def download_yunet() -> Optional[Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if YUNET_MODEL_PATH.exists() and YUNET_MODEL_PATH.stat().st_size > 1000:
        return YUNET_MODEL_PATH

    import urllib.request
    print("Downloading YuNet face detector...")
    try:
        urllib.request.urlretrieve(YUNET_URL, YUNET_MODEL_PATH)
        size = YUNET_MODEL_PATH.stat().st_size
        print(f"  Saved: {YUNET_MODEL_PATH} ({size // 1024} KB)")
        return YUNET_MODEL_PATH
    except Exception as e:
        print(f"  Download failed: {e}")
        YUNET_MODEL_PATH.unlink(missing_ok=True)
        return None


def ensure_models() -> bool:
    onnx_ok = download_onnx_model() is not None
    yunet_ok = download_yunet() is not None
    if onnx_ok and yunet_ok:
        print("All models ready!")
        return True
    print("Some models failed to download.")
    return False
