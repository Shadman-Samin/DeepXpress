import sys
import time
from pathlib import Path

from models.download_pretrained import ensure_models


def _init_paths() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).resolve().parent
    return base


def main():
    base = _init_paths()

    print("DeepXpress — Facial Emotion Recognition")
    print("=" * 40)
    print("Checking models...")

    if not ensure_models():
        input("\nModel download failed. Check your internet and try again.\nPress Enter to exit...")
        sys.exit(1)

    print("\nStarting webcam... (press 'q' in the window to quit)\n")
    time.sleep(1)

    from ui.webcam import run_webcam

    run_webcam()


if __name__ == "__main__":
    main()
