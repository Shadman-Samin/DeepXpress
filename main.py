import argparse
import sys
from pathlib import Path

from models.download_pretrained import ensure_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepXpress - Facial Emotion Recognition System"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="webcam",
        choices=["webcam", "api", "streamlit", "setup"],
        help="Operation mode",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API server host")

    args = parser.parse_args()

    if args.mode == "webcam":
        _run_webcam(args)
    elif args.mode == "api":
        _run_api(args)
    elif args.mode == "streamlit":
        _run_streamlit(args)
    elif args.mode == "setup":
        _run_setup()
    else:
        parser.print_help()


def _run_webcam(args) -> None:
    from ui.webcam import run_webcam

    ensure_models()
    run_webcam()


def _run_api(args) -> None:
    ensure_models()
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


def _run_streamlit(args) -> None:
    import subprocess
    import sys

    ensure_models()
    cmd = [sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py"]
    if args.port:
        cmd.extend(["--server.port", str(args.port)])
    subprocess.run(cmd)


def _run_setup() -> None:
    if ensure_models():
        print("Setup complete! All models ready.")
    else:
        print("Setup failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
