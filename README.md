# DeepXpress — Facial Emotion Recognition

Real-time multi-face emotion recognition at **88.27% accuracy** (RAF-DB).
Detects faces with YuNet (DNN), aligns with 5-point landmarks, and classifies
7 emotions via MobileFaceNet ONNX — all in a portable desktop .exe.

**Emotions:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised

## Quick Start

### Desktop App (Windows)

<a href="https://mega.nz/file/ho0EULDC#1RDr6-o9PRtihOJOYQChgSr0MopfwybUjPJcrCnAsA4" target="_blank">Download DeepXpress.exe (75 MB)</a>

Double-click and go. Models auto-download on first launch.

### From Source
```bash
pip install -r requirements.txt
python main.py webcam
```

## Usage

| Command | Description |
|---------|-------------|
| `python main.py webcam` | Real-time webcam (720p) |
| `python main.py api` | FastAPI server on port 8000 |
| `python main.py streamlit` | Streamlit dashboard |
| `python main.py setup` | Pre-download all models |

### Build .exe
```bash
pip install pyinstaller
python scripts/build.py
# Output: dist/DeepXpress.exe
```

## Features

- **88.27% accuracy** on RAF-DB (MobileFaceNet ONNX, 4.5 MB)
- **YuNet DNN face detector** — handles angles, occlusions, multiple faces
- **5-point face alignment** — similarity warp to canonical 112×112
- **Temporal smoothing** — EMA with IoU-based face tracking for stable predictions
- **Per-emotion colors** — distinct color per emotion (red for Angry, yellow for Happy, etc.)
- **720p webcam** — threaded capture + inference with FPS/face count overlay
- **Portable .exe** — single file, no Python or torch needed (75 MB)
- **Auto-download** — models fetch on first run and cache alongside the .exe

## Benchmarks

| Backend | Accuracy | Size | Speed |
|---------|----------|------|-------|
| MobileFaceNet ONNX | **88.27%** (RAF-DB) | 4.5 MB | Fast |

## Project Structure

```
deepxpress/
├── app.py                  # Desktop launcher for .exe
├── main.py                 # CLI entry (webcam/api/streamlit/setup)
├── inference/
│   └── onnx_predictor.py   # YuNet + MobileFaceNet ONNX pipeline
├── ui/webcam.py            # 720p real-time UI with overlay
├── models/download_pretrained.py  # Auto-download ONNX + YuNet
├── api/app.py              # FastAPI server
├── configs/config.yaml     # Configuration
├── scripts/build.py        # PyInstaller build script
└── tests/                  # Test suite (16 tests)
```

## Tests

```bash
pip install pytest       # if not already installed
python -m pytest tests/ -v
```

## Requirements

Python 3.10+. Dependencies: `numpy`, `opencv-python`, `Pillow`, `onnxruntime`, `pyyaml`.

See `requirements.txt`.
