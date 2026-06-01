# DeepXpress — Facial Emotion Recognition

Real-time multi-face emotion recognition with **88.27% accuracy** (RAF-DB). Detects faces using YuNet (DNN), aligns with 5-point landmarks, and classifies 7 emotions via MobileFaceNet ONNX.

**Emotions:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised

## Quick Start

```bash
pip install -r requirements.txt

# Recommended — MobileFaceNet ONNX (88.27% on RAF-DB, fast)
python main.py webcam --backend onnx

# Fast fallback — ResNet18
python main.py webcam

# Transformer — ViT (84.3%)
python main.py webcam --backend transformer
```

Models auto-download on first run.

## Usage

```bash
# Webcam (720p)
python main.py webcam --backend onnx               # Best accuracy
python main.py webcam --backend transformer         # ViT
python main.py webcam                               # ResNet18
python main.py webcam --camera 1

# Single image prediction
python main.py predict --image path/to/image.jpg

# Train from scratch
python main.py train --epochs 100 --batch-size 64

# API server
python main.py api --port 8000

# Streamlit dashboard
python main.py streamlit --port 8501
```

## Benchmarks

| Backend | Accuracy | Speed | Size |
|---------|----------|-------|------|
| MobileFaceNet ONNX | **88.27%** on RAF-DB | Fast | 4.5 MB |
| ViT (HuggingFace) | 84.3% on FER2013 | Slow | ~340 MB |
| ResNet18 | ~71% (target) | Fastest | 44 MB |

## Features

- **YuNet DNN face detector** — accurate detection, handles angles/occlusions
- **5-point face alignment** — warps faces to standard position before inference
- **Temporal smoothing** — EMA across frames for stable predictions
- **Multi-face** — all faces detected and classified
- **720p webcam** — threaded capture + inference with FPS overlay
- **Confidence bars** — per-face emotion probability distribution
- **Auto-setup** — all models download on first run

## Project Structure

```
deepxpress/
├── main.py                 # CLI entry
├── inference/
│   ├── onnx_predictor.py   #   YuNet + MobileFaceNet (recommended)
│   ├── predictor.py        #   ResNet18
│   ├── hf_predictor.py     #   HuggingFace ViT
│   └── detector.py         #   Haar cascade fallback
├── ui/webcam.py            # 720p real-time UI
├── models/                 # Model definitions + downloads
├── training/               # Training pipeline
├── data/                   # Dataset + preprocessing
├── api/                    # FastAPI server
├── configs/                # YAML configuration
└── tests/                  # Test suite
```

## Tests

```bash
python -m pytest tests/ -v
```

## Requirements

Python 3.10+. See `requirements.txt`.
