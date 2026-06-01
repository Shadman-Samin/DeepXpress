# AGENT.md

# DeepXpress — Facial Emotion Recognition System

## Project Overview

DeepXpress is a Computer Vision project that detects and classifies human
facial emotions from webcam feeds, images, and video streams in real time.

The system recognizes 7 emotional states:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprised

The pipeline: **YuNet face detection → 5-point landmark alignment →
MobileFaceNet ONNX inference → temporal EMA smoothing**.

Accuracy: **88.27%** on RAF-DB. Delivered as a portable Windows .exe.

---

# Agent Role

You are the AI Engineering Agent responsible for designing, developing,
testing, documenting, and improving the DeepXpress project.

---

# Core Responsibilities

## 1. System Architecture

```
Face Detection (YuNet) → Face Alignment → ONNX Inference → Temporal Smoothing → Display
```

Components:
- **models/download_pretrained.py** — auto-downloads ONNX + YuNet on first run
- **inference/onnx_predictor.py** — detection, alignment, inference, EMA smoothing
- **ui/webcam.py** — threaded capture, 720p visualization with overlay
- **api/app.py** — FastAPI server for remote image prediction
- **app.py** — desktop launcher for .exe distribution

All components remain loosely coupled and maintainable.

---

## 2. Model Pipeline

### Face Detection
- **YuNet DNN** from OpenCV Zoo (2023 March version)
- Handles angles, occlusions, multi-face
- Returns bounding box + 5 facial landmarks

### Face Alignment
- Similarity transform warps landmarks to canonical 112×112
- Same alignment the MobileFaceNet model was trained with

### Emotion Classification
- **MobileFaceNet ONNX** (88.27% on RAF-DB)
- Single forward pass per face (batch=1)
- Input: 1×3×112×112, normalized (0.5, 0.5)

### Temporal Smoothing
- EMA: `0.4 * raw + 0.6 * previous`
- IoU-based face tracking across frames
- Prevents emotion flickering

---

## 3. Inference Pipeline

1. Capture 720p frame from webcam
2. Run YuNet face detection
3. For each face: extract 5 landmarks, similarity-warp to 112×112
4. Preprocess: BGR→RGB, normalize to [-1, 1]
5. Run ONNX inference → softmax → 7-class probs
6. Apply EMA smoothing (tracked by IoU)
7. Render bounding boxes + emotion labels + FPS + face count

---

## 4. User Interface

### Desktop (default)
- OpenCV `cv2.imshow` with custom overlay
- 720p live feed, per-emotion color-coded boxes
- FPS counter, face count
- Press `q` or click X to exit

### Web
- **FastAPI** — `/predict/image` endpoint
- **Streamlit** — image upload with visualization

---

## 5. Deployment

### Distributable
- PyInstaller `--onefile --noconsole` → single `DeepXpress.exe` (75 MB)
- Models auto-download on first run (stored alongside .exe)
- Zero configuration, double-click to launch

### API
- `python main.py api` → FastAPI on port 8000
- Docker-ready

---

## 6. Documentation

Maintain:
- **README.md** — quick start, usage, benchmarks
- **AGENT.md** — architecture, workflow, conventions

Documentation must remain updated with code changes.

---

# Coding Standards

## Python Style

- PEP 8
- Type hints
- Modular structure
- Meaningful naming

## Project Structure

```
deepxpress/
├── app.py                     # Desktop launcher for .exe
├── main.py                    # CLI entry point
├── inference/
│   ├── __init__.py
│   └── onnx_predictor.py     # Core inference pipeline
├── ui/
│   ├── __init__.py
│   ├── webcam.py             # Real-time visualization
│   └── streamlit_app.py      # Streamlit dashboard
├── api/
│   ├── __init__.py
│   └── app.py                # FastAPI server
├── models/
│   ├── __init__.py
│   └── download_pretrained.py  # Auto-download models
├── configs/
│   ├── __init__.py
│   ├── config.py             # Configuration handler
│   └── config.yaml           # Settings
├── scripts/
│   └── build.py              # PyInstaller build script
├── tests/
│   ├── test_onnx_predictor.py
│   ├── test_config.py
│   └── test_api.py
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── AGENT.md
└── pyproject.toml
```

---

# Testing Requirements

Create tests for:
- ONNX inference pipeline (softmax, preprocessing, predict)
- Configuration loading
- API endpoints

Run: `python -m pytest tests/ -v`

Minimum: **16 tests passing**.

---

# Performance Targets

| Metric         | Current  |
| -------------- | -------- |
| Accuracy       | **88.27%** (RAF-DB) |
| Inference Time | < 20 ms  |
| FPS            | > 20     |

---

# Future Enhancements

- Emotion trend analysis over time
- Emotion analytics dashboard
- Mobile deployment (ONNX is already mobile-friendly)
- TensorRT acceleration
- Multimodal emotion recognition (face + voice)

---

# Success Criteria

The project is considered successful when:
- Facial emotions are accurately classified (88%+).
- Real-time webcam inference runs at 20+ FPS.
- Desktop .exe launches without any setup.
- Codebase is maintainable, tested, and well documented.

---

# Mission Statement

DeepXpress makes emotion recognition accessible, accurate, and deployable
through modern computer vision and ONNX runtime, with a focus on clean
architecture and zero-friction deployment.
