from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from inference.onnx_predictor import ONNXEmotionPredictor

predictor: Optional[ONNXEmotionPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = ONNXEmotionPredictor()
    yield


app = FastAPI(title="DeepXpress — Emotion Recognition API", lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "DeepXpress API", "endpoints": ["/predict", "/predict/image"]}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "MobileFaceNet ONNX (88.27% RAF-DB)"}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(
            status_code=400, content={"error": "Invalid image file"}
        )

    faces, emotions, confidences = predictor.predict_emotions(image)

    results = []
    for face, emotion, confidence in zip(faces, emotions, confidences):
        x1, y1, w, h = face[:4]
        results.append(
            {
                "bbox": {"x": int(x1), "y": int(y1), "w": int(w), "h": int(h)},
                "emotion": emotion,
                "confidence": float(confidence),
            }
        )

    return {"faces": len(results), "results": results}
