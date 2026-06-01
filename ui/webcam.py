import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from inference.onnx_predictor import ONNXEmotionPredictor


@dataclass
class DetectionResult:
    faces: list = field(default_factory=list)
    emotions: list = field(default_factory=list)
    confidences: list = field(default_factory=list)


@dataclass
class FrameData:
    frame: np.ndarray
    detections: DetectionResult = field(default_factory=DetectionResult)
    fps: float = 0.0
    inference_time: float = 0.0
    timestamp: float = 0.0


class WebcamEmotionDetector:
    def __init__(
        self,
        source: int | str = 0,
        model_path: Optional[Path] = None,
        yunet_path: Optional[Path] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        inference_interval: float = 0.0,
        show_visualization: bool = True,
    ):
        self.source = source
        self.model_path = model_path
        self.yunet_path = yunet_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.inference_interval = inference_interval
        self.show_visualization = show_visualization

        self.predictor: Optional[ONNXEmotionPredictor] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False

        # Threading
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.result_queue: queue.Queue = queue.Queue(maxsize=2)
        self.current_frame: Optional[np.ndarray] = None
        self.current_detection = DetectionResult()
        self.lock = threading.Lock()

        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        self.callback: Optional[Callable] = None

    def start(self) -> None:
        self.running = True
        self._init_camera()
        self.predictor = ONNXEmotionPredictor(
            model_path=str(self.model_path) if self.model_path else None,
            yunet_path=str(self.yunet_path) if self.yunet_path else None,
        )

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )

        self.capture_thread.start()
        self.inference_thread.start()

        if self.show_visualization:
            self._visualization_loop()

    def stop(self) -> None:
        self.running = False
        if self.cap:
            self.cap.release()

    def set_callback(self, callback: Callable) -> None:
        self.callback = callback

    def _init_camera(self) -> None:
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def _capture_loop(self) -> None:
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    continue

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)

                self.current_frame = frame
                with self.lock:
                    self.frame_count += 1
                    elapsed = time.time() - self.fps_start_time
                    if elapsed >= 1.0:
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.fps_start_time = time.time()

    def _inference_loop(self) -> None:
        last_inference_time = 0.0

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            current_time = time.time()
            if current_time - last_inference_time >= self.inference_interval:
                if frame is not None:
                    start_time = time.time()
                    faces, emotions, confidences = (
                        self.predictor.predict_emotions(frame)
                    )
                    inference_time = (time.time() - start_time) * 1000

                    result = DetectionResult(
                        faces=faces,
                        emotions=emotions,
                        confidences=confidences,
                    )

                    with self.lock:
                        self.current_detection = result

                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.result_queue.put(result)

                    if self.callback:
                        self.callback(result)

                    last_inference_time = current_time

    EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
        "Angry": (0, 0, 255),
        "Disgust": (0, 102, 51),
        "Fear": (128, 128, 128),
        "Happy": (0, 255, 255),
        "Neutral": (255, 255, 255),
        "Sad": (255, 128, 0),
        "Surprised": (255, 0, 255),
    }

    def _visualization_loop(self) -> None:
        window_name = "DeepXpress - Facial Emotion Recognition v1.0"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while self.running:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.stop()
                break

            frame = self.current_frame
            if frame is None:
                continue

            display = frame.copy()
            with self.lock:
                det = self.current_detection

            for face, emotion, confidence in zip(
                det.faces, det.emotions, det.confidences
            ):
                x1, y1, w, h = face[:4]
                x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                label = f"{emotion} ({confidence:.1%})"

                color = self.EMOTION_COLORS.get(emotion, (0, 255, 0))
                cv2.rectangle(display, (x1, y1), (x1 + w, y1 + h), color, 2)

                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(
                    display,
                    (x1, y1 - label_h - 4),
                    (x1 + label_w, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    display,
                    label,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                )

            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(
                display,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            faces_text = f"Faces: {len(det.faces)}"
            cv2.putText(
                display,
                faces_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop()
                break

        cv2.destroyAllWindows()


def run_webcam(
    model_path: Optional[Path] = None,
    yunet_path: Optional[Path] = None,
    show_visualization: bool = True,
) -> None:
    detector = WebcamEmotionDetector(
        source=0,
        model_path=model_path,
        yunet_path=yunet_path,
        inference_interval=0.0,
        show_visualization=show_visualization,
    )
    detector.start()
