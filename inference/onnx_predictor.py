import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


EMOTION_LABELS: list[str] = [
    "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"
]

# Standard 5 landmarks for a 112x112 aligned face
_STD_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)

FaceResult = tuple[str, float, tuple[int, int, int, int]]
MultiFaceResult = list[
    tuple[str, float, tuple[int, int, int, int], list[tuple[str, float]]]
]


def _resolve_path(path: str) -> str:
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path.cwd()
    candidate = base / path
    if candidate.exists():
        return str(candidate.resolve())
    return path


class ONNXEmotionPredictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        yunet_path: Optional[str] = None,
        confidence_threshold: float = 0.0,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self._weights_loaded = True

        _model_path = model_path or "models/saved/raf_mobilefacenet.onnx"
        _yunet_path = yunet_path or "models/saved/face_detection_yunet.onnx"

        _model_path = _resolve_path(_model_path)
        _yunet_path = _resolve_path(_yunet_path)

        # Emotion model
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(_model_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name

        # YuNet face detector
        self._yunet_path = _yunet_path
        self._yunet = None
        self._yunet_input_size = None

        # Temporal smoothing state
        self._tracks: list[dict] = []
        self._smooth_alpha = 0.4
        self._track_max_age = 5
        self._frame_count = 0

        self._warmed_up = False

    def _ensure_yunet(self, frame_w: int, frame_h: int):
        if (self._yunet is not None
                and self._yunet_input_size == (frame_w, frame_h)):
            return
        import os
        if not os.path.exists(self._yunet_path):
            self._download_yunet()
        self._yunet = cv2.FaceDetectorYN.create(
            model=self._yunet_path,
            config="",
            input_size=(frame_w, frame_h),
            score_threshold=0.7,
            nms_threshold=0.3,
            top_k=5000,
        )
        self._yunet_input_size = (frame_w, frame_h)

    def _download_yunet(self):
        import os
        import urllib.request
        url = (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        )
        os.makedirs(os.path.dirname(self._yunet_path), exist_ok=True)
        print("  Downloading YuNet face detector...")
        urllib.request.urlretrieve(url, self._yunet_path)
        size = os.path.getsize(self._yunet_path)
        print(f"  YuNet saved ({size // 1024} KB)")

    # ── Face alignment (from opencv_zoo) ──────────────────────────────

    @staticmethod
    def _similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray
                              ) -> np.ndarray:
        """Returns 2x3 affine matrix mapping src_pts -> dst_pts."""
        src = src_pts.astype(np.float64)
        dst = dst_pts.astype(np.float64)
        x1, y1 = src[:, 0], src[:, 1]
        x2, y2 = dst[:, 0], dst[:, 1]
        n = len(src)
        A = np.zeros((2 * n, 4))
        b = np.zeros(2 * n)
        for i in range(n):
            A[2 * i] = [x1[i], -y1[i], 1, 0]
            A[2 * i + 1] = [y1[i], x1[i], 0, 1]
            b[2 * i] = x2[i]
            b[2 * i + 1] = y2[i]
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)
        sc, ss, tx, ty = sol
        return np.array([[sc, -ss, tx], [ss, sc, ty]], dtype=np.float32)

    def _align_face(self, image: np.ndarray,
                    landmarks: np.ndarray) -> np.ndarray:
        """Warp face region to 112x112 aligned crop."""
        tfm = self._similarity_transform(landmarks, _STD_LANDMARKS)
        aligned = cv2.warpAffine(
            image, tfm, (112, 112), flags=cv2.INTER_LINEAR
        )
        return aligned

    # ── Preprocessing ─────────────────────────────────────────────────

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR aligned face -> [1,3,112,112] normalized tensor."""
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255.0
        norm = (norm - 0.5) / 0.5
        return norm.transpose(2, 0, 1)[np.newaxis, ...]

    # ── Temporal smoothing ────────────────────────────────────────────

    def _match_tracks(self, rects: list[tuple]
                      ) -> tuple[list[dict], list[int]]:
        """Match detections to existing tracks by IoU. Returns matched
        tracks list and list of unmatched detection indices."""
        matched = [None] * len(rects)
        used = [False] * len(self._tracks)
        for i, (x, y, w, h) in enumerate(rects):
            best_iou = 0.3
            best_t = None
            for ti, t in enumerate(self._tracks):
                if used[ti]:
                    continue
                tx, ty, tw, th = t["rect"]
                iou = self._iou((x, y, w, h), (tx, ty, tw, th))
                if iou > best_iou:
                    best_iou = iou
                    best_t = ti
            if best_t is not None:
                used[best_t] = True
                matched[i] = self._tracks[best_t]
        unmatched = [i for i, m in enumerate(matched) if m is None]
        return matched, unmatched

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        xi = max(ax, bx)
        yi = max(ay, by)
        wi = min(ax + aw, bx + bw) - xi
        hi = min(ay + ah, by + bh) - yi
        if wi <= 0 or hi <= 0:
            return 0.0
        inter = wi * hi
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    # ── Main inference ────────────────────────────────────────────────

    def warmup(self, n: int = 3) -> None:
        if self._warmed_up:
            return
        dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)
        for _ in range(n):
            self.session.run(None, {self.input_name: dummy})
        self._warmed_up = True

    def predict_all_faces(self, image: np.ndarray) -> MultiFaceResult:
        h, w = image.shape[:2]
        self._ensure_yunet(w, h)
        self._frame_count += 1

        self._yunet.setInputSize((w, h))
        _, raw_faces = self._yunet.detect(image)
        if raw_faces is None:
            self._tracks = []
            return []

        # Parse YuNet output: each row = [x, y, w, h, x1..y5, conf]
        face_rects = []
        face_landmarks = []
        for row in raw_faces:
            x, y, fw, fh = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            lm = row[4:14].reshape(5, 2).astype(np.float64)
            face_rects.append((x, y, fw, fh))
            face_landmarks.append(lm)

        # Run inference + smoothing
        results: MultiFaceResult = []
        matched, unmatched_idx = self._match_tracks(face_rects)

        for i, (rect, lm) in enumerate(zip(face_rects, face_landmarks)):
            aligned = self._align_face(image, lm)
            tensor = self._preprocess(aligned)
            outputs = self.session.run(None, {self.input_name: tensor})
            raw_probs = self._softmax(outputs[0][0])

            # Temporal smoothing
            track = matched[i]
            if track is not None:
                track["rect"] = rect
                track["age"] = 0
                prev = track["smooth"]
                track["smooth"] = (self._smooth_alpha * raw_probs
                                   + (1 - self._smooth_alpha) * prev)
            else:
                self._tracks.append({
                    "rect": rect,
                    "smooth": raw_probs.copy(),
                    "age": 0,
                })
                track = self._tracks[-1]

            probs = track["smooth"]
            top_idx = int(probs.argmax())
            confidence = float(probs[top_idx])
            emotion = EMOTION_LABELS[top_idx]
            all_scores = []
            for j in range(len(EMOTION_LABELS)):
                all_scores.append((EMOTION_LABELS[j], float(probs[j])))
            results.append((emotion, confidence, rect, all_scores))

        # Expire old tracks
        self._tracks = [
            t for t in self._tracks if t["age"] < self._track_max_age
        ]
        for t in self._tracks:
            t["age"] += 1

        results.sort(key=lambda r: r[2][2] * r[2][3], reverse=True)
        return results

    def predict_emotions(
        self, image: np.ndarray
    ) -> tuple[list, list[str], list[float]]:
        results = self.predict_all_faces(image)
        faces = []
        emotions = []
        confidences = []
        for emotion, confidence, rect, _ in results:
            x, y, w, h = rect
            faces.append([x, y, w, h, confidence])
            emotions.append(emotion)
            confidences.append(confidence)
        return faces, emotions, confidences

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @property
    def has_trained_weights(self) -> bool:
        return self._weights_loaded
