import numpy as np
import pytest

from inference.onnx_predictor import ONNXEmotionPredictor, EMOTION_LABELS


class TestONNXEmotionPredictor:
    def test_labels_defined(self):
        assert len(EMOTION_LABELS) == 7

    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        probs = ONNXEmotionPredictor._softmax(x)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert probs[2] > probs[1] > probs[0]

    def test_predict_no_face(self):
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            predictor = ONNXEmotionPredictor(
                model_path="models/saved/raf_mobilefacenet.onnx"
            )
        except Exception:
            pytest.skip("ONNX model not downloaded")
        results = predictor.predict_all_faces(blank)
        assert results == []

    def test_predict_random_noise(self):
        noisy = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        try:
            predictor = ONNXEmotionPredictor(
                model_path="models/saved/raf_mobilefacenet.onnx"
            )
        except Exception:
            pytest.skip("ONNX model not downloaded")
        results = predictor.predict_all_faces(noisy)
        assert isinstance(results, list)

    def test_preprocess_face_shape(self):
        face = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        try:
            predictor = ONNXEmotionPredictor(
                model_path="models/saved/raf_mobilefacenet.onnx"
            )
        except Exception:
            pytest.skip("ONNX model not downloaded")
        tensor = predictor._preprocess(face)
        assert tensor.shape == (1, 3, 112, 112)
        assert tensor.dtype == np.float32
        assert tensor.min() >= -1.0 and tensor.max() <= 1.0
