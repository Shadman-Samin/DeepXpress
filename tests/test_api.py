import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestAPI:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_predict_no_file(self, client):
        response = client.post("/predict/image")
        assert response.status_code == 422

    def test_predict_invalid_file(self, client):
        response = client.post(
            "/predict/image",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_blank_image(self, client):
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", blank)
        response = client.post(
            "/predict/image",
            files={"file": ("blank.jpg", buffer.tobytes(), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "faces" in data

    def test_model_loaded(self, client):
        from api.app import predictor
        assert predictor is not None
