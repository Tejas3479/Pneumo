from fastapi.testclient import TestClient
import pytest
from app.main import app

class DummyModelManager:
    """
    Mock ModelManager to allow API testing without model files on disk.
    """
    def predict(self, image_bytes):
        return {
            "probability": 0.85,
            "prediction": "POSITIVE",
            "heatmap_base64": "data:image/jpeg;base64,abc"
        }

def test_root_endpoint():
    """
    Tests that the root endpoint serves the index.html page correctly.
    """
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "PneumoDetect" in response.text

def test_predict_endpoint_success(monkeypatch):
    """
    Tests that the /predict endpoint successfully parses input files,
    enqueues the prediction task, and returns the task ID.
    """
    class MockTask:
        id = "mock-task-id-123"

    monkeypatch.setattr("app.main.run_inference_task.delay", lambda *args, **kwargs: MockTask())
    
    client = TestClient(app)
    
    # 1-pixel transparent PNG base64 decoded as bytes
    dummy_png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    
    response = client.post(
        "/predict",
        files={"file": ("test.png", dummy_png_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    json_data = response.json()
    assert json_data == {"task_id": "mock-task-id-123", "status": "PENDING"}

def test_predict_endpoint_empty_file():
    """
    Tests that the API handles empty files by returning 400 Bad Request.
    """
    client = TestClient(app)
    
    response = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")}
    )
    
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
