import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Force Celery to execute tasks synchronously (in-process) during testing
from app.tasks import celery_app
celery_app.conf.update(task_always_eager=True)

from app.main import app
from app.circuit_breaker import predict_with_breaker, inference_breaker

def test_health_endpoint():
    """
    Verifies that the /health liveness endpoint returns UP.
    """
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}

def test_ready_endpoint_offline():
    """
    Verifies that /ready returns HTTP 503 if Redis connection ping fails.
    """
    with patch("redis.from_url") as mock_from_url:
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Connection refused")
        mock_from_url.return_value = mock_redis
        
        client = TestClient(app)
        response = client.get("/ready")
        assert response.status_code == 503
        assert "Redis connection offline" in response.json()["detail"]

def test_ready_endpoint_online():
    """
    Verifies that /ready returns HTTP 200 if Redis connection ping succeeds.
    """
    with patch("redis.from_url") as mock_from_url:
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_from_url.return_value = mock_redis
        
        client = TestClient(app)
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "READY"}

def test_circuit_breaker_transitions():
    """
    Verifies that the circuit breaker opens after 5 failures and immediately
    degrades to the fallback prediction logic.
    """
    # Reset breaker state to clean start
    inference_breaker.close()
    
    mock_manager = MagicMock()
    mock_manager.predict.side_effect = Exception("Corrupted ONNX file")
    
    image_bytes = b"fake_bytes"
    
    # 1. Trigger 5 sequential failures to open the circuit breaker
    for _ in range(5):
        res = predict_with_breaker(mock_manager, image_bytes)
        assert res["prediction"] == "UNKNOWN (FALLBACK)"
        assert "degraded fallback mode" in res["text_justification"]
        
    # 2. Re-assert that the circuit is now open (6th call returns fallback immediately without executing inner function)
    mock_manager.predict.reset_mock()
    res = predict_with_breaker(mock_manager, image_bytes)
    assert res["prediction"] == "UNKNOWN (FALLBACK)"
    mock_manager.predict.assert_not_called()
    
    # Restore breaker state
    inference_breaker.close()

def test_async_prediction_flow():
    """
    Verifies the async predict and polling flow end-to-end using eager Celery execution and mocked AsyncResult.
    """
    client = TestClient(app)
    
    with open("tests/sample_chest.png", "rb") as f:
        image_bytes = f.read()
        
    files = {"file": ("sample_chest.png", image_bytes, "image/png")}
    
    # 1. Enqueue predict task
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    task_id = data["task_id"]
    
    # 2. Retrieve result from polling endpoint by mocking AsyncResult
    with patch("app.main.AsyncResult") as mock_async_result:
        mock_res = MagicMock()
        mock_res.state = "SUCCESS"
        mock_res.result = {
            "probability": 0.85,
            "uncertainty": 0.1,
            "prediction": "POSITIVE",
            "heatmap_base64": "",
            "counterfactual_base64": "",
            "text_justification": "Eager execution test success",
            "tcav_scores": {},
            "image_path": None
        }
        mock_async_result.return_value = mock_res
        
        res_response = client.get(f"/result/{task_id}")
        assert res_response.status_code == 200
        res_data = res_response.json()
        assert res_data["status"] == "SUCCESS"
        assert res_data["result"]["probability"] == 0.85
        assert res_data["result"]["prediction"] == "POSITIVE"
