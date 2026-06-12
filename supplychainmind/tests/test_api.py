from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    payload = [{
        "ShipmentID": "SHP001",
        "Origin": "Shanghai",
        "Destination": "Rotterdam",
        "Carrier": "Maersk",
        "ProductCategory": "Electronics",
        "DepartureDate": "2024-05-01",
        "ExpectedDelivery": "2024-05-10",
        "Weight_kg": 500.0,
        "WeatherRisk": 0.1,
        "PortCongestion": 0.2,
        "GeopoliticalSentiment": 0.05
    }]
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["ShipmentID"] == "SHP001"
    assert "PredictedDelay" in data["predictions"][0]
    assert "RiskLevel" in data["predictions"][0]
    assert "Explanation" in data["predictions"][0]

def test_heatmap():
    response = client.get("/api/heatmap")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert len(data["features"]) > 0

def test_simulate():
    shipments = [{
        "ShipmentID": "SHP001",
        "Origin": "Shanghai",
        "Destination": "Rotterdam",
        "Carrier": "Maersk",
        "ProductCategory": "Electronics",
        "DepartureDate": "2024-05-01",
        "ExpectedDelivery": "2024-05-10",
        "Weight_kg": 500.0,
        "WeatherRisk": 0.1,
        "PortCongestion": 0.2,
        "GeopoliticalSentiment": 0.05
    }]
    payload = {
        "affected_port": "Shanghai",
        "delay_days": 3,
        "shipments": shipments
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
