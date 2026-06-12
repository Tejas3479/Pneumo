import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    csv_content = (
        "shipment_id,origin,destination,carrier,departure_date,product_category,actual_delay_days\n"
        "SHP00001,Shanghai,Rotterdam,Maersk,2024-05-01,Electronics,1.2\n"
    )
    
    mock_enriched = pd.DataFrame([{
        "shipment_id": "SHP00001",
        "origin": "Shanghai",
        "destination": "Rotterdam",
        "carrier": "Maersk",
        "departure_date": "2024-05-01",
        "product_category": "Electronics",
        "actual_delay_days": 1.2,
        "origin_lat": 31.2304,
        "origin_lon": 121.4737,
        "dest_lat": 51.9244,
        "dest_lon": 4.4777,
        "supplier_health": 85.0,
        "weather_risk": 0.1,
        "congestion": 0.2,
        "geopolitical_risk": 0.05
    }])
    
    mock_predictions = {
        "shipments": [{
            "shipment_id": "SHP00001",
            "origin": "Shanghai",
            "destination": "Rotterdam",
            "carrier": "Maersk",
            "departure_date": "2024-05-01",
            "product_category": "Electronics",
            "predicted_delay": 2.5,
            "lower_bound": 1.0,
            "upper_bound": 4.5,
            "confidence": "1.0 to 4.5 days",
            "risk_level": "Medium",
            "explanation": "Shipment SHP00001 from Shanghai to Rotterdam via Maersk has a 25% chance of being delayed by 2.5 days."
        }]
    }

    # Patch enrich_shipments and pipeline.predict
    with patch("app.main.enrich_shipments", return_value=mock_enriched), \
         patch("app.main.pipeline.predict", return_value=mock_predictions):
        
        response = client.post(
            "/predict",
            files={"file": ("test.csv", csv_content, "text/csv")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "shipments" in data
        assert len(data["shipments"]) == 1
        assert data["shipments"][0]["shipment_id"] == "SHP00001"
        assert data["shipments"][0]["risk_level"] == "Medium"
        assert "predicted_delay" in data["shipments"][0]

def test_heatmap_endpoint():
    mock_ports = pd.DataFrame([
        {"port_name": "Shanghai", "country": "China", "latitude": 31.2304, "longitude": 121.4737},
        {"port_name": "Rotterdam", "country": "Netherlands", "latitude": 51.9244, "longitude": 4.4777}
    ])
    
    mock_external = pd.DataFrame([
        {"origin": "Shanghai", "destination": "Rotterdam", "weather_risk": 0.1, "congestion": 0.2, "geopolitical_risk": 0.05}
    ])
    
    with patch("pandas.read_csv") as mock_read_csv:
        # Side effect to return ports_df on first call, external_df on second call
        mock_read_csv.side_effect = [mock_ports, mock_external]
        
        response = client.get("/heatmap")
        
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        assert "features" in data
        assert len(data["features"]) == 2
        assert data["features"][0]["properties"]["name"] in ["Shanghai", "Rotterdam"]
        assert "risk_score" in data["features"][0]["properties"]
        assert "explanation" in data["features"][0]["properties"]

def test_simulate_endpoint():
    mock_shipments = pd.DataFrame([
        {"shipment_id": "SHP00001", "origin": "Shanghai", "destination": "Rotterdam", "carrier": "Maersk", "departure_date": "2024-05-01", "product_category": "Electronics"}
    ])
    
    mock_enriched = pd.DataFrame([{
        "shipment_id": "SHP00001", "origin": "Shanghai", "destination": "Rotterdam", "carrier": "Maersk", "departure_date": "2024-05-01", "product_category": "Electronics", "congestion": 0.2
    }])
    
    mock_predictions = {
        "shipments": [{
            "shipment_id": "SHP00001", "origin": "Shanghai", "destination": "Rotterdam", "predicted_delay": 5.2, "risk_level": "High"
        }]
    }
    
    with patch("pandas.read_csv", return_value=mock_shipments), \
         patch("app.main.enrich_shipments", return_value=mock_enriched), \
         patch("app.main.pipeline.predict", return_value=mock_predictions):
         
        payload = {"port": "Shanghai", "days_closed": 3}
        response = client.post("/simulate", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "shipments" in data
        assert len(data["shipments"]) == 1
        assert data["shipments"][0]["risk_level"] == "High"
