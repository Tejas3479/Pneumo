import pytest
from src.data import generate_mock_shipment_data, enrich_with_external_data

def test_generate_data():
    df = generate_mock_shipment_data(100)
    assert df.shape[0] == 100
    assert "ShipmentID" in df.columns
    assert "ActualDelay" in df.columns

def test_enrich():
    df = generate_mock_shipment_data(10)
    enriched = enrich_with_external_data(df)
    assert "WeatherRisk" in enriched.columns
    assert "PortCongestion" in enriched.columns
    assert "GeopoliticalSentiment" in enriched.columns
