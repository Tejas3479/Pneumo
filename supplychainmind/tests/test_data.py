import os
import pandas as pd
import pytest
from src.data import load_shipment_data, enrich_shipments, split_train_test
import generate_mock_data

def test_data_pipeline(tmp_path):
    # 1. Run generate_mock_data inside a temp path
    # We override the output paths to temp directory
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        generate_mock_data.main()
        
        # Check files exist
        assert os.path.exists("data/shipments.csv")
        assert os.path.exists("data/ports.csv")
        assert os.path.exists("data/suppliers.csv")
        assert os.path.exists("data/external_factors.csv")
        
        # 2. Test load_shipment_data
        df_shipments = load_shipment_data("data/shipments.csv")
        assert len(df_shipments) == 5000
        assert "shipment_id" in df_shipments.columns
        assert "actual_delay_days" in df_shipments.columns
        
        # 3. Test enrich_shipments
        df_enriched = enrich_shipments(
            df_shipments,
            "data/ports.csv",
            "data/suppliers.csv",
            "data/external_factors.csv"
        )
        assert "origin_lat" in df_enriched.columns
        assert "dest_lat" in df_enriched.columns
        assert "supplier_health" in df_enriched.columns
        assert "weather_risk" in df_enriched.columns
        
        # 4. Test split_train_test
        X_train, X_test, y_train, y_test = split_train_test(
            df_enriched, target="actual_delay_days", test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 4000
        assert len(X_test) == 1000
        assert len(y_train) == 4000
        assert len(y_test) == 1000
        
    finally:
        os.chdir(original_cwd)
