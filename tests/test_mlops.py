import os
import sqlite3
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from mlops.mlflow_tracking import get_mlflow_logger
from mlops.drift_monitor import compute_psi
from src.regulatory import log_prediction_audit, verify_audit_trail

def test_mlflow_logger_init():
    """
    Test 1: Verifies that the MLflow Logger can be successfully created.
    """
    logger = get_mlflow_logger(experiment_name="test_experiment", run_name="test_run")
    assert logger is not None
    assert logger._experiment_name == "test_experiment"
    assert "mlruns" in logger._tracking_uri

def test_psi_calculation():
    """
    Test 2: Verifies PSI behaves correctly under similar/different distributions.
    """
    # Identical distributions should have PSI close to zero
    expected = [0.1, 0.2, 0.3, 0.4, 0.5] * 20
    actual = [0.1, 0.2, 0.3, 0.4, 0.5] * 20
    psi_val = compute_psi(expected, actual)
    assert psi_val < 0.05
    
    # Different distributions should have a positive PSI
    different_actual = [0.8, 0.9, 0.7, 0.85, 0.9] * 20
    psi_diff = compute_psi(expected, different_actual)
    assert psi_diff > 0.25

def test_audit_ledger_tampering():
    """
    Test 3: Verifies that manual updates to database records break row hash verification.
    """
    # Initialize / clean database
    db_path = "data/audit_ledger.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    img_bytes = b"fake_chest_xray_image_data_bytes"
    
    # Log two predictions
    log_prediction_audit(img_bytes, 0.123, "NEGATIVE", 0.45, 0.12)
    log_prediction_audit(img_bytes, 0.987, "POSITIVE", 0.46, 0.13)
    
    # Assert ledger starts off as valid
    valid, mismatches = verify_audit_trail()
    assert valid is True
    assert len(mismatches) == 0
    
    # Tamper with the first row's probability score in SQLite directly
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE audit_ledger SET probability = 0.500 WHERE id = 1")
    conn.commit()
    conn.close()
    
    # Assert verification now fails, flagging the mismatched row ID 1
    valid_after_tamper, mismatches_after_tamper = verify_audit_trail()
    assert valid_after_tamper is False
    assert 1 in mismatches_after_tamper

def test_endpoints():
    """
    Test 4: Hits /metrics/drift, /audit-ledger/verify, and /regulatory/model-card endpoints.
    """
    client = TestClient(app)
    
    # 1. Test model card endpoint
    response = client.post("/regulatory/model-card")
    assert response.status_code == 200
    res_data = response.json()
    assert "task_id" in res_data
    task_id = res_data["task_id"]
    
    # Wait for task completion
    res_response = client.get(f"/result/{task_id}")
    assert res_response.status_code == 200
    assert res_response.json()["status"] == "SUCCESS"
    assert res_response.json()["result"]["status"] == "success"
    assert os.path.exists("model_card.md")
    
    # 2. Test audit verification endpoint
    # Tampered from the previous test, so it should be invalid
    response = client.get("/audit-ledger/verify")
    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert 1 in response.json()["mismatches"]
    
    # Let's reset the DB to verify valid responses
    db_path = "data/audit_ledger.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    log_prediction_audit(b"test", 0.1, "NEGATIVE", 0.4, 0.1)
    
    response = client.get("/audit-ledger/verify")
    assert response.status_code == 200
    assert response.json()["valid"] is True
    
    # 3. Test metrics drift endpoint
    response = client.get("/metrics/drift")
    assert response.status_code == 200
    res_data = response.json()
    assert "task_id" in res_data
    drift_task_id = res_data["task_id"]
    
    # Wait for drift task completion
    res_response = client.get(f"/result/{drift_task_id}")
    assert res_response.status_code == 200
    result_json = res_response.json()
    assert result_json["status"] == "SUCCESS"
    data = result_json["result"]
    assert "psi_mean" in data
    assert "psi_std" in data
    assert "drift_detected" in data
