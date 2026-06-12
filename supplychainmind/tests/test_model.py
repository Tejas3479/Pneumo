import numpy as np
import pandas as pd
import pytest
import os
from src.model import SupplyChainDelayPredictor

def test_fit_predict():
    # Generate tiny synthetic feature dataset
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 20))
    y = np.random.rand(100) * 5.0
    
    model = SupplyChainDelayPredictor()
    model.fit(X, y)
    
    # Check predictions
    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert isinstance(preds, np.ndarray)
    
    # Check predictions with uncertainty
    mean_preds, lower_preds, upper_preds = model.predict_with_uncertainty(X[:5])
    assert len(mean_preds) == 5
    assert len(lower_preds) == 5
    assert len(upper_preds) == 5
    
    # Bounds check
    assert np.all(lower_preds <= mean_preds)
    assert np.all(mean_preds <= upper_preds)

def test_save_load(tmp_path):
    X = pd.DataFrame(np.random.rand(50, 10))
    y = np.random.rand(50) * 5.0
    
    model = SupplyChainDelayPredictor()
    model.fit(X, y)
    
    model_path = os.path.join(tmp_path, "xgb_model.json")
    model.save(model_path)
    
    assert os.path.exists(model_path)
    
    loaded = SupplyChainDelayPredictor.load(model_path)
    assert loaded.model_mean is not None
    assert loaded.model_lower is not None
    assert loaded.model_upper is not None
    
    preds_orig = model.predict(X[:5])
    preds_loaded = loaded.predict(X[:5])
    
    np.testing.assert_array_almost_equal(preds_orig, preds_loaded)
