import numpy as np
import pytest
from src.model import ShipmentDelayPredictor

def test_fit_predict():
    X = np.random.rand(100, 5)
    y = np.random.rand(100) * 10
    model = ShipmentDelayPredictor()
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5

def test_save_load(tmp_path):
    model = ShipmentDelayPredictor()
    model.fit(np.random.rand(50, 5), np.random.rand(50))
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = ShipmentDelayPredictor.load(path)
    assert loaded.model is not None
