import pandas as pd
from src.features import FeatureEngineering
from src.model import ShipmentDelayPredictor
from src.explanation import classify_risk

class ShipmentPredictionManager:
    def __init__(self, model_path, feature_path):
        self.model = ShipmentDelayPredictor.load(model_path)
        self.fe = FeatureEngineering.load(feature_path)

    def predict_one(self, df):
        X = self.fe.transform(df)
        delay = self.model.predict(X)[0]
        risk = classify_risk(delay)
        return delay, risk
