import xgboost as xgb
import numpy as np
import joblib

class ShipmentDelayPredictor:
    def __init__(self, params=None):
        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": 42,
            "verbosity": 0
        }
        self.params = params if params else default_params
        self.model = None
        self.feature_importance = None

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X, y)
        self.feature_importance = self.model.feature_importances_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
