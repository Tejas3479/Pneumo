import xgboost as xgb
import numpy as np
import joblib

class SupplyChainDelayPredictor:
    def __init__(self, params=None):
        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "random_state": 42,
            "verbosity": 0
        }
        self.params = params if params else default_params
        self.model_mean = None
        self.model_lower = None
        self.model_upper = None
        self.feature_importances_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # 1. Fit mean model
        self.model_mean = xgb.XGBRegressor(**self.params)
        if X_val is not None and y_val is not None:
            self.model_mean.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model_mean.fit(X_train, y_train)
        
        # 2. Fit lower quantile (5th percentile) model
        lower_params = self.params.copy()
        lower_params["objective"] = "reg:quantileerror"
        lower_params["quantile_alpha"] = 0.05
        if "eval_metric" in lower_params:
            lower_params["eval_metric"] = "mphe"
        self.model_lower = xgb.XGBRegressor(**lower_params)
        self.model_lower.fit(X_train, y_train)

        # 3. Fit upper quantile (95th percentile) model
        upper_params = self.params.copy()
        upper_params["objective"] = "reg:quantileerror"
        upper_params["quantile_alpha"] = 0.95
        if "eval_metric" in upper_params:
            upper_params["eval_metric"] = "mphe"
        self.model_upper = xgb.XGBRegressor(**upper_params)
        self.model_upper.fit(X_train, y_train)

        self.feature_importances_ = self.model_mean.feature_importances_
        return self

    def predict(self, X):
        if self.model_mean is None:
            raise ValueError("Model is not fitted yet.")
        return self.model_mean.predict(X)

    def predict_with_uncertainty(self, X):
        if self.model_mean is None or self.model_lower is None or self.model_upper is None:
            raise ValueError("All quantile models must be fitted.")
        mean_pred = self.model_mean.predict(X)
        lower_pred = self.model_lower.predict(X)
        upper_pred = self.model_upper.predict(X)
        
        # Ensure logical bounds
        lower_pred = np.minimum(lower_pred, mean_pred)
        upper_pred = np.maximum(upper_pred, mean_pred)
        
        return mean_pred, lower_pred, upper_pred

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

