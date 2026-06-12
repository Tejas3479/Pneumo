import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from geopy.distance import geodesic
import joblib

class FeatureEngineering:
    def __init__(self, target_col="ActualDelay"):
        self.target_col = target_col
        self.categorical_cols = ["Origin", "Destination", "Carrier", "ProductCategory"]
        self.numeric_cols = ["Weight_kg", "WeatherRisk", "PortCongestion", "GeopoliticalSentiment"]
        self.distance_col = "Distance_km"
        self.derived_cols = []
        self.preprocessor = None
        self.fitted = False

    def _add_derived_features(self, df):
        df = df.copy()
        # Distance in km
        df[self.distance_col] = df.apply(
            lambda r: geodesic(
                (r["Origin_Lat"], r["Origin_Lon"]),
                (r["Dest_Lat"], r["Dest_Lon"])
            ).km, axis=1
        )
        # Days between departure and expected delivery
        df["DepToExpDays"] = (
            pd.to_datetime(df["ExpectedDelivery"]) - pd.to_datetime(df["DepartureDate"])
        ).dt.days
        self.derived_cols = [self.distance_col, "DepToExpDays"]
        return df

    def fit(self, df, y=None):
        df = self._add_derived_features(df)
        # Use target encoding for categoricals
        cat_pipe = Pipeline([
            ("target_enc", TargetEncoder(target_type="continuous", smooth=10, random_state=42))
        ])
        num_pipe = Pipeline([
            ("scaler", StandardScaler())
        ])
        self.preprocessor = ColumnTransformer([
            ("cat", cat_pipe, self.categorical_cols),
            ("num", num_pipe, self.numeric_cols + self.derived_cols)
        ])
        self.preprocessor.fit(df, y)
        self.fitted = True
        return self

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Must fit before transform")
        df = self._add_derived_features(df)
        X = self.preprocessor.transform(df)
        return X

    def get_feature_names(self):
        return self.categorical_cols + self.numeric_cols + self.derived_cols

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
