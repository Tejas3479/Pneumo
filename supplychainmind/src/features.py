import pandas as pd
import numpy as np
from geopy.distance import geodesic

CITIES = [
    ("Shanghai", "China", 31.2304, 121.4737),
    ("Singapore", "Singapore", 1.3521, 103.8198),
    ("Rotterdam", "Netherlands", 51.9244, 4.4777),
    ("Los Angeles", "USA", 33.9425, -118.4081),
    ("Hamburg", "Germany", 53.5511, 9.9937),
    ("Dubai", "UAE", 25.2048, 55.2708),
    ("New York", "USA", 40.7128, -74.0060),
    ("Shenzhen", "China", 22.5431, 114.0579),
    ("Antwerp", "Belgium", 51.2194, 4.4025),
    ("Mumbai", "India", 19.0760, 72.8777),
]

CARRIERS = ["Maersk", "MSC", "CMA CGM", "COSCO", "Hapag-Lloyd", "ONE", "Evergreen", "HMM"]
PRODUCT_CATEGORIES = ["Electronics", "Automotive", "Pharmaceuticals", "Apparel", "Food & Beverage", "Heavy Machinery"]

CARRIER_RELIABILITY = {
    "Maersk": 0.92, "MSC": 0.88, "CMA CGM": 0.89, "COSCO": 0.85,
    "Hapag-Lloyd": 0.90, "ONE": 0.87, "Evergreen": 0.84, "HMM": 0.82
}

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms enriched raw shipment DataFrame to a numeric feature matrix.
    """
    df = df.copy()
    
    # 1. Geodesic distance calculation
    df["distance_km"] = df.apply(
        lambda r: geodesic(
            (r["origin_lat"], r["origin_lon"]),
            (r["dest_lat"], r["dest_lon"])
        ).km, axis=1
    )
    
    # 2. Carrier reliability lookup
    df["carrier_reliability"] = df["carrier"].map(CARRIER_RELIABILITY).fillna(0.85)
    
    # 3. Port congestion, weather, and geopolitical risks (origin & destination estimates)
    df["port_congestion_origin"] = df["congestion"]
    df["port_congestion_destination"] = df["congestion"] * 0.9
    df["weather_risk_origin"] = df["weather_risk"]
    df["weather_risk_destination"] = df["weather_risk"] * 0.85
    df["geopolitical_risk_origin"] = df["geopolitical_risk"]
    df["geopolitical_risk_destination"] = df["geopolitical_risk"] * 0.95
    
    # 4. Departure Date attributes
    dep_dates = pd.to_datetime(df["departure_date"])
    df["day_of_week"] = dep_dates.dt.dayofweek
    df["month"] = dep_dates.dt.month
    
    # 5. One-hot encode product categories (static loop for output shape stability)
    for prod in PRODUCT_CATEGORIES:
        df[f"product_category_{prod}"] = (df["product_category"] == prod).astype(float)
        
    # 6. Stable label encoding mapping for high-cardinality categoricals
    df["origin_code"] = df["origin"].map({p[0]: idx for idx, p in enumerate(CITIES)}).fillna(-1)
    df["dest_code"] = df["destination"].map({p[0]: idx for idx, p in enumerate(CITIES)}).fillna(-1)
    df["carrier_code"] = df["carrier"].map({c: idx for idx, c in enumerate(CARRIERS)}).fillna(-1)
    
    # 7. Select only numeric features
    feature_cols = [
        "distance_km",
        "carrier_reliability",
        "port_congestion_origin",
        "port_congestion_destination",
        "weather_risk_origin",
        "weather_risk_destination",
        "geopolitical_risk_origin",
        "geopolitical_risk_destination",
        "day_of_week",
        "month",
        "origin_code",
        "dest_code",
        "carrier_code",
        "supplier_health"
    ] + [f"product_category_{prod}" for prod in PRODUCT_CATEGORIES]
    
    X = df[feature_cols].copy()
    
    # Impute missing values with column median
    for col in X.columns:
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X[col] = X[col].fillna(median_val)
        
    return X
