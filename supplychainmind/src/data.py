import os
import pandas as pd
import numpy as np

def load_shipment_data(csv_path: str) -> pd.DataFrame:
    """
    Reads the shipment CSV and returns it as a DataFrame.
    """
    df = pd.read_csv(csv_path)
    df["departure_date"] = pd.to_datetime(df["departure_date"])
    return df

def enrich_shipments(shipments_df, ports_csv, suppliers_csv, external_csv) -> pd.DataFrame:
    """
    Merges shipment data with port coordinates, supplier financial health, and external factors.
    """
    ports_df = pd.read_csv(ports_csv)
    suppliers_df = pd.read_csv(suppliers_csv)
    external_df = pd.read_csv(external_csv)
    
    # 1. Merge origin coordinates
    ports_origin = ports_df.rename(columns={
        "port_name": "origin",
        "latitude": "origin_lat",
        "longitude": "origin_lon",
        "country": "origin_country"
    })
    df = pd.merge(shipments_df, ports_origin, on="origin", how="left")
    
    # 2. Merge destination coordinates
    ports_dest = ports_df.rename(columns={
        "port_name": "destination",
        "latitude": "dest_lat",
        "longitude": "dest_lon",
        "country": "dest_country"
    })
    df = pd.merge(df, ports_dest, on="destination", how="left")
    
    # 3. Merge supplier financial health based on origin matching supplier location
    # If multiple suppliers exist in a location, take the mean health
    supplier_health = suppliers_df.groupby("location")["financial_health"].mean().reset_index()
    supplier_health = supplier_health.rename(columns={
        "location": "origin",
        "financial_health": "supplier_health"
    })
    df = pd.merge(df, supplier_health, on="origin", how="left")
    # Fill missing supplier health with median
    df["supplier_health"] = df["supplier_health"].fillna(80.0)
    
    # 4. Merge external factors based on origin-destination pairs
    df = pd.merge(df, external_df, on=["origin", "destination"], how="left")
    
    # Fill remaining missing external factors with defaults
    df["weather_risk"] = df["weather_risk"].fillna(0.1)
    df["congestion"] = df["congestion"].fillna(0.2)
    df["geopolitical_risk"] = df["geopolitical_risk"].fillna(0.05)
    
    return df

def split_train_test(df, target='actual_delay_days', test_size=0.2, random_state=42):
    """
    Splits features and target into train and test sets.
    """
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
