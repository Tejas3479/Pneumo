import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.locations import CITIES, CARRIERS, PRODUCT_CATEGORIES

def generate_mock_shipment_data(num_samples=1000, seed=42):
    np.random.seed(seed)
    data = []
    for i in range(num_samples):
        origin = CITIES[np.random.randint(len(CITIES))]
        dest = CITIES[np.random.randint(len(CITIES))]
        while dest == origin:
            dest = CITIES[np.random.randint(len(CITIES))]
        carrier = np.random.choice(CARRIERS)
        product = np.random.choice(PRODUCT_CATEGORIES)
        dep_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
        transit_base = np.random.uniform(2, 15)  # base transit time in days
        # External risk factors (0 to 1)
        weather_risk = np.random.beta(2, 5)  # mostly low
        port_congestion = np.random.beta(2, 5)
        geopolitical_sentiment = np.random.beta(1, 3)
        # Delay = base_transit * (1 + 0.5*weather + 0.3*congestion + 0.2*geopolitical) + noise
        delay = transit_base * (1 + 0.5*weather_risk + 0.3*port_congestion + 0.2*geopolitical_sentiment) + np.random.normal(0, 0.5)
        delay = max(0, delay)
        weight = np.random.uniform(10, 20000)
        data.append({
            "ShipmentID": f"SHP{i:05d}",
            "Origin": origin[0],
            "Origin_Country": origin[1],
            "Origin_Lat": origin[2],
            "Origin_Lon": origin[3],
            "Destination": dest[0],
            "Dest_Country": dest[1],
            "Dest_Lat": dest[2],
            "Dest_Lon": dest[3],
            "Carrier": carrier,
            "ProductCategory": product,
            "DepartureDate": dep_date.strftime("%Y-%m-%d"),
            "ExpectedDelivery": (dep_date + timedelta(days=int(transit_base))).strftime("%Y-%m-%d"),
            "Weight_kg": weight,
            "WeatherRisk": round(weather_risk, 3),
            "PortCongestion": round(port_congestion, 3),
            "GeopoliticalSentiment": round(geopolitical_sentiment, 3),
            "ActualDelay": round(delay, 2)
        })
    return pd.DataFrame(data)

def enrich_with_external_data(df):
    # In mock mode, the data is already enriched. For real integration, APIs would be called here.
    # We simply ensure columns exist.
    required = ["WeatherRisk", "PortCongestion", "GeopoliticalSentiment"]
    for col in required:
        if col not in df.columns:
            df[col] = 0.0
    return df

def train_val_test_split(df, val_frac=0.1, test_frac=0.1):
    n = len(df)
    idx = np.random.permutation(n)
    val_end = int(n * val_frac)
    test_end = val_end + int(n * test_frac)
    train_idx = idx[test_end:]
    val_idx = idx[:val_end]
    test_idx = idx[val_end:test_end]
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
