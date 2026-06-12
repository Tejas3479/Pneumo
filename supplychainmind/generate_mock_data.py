import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def main():
    np.random.seed(42)
    os.makedirs("data", exist_ok=True)

    # 1. Ports List
    ports = [
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
    ports_df = pd.DataFrame(ports, columns=["port_name", "country", "latitude", "longitude"])
    ports_df.to_csv("data/ports.csv", index=False)
    print("Saved data/ports.csv")

    # 2. Suppliers
    suppliers = [
        ("Shenzhen Electronics Co.", "Shenzhen", 85.0),
        ("Shanghai Precision Parts", "Shanghai", 92.5),
        ("Mumbai Pharma Lab", "Mumbai", 74.0),
        ("Singapore Silicon", "Singapore", 96.0),
        ("Rotterdam Chemical Corp", "Rotterdam", 81.0),
        ("Hamburg Castings GmbH", "Hamburg", 88.0),
        ("Detroit Automotive Assembly", "Los Angeles", 79.5), # Suppliers mapped near LA port
        ("New Jersey Textiles", "New York", 68.0),
        ("Dubai Heavy Machinery", "Dubai", 90.0),
        ("Antwerp Metallurgy", "Antwerp", 83.5)
    ]
    suppliers_df = pd.DataFrame(suppliers, columns=["supplier_name", "location", "financial_health"])
    suppliers_df.to_csv("data/suppliers.csv", index=False)
    print("Saved data/suppliers.csv")

    # 3. External Factors
    external_factors = []
    for o_name, _, _, _ in ports:
        for d_name, _, _, _ in ports:
            if o_name == d_name:
                continue
            weather_risk = np.random.beta(2, 5)
            congestion = np.random.beta(2, 5)
            geopolitical_risk = np.random.beta(1, 3)
            external_factors.append({
                "origin": o_name,
                "destination": d_name,
                "weather_risk": round(weather_risk, 3),
                "congestion": round(congestion, 3),
                "geopolitical_risk": round(geopolitical_risk, 3)
            })
    external_df = pd.DataFrame(external_factors)
    external_df.to_csv("data/external_factors.csv", index=False)
    print("Saved data/external_factors.csv")

    # 4. Shipments (5,000 samples)
    carriers = ["Maersk", "MSC", "CMA CGM", "COSCO", "Hapag-Lloyd", "ONE", "Evergreen", "HMM"]
    products = ["Electronics", "Automotive", "Pharmaceuticals", "Apparel", "Food & Beverage", "Heavy Machinery"]
    
    shipments = []
    for i in range(5000):
        # Pick random origin and destination
        o_idx = np.random.randint(len(ports))
        d_idx = np.random.randint(len(ports))
        while d_idx == o_idx:
            d_idx = np.random.randint(len(ports))
            
        origin_port = ports[o_idx][0]
        dest_port = ports[d_idx][0]
        carrier = np.random.choice(carriers)
        product = np.random.choice(products)
        dep_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
        
        # Load external factors to build delay dependency
        ext = next(item for item in external_factors if item["origin"] == origin_port and item["destination"] == dest_port)
        w_risk = ext["weather_risk"]
        cong = ext["congestion"]
        geo_risk = ext["geopolitical_risk"]
        
        # Base transit time depends on distance/random base
        transit_base = np.random.uniform(5, 20)
        
        # Delay function based on external variables + noise
        delay_days = transit_base * (0.1 * w_risk + 0.25 * cong + 0.15 * geo_risk) + np.random.normal(1.0, 0.8)
        delay_days = max(0, round(delay_days, 2))
        
        shipments.append({
            "shipment_id": f"SHP{i:05d}",
            "origin": origin_port,
            "destination": dest_port,
            "carrier": carrier,
            "departure_date": dep_date.strftime("%Y-%m-%d"),
            "product_category": product,
            "actual_delay_days": delay_days
        })
        
    shipments_df = pd.DataFrame(shipments)
    shipments_df.to_csv("data/shipments.csv", index=False)
    print("Saved data/shipments.csv (5,000 samples)")

if __name__ == "__main__":
    main()
