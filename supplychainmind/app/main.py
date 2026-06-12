import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from app.utils import parse_uploaded_csv
from app.models import SupplyChainPredictionPipeline
from src.data import enrich_shipments

app = FastAPI(title="SupplyChainMind — API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.environ.get("DATA_DIR", "data")
pipeline = SupplyChainPredictionPipeline()

class SimulationRequest(BaseModel):
    port: str
    days_closed: int

@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts a CSV file upload, enriches it, builds features, runs prediction, and returns predictions.
    """
    try:
        shipments_df = parse_uploaded_csv(file)
        
        # Paths for enrichment
        ports_csv = os.path.join(DATA_DIR, "ports.csv")
        suppliers_csv = os.path.join(DATA_DIR, "suppliers.csv")
        external_csv = os.path.join(DATA_DIR, "external_factors.csv")
        
        if not all(os.path.exists(p) for p in [ports_csv, suppliers_csv, external_csv]):
            raise HTTPException(status_code=500, detail="Reference data files (ports, suppliers, external_factors) missing in data directory.")
            
        enriched_df = enrich_shipments(shipments_df, ports_csv, suppliers_csv, external_csv)
        results = pipeline.predict(enriched_df)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/heatmap")
def heatmap_endpoint():
    """
    Returns a GeoJSON FeatureCollection of ports with risk scores and explanations.
    """
    try:
        ports_csv = os.path.join(DATA_DIR, "ports.csv")
        external_csv = os.path.join(DATA_DIR, "external_factors.csv")
        
        if not os.path.exists(ports_csv) or not os.path.exists(external_csv):
            raise HTTPException(status_code=500, detail="Data files for heatmap generation are missing.")
            
        ports_df = pd.read_csv(ports_csv)
        external_df = pd.read_csv(external_csv)
        
        features = []
        for _, port_row in ports_df.iterrows():
            port_name = port_row["port_name"]
            country = port_row["country"]
            lat = float(port_row["latitude"])
            lon = float(port_row["longitude"])
            
            # Find composite risk score for the port based on routes originating from it
            port_routes = external_df[external_df["origin"] == port_name]
            if not port_routes.empty:
                mean_w = float(port_routes["weather_risk"].mean())
                mean_c = float(port_routes["congestion"].mean())
                mean_g = float(port_routes["geopolitical_risk"].mean())
                risk_score = round((mean_w + mean_c + mean_g) / 3.0, 3)
            else:
                mean_w, mean_c, mean_g = 0.1, 0.2, 0.05
                risk_score = 0.117
                
            # Build narrative explanation
            factors = []
            if mean_c > 0.4:
                factors.append("high port congestion")
            if mean_w > 0.4:
                factors.append("severe weather patterns")
            if mean_g > 0.3:
                factors.append("elevated geopolitical risk")
                
            if not factors:
                explanation = f"Stable transit conditions. Low risk composite ({risk_score:.2f})."
            else:
                explanation = f"Port is currently experiencing {', '.join(factors)}. Risk composite ({risk_score:.2f})."
                
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]  # GeoJSON coordinates are [lon, lat]
                },
                "properties": {
                    "name": port_name,
                    "country": country,
                    "risk_score": risk_score,
                    "explanation": explanation
                }
            })
            
        return {"type": "FeatureCollection", "features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
def simulate_endpoint(request: SimulationRequest):
    """
    Reruns predictions on a cached dataset (default shipments.csv) simulating port closure.
    """
    try:
        shipments_csv = os.path.join(DATA_DIR, "shipments.csv")
        ports_csv = os.path.join(DATA_DIR, "ports.csv")
        suppliers_csv = os.path.join(DATA_DIR, "suppliers.csv")
        external_csv = os.path.join(DATA_DIR, "external_factors.csv")
        
        if not all(os.path.exists(p) for p in [shipments_csv, ports_csv, suppliers_csv, external_csv]):
            raise HTTPException(status_code=500, detail="Data files needed for simulation are missing.")
            
        shipments_df = pd.read_csv(shipments_csv)
        
        # Filter shipments affected by the target port
        affected_mask = (shipments_df["origin"] == request.port) | (shipments_df["destination"] == request.port)
        affected_shipments = shipments_df[affected_mask].copy()
        
        if affected_shipments.empty:
            return {"shipments": []}
            
        # Enrich the affected shipments
        enriched_df = enrich_shipments(affected_shipments, ports_csv, suppliers_csv, external_csv)
        
        # Override congestion risk to 1.0 (simulate full port disruption)
        # We can also increase weather risk/other factors depending on closure days
        enriched_df["congestion"] = 1.0
        
        # Recalculate predictions
        results = pipeline.predict(enriched_df)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend build files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
