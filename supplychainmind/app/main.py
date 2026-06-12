import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import json

from app.models import ShipmentPredictionManager
from src.features import FeatureEngineering
from src.model import ShipmentDelayPredictor
from src.explanation import generate_risk_explanation, classify_risk
from src.locations import CITIES  # for heatmap generation

app = FastAPI(title="SupplyChainMind")

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model_path = os.environ.get("MODEL_PATH", "models/xgb_model.pkl")
feat_path = os.environ.get("FEATURE_PATH", "models/feature_eng.pkl")

# If files don't exist, we try loading them from the default supplychainmind directory
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_model.pkl")
if not os.path.exists(feat_path):
    feat_path = os.path.join(os.path.dirname(__file__), "..", "models", "feature_eng.pkl")

# Lazy load/singleton prediction manager
manager = None

def get_manager():
    global manager
    if manager is None:
        if not os.path.exists(model_path) or not os.path.exists(feat_path):
            raise HTTPException(
                status_code=503,
                detail="ML model or feature preprocessor not found. Please run training pipeline first."
            )
        manager = ShipmentPredictionManager(model_path, feat_path)
    return manager

class PredictionInput(BaseModel):
    ShipmentID: str
    Origin: str
    Destination: str
    Carrier: str
    ProductCategory: str
    DepartureDate: str
    ExpectedDelivery: str
    Weight_kg: float
    Origin_Lat: float = 0.0
    Origin_Lon: float = 0.0
    Dest_Lat: float = 0.0
    Dest_Lon: float = 0.0
    WeatherRisk: float = 0.0
    PortCongestion: float = 0.0
    GeopoliticalSentiment: float = 0.0

class ShipmentPrediction(BaseModel):
    ShipmentID: str
    Origin: str
    Destination: str
    PredictedDelay: float
    RiskLevel: str
    Explanation: str

class PredictResponse(BaseModel):
    predictions: List[ShipmentPrediction]

class SimulationRequest(BaseModel):
    affected_port: str
    delay_days: int
    shipments: List[PredictionInput]

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict", response_model=PredictResponse)
def predict(shipments: List[PredictionInput]):
    pred_manager = get_manager()
    results = []
    
    # Pre-map cities for coordinates to ensure features.py distance calculation doesn't receive zeros
    city_coords = {c[0]: (c[2], c[3]) for c in CITIES}
    
    for s in shipments:
        s_dict = s.dict()
        # Fallback coordinates if not explicitly passed
        if s_dict["Origin_Lat"] == 0.0 and s_dict["Origin_Lon"] == 0.0:
            coords = city_coords.get(s.Origin, (31.2304, 121.4737))
            s_dict["Origin_Lat"], s_dict["Origin_Lon"] = coords
        if s_dict["Dest_Lat"] == 0.0 and s_dict["Dest_Lon"] == 0.0:
            coords = city_coords.get(s.Destination, (51.9244, 4.4777))
            s_dict["Dest_Lat"], s_dict["Dest_Lon"] = coords
            
        df = pd.DataFrame([s_dict])
        delay, risk = pred_manager.predict_one(df)
        
        # Feature importance dict for explanation (mock/static summary)
        feature_importance = {"WeatherRisk": s.WeatherRisk, "PortCongestion": s.PortCongestion, "GeopoliticalSentiment": s.GeopoliticalSentiment}
        explanation = generate_risk_explanation(
            s_dict, delay, risk, feature_importance, list(feature_importance.keys())
        )
        results.append(ShipmentPrediction(
            ShipmentID=s.ShipmentID,
            Origin=s.Origin,
            Destination=s.Destination,
            PredictedDelay=round(delay, 2),
            RiskLevel=risk,
            Explanation=explanation
        ))
    return {"predictions": results}

@app.get("/api/heatmap")
def heatmap():
    # Return GeoJSON of ports with dummy risk scores
    features = []
    for idx, city in enumerate(CITIES):
        risk_score = round(0.1 + 0.9 * (idx % 5) / 5, 2)  # mock risk
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [city[3], city[2]]
            },
            "properties": {
                "name": city[0],
                "country": city[1],
                "risk_score": risk_score
            }
        })
    return {"type": "FeatureCollection", "features": features}

@app.post("/api/simulate")
def simulate(request: SimulationRequest):
    # Simple simulation: increase congestion risk for shipments transiting the affected port
    modified_shipments = []
    for s in request.shipments:
        s_dict = s.dict()
        if s.Origin == request.affected_port or s.Destination == request.affected_port:
            s_dict["PortCongestion"] = min(s_dict.get("PortCongestion", 0.0) + request.delay_days * 0.1, 1.0)
            s_dict["WeatherRisk"] = min(s_dict.get("WeatherRisk", 0.0) + request.delay_days * 0.05, 1.0)
        modified_shipments.append(PredictionInput(**s_dict))
    return predict(modified_shipments)

# Serve static frontend files if app/static directory exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
