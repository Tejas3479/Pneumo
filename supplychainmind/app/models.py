import os
import pandas as pd
from src.features import build_features
from src.model import SupplyChainDelayPredictor
from src.explanation import generate_risk_explanation

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_model.json")

class SupplyChainPredictionPipeline:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                # Fallback to local execution directory or absolute check
                alt_path = os.path.abspath(self.model_path)
                if not os.path.exists(alt_path):
                    raise FileNotFoundError(f"Model not found at {self.model_path} or {alt_path}")
                self.model_path = alt_path
            self.model = SupplyChainDelayPredictor.load(self.model_path)

    def predict(self, enriched_df: pd.DataFrame) -> dict:
        self.load_model()
        
        # 1. Transform df to features
        X = build_features(enriched_df)
        
        # 2. Run model inference
        mean_pred, lower_pred, upper_pred = self.model.predict_with_uncertainty(X)
        
        # 3. Formulate response for each shipment
        shipments = []
        for idx, (_, row) in enumerate(enriched_df.iterrows()):
            shipment_row = row.to_dict()
            pred_delay = float(mean_pred[idx])
            low_bound = float(lower_pred[idx])
            up_bound = float(upper_pred[idx])
            
            confidence = f"{low_bound:.1f} to {up_bound:.1f} days"
            explanation = generate_risk_explanation(shipment_row, pred_delay, confidence)
            
            # Determine risk level based on delay thresholds
            if pred_delay < 1.5:
                risk_level = "Low"
            elif pred_delay < 3.5:
                risk_level = "Medium"
            else:
                risk_level = "High"
                
            shipments.append({
                "shipment_id": shipment_row.get("shipment_id", "N/A"),
                "origin": shipment_row.get("origin", "Unknown"),
                "destination": shipment_row.get("destination", "Unknown"),
                "carrier": shipment_row.get("carrier", "Unknown"),
                "departure_date": str(shipment_row.get("departure_date"))[:10],
                "product_category": shipment_row.get("product_category", "Unknown"),
                "predicted_delay": round(pred_delay, 2),
                "lower_bound": round(low_bound, 2),
                "upper_bound": round(up_bound, 2),
                "confidence": confidence,
                "risk_level": risk_level,
                "explanation": explanation
            })
            
        return {"shipments": shipments}
