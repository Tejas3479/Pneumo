import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.models import ModelManager

app = FastAPI(
    title="PneumoDetect AI",
    description="Medical-grade Chest X-ray Pneumothorax detection served via FastAPI and ONNX.",
    version="1.0.0"
)

# Enable CORS for cross-origin client interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager initialized lazily
_model_manager = None

from pydantic import BaseModel
import threading
import torch

class FeedbackRequest(BaseModel):
    image_path: str
    clinician_label: int

def federated_worker(model, model_type, models_dir):
    # Apply NumPy monkeypatch in background thread
    import numpy as np
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
        
    import flwr as fl
    from src.federated import PneumoFlowerClient
    from export_onnx import export_onnx_model
    
    try:
        print("Background Federated Client starting...")
        client = PneumoFlowerClient(model=model, epochs=1, batch_size=4, lr=1e-4)
        
        # Connect client to the local Flower server
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
        print("Background Federated Client completed successfully.")
        
        # Save updated weights back to checkpoint
        ckpt_path = os.path.join(models_dir, "best_seed_0.ckpt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(models_dir, "best.ckpt")
            
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            checkpoint["state_dict"] = model.state_dict()
            torch.save(checkpoint, ckpt_path)
            print(f"Persisted updated weights to checkpoint {ckpt_path}")
            
            # Save PEFT adapter weights
            lora_dir = os.path.join(models_dir, "lora_weights")
            os.makedirs(lora_dir, exist_ok=True)
            model.resnet_or_vit.save_pretrained(lora_dir)
            print(f"Saved LoRA adapter weights to {lora_dir}")
            
            # Re-export to ONNX programmatically
            weights_path = os.path.join(models_dir, "model_weights.npy")
            export_onnx_model(model_type, ckpt_path, weights_path)
            print("ONNX model re-exported successfully after federated round.")
    except Exception as e:
        print(f"Error in background federated client thread: {e}")

def get_model_manager():
    """
    Lazy loader for ModelManager to prevent API server startup failures
    if model training and export have not been executed yet.
    """
    global _model_manager
    if _model_manager is None:
        try:
            _model_manager = ModelManager(models_dir="models")
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model files not available. Please run training and ONNX export. Error: {str(e)}"
            )
    return _model_manager

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serves the single-page dashboard HTML from app/static/index.html.
    """
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(
            status_code=404, 
            detail="Frontend index.html file not found in app/static/."
        )
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...), uncertainty: bool = True):
    """
    Accepts an uploaded chest X-ray image (DICOM, PNG, or JPEG),
    runs model prediction with Grad-CAM overlays, and returns results.
    """
    manager = get_model_manager()
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        result = manager.predict(image_bytes)
        if not uncertainty:
            result.pop("uncertainty", None)
        return result
    except HTTPException as he:
        raise he
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """
    Accepts feedback on model predictions and logs correct labels to active learning database.
    """
    try:
        from src.active_learning import save_clinician_feedback
        save_clinician_feedback(req.image_path, req.clinician_label)
        return {"status": "success", "message": f"Clinician feedback successfully saved for {req.image_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

@app.get("/fairness-audit")
async def fairness_audit():
    """
    Executes serving-side bias audit using ONNX model on validation subset.
    """
    try:
        import numpy as np
        import pandas as pd
        from src.fairness import FairnessAudit
        
        val_indices_path = "data/val_indices.npy"
        train_csv_path = "data/train.csv"
        
        if not os.path.exists(val_indices_path) or not os.path.exists(train_csv_path):
            raise HTTPException(
                status_code=400, 
                detail="Validation split indices or training CSV not found. Please train model first."
            )
            
        val_indices = np.load(val_indices_path)
        df_all = pd.read_csv(train_csv_path)
        val_df = df_all.iloc[val_indices].reset_index(drop=True)
        
        manager = get_model_manager()
        y_trues = val_df['Label'].to_numpy()
        sexes = val_df['Sex'].to_numpy()
        y_pred_probs = []
        
        for _, row in val_df.iterrows():
            image_rel_path = row['ImagePath']
            image_abs_path = os.path.join("data", image_rel_path)
            
            if not os.path.exists(image_abs_path):
                y_pred_probs.append(0.0)
                continue
                
            with open(image_abs_path, "rb") as f:
                img_bytes = f.read()
                
            from app.utils import preprocess_image
            batch_img, _ = preprocess_image(img_bytes)
            prob, _, _, _ = manager.ensemble.predict_ensemble(batch_img)
            y_pred_probs.append(prob)
            
        y_pred_probs = np.array(y_pred_probs)
        metrics = FairnessAudit.compute_metrics(y_trues, y_pred_probs, sexes)
        
        # Fetch active learning stats
        import sqlite3
        conn = sqlite3.connect("data/active_learning.db")
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM feedback_samples WHERE status = 'flagged'")
            num_flagged = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM feedback_samples WHERE status = 'corrected'")
            num_corrected = cursor.fetchone()[0]
        except Exception:
            num_flagged = 0
            num_corrected = 0
        finally:
            conn.close()
            
        return {
            "metrics": metrics,
            "db_stats": {
                "num_flagged": num_flagged,
                "num_corrected": num_corrected
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fairness audit execution failed: {str(e)}")

@app.post("/run-federated-round")
async def run_federated_round():
    """
    Spawns background thread running Flower client for on-device/collaborative parameter updates.
    """
    manager = get_model_manager()
    try:
        model = manager._get_pytorch_model()
        t = threading.Thread(
            target=federated_worker,
            args=(model, manager.model_type, manager.models_dir),
            daemon=True
        )
        t.start()
        return {"status": "success", "message": "Federated training client started in isolated background thread."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start federated round client: {str(e)}")
