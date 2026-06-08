import os
import sqlite3
import cv2
import torch
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import ModelManager
from app.dicomweb import router as dicomweb_router

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

# Mount DICOMweb router
app.include_router(dicomweb_router)

# Mount static files for the Cornerstone viewer
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

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
    If input is a clinical DICOM, returns a binary Basic Text Structured Report (SR).
    """
    manager = get_model_manager()
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        # Check if it parses as DICOM
        is_dicom = False
        try:
            from io import BytesIO
            import pydicom
            pydicom.dcmread(BytesIO(image_bytes), stop_before_pixels=True)
            is_dicom = True
        except Exception:
            is_dicom = False

        if is_dicom:
            from io import BytesIO
            import pydicom
            from pydicom.uid import generate_uid
            import sqlite3
            
            ds = pydicom.dcmread(BytesIO(image_bytes))
            study_uid = str(ds.get("StudyInstanceUID", generate_uid()))
            series_uid = str(ds.get("SeriesInstanceUID", generate_uid()))
            sop_uid = str(ds.get("SOPInstanceUID", generate_uid()))
            sop_class_uid = str(ds.get("SOPClassUID", "1.2.840.10008.5.1.4.1.1.1"))
            
            pat_id = str(ds.get("PatientID", "N/A"))
            pat_name = str(ds.get("PatientName", "N/A"))
            study_date = str(ds.get("StudyDate", ""))
            study_time = str(ds.get("StudyTime", ""))
            study_desc = str(ds.get("StudyDescription", "No Description"))
            modality = str(ds.get("Modality", "DX"))
            
            # Ensure original study folder exists
            study_folder = os.path.join("data", "studies", study_uid)
            os.makedirs(study_folder, exist_ok=True)
            orig_path = os.path.join(study_folder, f"{sop_uid}.dcm")
            
            # Check database for existing index
            conn = sqlite3.connect("data/dicomweb.db")
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM dicom_instances WHERE sop_instance_uid = ?", (sop_uid,))
            row = cursor.fetchone()
            
            if not row:
                with open(orig_path, "wb") as f_out:
                    f_out.write(image_bytes)
                cursor.execute("""
                    INSERT OR REPLACE INTO dicom_instances 
                    (sop_instance_uid, series_instance_uid, study_instance_uid, patient_id, patient_name, study_date, study_time, study_description, modality, file_path, file_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'original')
                """, (sop_uid, series_uid, study_uid, pat_id, pat_name, study_date, study_time, study_desc, modality, orig_path))
                conn.commit()
            else:
                orig_path = row[0]
                
            # Check if prediction is already cached
            cursor.execute("SELECT sr_path FROM study_predictions WHERE study_instance_uid = ?", (study_uid,))
            pred_row = cursor.fetchone()
            if pred_row and os.path.exists(pred_row[0]):
                conn.close()
                return FileResponse(pred_row[0], media_type="application/dicom")
                
            # Run prediction and generate report
            from app.utils import preprocess_image
            batch_img, img_resized = preprocess_image(image_bytes)
            prob, uncertainty_val, logits, cls_or_fm = manager.ensemble.predict_ensemble(batch_img)
            prediction_label = "POSITIVE" if prob > 0.5 else "NEGATIVE"
            
            # Generate heatmap matrix
            if manager.model_type == "vit":
                model = manager._get_pytorch_model()
                image_tensor = torch.tensor(batch_img, dtype=torch.float32)
                from src.xai import ViTAttentionGradCAM
                explainer = ViTAttentionGradCAM(model)
                heatmap, _ = explainer.generate_heatmap(image_tensor)
            else:
                weights_path = os.path.join(manager.models_dir, "model_weights.npy")
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(f"FC weights file not found at {weights_path}.")
                model_weights = np.load(weights_path, allow_pickle=True).item()
                weight = model_weights["weight"][0]
                bias = model_weights["bias"]
                fm = cls_or_fm[0]
                cam = np.dot(fm.transpose(1, 2, 0), weight) + bias[0]
                cam = np.maximum(cam, 0)
                cam_min, cam_max = cam.min(), cam.max()
                if cam_max - cam_min > 0:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                else:
                    cam = np.zeros_like(cam)
                heatmap = cv2.resize(cam, (224, 224))
                
            from src.xai import generate_clinical_justification
            narrative = generate_clinical_justification(prob, prediction_label, heatmap)
            
            # Save upsampled transparent heatmap PNG
            heatmap_png_name = f"{sop_uid}_heatmap.png"
            heatmap_png_path = os.path.join(study_folder, heatmap_png_name)
            
            orig_height = int(ds.get("Rows", 224))
            orig_width = int(ds.get("Columns", 224))
            upsampled_cam = cv2.resize(heatmap, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
            upsampled_cam = np.clip(upsampled_cam, 0.0, 1.0)
            
            heatmap_color = cv2.applyColorMap(np.uint8(255 * upsampled_cam), cv2.COLORMAP_JET)
            bgra = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = np.uint8(255 * upsampled_cam * 0.75)
            cv2.imwrite(heatmap_png_path, bgra)
            
            # Generate Secondary Capture (SC) DICOM
            from src.report_dicom import create_secondary_capture, create_dicom_sr
            sc_sop_uid = generate_uid()
            sc_series_uid = generate_uid()
            sc_filename = f"{sc_sop_uid}.dcm"
            sc_path = os.path.join(study_folder, sc_filename)
            
            sc_ds = create_secondary_capture(ds, heatmap, sc_path)
            sc_ds.SOPInstanceUID = sc_sop_uid
            sc_ds.SeriesInstanceUID = sc_series_uid
            sc_ds.file_meta.MediaStorageSOPInstanceUID = sc_sop_uid
            sc_ds.save_as(sc_path, write_like_original=False)
            
            # Generate Structured Report (SR) DICOM
            sr_sop_uid = generate_uid()
            sr_series_uid = generate_uid()
            sr_filename = f"{sr_sop_uid}.dcm"
            sr_path = os.path.join(study_folder, sr_filename)
            
            sr_ds = create_dicom_sr(ds, float(prob), prediction_label, sc_sop_uid, sc_series_uid)
            sr_ds.SOPInstanceUID = sr_sop_uid
            sr_ds.SeriesInstanceUID = sr_series_uid
            sr_ds.file_meta.MediaStorageSOPInstanceUID = sr_sop_uid
            sr_ds.save_as(sr_path, write_like_original=False)
            
            # Index SC and SR in dicom_instances
            cursor.execute("""
                INSERT OR REPLACE INTO dicom_instances 
                (sop_instance_uid, series_instance_uid, study_instance_uid, patient_id, patient_name, study_date, study_time, study_description, modality, file_path, file_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'secondary_capture')
            """, (sc_sop_uid, sc_series_uid, study_uid, pat_id, pat_name, study_date, study_time, study_desc, "OT", sc_path))
            
            cursor.execute("""
                INSERT OR REPLACE INTO dicom_instances 
                (sop_instance_uid, series_instance_uid, study_instance_uid, patient_id, patient_name, study_date, study_time, study_description, modality, file_path, file_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'report')
            """, (sr_sop_uid, sr_series_uid, study_uid, pat_id, pat_name, study_date, study_time, study_desc, "SR", sr_path))
            
            # Store prediction
            cursor.execute("""
                INSERT OR REPLACE INTO study_predictions 
                (study_instance_uid, probability, prediction, text_justification, heatmap_path, sc_path, sr_path, sc_series_uid, sc_sop_uid, sr_series_uid, sr_sop_uid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (study_uid, float(prob), prediction_label, narrative, heatmap_png_path, sc_path, sr_path, sc_series_uid, sc_sop_uid, sr_series_uid, sr_sop_uid))
            
            conn.commit()
            conn.close()
            
            return FileResponse(sr_path, media_type="application/dicom")

        # Standard non-DICOM prediction flow
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

@app.get("/studies")
async def get_studies():
    """
    List all stored studies from the index database.
    """
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT study_instance_uid, patient_id, patient_name, study_date, study_time, study_description, modality, COUNT(sop_instance_uid) 
        FROM dicom_instances
        WHERE file_type = 'original'
        GROUP BY study_instance_uid
    """)
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "study_instance_uid": row[0],
            "patient_id": row[1],
            "patient_name": row[2],
            "study_date": row[3],
            "study_time": row[4],
            "study_description": row[5],
            "modality": row[6],
            "instances_count": row[7]
        })
    return results

@app.get("/studies/{study_uid}")
async def get_study_details(study_uid: str):
    """
    Return detailed instance tree for a study.
    """
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sop_instance_uid, series_instance_uid, modality, file_type, file_path 
        FROM dicom_instances
        WHERE study_instance_uid = ?
    """, (study_uid,))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise HTTPException(status_code=404, detail="Study not found.")
        
    instances = []
    for row in rows:
        instances.append({
            "sop_instance_uid": row[0],
            "series_instance_uid": row[1],
            "modality": row[2],
            "file_type": row[3],
            "file_path": row[4]
        })
    return {
        "study_instance_uid": study_uid,
        "instances": instances
    }

@app.get("/studies/{study_uid}/prediction")
async def get_study_prediction(study_uid: str):
    """
    Retrieves previously cached prediction results.
    """
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT probability, prediction, text_justification, sc_series_uid, sc_sop_uid, sr_series_uid, sr_sop_uid 
        FROM study_predictions 
        WHERE study_instance_uid = ?
    """, (study_uid,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return {"status": "not_predicted"}
        
    prob, pred, justification, sc_series, sc_sop, sr_series, sr_sop = row
    
    # Fetch original instance info to construct the heatmap URL
    cursor.execute("""
        SELECT series_instance_uid, sop_instance_uid 
        FROM dicom_instances 
        WHERE study_instance_uid = ? AND file_type = 'original' 
        LIMIT 1
    """, (study_uid,))
    orig_row = cursor.fetchone()
    conn.close()
    
    if not orig_row:
        raise HTTPException(status_code=404, detail="Original DICOM instance not found for this study.")
        
    orig_series, orig_sop = orig_row
    
    return {
        "status": "completed",
        "probability": float(prob),
        "prediction": pred,
        "text_justification": justification,
        "heatmap_url": f"/dicomweb/studies/{study_uid}/series/{orig_series}/instances/{orig_sop}/heatmap",
        "sc_url": f"/dicomweb/studies/{study_uid}/series/{sc_series}/instances/{sc_sop}" if sc_sop else None,
        "sr_url": f"/dicomweb/studies/{study_uid}/series/{sr_series}/instances/{sr_sop}" if sr_sop else None
    }

@app.post("/studies/{study_uid}/predict")
async def predict_study(study_uid: str):
    """
    Synchronously runs prediction on the study's primary instance.
    """
    # Check if prediction is already cached
    cached = await get_study_prediction(study_uid)
    if cached.get("status") == "completed":
        return cached
        
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT series_instance_uid, sop_instance_uid, file_path 
        FROM dicom_instances 
        WHERE study_instance_uid = ? AND file_type = 'original' 
        LIMIT 1
    """, (study_uid,))
    orig_row = cursor.fetchone()
    if not orig_row:
        conn.close()
        raise HTTPException(status_code=404, detail="No original DICOM instance found for study.")
        
    orig_series, orig_sop, file_path = orig_row
    
    if not os.path.exists(file_path):
        conn.close()
        raise HTTPException(status_code=404, detail=f"DICOM file not found on disk at {file_path}")
        
    try:
        ds = pydicom.dcmread(file_path)
        with open(file_path, "rb") as f:
            img_bytes = f.read()
            
        manager = get_model_manager()
        from app.utils import preprocess_image
        batch_img, img_resized = preprocess_image(img_bytes)
        
        prob, uncertainty_val, logits, cls_or_fm = manager.ensemble.predict_ensemble(batch_img)
        prediction_label = "POSITIVE" if prob > 0.5 else "NEGATIVE"
        
        # Generate heatmap matrix
        if manager.model_type == "vit":
            model = manager._get_pytorch_model()
            image_tensor = torch.tensor(batch_img, dtype=torch.float32)
            from src.xai import ViTAttentionGradCAM
            explainer = ViTAttentionGradCAM(model)
            heatmap, _ = explainer.generate_heatmap(image_tensor)
        else:
            weights_path = os.path.join(manager.models_dir, "model_weights.npy")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"FC weights file not found at {weights_path}.")
            model_weights = np.load(weights_path, allow_pickle=True).item()
            weight = model_weights["weight"][0]
            bias = model_weights["bias"]
            fm = cls_or_fm[0]
            cam = np.dot(fm.transpose(1, 2, 0), weight) + bias[0]
            cam = np.maximum(cam, 0)
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 0:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
            heatmap = cv2.resize(cam, (224, 224))
            
        from src.xai import generate_clinical_justification
        narrative = generate_clinical_justification(prob, prediction_label, heatmap)
        
        # Save upsampled transparent heatmap PNG
        study_folder = os.path.join("data", "studies", study_uid)
        os.makedirs(study_folder, exist_ok=True)
        heatmap_png_name = f"{orig_sop}_heatmap.png"
        heatmap_png_path = os.path.join(study_folder, heatmap_png_name)
        
        orig_height = int(ds.get("Rows", 224))
        orig_width = int(ds.get("Columns", 224))
        upsampled_cam = cv2.resize(heatmap, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
        upsampled_cam = np.clip(upsampled_cam, 0.0, 1.0)
        
        heatmap_color = cv2.applyColorMap(np.uint8(255 * upsampled_cam), cv2.COLORMAP_JET)
        bgra = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = np.uint8(255 * upsampled_cam * 0.75)
        cv2.imwrite(heatmap_png_path, bgra)
        
        # Generate Secondary Capture (SC) DICOM
        from src.report_dicom import create_secondary_capture, create_dicom_sr
        sc_sop_uid = generate_uid()
        sc_series_uid = generate_uid()
        sc_filename = f"{sc_sop_uid}.dcm"
        sc_path = os.path.join(study_folder, sc_filename)
        
        sc_ds = create_secondary_capture(ds, heatmap, sc_path)
        sc_ds.SOPInstanceUID = sc_sop_uid
        sc_ds.SeriesInstanceUID = sc_series_uid
        sc_ds.file_meta.MediaStorageSOPInstanceUID = sc_sop_uid
        sc_ds.save_as(sc_path, write_like_original=False)
        
        # Generate Structured Report (SR) DICOM
        sr_sop_uid = generate_uid()
        sr_series_uid = generate_uid()
        sr_filename = f"{sr_sop_uid}.dcm"
        sr_path = os.path.join(study_folder, sr_filename)
        
        sr_ds = create_dicom_sr(ds, float(prob), prediction_label, sc_sop_uid, sc_series_uid)
        sr_ds.SOPInstanceUID = sr_sop_uid
        sr_ds.SeriesInstanceUID = sr_series_uid
        sr_ds.file_meta.MediaStorageSOPInstanceUID = sr_sop_uid
        sr_ds.save_as(sr_path, write_like_original=False)
        
        # Index SC and SR in dicom_instances
        pat_id = str(ds.get("PatientID", "N/A"))
        pat_name = str(ds.get("PatientName", "N/A"))
        study_date = str(ds.get("StudyDate", ""))
        study_time = str(ds.get("StudyTime", ""))
        study_desc = str(ds.get("StudyDescription", "No Description"))
        
        cursor.execute("""
            INSERT OR REPLACE INTO dicom_instances 
            (sop_instance_uid, series_instance_uid, study_instance_uid, patient_id, patient_name, study_date, study_time, study_description, modality, file_path, file_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'secondary_capture')
        """, (sc_sop_uid, sc_series_uid, study_uid, pat_id, pat_name, study_date, study_time, study_desc, "OT", sc_path))
        
        cursor.execute("""
            INSERT OR REPLACE INTO dicom_instances 
            (sop_instance_uid, series_instance_uid, study_instance_uid, patient_id, patient_name, study_date, study_time, study_description, modality, file_path, file_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'report')
        """, (sr_sop_uid, sr_series_uid, study_uid, pat_id, pat_name, study_date, study_time, study_desc, "SR", sr_path))
        
        # Store prediction
        cursor.execute("""
            INSERT OR REPLACE INTO study_predictions 
            (study_instance_uid, probability, prediction, text_justification, heatmap_path, sc_path, sr_path, sc_series_uid, sc_sop_uid, sr_series_uid, sr_sop_uid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (study_uid, float(prob), prediction_label, narrative, heatmap_png_path, sc_path, sr_path, sc_series_uid, sc_sop_uid, sr_series_uid, sr_sop_uid))
        
        conn.commit()
        conn.close()
        
        return {
            "status": "completed",
            "probability": float(prob),
            "prediction": prediction_label,
            "text_justification": narrative,
            "heatmap_url": f"/dicomweb/studies/{study_uid}/series/{orig_series}/instances/{orig_sop}/heatmap",
            "sc_url": f"/dicomweb/studies/{study_uid}/series/{sc_series_uid}/instances/{sc_sop_uid}",
            "sr_url": f"/dicomweb/studies/{study_uid}/series/{sr_series_uid}/instances/{sr_sop_uid}"
        }
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        raise HTTPException(status_code=500, detail=f"Prediction failed for study {study_uid}: {str(e)}")

@app.get("/metrics/drift")
async def get_drift_metrics():
    """
    Computes and returns population stability index drift metrics.
    """
    try:
        from mlops.drift_monitor import check_data_drift
        drift_stats = check_data_drift()
        return drift_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check data drift: {str(e)}")

@app.get("/audit-ledger/verify")
async def verify_ledger():
    """
    Triggers Row Hash Chain audit log verification and returns validity status.
    """
    try:
        from src.regulatory import verify_audit_trail
        valid, mismatches = verify_audit_trail()
        return {"valid": valid, "mismatches": mismatches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ledger verification failed: {str(e)}")

@app.post("/regulatory/model-card")
async def create_model_card():
    """
    Generates and saves standard-compliant model_card.md to disk.
    """
    try:
        from src.regulatory import generate_model_card
        path = generate_model_card()
        return {"status": "ok", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model card generation failed: {str(e)}")
