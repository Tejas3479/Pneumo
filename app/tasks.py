import os
import io
import base64
import json
import sqlite3
import datetime
import pydicom
import cv2
import numpy as np
import torch
from celery import Celery
from pydicom.uid import generate_uid

# Initialize Celery app
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# Lazy-loaded ModelManager
_model_manager = None

def get_task_model_manager():
    global _model_manager
    if _model_manager is None:
        from app.models import ModelManager
        _model_manager = ModelManager(models_dir="models")
    return _model_manager

def init_dicomweb_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dicom_instances (
            sop_instance_uid TEXT PRIMARY KEY,
            series_instance_uid TEXT NOT NULL,
            study_instance_uid TEXT NOT NULL,
            patient_id TEXT,
            patient_name TEXT,
            study_date TEXT,
            study_time TEXT,
            study_description TEXT,
            modality TEXT,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_predictions (
            study_instance_uid TEXT PRIMARY KEY,
            probability REAL,
            prediction TEXT,
            text_justification TEXT,
            heatmap_path TEXT,
            sc_path TEXT,
            sr_path TEXT,
            sc_series_uid TEXT,
            sc_sop_uid TEXT,
            sr_series_uid TEXT,
            sr_sop_uid TEXT
        )
    """)
    conn.commit()
    conn.close()

@celery_app.task
def run_inference_task(image_b64: str, is_dicom: bool) -> dict:
    """
    Inference Celery task. Decodes base64, runs prediction under the circuit breaker,
    handles DICOM creation and database registrations, and returns the result.
    """
    image_bytes = base64.b64decode(image_b64)
    manager = get_task_model_manager()
    from app.circuit_breaker import predict_with_breaker, inference_breaker

    if is_dicom:
        init_dicomweb_db()
        ds = pydicom.dcmread(io.BytesIO(image_bytes))
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
        
        study_folder = os.path.join("data", "studies", study_uid)
        os.makedirs(study_folder, exist_ok=True)
        orig_path = os.path.join(study_folder, f"{sop_uid}.dcm")
        
        conn = sqlite3.connect("data/dicomweb.db")
        cursor = conn.cursor()
        
        # Check database for existing index
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
            
        # Check if prediction is cached
        cursor.execute("SELECT sr_path FROM study_predictions WHERE study_instance_uid = ?", (study_uid,))
        pred_row = cursor.fetchone()
        if pred_row and os.path.exists(pred_row[0]):
            with open(pred_row[0], "rb") as f_sr:
                sr_bytes = f_sr.read()
            conn.close()
            return {
                "type": "dicom",
                "filename": os.path.basename(pred_row[0]),
                "data_b64": base64.b64encode(sr_bytes).decode("utf-8")
            }
            
        # Run prediction
        try:
            # Wrap standard prediction call under the circuit breaker
            pred_res = predict_with_breaker(manager, image_bytes)
            prob = pred_res["probability"]
            prediction_label = pred_res["prediction"]
            narrative = pred_res["text_justification"]
            
            # Re-create heatmap matrix from standard logic
            # Use fallback empty heatmap if breaker tripped or heatmap unavailable
            heatmap = np.zeros((224, 224), dtype=np.float32)
            if prediction_label != "UNKNOWN (FALLBACK)":
                from app.utils import preprocess_image
                batch_img, _ = preprocess_image(image_bytes, model_type=manager.model_type)
                if manager.model_type == "vit":
                    model = manager._get_pytorch_model()
                    image_tensor = torch.tensor(batch_img, dtype=torch.float32)
                    from src.xai import ViTAttentionGradCAM
                    explainer = ViTAttentionGradCAM(model)
                    heatmap, _ = explainer.generate_heatmap(image_tensor)
                else:
                    weights_path = os.path.join(manager.models_dir, "model_weights.npy")
                    if os.path.exists(weights_path):
                        model_weights = np.load(weights_path, allow_pickle=True).item()
                        weight = model_weights["weight"][0]
                        bias = model_weights["bias"]
                        _, _, _, cls_or_fm = manager.ensemble.predict_ensemble(batch_img)
                        fm = cls_or_fm[0]
                        cam = np.dot(fm.transpose(1, 2, 0), weight) + bias[0]
                        cam = np.maximum(cam, 0)
                        cam_min, cam_max = cam.min(), cam.max()
                        if cam_max - cam_min > 0:
                            cam = (cam - cam_min) / (cam_max - cam_min)
                        else:
                            cam = np.zeros_like(cam)
                        heatmap = cv2.resize(cam, (224, 224))
        except Exception as e:
            print(f"Prediction failed in Celery task, applying fallback: {e}")
            prob = 0.5
            prediction_label = "UNKNOWN (FALLBACK)"
            narrative = "Inference failed. Running on degraded fallback."
            heatmap = np.zeros((224, 224), dtype=np.float32)
            
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
        sc_path = os.path.join(study_folder, f"{sc_sop_uid}.dcm")
        
        sc_ds = create_secondary_capture(ds, heatmap, sc_path)
        sc_ds.SOPInstanceUID = sc_sop_uid
        sc_ds.SeriesInstanceUID = sc_series_uid
        sc_ds.file_meta.MediaStorageSOPInstanceUID = sc_sop_uid
        sc_ds.save_as(sc_path, write_like_original=False)
        
        # Generate Structured Report (SR) DICOM
        sr_sop_uid = generate_uid()
        sr_series_uid = generate_uid()
        sr_path = os.path.join(study_folder, f"{sr_sop_uid}.dcm")
        
        sr_ds = create_dicom_sr(ds, float(prob), prediction_label, sc_sop_uid, sc_series_uid)
        sr_ds.SOPInstanceUID = sr_sop_uid
        sr_ds.SeriesInstanceUID = sr_series_uid
        sr_ds.file_meta.MediaStorageSOPInstanceUID = sr_sop_uid
        sr_ds.save_as(sr_path, write_like_original=False)
        
        # Index SC and SR
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
        
        with open(sr_path, "rb") as f_sr:
            sr_bytes = f_sr.read()
            
        return {
            "type": "dicom",
            "filename": f"{sr_sop_uid}.dcm",
            "data_b64": base64.b64encode(sr_bytes).decode("utf-8")
        }
        
    else:
        # Standard non-DICOM prediction
        result = predict_with_breaker(manager, image_bytes)
        return result

from email.parser import BytesParser

def parse_multipart_related(content_type: str, body: bytes):
    headers = {"Content-Type": content_type}
    mime_msg = b"\r\n".join([f"{k}: {v}".encode() for k, v in headers.items()]) + b"\r\n\r\n" + body
    msg = BytesParser().parsebytes(mime_msg)
    parts = []
    if msg.is_multipart():
        for part in msg.get_payload():
            part_bytes = part.get_payload(decode=True)
            if part_bytes:
                parts.append(part_bytes)
    else:
        parts.append(body)
    return parts

@celery_app.task
def stow_rs_task(content_type: str, body_b64: str, study_uid: str = None) -> dict:
    """
    Task to store and index multipart STOW-RS uploaded instances.
    Accepts content_type and base64-encoded multipart body.
    """
    init_dicomweb_db()
    body = base64.b64decode(body_b64)
    parts = parse_multipart_related(content_type, body)
    
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    
    stowed = []
    for part in parts:
        if len(part) < 132:
            continue
        try:
            ds = pydicom.dcmread(io.BytesIO(part), force=True)
            study_inst_uid = str(ds.get("StudyInstanceUID", study_uid or generate_uid()))
            series_inst_uid = str(ds.get("SeriesInstanceUID", generate_uid()))
            sop_inst_uid = str(ds.get("SOPInstanceUID", generate_uid()))
            sop_class_uid = str(ds.get("SOPClassUID", "1.2.840.10008.5.1.4.1.1.1"))
            
            pat_id = str(ds.get("PatientID", "N/A"))
            pat_name = str(ds.get("PatientName", "N/A"))
            study_date = str(ds.get("StudyDate", ""))
            study_time = str(ds.get("StudyTime", ""))
            study_desc = str(ds.get("StudyDescription", "No Description"))
            modality = str(ds.get("Modality", "DX"))
            
            study_folder = os.path.join("data", "studies", study_inst_uid)
            os.makedirs(study_folder, exist_ok=True)
            file_path = os.path.join(study_folder, f"{sop_inst_uid}.dcm")
            
            with open(file_path, "wb") as f_out:
                f_out.write(part)
                
            cursor.execute("""
                INSERT OR REPLACE INTO dicom_instances 
                (sop_instance_uid, series_instance_uid, study_instance_uid, patient_id, patient_name, study_date, study_time, study_description, modality, file_path, file_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'original')
            """, (sop_inst_uid, series_inst_uid, study_inst_uid, pat_id, pat_name, study_date, study_time, study_desc, modality, file_path))
            
            wado_url = f"/dicomweb/studies/{study_inst_uid}/series/{series_inst_uid}/instances/{sop_inst_uid}"
            stowed.append((sop_class_uid, sop_inst_uid, wado_url))
        except Exception as e:
            print(f"Error parsing DICOM part: {e}")
            
    conn.commit()
    conn.close()
    
    if not stowed:
        return {"status": "error", "message": "No valid DICOM instances were parsed from STOW payload."}
        
    return {
        "status": "success",
        "stowed": stowed
    }

@celery_app.task
def feedback_task(image_path: str, clinician_label: int) -> dict:
    """
    Task to log clinician feedback to active learning database.
    """
    from src.active_learning import save_clinician_feedback
    save_clinician_feedback(image_path, clinician_label)
    return {"status": "success"}

@celery_app.task
def drift_check_task() -> dict:
    """
    Task to execute data drift validation.
    """
    from mlops.drift_monitor import check_data_drift
    return check_data_drift()

@celery_app.task
def model_card_task() -> dict:
    """
    Task to generate standard model card.
    """
    from src.regulatory import generate_model_card
    path = generate_model_card()
    return {"status": "success", "path": path}

@celery_app.task
def query_studies_task() -> list:
    """
    Read task: returns list of indexed DICOM studies.
    """
    init_dicomweb_db()
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

@celery_app.task
def get_study_details_task(study_uid: str) -> dict:
    """
    Read task: returns detailed instance tree for a study.
    """
    init_dicomweb_db()
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
        return {"status": "not_found"}
        
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
        "status": "success",
        "study_instance_uid": study_uid,
        "instances": instances
    }

@celery_app.task
def get_study_prediction_task(study_uid: str) -> dict:
    """
    Read task: retrieves cached prediction results for a study.
    """
    init_dicomweb_db()
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
    
    # Fetch original instance info
    cursor.execute("""
        SELECT series_instance_uid, sop_instance_uid 
        FROM dicom_instances 
        WHERE study_instance_uid = ? AND file_type = 'original' 
        LIMIT 1
    """, (study_uid,))
    orig_row = cursor.fetchone()
    conn.close()
    
    if not orig_row:
        return {"status": "error", "message": "Original DICOM instance not found for this study."}
        
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

@celery_app.task
def get_dicom_file_task(study_uid: str, series_uid: str, sop_uid: str, file_type: str = "original") -> dict:
    """
    Read task: retrieves a file (DICOM / rendered frame) from worker storage and returns it base64-encoded.
    """
    init_dicomweb_db()
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    
    if file_type == "heatmap":
        # Heatmap PNG retrieval
        cursor.execute("SELECT heatmap_path FROM study_predictions WHERE study_instance_uid = ?", (study_uid,))
        row = cursor.fetchone()
        file_path = row[0] if (row and os.path.exists(row[0])) else None
        media_type = "image/png"
    else:
        cursor.execute("SELECT file_path FROM dicom_instances WHERE sop_instance_uid = ?", (sop_uid,))
        row = cursor.fetchone()
        file_path = row[0] if (row and os.path.exists(row[0])) else None
        media_type = "application/dicom"
        
    conn.close()
    
    if not file_path or not os.path.exists(file_path):
        return {"status": "not_found"}
        
    with open(file_path, "rb") as f:
        bytes_data = f.read()
        
    return {
        "status": "success",
        "media_type": media_type,
        "filename": os.path.basename(file_path),
        "data_b64": base64.b64encode(bytes_data).decode("utf-8")
    }

@celery_app.task
def verify_audit_ledger_task() -> dict:
    """
    Read task: verify SQLite audit ledger Row Hash Chain integrity.
    """
    from src.regulatory import verify_audit_trail
    valid, mismatches = verify_audit_trail()
    return {"valid": valid, "mismatches": mismatches}

@celery_app.task
def predict_study_task(study_uid: str) -> dict:
    """
    Task to execute prediction on a DICOM study from its stored instance.
    """
    init_dicomweb_db()
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT file_path 
        FROM dicom_instances 
        WHERE study_instance_uid = ? AND file_type = 'original' 
        LIMIT 1
    """, (study_uid,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not os.path.exists(row[0]):
        return {"status": "error", "message": "Original DICOM instance not found on disk."}
        
    with open(row[0], "rb") as f:
        img_bytes = f.read()
        
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return run_inference_task(img_b64, is_dicom=True)

@celery_app.task
def wado_rendered_task(study_uid: str, series_uid: str, sop_uid: str) -> dict:
    """
    Read task: renders a DICOM instance frame to a JPEG buffer on the worker.
    """
    init_dicomweb_db()
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM dicom_instances WHERE sop_instance_uid = ?", (sop_uid,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not os.path.exists(row[0]):
        return {"status": "not_found"}
        
    try:
        ds = pydicom.dcmread(row[0])
        pixel_array = ds.pixel_array.astype(np.float32)
        
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope) if ds.RescaleSlope is not None else 1.0
            intercept = float(ds.RescaleIntercept) if ds.RescaleIntercept is not None else 0.0
            pixel_array = pixel_array * slope + intercept
            
        min_val = pixel_array.min()
        max_val = pixel_array.max()
        if max_val - min_val > 0:
            pixel_array = (pixel_array - min_val) / (max_val - min_val)
        else:
            pixel_array = np.zeros_like(pixel_array)
            
        img_uint8 = (pixel_array * 255.0).astype(np.uint8)
        _, img_encoded = cv2.imencode(".jpg", img_uint8)
        
        return {
            "status": "success",
            "media_type": "image/jpeg",
            "data_b64": base64.b64encode(img_encoded.tobytes()).decode("utf-8")
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to render image: {str(e)}"}

@celery_app.task
def wado_heatmap_task(study_uid: str, series_uid: str, sop_uid: str) -> dict:
    """
    Read task: serves or generates a transparent heatmap overlay PNG.
    """
    init_dicomweb_db()
    heatmap_png_name = f"{sop_uid}_heatmap.png"
    heatmap_png_path = os.path.join("data", "studies", study_uid, heatmap_png_name)
    
    if os.path.exists(heatmap_png_path):
        with open(heatmap_png_path, "rb") as f:
            bytes_data = f.read()
        return {
            "status": "success",
            "media_type": "image/png",
            "data_b64": base64.b64encode(bytes_data).decode("utf-8")
        }
        
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM dicom_instances WHERE sop_instance_uid = ?", (sop_uid,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not os.path.exists(row[0]):
        return {"status": "not_found"}
        
    try:
        with open(row[0], "rb") as f:
            img_bytes = f.read()
            
        manager = get_task_model_manager()
        from app.utils import preprocess_image
        batch_img, _ = preprocess_image(img_bytes, model_type=manager.model_type)
        
        if manager.model_type == "vit":
            import torch
            model = manager._get_pytorch_model()
            image_tensor = torch.tensor(batch_img, dtype=torch.float32)
            from src.xai import ViTAttentionGradCAM
            explainer = ViTAttentionGradCAM(model)
            heatmap, _ = explainer.generate_heatmap(image_tensor)
        else:
            weights_path = os.path.join(manager.models_dir, "model_weights.npy")
            if os.path.exists(weights_path):
                model_weights = np.load(weights_path, allow_pickle=True).item()
                weight = model_weights["weight"][0]
                bias = model_weights["bias"]
                _, _, _, cls_or_fm = manager.ensemble.predict_ensemble(batch_img)
                fm = cls_or_fm[0]
                cam = np.dot(fm.transpose(1, 2, 0), weight) + bias[0]
                cam = np.maximum(cam, 0)
                cam_min, cam_max = cam.min(), cam.max()
                if cam_max - cam_min > 0:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                else:
                    cam = np.zeros_like(cam)
                heatmap = cv2.resize(cam, (224, 224))
            else:
                heatmap = np.zeros((224, 224), dtype=np.float32)
                
        ds = pydicom.dcmread(row[0])
        orig_height = int(ds.get("Rows", 224))
        orig_width = int(ds.get("Columns", 224))
        
        upsampled_cam = cv2.resize(heatmap, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
        upsampled_cam = np.clip(upsampled_cam, 0.0, 1.0)
        
        heatmap_color = cv2.applyColorMap(np.uint8(255 * upsampled_cam), cv2.COLORMAP_JET)
        bgra = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = np.uint8(255 * upsampled_cam * 0.75)
        
        os.makedirs(os.path.dirname(heatmap_png_path), exist_ok=True)
        cv2.imwrite(heatmap_png_path, bgra)
        
        with open(heatmap_png_path, "rb") as f:
            bytes_data = f.read()
            
        return {
            "status": "success",
            "media_type": "image/png",
            "data_b64": base64.b64encode(bytes_data).decode("utf-8")
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to generate heatmap: {str(e)}"}

@celery_app.task
def qido_studies_task() -> list:
    """
    Read task: returns QIDO-RS formatted list of studies in DICOM JSON format.
    """
    init_dicomweb_db()
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
            "0020000D": {"vr": "UI", "Value": [row[0]]},
            "00100020": {"vr": "LO", "Value": [row[1]]},
            "00100010": {"vr": "PN", "Value": [{"Alphabetic": row[2]}]},
            "00080020": {"vr": "DA", "Value": [row[3]]},
            "00080030": {"vr": "TM", "Value": [row[4]]},
            "00081030": {"vr": "LO", "Value": [row[5]]},
            "00080060": {"vr": "CS", "Value": [row[6]]},
            "00201208": {"vr": "IS", "Value": [row[7]]}
        })
    return results

@celery_app.task
def qido_series_task(study_uid: str) -> list:
    """
    Read task: returns QIDO-RS formatted list of series.
    """
    init_dicomweb_db()
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT study_instance_uid, series_instance_uid, modality, COUNT(sop_instance_uid)
        FROM dicom_instances
        WHERE study_instance_uid = ?
        GROUP BY series_instance_uid
    """, (study_uid,))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "0020000D": {"vr": "UI", "Value": [row[0]]},
            "0020000E": {"vr": "UI", "Value": [row[1]]},
            "00080060": {"vr": "CS", "Value": [row[2]]},
            "00201209": {"vr": "IS", "Value": [row[3]]}
        })
    return results

@celery_app.task
def qido_instances_task(study_uid: str, series_uid: str) -> list:
    """
    Read task: returns QIDO-RS formatted list of instances.
    """
    init_dicomweb_db()
    conn = sqlite3.connect("data/dicomweb.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT study_instance_uid, series_instance_uid, sop_instance_uid, file_path, file_type
        FROM dicom_instances
        WHERE study_instance_uid = ? AND series_instance_uid = ?
    """, (study_uid, series_uid))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "0020000D": {"vr": "UI", "Value": [row[0]]},
            "0020000E": {"vr": "UI", "Value": [row[1]]},
            "00080018": {"vr": "UI", "Value": [row[2]]},
            "00080016": {"vr": "UI", "Value": ["1.2.840.10008.5.1.4.1.1.88.11" if row[4] == 'report' else "1.2.840.10008.5.1.4.1.1.7" if row[4] == 'secondary_capture' else "1.2.840.10008.5.1.4.1.1.1"]}
        })
    return results

@celery_app.task
def fairness_audit_task() -> dict:
    """
    Read task: executes demographic fairness bias audit on validation data.
    """
    import numpy as np
    import pandas as pd
    from src.fairness import FairnessAudit
    
    val_indices_path = "data/val_indices.npy"
    train_csv_path = "data/train.csv"
    
    if not os.path.exists(val_indices_path) or not os.path.exists(train_csv_path):
        return {"status": "error", "message": "Validation split indices or training CSV not found."}
        
    val_indices = np.load(val_indices_path)
    df_all = pd.read_csv(train_csv_path)
    val_df = df_all.iloc[val_indices].reset_index(drop=True)
    
    manager = get_task_model_manager()
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
        batch_img, _ = preprocess_image(img_bytes, model_type=manager.model_type)
        prob, _, _, _ = manager.ensemble.predict_ensemble(batch_img)
        y_pred_probs.append(prob)
        
    y_pred_probs = np.array(y_pred_probs)
    metrics = FairnessAudit.compute_metrics(y_trues, y_pred_probs, sexes)
    
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

@celery_app.task
def run_federated_round_task() -> dict:
    """
    Task to execute a single federated learning round on the worker.
    """
    import numpy as np
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
        
    import flwr as fl
    from src.federated import PneumoFlowerClient
    from export_onnx import export_onnx_model
    import torch
    
    manager = get_task_model_manager()
    model = manager._get_pytorch_model()
    
    try:
        print("Federated Client starting in Celery task...")
        client = PneumoFlowerClient(model=model, epochs=1, batch_size=4, lr=1e-4)
        
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
        print("Federated Client completed successfully.")
        
        ckpt_path = os.path.join(manager.models_dir, "best_seed_0.ckpt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(manager.models_dir, "best.ckpt")
            
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            checkpoint["state_dict"] = model.state_dict()
            torch.save(checkpoint, ckpt_path)
            print(f"Persisted updated weights to checkpoint {ckpt_path}")
            
            lora_dir = os.path.join(manager.models_dir, "lora_weights")
            os.makedirs(lora_dir, exist_ok=True)
            model.resnet_or_vit.save_pretrained(lora_dir)
            print(f"Saved LoRA adapter weights to {lora_dir}")
            
            weights_path = os.path.join(manager.models_dir, "model_weights.npy")
            export_onnx_model(manager.model_type, ckpt_path, weights_path)
            print("ONNX model re-exported successfully after federated round.")
            return {"status": "success", "message": "Federated round completed and model re-exported."}
        else:
            return {"status": "error", "message": "Checkpoint file not found to persist updated weights."}
    except Exception as e:
        print(f"Error in federated task execution: {e}")
        return {"status": "error", "message": str(e)}

