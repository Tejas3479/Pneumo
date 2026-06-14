import os
import base64
import redis
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from celery.result import AsyncResult

from app.dicomweb import router as dicomweb_router
from app.tasks import (
    run_inference_task,
    feedback_task,
    fairness_audit_task,
    run_federated_round_task,
    query_studies_task,
    get_study_details_task,
    get_study_prediction_task,
    predict_study_task,
    drift_check_task,
    verify_audit_ledger_task,
    model_card_task,
    trigger_retrain_task,
    celery_app,
    REDIS_URL
)

app = FastAPI(
    title="PneumoDetect AI",
    description="Medical-grade Chest X-ray Pneumothorax detection served via FastAPI, Celery, and ONNX.",
    version="1.0.0"
)

# Enable CORS for cross-origin client interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def verify_api_key_middleware(request: Request, call_next):
    # Exclude health, ready, static files, and root path from API Key check
    path = request.url.path
    if path in ["/", "/health", "/ready"] or path.startswith("/static"):
        return await call_next(request)
        
    expected_key = os.getenv("PNEUMO_API_KEY")
    if expected_key:
        api_key = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization")
        
        auth_token = None
        if auth_header:
            if auth_header.startswith("Bearer "):
                auth_token = auth_header[7:]
            else:
                auth_token = auth_header
                
        if api_key != expected_key and auth_token != expected_key:
            return Response(
                content='{"detail":"Could not validate credentials / Invalid API Key"}',
                status_code=403,
                media_type="application/json"
            )
            
    return await call_next(request)

# Mount DICOMweb router
app.include_router(dicomweb_router)

# Mount static files for the Cornerstone viewer
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

class FeedbackRequest(BaseModel):
    image_path: str
    clinician_label: int

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
async def predict(file: UploadFile = File(...), uncertainty: bool = True, model_type: str = "vit", save_image: bool = False):
    """
    Accepts an uploaded chest X-ray image (DICOM, PNG, or JPEG) and enqueues prediction.
    If input is a clinical DICOM, returns a task_id that yields a binary Structured Report (SR).
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
    # Check if it parses as DICOM
    is_dicom = False
    if image_bytes.startswith(b"\x00" * 128 + b"DICM"):
        is_dicom = True
    else:
        # Standard fallback check for DICM signature later in file (preamble offset)
        try:
            if len(image_bytes) >= 132 and image_bytes[128:132] == b"DICM":
                is_dicom = True
        except Exception:
            is_dicom = False
            
    # Base64 encode the image bytes for Celery JSON serialization
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Enqueue task
    task = run_inference_task.delay(image_b64, is_dicom=is_dicom, model_type=model_type, save_image=save_image)
    return {"task_id": task.id, "status": "PENDING"}

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """
    Accepts feedback on model predictions and enqueues Active Learning DB logging.
    """
    task = feedback_task.delay(req.image_path, req.clinician_label)
    return {"task_id": task.id, "status": "PENDING"}

@app.get("/fairness-audit")
async def fairness_audit():
    """
    Executes serving-side bias audit using ONNX model on validation subset.
    Blocks synchronously on Celery read task.
    """
    task = fairness_audit_task.delay()
    try:
        # Give audit up to 30s as it iterates over multiple images
        return task.get(timeout=30.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fairness audit execution failed: {str(e)}")

@app.post("/run-federated-round")
async def run_federated_round():
    """
    Enqueues a background federated training round on the worker.
    """
    task = run_federated_round_task.delay()
    return {"task_id": task.id, "status": "PENDING"}

@app.get("/studies")
def get_studies():
    """
    List all stored studies from the index database.
    """
    try:
        return query_studies_task()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query studies: {e}")

@app.get("/studies/{study_uid}")
def get_study_details(study_uid: str):
    """
    Return detailed instance tree for a study.
    """
    try:
        res = get_study_details_task(study_uid)
        if res.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Study not found.")
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query study details: {e}")

@app.get("/studies/{study_uid}/prediction")
def get_study_prediction(study_uid: str):
    """
    Retrieves previously cached prediction results.
    """
    try:
        res = get_study_prediction_task(study_uid)
        if res.get("status") == "error":
            raise HTTPException(status_code=404, detail=res.get("message"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch study prediction: {e}")

@app.post("/studies/{study_uid}/predict")
async def predict_study(study_uid: str, model_type: str = "vit"):
    """
    Enqueues prediction on the study's primary instance.
    """
    task = predict_study_task.delay(study_uid, model_type=model_type)
    return {"task_id": task.id, "status": "PENDING"}

@app.get("/metrics/drift")
async def get_drift_metrics():
    """
    Enqueues data drift analysis task and returns a task ID.
    """
    task = drift_check_task.delay()
    return {"task_id": task.id, "status": "PENDING"}

@app.get("/audit-ledger/verify")
def verify_ledger():
    """
    Triggers Row Hash Chain audit log verification. Blocks synchronously.
    """
    try:
        return verify_audit_ledger_task()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ledger verification failed: {e}")

@app.post("/regulatory/model-card")
async def create_model_card():
    """
    Enqueues generation of standard-compliant model_card.md.
    """
    task = model_card_task.delay()
    return {"task_id": task.id, "status": "PENDING"}

@app.get("/result/{task_id}")
async def get_result(task_id: str, format: str = None):
    """
    Result polling endpoint. Retrieves task state from Redis.
    If the task processed a DICOM file successfully, returns it directly as a binary FileResponse payload.
    """
    res = AsyncResult(task_id, app=celery_app)
    if res.state == "PENDING" or res.state == "STARTED":
        return {"status": "PENDING"}
    elif res.state == "FAILURE":
        return {"status": "FAILED", "error": str(res.info)}
    elif res.state == "SUCCESS":
        result_data = res.result
        if format != "json" and isinstance(result_data, dict) and result_data.get("type") == "dicom":
            filename = result_data.get("filename", "result.dcm")
            data_bytes = base64.b64decode(result_data.get("data_b64", ""))
            return Response(
                content=data_bytes,
                media_type="application/dicom",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        return {"status": "SUCCESS", "result": result_data}
    return {"status": res.state}

@app.get("/active-learning/flagged")
def get_flagged_samples():
    """
    Exposes all flagged predictions for clinician feedback review.
    """
    from src.active_learning import get_feedback_dataset
    import sqlite3
    import pandas as pd
    # Query flagged samples directly (status = 'flagged')
    db_path = os.path.join("data", "active_learning.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        df = pd.read_sql_query("""
            SELECT id, image_path AS ImagePath, prediction_prob AS Probability, prediction_label AS Label, sex AS Sex, age AS Age, timestamp AS Timestamp
            FROM feedback_samples WHERE status = 'flagged'
        """, conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    
    return df.to_dict(orient="records")


@app.get("/active-learning/status")
def get_active_learning_status():
    """
    Returns the current Active Learning dataset statistics:
    flagged count, corrected count, and the auto-retrain threshold.
    """
    import sqlite3
    threshold = int(os.getenv("AL_RETRAIN_THRESHOLD", "50"))
    db_path = os.path.join("data", "active_learning.db")
    if not os.path.exists(db_path):
        return {"flagged": 0, "corrected": 0, "threshold": threshold, "next_retrain_at": threshold}
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM feedback_samples WHERE status = 'flagged'")
        flagged = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM feedback_samples WHERE status = 'corrected'")
        corrected = cursor.fetchone()[0]
    except Exception:
        flagged = 0
        corrected = 0
    finally:
        conn.close()
    # Calculate how many more corrections until the next auto-retrain fires
    next_milestone = ((corrected // threshold) + 1) * threshold
    return {
        "flagged": flagged,
        "corrected": corrected,
        "threshold": threshold,
        "next_retrain_at": next_milestone
    }


@app.post("/active-learning/trigger-retrain")
async def trigger_retrain():
    """
    Manually triggers an Active Learning fine-tuning job on the Celery worker.
    Enqueues trigger_retrain_task and returns a task_id for polling.
    """
    task = trigger_retrain_task.delay()
    return {"task_id": task.id, "status": "PENDING"}

@app.get("/metrics/drift/history")
def get_drift_history():
    """
    Returns historical Population Stability Index (PSI) records from the audit database.
    """
    import sqlite3
    import datetime
    db_path = "data/audit_ledger.db"
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT timestamp, psi_mean, psi_std, alert_flag, samples_count FROM drift_metrics ORDER BY id ASC")
        rows = cursor.fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    
    history = []
    for r in rows:
        try:
            dt = datetime.datetime.fromisoformat(r[0])
            day_str = dt.strftime("%a %H:%M")
        except Exception:
            day_str = r[0][:16]
        history.append({
            "timestamp": r[0],
            "day": day_str,
            "meanPsi": r[1],
            "stdPsi": r[2],
            "drift_detected": bool(r[3]),
            "samples_count": r[4]
        })
    return history

@app.get("/rendered-image")
def get_rendered_image(path: str):
    """
    Utility endpoint to render any DICOM or image file from the data directory.
    Note: Unprotected for demo/development purposes.
    """
    import pydicom
    import numpy as np
    import cv2
    from PIL import Image
    import io
    
    abs_path = os.path.join("data", path)
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        if path.lower().endswith(".dcm"):
            ds = pydicom.dcmread(abs_path)
            pixel_array = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope) if ds.RescaleSlope is not None else 1.0
                intercept = float(ds.RescaleIntercept) if ds.RescaleIntercept is not None else 0.0
                pixel_array = pixel_array * slope + intercept
            min_val, max_val = pixel_array.min(), pixel_array.max()
            if max_val - min_val > 0:
                pixel_array = (pixel_array - min_val) / (max_val - min_val)
            else:
                pixel_array = np.zeros_like(pixel_array)
            img_uint8 = (pixel_array * 255.0).astype(np.uint8)
            _, img_encoded = cv2.imencode(".jpg", img_uint8)
            return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
        else:
            img = Image.open(abs_path).convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return Response(content=buffer.getvalue(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {str(e)}")

@app.get("/health")
async def health():
    """
    Fast liveness health check endpoint.
    """
    return {"status": "UP"}

@app.get("/ready")
async def ready():
    """
    Readiness health check endpoint verifying Redis broker availability.
    """
    try:
        r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
        if r.ping():
            return {"status": "READY"}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service Unavailable. Redis connection offline: {e}"
        )
