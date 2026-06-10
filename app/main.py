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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def predict(file: UploadFile = File(...), uncertainty: bool = True):
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
    task = run_inference_task.delay(image_b64, is_dicom=is_dicom)
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
async def get_studies():
    """
    List all stored studies from the index database.
    """
    try:
        return query_studies_task()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query studies: {e}")

@app.get("/studies/{study_uid}")
async def get_study_details(study_uid: str):
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
async def get_study_prediction(study_uid: str):
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
async def predict_study(study_uid: str):
    """
    Enqueues prediction on the study's primary instance.
    """
    task = predict_study_task.delay(study_uid)
    return {"task_id": task.id, "status": "PENDING"}

@app.get("/metrics/drift")
async def get_drift_metrics():
    """
    Enqueues data drift analysis task and returns a task ID.
    """
    task = drift_check_task.delay()
    return {"task_id": task.id, "status": "PENDING"}

@app.get("/audit-ledger/verify")
async def verify_ledger():
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
async def get_result(task_id: str):
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
        if isinstance(result_data, dict) and result_data.get("type") == "dicom":
            filename = result_data.get("filename", "result.dcm")
            data_bytes = base64.b64decode(result_data.get("data_b64", ""))
            return Response(
                content=data_bytes,
                media_type="application/dicom",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        return {"status": "SUCCESS", "result": result_data}
    return {"status": res.state}

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
