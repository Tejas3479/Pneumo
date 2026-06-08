import base64
from fastapi import APIRouter, Request, HTTPException, Response
from app.tasks import (
    stow_rs_task,
    qido_studies_task,
    qido_series_task,
    qido_instances_task,
    get_dicom_file_task,
    wado_rendered_task,
    wado_heatmap_task
)

router = APIRouter(prefix="/dicomweb", tags=["DICOMweb"])

@router.post("/studies")
@router.post("/studies/{study_uid}")
async def stow_rs(request: Request, study_uid: str = None):
    """
    STOW-RS: Receive studies directly, and enqueue task on the Celery worker.
    Returns a task ID for the client to poll.
    """
    content_type = request.headers.get("Content-Type", "")
    body = await request.body()
    body_b64 = base64.b64encode(body).decode("utf-8")
    
    # Enqueue storage task on the worker
    task = stow_rs_task.delay(content_type, body_b64, study_uid)
    return {"task_id": task.id, "status": "PENDING"}

@router.get("/studies")
async def qido_studies():
    """
    QIDO-RS: Query studies. Returns list of studies in DICOM JSON format.
    Runs synchronously on the web layer by blocking on the read task.
    """
    task = qido_studies_task.delay()
    try:
        return task.get(timeout=5.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query studies: {e}")

@router.get("/studies/{study_uid}/series")
async def qido_series(study_uid: str):
    """
    QIDO-RS: Query series in study.
    Runs synchronously by blocking on Celery.
    """
    task = qido_series_task.delay(study_uid)
    try:
        return task.get(timeout=5.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query series: {e}")

@router.get("/studies/{study_uid}/series/{series_uid}/instances")
async def qido_instances(study_uid: str, series_uid: str):
    """
    QIDO-RS: Query instances in series.
    Runs synchronously by blocking on Celery.
    """
    task = qido_instances_task.delay(study_uid, series_uid)
    try:
        return task.get(timeout=5.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query instances: {e}")

@router.get("/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}")
async def wado_instance(study_uid: str, series_uid: str, instance_uid: str):
    """
    WADO-RS: Serve raw DICOM instances by fetching file bytes from the worker.
    """
    task = get_dicom_file_task.delay(study_uid, series_uid, instance_uid, "original")
    try:
        res = task.get(timeout=5.0)
        if res.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="DICOM instance not found.")
        data_bytes = base64.b64decode(res.get("data_b64", ""))
        filename = res.get("filename", "instance.dcm")
        return Response(
            content=data_bytes,
            media_type=res.get("media_type"),
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch DICOM instance: {e}")

@router.get("/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}/rendered")
async def wado_rendered(study_uid: str, series_uid: str, instance_uid: str):
    """
    WADO-RS Rendered: Extract pixel data and render it as a JPEG response.
    Delegated completely to the worker to preserve stateless web layer.
    """
    task = wado_rendered_task.delay(study_uid, series_uid, instance_uid)
    try:
        res = task.get(timeout=5.0)
        if res.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Instance not found.")
        if res.get("status") == "error":
            raise HTTPException(status_code=500, detail=res.get("message"))
        data_bytes = base64.b64decode(res.get("data_b64", ""))
        return Response(content=data_bytes, media_type=res.get("media_type"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render instance: {e}")

@router.get("/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}/heatmap")
async def wado_heatmap(study_uid: str, series_uid: str, instance_uid: str):
    """
    WADO-RS Heatmap: Serve the transparent overlay PNG for Cornerstone canvas overlays.
    Delegates generation or retrieval to the worker.
    """
    task = wado_heatmap_task.delay(study_uid, series_uid, instance_uid)
    try:
        res = task.get(timeout=5.0)
        if res.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Instance file not found.")
        if res.get("status") == "error":
            raise HTTPException(status_code=500, detail=res.get("message"))
        data_bytes = base64.b64decode(res.get("data_b64", ""))
        return Response(content=data_bytes, media_type=res.get("media_type"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch/generate heatmap: {e}")
