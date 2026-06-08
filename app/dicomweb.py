import os
import sqlite3
import numpy as np
import cv2
import pydicom
import torch
from email.parser import BytesParser
from fastapi import APIRouter, Request, HTTPException, Response, UploadFile, File
from fastapi.responses import FileResponse
from pydicom.uid import generate_uid

# Database and file path configs
DB_PATH = os.path.join("data", "dicomweb.db")
STUDIES_DIR = os.path.join("data", "studies")

router = APIRouter(prefix="/dicomweb", tags=["DICOMweb"])

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(STUDIES_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Store general DICOM metadata for fast QIDO queries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dicom_instances (
            sop_instance_uid TEXT PRIMARY KEY,
            series_instance_uid TEXT,
            study_instance_uid TEXT,
            patient_id TEXT,
            patient_name TEXT,
            study_date TEXT,
            study_time TEXT,
            study_description TEXT,
            modality TEXT,
            file_path TEXT,
            file_type TEXT DEFAULT 'original'
        )
    """)
    
    # Cache study-level prediction reports and links
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

# Initialize upon import
init_db()

def parse_multipart_related(content_type: str, body: bytes):
    """
    Parses application/dicom multipart related body using standard BytesParser.
    """
    headers = {"Content-Type": content_type}
    # Assemble a raw MIME email structure to trigger Python's built-in multipart parser
    mime_msg = b"\r\n".join([f"{k}: {v}".encode() for k, v in headers.items()]) + b"\r\n\r\n" + body
    msg = BytesParser().parsebytes(mime_msg)
    
    parts = []
    if msg.is_multipart():
        for part in msg.get_payload():
            # Get part contents
            part_bytes = part.get_payload(decode=True)
            if part_bytes:
                parts.append(part_bytes)
    else:
        # If payload is sent directly without multi-part boundaries, fallback to raw body
        parts.append(body)
    return parts

@router.post("/studies")
@router.post("/studies/{study_uid}")
async def stow_rs(request: Request, study_uid: str = None):
    """
    STOW-RS: Receive studies directly, store files to disk, and index metadata in SQLite.
    """
    content_type = request.headers.get("Content-Type", "")
    if "multipart/related" not in content_type:
        # Fallback to check if raw DICOM was posted
        body = await request.body()
        parts = [body]
    else:
        body = await request.body()
        parts = parse_multipart_related(content_type, body)
        
    saved_instances = []
    
    for part in parts:
        if len(part) < 132:
            continue  # Too small to be a DICOM file
        try:
            # Attempt to parse DICOM header
            from io import BytesIO
            f = BytesIO(part)
            ds = pydicom.dcmread(f, force=True)
            
            # Read critical tags
            study_inst_uid = str(ds.get("StudyInstanceUID", study_uid or generate_uid()))
            series_inst_uid = str(ds.get("SeriesInstanceUID", generate_uid()))
            sop_inst_uid = str(ds.get("SOPInstanceUID", generate_uid()))
            sop_class_uid = str(ds.get("SOPClassUID", "1.2.840.10008.5.1.4.1.1.1")) # Default CR Image Storage
            
            pat_id = str(ds.get("PatientID", "N/A"))
            # Format PatientName safely
            pat_name = str(ds.get("PatientName", "N/A"))
            study_date = str(ds.get("StudyDate", ""))
            study_time = str(ds.get("StudyTime", ""))
            study_desc = str(ds.get("StudyDescription", "No Description"))
            modality = str(ds.get("Modality", "DX"))
            
            # Save file to disk
            study_folder = os.path.join(STUDIES_DIR, study_inst_uid)
            os.makedirs(study_folder, exist_ok=True)
            file_path = os.path.join(study_folder, f"{sop_inst_uid}.dcm")
            
            with open(file_path, "wb") as f_out:
                f_out.write(part)
                
            # Store in database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dicom_instances 
                (sop_instance_uid, series_instance_uid, study_instance_uid, patient_id, patient_name, study_date, study_time, study_description, modality, file_path, file_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'original')
            """, (sop_inst_uid, series_inst_uid, study_inst_uid, pat_id, pat_name, study_date, study_time, study_desc, modality, file_path))
            conn.commit()
            conn.close()
            
            wado_url = f"/dicomweb/studies/{study_inst_uid}/series/{series_inst_uid}/instances/{sop_inst_uid}"
            saved_instances.append((sop_class_uid, sop_inst_uid, wado_url))
        except Exception as e:
            print(f"Failed to process STOW-RS part: {e}")
            
    if not saved_instances:
        raise HTTPException(status_code=400, detail="No valid DICOM instances were parsed from STOW payload.")
        
    # Return standard STOW-RS JSON response structure
    stow_response = {
        "00081199": {
            "vr": "SQ",
            "Value": [
                {
                    "00081150": {"vr": "UI", "Value": [item[0]]},
                    "00081155": {"vr": "UI", "Value": [item[1]]},
                    "00081190": {"vr": "UR", "Value": [item[2]]}
                } for item in saved_instances
            ]
        }
    }
    return stow_response

@router.get("/studies")
async def qido_studies():
    """
    QIDO-RS: Query studies. Returns list of studies in DICOM JSON format.
    """
    conn = sqlite3.connect(DB_PATH)
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

@router.get("/studies/{study_uid}/series")
async def qido_series(study_uid: str):
    """
    QIDO-RS: Query series in study.
    """
    conn = sqlite3.connect(DB_PATH)
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

@router.get("/studies/{study_uid}/series/{series_uid}/instances")
async def qido_instances(study_uid: str, series_uid: str):
    """
    QIDO-RS: Query instances in series.
    """
    conn = sqlite3.connect(DB_PATH)
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
        # Find SOP class by reading minimal metadata or defaulting
        results.append({
            "0020000D": {"vr": "UI", "Value": [row[0]]},
            "0020000E": {"vr": "UI", "Value": [row[1]]},
            "00080018": {"vr": "UI", "Value": [row[2]]},
            # Map report file type to correct Basic Text SR SOP Class if file_type is 'report'
            "00080016": {"vr": "UI", "Value": ["1.2.840.10008.5.1.4.1.1.88.11" if row[4] == 'report' else "1.2.840.10008.5.1.4.1.1.7" if row[4] == 'secondary_capture' else "1.2.840.10008.5.1.4.1.1.1"]}
        })
    return results

@router.get("/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}")
async def wado_instance(study_uid: str, series_uid: str, instance_uid: str):
    """
    WADO-RS: Serve raw DICOM instances.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM dicom_instances WHERE sop_instance_uid = ?", (instance_uid,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not os.path.exists(row[0]):
        raise HTTPException(status_code=404, detail="DICOM instance not found.")
        
    return FileResponse(row[0], media_type="application/dicom")

@router.get("/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}/rendered")
async def wado_rendered(study_uid: str, series_uid: str, instance_uid: str):
    """
    WADO-RS Rendered: Extract pixel data and render it as a JPEG response.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM dicom_instances WHERE sop_instance_uid = ?", (instance_uid,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not os.path.exists(row[0]):
        raise HTTPException(status_code=404, detail="Instance not found.")
        
    try:
        ds = pydicom.dcmread(row[0])
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Rescale intercept/slope
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
        
        # Encode to JPEG
        _, img_encoded = cv2.imencode(".jpg", img_uint8)
        return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render image: {str(e)}")

@router.get("/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}/heatmap")
async def wado_heatmap(study_uid: str, series_uid: str, instance_uid: str):
    """
    WADO-RS Heatmap: Serve the transparent overlay PNG for Cornerstone canvas overlays.
    Dynamic generation + caching.
    """
    heatmap_png_name = f"{instance_uid}_heatmap.png"
    heatmap_png_path = os.path.join(STUDIES_DIR, study_uid, heatmap_png_name)
    
    if os.path.exists(heatmap_png_path):
        return FileResponse(heatmap_png_path, media_type="image/png")
        
    # Not cached, run prediction synchronously to generate
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM dicom_instances WHERE sop_instance_uid = ?", (instance_uid,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not os.path.exists(row[0]):
        raise HTTPException(status_code=404, detail="Instance file not found.")
        
    try:
        # Load image bytes
        with open(row[0], "rb") as f:
            img_bytes = f.read()
            
        # Trigger prediction logic
        # Import globally loaded ModelManager to keep state
        from app.main import get_model_manager
        manager = get_model_manager()
        
        # We need the 2D Grad-CAM heatmap array from model
        from app.utils import preprocess_image
        batch_img, _ = preprocess_image(img_bytes)
        
        # Get PyTorch model CPU ref for Grad-CAM
        pytorch_model = manager._get_pytorch_model()
        image_tensor = torch_tensor = torch.tensor(batch_img, dtype=torch.float32)
        
        from src.xai import ViTAttentionGradCAM
        explainer = ViTAttentionGradCAM(pytorch_model)
        heatmap, _ = explainer.generate_heatmap(image_tensor)
        
        # Load original DICOM metadata for exact matrix resolution
        ds = pydicom.dcmread(row[0])
        orig_height = int(ds.get("Rows", 224))
        orig_width = int(ds.get("Columns", 224))
        
        # Upsample heatmap to exact original dimensions
        upsampled_cam = cv2.resize(heatmap, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
        upsampled_cam = np.clip(upsampled_cam, 0.0, 1.0)
        
        # Apply JET colormap
        heatmap_color = cv2.applyColorMap(np.uint8(255 * upsampled_cam), cv2.COLORMAP_JET)
        bgra = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)
        
        # Set alpha transparency proportional to CAM intensity (fade out lower values)
        bgra[:, :, 3] = np.uint8(255 * upsampled_cam * 0.75)
        
        # Cache to disk
        os.makedirs(os.path.dirname(heatmap_png_path), exist_ok=True)
        cv2.imwrite(heatmap_png_path, bgra)
        
        return FileResponse(heatmap_png_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate heatmap: {str(e)}")
