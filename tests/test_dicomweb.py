import os
import io
import shutil
import sqlite3
import numpy as np
import pytest
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture(scope="module", autouse=True)
def setup_test_db_and_folders():
    # Back up existing DB and directories if they exist, or clean up after test
    db_backup = "data/dicomweb_test.db.bak"
    orig_db = "data/dicomweb.db"
    
    if os.path.exists(orig_db):
        shutil.copy(orig_db, db_backup)
        os.remove(orig_db)
        
    # Reinitialize DB
    from app.tasks import init_dicomweb_db as init_db
    init_db()
    
    yield
    
    # Clean up test files
    if os.path.exists(orig_db):
        os.remove(orig_db)
    if os.path.exists(db_backup):
        shutil.move(db_backup, orig_db)

def create_mock_dicom():
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.1') # DX Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "DX"
    ds.StudyDate = "20260608"
    ds.StudyTime = "120000"
    ds.StudyDescription = "Test chest radiograph"
    
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 224
    ds.Columns = 224
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = "1.0"
    ds.RescaleIntercept = "0.0"
    
    pixels = np.random.randint(0, 4096, (224, 224), dtype=np.uint16)
    ds.PixelData = pixels.tobytes()
    
    out = io.BytesIO()
    ds.save_as(out, write_like_original=False)
    return out.getvalue(), ds

def test_stow_and_qido():
    client = TestClient(app)
    dicom_bytes, ds = create_mock_dicom()
    original_study_uid = ds.StudyInstanceUID
    
    # 1. Test STOW-RS via POST /dicomweb/studies
    response = client.post(
        f"/dicomweb/studies/{original_study_uid}",
        content=dicom_bytes,
        headers={"Content-Type": "application/dicom"}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "task_id" in res_data
    task_id = res_data["task_id"]
    
    # Wait for completion (via mock eager backend, it completes instantly)
    res_response = client.get(f"/result/{task_id}")
    assert res_response.status_code == 200
    result_json = res_response.json()
    assert result_json["status"] == "SUCCESS"
    assert result_json["result"]["status"] == "success"
    assert len(result_json["result"]["stowed"]) > 0
    
    # Extract the regenerated UIDs from the stowed instance info
    stowed_item = result_json["result"]["stowed"][0]
    wado_url = stowed_item[2]
    parts = wado_url.split("/")
    study_uid = parts[3]
    series_uid = parts[5]
    sop_uid = parts[7]
    
    # 2. Test QIDO-RS studies query
    response = client.get("/dicomweb/studies")
    assert response.status_code == 200
    res_data = response.json()
    assert len(res_data) > 0
    assert any(study["0020000D"]["Value"][0] == study_uid for study in res_data)
    
    # 3. Test QIDO-RS series query
    response = client.get(f"/dicomweb/studies/{study_uid}/series")
    assert response.status_code == 200
    res_data = response.json()
    assert len(res_data) > 0
    assert res_data[0]["0020000E"]["Value"][0] == series_uid
    
    # 4. Test QIDO-RS instances query
    response = client.get(f"/dicomweb/studies/{study_uid}/series/{series_uid}/instances")
    assert response.status_code == 200
    res_data = response.json()
    assert len(res_data) > 0
    assert any(inst["00080018"]["Value"][0] == sop_uid for inst in res_data)

def test_wado():
    client = TestClient(app)
    dicom_bytes, ds = create_mock_dicom()
    original_study_uid = ds.StudyInstanceUID
    
    # STOW first
    response = client.post(
        f"/dicomweb/studies/{original_study_uid}",
        content=dicom_bytes,
        headers={"Content-Type": "application/dicom"}
    )
    task_id = response.json()["task_id"]
    res_response = client.get(f"/result/{task_id}")
    result_json = res_response.json()
    
    wado_url = result_json["result"]["stowed"][0][2]
    parts = wado_url.split("/")
    study_uid = parts[3]
    series_uid = parts[5]
    sop_uid = parts[7]
    
    # 1. Retrieve raw DICOM
    response = client.get(f"/dicomweb/studies/{study_uid}/series/{series_uid}/instances/{sop_uid}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/dicom"
    ret_ds = pydicom.dcmread(io.BytesIO(response.content))
    assert ret_ds.SOPInstanceUID == sop_uid
    
    # 2. Retrieve rendered image
    response = client.get(f"/dicomweb/studies/{study_uid}/series/{series_uid}/instances/{sop_uid}/rendered")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    # 3. Retrieve transparent heatmap
    response = client.get(f"/dicomweb/studies/{study_uid}/series/{series_uid}/instances/{sop_uid}/heatmap")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_prediction_endpoints():
    client = TestClient(app)
    dicom_bytes, ds = create_mock_dicom()
    original_study_uid = ds.StudyInstanceUID
    
    # STOW first
    response = client.post(
        f"/dicomweb/studies/{original_study_uid}",
        content=dicom_bytes,
        headers={"Content-Type": "application/dicom"}
    )
    task_id = response.json()["task_id"]
    res_response = client.get(f"/result/{task_id}")
    result_json = res_response.json()
    
    wado_url = result_json["result"]["stowed"][0][2]
    parts = wado_url.split("/")
    study_uid = parts[3]
    series_uid = parts[5]
    sop_uid = parts[7]
    
    # Check initial prediction status (should be not_predicted)
    response = client.get(f"/studies/{study_uid}/prediction")
    assert response.status_code == 200
    assert response.json()["status"] == "not_predicted"
    
    # Run prediction
    response = client.post(f"/studies/{study_uid}/predict")
    assert response.status_code == 200
    pred_data = response.json()
    assert pred_data["status"] == "PENDING"
    pred_task_id = pred_data["task_id"]
    
    # Retrieve prediction file result
    res_response = client.get(f"/result/{pred_task_id}")
    assert res_response.status_code == 200
    assert res_response.headers["content-type"] == "application/dicom"
    
    # Check updated prediction status
    response = client.get(f"/studies/{study_uid}/prediction")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"

def test_predict_on_dicom_upload():
    client = TestClient(app)
    dicom_bytes, ds = create_mock_dicom()
    
    # POST to default /predict
    response = client.post(
        "/predict",
        files={"file": ("test.dcm", dicom_bytes, "application/dicom")}
    )
    
    assert response.status_code == 200
    pred_data = response.json()
    assert pred_data["status"] == "PENDING"
    task_id = pred_data["task_id"]
    
    # Get async prediction result
    res_response = client.get(f"/result/{task_id}")
    assert res_response.status_code == 200
    assert res_response.headers["content-type"] == "application/dicom"
    
    # Read returned bytes as DICOM and verify it is a Basic Text SR
    sr_ds = pydicom.dcmread(io.BytesIO(res_response.content))
    assert sr_ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.11' # Basic Text SR Storage
    assert sr_ds.ValueType == "CONTAINER"
