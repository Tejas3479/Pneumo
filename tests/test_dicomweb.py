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
    from app.dicomweb import init_db
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
    study_uid = ds.StudyInstanceUID
    series_uid = ds.SeriesInstanceUID
    sop_uid = ds.SOPInstanceUID
    
    # 1. Test STOW-RS via POST /dicomweb/studies
    response = client.post(
        f"/dicomweb/studies/{study_uid}",
        content=dicom_bytes,
        headers={"Content-Type": "application/dicom"}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "00081199" in res_data
    
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
    study_uid = ds.StudyInstanceUID
    series_uid = ds.SeriesInstanceUID
    sop_uid = ds.SOPInstanceUID
    
    # STOW first
    client.post(
        f"/dicomweb/studies/{study_uid}",
        content=dicom_bytes,
        headers={"Content-Type": "application/dicom"}
    )
    
    # 1. Retrieve raw DICOM
    response = client.get(f"/dicomweb/studies/{study_uid}/series/{series_uid}/instances/{sop_uid}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/dicom"
    assert len(response.content) == len(dicom_bytes)
    
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
    study_uid = ds.StudyInstanceUID
    series_uid = ds.SeriesInstanceUID
    sop_uid = ds.SOPInstanceUID
    
    # STOW first
    client.post(
        f"/dicomweb/studies/{study_uid}",
        content=dicom_bytes,
        headers={"Content-Type": "application/dicom"}
    )
    
    # Check initial prediction status (should be not_predicted)
    response = client.get(f"/studies/{study_uid}/prediction")
    assert response.status_code == 200
    assert response.json()["status"] == "not_predicted"
    
    # Run prediction
    response = client.post(f"/studies/{study_uid}/predict")
    assert response.status_code == 200
    pred_data = response.json()
    assert pred_data["status"] == "completed"
    assert "probability" in pred_data
    assert "prediction" in pred_data
    assert "text_justification" in pred_data
    assert "heatmap_url" in pred_data
    assert "sc_url" in pred_data
    assert "sr_url" in pred_data
    
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
    assert response.headers["content-type"] == "application/dicom"
    
    # Read returned bytes as DICOM and verify it is a Basic Text SR
    sr_ds = pydicom.dcmread(io.BytesIO(response.content))
    assert sr_ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.11' # Basic Text SR Storage
    assert sr_ds.ValueType == "CONTAINER"
