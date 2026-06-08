import os
import torch
import torch.nn as nn
import numpy as np
from src.federated import PneumoFlowerClient

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Trainable parameters (fc1)
        self.fc1 = nn.Linear(10, 5)
        # Frozen parameters (fc2)
        self.fc2 = nn.Linear(5, 1)
        for p in self.fc2.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        return self.fc2(self.fc1(x))

def test_dp_sgd_and_parameter_filtering():
    model = MockModel()
    
    client = PneumoFlowerClient(
        model=model,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        max_grad_norm=0.5,
        noise_multiplier=0.1
    )
    
    # Check that only fc1 parameters are returned
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    
    parameters = client.get_parameters(config={})
    assert len(parameters) == 2
    assert len(trainable_params) == 2
    
    # Test setting parameters
    new_vals = [np.ones_like(p.detach().cpu().numpy()) * 2.0 for p in trainable_params]
    client.set_parameters(new_vals)
    for p in trainable_params:
        assert torch.all(p == 2.0)
        
    # Check that frozen parameters did NOT change
    for p in frozen_params:
        assert not torch.all(p == 2.0)

def test_client_fit_evaluation(tmp_path):
    # Setup test database
    test_db = os.path.join(tmp_path, "test_active_learning.db")
    import src.active_learning
    original_db_path = src.active_learning.DB_PATH
    src.active_learning.DB_PATH = test_db
    
    try:
        src.active_learning.init_db()
        
        # Write a dummy DICOM image to data/dicoms/
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian
        
        dicoms_dir = os.path.join("data", "dicoms")
        os.makedirs(dicoms_dir, exist_ok=True)
        img_path = os.path.join(dicoms_dir, "test_fit_img.dcm")
        
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.1')
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = file_meta
        ds.preamble = b"\0" * 128
        ds.Rows = 224
        ds.Columns = 224
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.zeros((224, 224), dtype=np.uint16).tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(img_path, write_like_original=False)
        
        # Save feedback in DB
        src.active_learning.save_clinician_feedback("dicoms/test_fit_img.dcm", 1)
        
        # Instantiate model and client
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224*224*3, 1)
        )
        for p in model.parameters():
            p.requires_grad = True
            
        client = PneumoFlowerClient(
            model=model,
            epochs=1,
            batch_size=1,
            lr=1e-3,
            max_grad_norm=1.0,
            noise_multiplier=0.1
        )
        
        parameters = client.get_parameters(config={})
        
        # Run fit
        fit_params, num_samples, fit_metrics = client.fit(parameters, config={})
        assert num_samples == 1
        assert "loss" in fit_metrics
        
        # Run evaluate
        loss, num_samples_eval, eval_metrics = client.evaluate(fit_params, config={})
        assert num_samples_eval == 1
        assert "accuracy" in eval_metrics
        
    finally:
        # Restore DB_PATH
        src.active_learning.DB_PATH = original_db_path
        img_path = os.path.join("data", "dicoms", "test_fit_img.dcm")
        if os.path.exists(img_path):
            os.remove(img_path)
