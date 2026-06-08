import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
from torchvision import transforms

from generate_mock_data import create_mock_dicom
from src.data import PneumothoraxDataset

def test_pneumothorax_dataset():
    # Setup temporary directory and files
    with tempfile.TemporaryDirectory() as tmp_dir:
        dicoms_dir = os.path.join(tmp_dir, "dicoms")
        os.makedirs(dicoms_dir, exist_ok=True)
        
        # Create a mock DICOM image
        mock_img_path = "dicoms/test_img.dcm"
        full_img_path = os.path.join(tmp_dir, mock_img_path)
        
        # 224x224 mock image with a bright circle shape
        img_data = np.zeros((224, 224), dtype=np.float32)
        y, x = np.ogrid[:224, :224]
        mask = (x - 112)**2 + (y - 112)**2 < 50**2
        img_data[mask] = 1.0
        
        create_mock_dicom(full_img_path, img_data)
        
        # Create dummy dataframe
        df = pd.DataFrame([{
            "ImagePath": mock_img_path,
            "Label": 1
        }])
        
        # Define transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Instantiate dataset
        dataset = PneumothoraxDataset(dataframe=df, data_dir=tmp_dir, transform=transform)
        
        # Verify length
        assert len(dataset) == 1
        
        # Retrieve item
        image_tensor, label_tensor = dataset[0]
        
        # Verify types and dimensions
        assert isinstance(image_tensor, torch.Tensor)
        assert isinstance(label_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 224, 224)
        assert label_tensor.item() == 1.0
        
        # Verify values are normalized (not in [0, 255])
        assert image_tensor.max() < 10.0
        assert image_tensor.min() > -10.0
