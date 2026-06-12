import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import cv2
from torchvision import transforms

from src.siim_data import SIIMPneumothoraxDataset

def test_siim_dataset():
    # Setup temporary directory and files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy PNG file
        png_dir = os.path.join(tmp_dir, "png")
        os.makedirs(png_dir, exist_ok=True)
        
        img_id = "test_img_123"
        img_path = os.path.join(png_dir, f"{img_id}.png")
        
        # 224x224 grayscale image
        img_data = np.zeros((224, 224), dtype=np.uint8)
        img_data[50:150, 50:150] = 255
        cv2.imwrite(img_path, img_data)
        
        # Create dummy CSV
        csv_path = os.path.join(tmp_dir, "train.csv")
        df = pd.DataFrame([{
            "ImageId": img_id,
            "has_pneumo": 1
        }])
        df.to_csv(csv_path, index=False)
        
        # Define transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Instantiate dataset
        dataset = SIIMPneumothoraxDataset(csv_path=csv_path, img_dir=png_dir, transform=transform)
        
        # Verify length
        assert len(dataset) == 1
        
        # Retrieve item
        image_tensor, label_tensor = dataset[0]
        
        # Verify types and dimensions
        assert isinstance(image_tensor, torch.Tensor)
        assert isinstance(label_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 224, 224)
        assert label_tensor.item() == 1.0
        
        # Verify default fallback if image is missing
        missing_csv_path = os.path.join(tmp_dir, "missing.csv")
        df_missing = pd.DataFrame([{
            "ImageId": "non_existent",
            "has_pneumo": 0
        }])
        df_missing.to_csv(missing_csv_path, index=False)
        
        dataset_missing = SIIMPneumothoraxDataset(csv_path=missing_csv_path, img_dir=png_dir, transform=transform)
        img_tensor_missing, label_tensor_missing = dataset_missing[0]
        assert img_tensor_missing.shape == (3, 224, 224)
        assert label_tensor_missing.item() == 0.0
