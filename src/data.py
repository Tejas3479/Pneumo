import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class PneumothoraxDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Chest X-Ray DICOM images.
    """
    def __init__(self, dataframe: pd.DataFrame, data_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.data_dir, row['ImagePath'])
        label = row['Label']
        
        # Load DICOM using pydicom
        try:
            ds = pydicom.dcmread(image_path)
            pixel_array = ds.pixel_array.astype(np.float32)
        except Exception as e:
            # Fallback to zeros if file reading fails
            print(f"Error reading DICOM {image_path}: {e}")
            pixel_array = np.zeros((224, 224), dtype=np.float32)
            ds = None
            
        # Apply DICOM rescale if present, otherwise do min-max normalization
        if ds is not None and hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope) if ds.RescaleSlope is not None else 1.0
            intercept = float(ds.RescaleIntercept) if ds.RescaleIntercept is not None else 0.0
            pixel_array = pixel_array * slope + intercept

        # Normalize to [0, 1] range
        min_val = pixel_array.min()
        max_val = pixel_array.max()
        if max_val - min_val > 0:
            pixel_array = (pixel_array - min_val) / (max_val - min_val)
        else:
            pixel_array = np.zeros_like(pixel_array)

        # Convert monochrome (1 channel) to PIL RGB Image
        # Scale back to [0, 255] for PIL representation
        pixel_array_uint8 = (pixel_array * 255.0).astype(np.uint8)
        image = Image.fromarray(pixel_array_uint8).convert("RGB")

        # Apply transformations
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Default fallback transformation
            fallback_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = fallback_transform(image)

        # Convert label to float tensor for BCEWithLogitsLoss
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image_tensor, label_tensor

def get_dataloaders(csv_file: str, data_dir: str, batch_size: int = 32, val_split: float = 0.2, seed: int = 42, model_type: str = "vit"):
    """
    Reads the dataset description from a CSV, performs a stratified train/val split,
    and returns PyTorch DataLoader instances.
    """
    df = pd.read_csv(csv_file)
    
    # Stratified split to maintain class ratios in train and val sets
    train_df, val_df = train_test_split(
        df, 
        test_size=val_split, 
        random_state=seed, 
        stratify=df['Label']
    )

    # Set mean/std normalization based on model type
    if model_type.lower() == "vit":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    # Define transformations in strict order
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Create datasets
    train_dataset = PneumothoraxDataset(train_df, data_dir, transform=train_transform)
    val_dataset = PneumothoraxDataset(val_df, data_dir, transform=val_transform)
    
    # Create dataloaders
    # num_workers=0 is selected to prevent multiprocessing/hanging issues on Windows
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader
