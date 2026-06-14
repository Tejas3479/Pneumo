import os
import sys
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# On Windows, multiprocessing workers cause deadlocks; force single-threaded loading.
# On Linux/Docker containers (production), allow parallel workers for throughput.
_NUM_WORKERS = 0 if sys.platform.startswith('win') else 4

class PneumothoraxDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Chest X-Ray DICOM images.
    """
    def __init__(self, dataframe: pd.DataFrame, data_dir: str, transform=None, include_metadata: bool = False):
        self.df = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.include_metadata = include_metadata

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

        if self.include_metadata:
            # pandas Series.get() does not exist — use .get() from dict or check index
            sex = int(row['Sex']) if 'Sex' in row.index and not pd.isna(row['Sex']) else 0
            age = int(row['Age']) if 'Age' in row.index and not pd.isna(row['Age']) else 40
            sex_tensor = torch.tensor(sex, dtype=torch.float32)
            age_tensor = torch.tensor(age, dtype=torch.float32)
            return image_tensor, label_tensor, sex_tensor, age_tensor

        return image_tensor, label_tensor

NORMALIZATION = {
    "resnet": {
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225]
    },
    "vit": {
        "mean": [0.5, 0.5, 0.5],
        "std":  [0.5, 0.5, 0.5]
    },
    "medfound": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std":  [0.26862954, 0.26130258, 0.27577711]
    }
}

def get_normalization_params(model_type, medfound_model=None):
    """Return mean, std for the given model_type."""
    m_type = model_type.lower()
    if m_type == "medfound":
        if medfound_model and "chexzero" in medfound_model.lower():
            norm = NORMALIZATION["medfound"]
        else:
            # microsoft/Biovil-T uses ViT normalization
            norm = NORMALIZATION["vit"]
    else:
        norm = NORMALIZATION.get(m_type, NORMALIZATION["resnet"])
    return norm["mean"], norm["std"]

def get_train_transforms(model_type, medfound_model=None):
    mean, std = get_normalization_params(model_type, medfound_model)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_val_transforms(model_type, medfound_model=None):
    mean, std = get_normalization_params(model_type, medfound_model)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_dataloaders(csv_file: str, data_dir: str, batch_size: int = 32, val_split: float = 0.2, seed: int = 42, model_type: str = "vit", include_metadata: bool = False, dataset_type: str = "mock", medfound_model: str = "microsoft/Biovil-T"):
    """
    Reads the dataset description from a CSV/dir, performs split if mock,
    and returns PyTorch DataLoader instances.
    """
    if dataset_type.lower() == "siim":
        from src.siim_data import SIIMPneumothoraxDataset
        train_csv = os.path.join(data_dir, "siim", "train.csv")
        val_csv = os.path.join(data_dir, "siim", "val.csv")
        img_dir = os.path.join(data_dir, "siim", "png")
        
        train_dataset = SIIMPneumothoraxDataset(
            csv_path=train_csv,
            img_dir=img_dir,
            transform=get_train_transforms(model_type, medfound_model)
        )
        val_dataset = SIIMPneumothoraxDataset(
            csv_path=val_csv,
            img_dir=img_dir,
            transform=get_val_transforms(model_type, medfound_model)
        )
    else:
        df = pd.read_csv(csv_file)
        
        # Stratified split to maintain class ratios in train and val sets
        train_df, val_df = train_test_split(
            df, 
            test_size=val_split, 
            random_state=seed, 
            stratify=df['Label']
        )

        # Save validation indices to file for fairness audits
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, "val_indices.npy"), val_df.index.to_numpy())

        train_transform = get_train_transforms(model_type, medfound_model)
        val_transform = get_val_transforms(model_type, medfound_model)
        
        # Create datasets
        train_dataset = PneumothoraxDataset(train_df, data_dir, transform=train_transform, include_metadata=include_metadata)
        val_dataset = PneumothoraxDataset(val_df, data_dir, transform=val_transform, include_metadata=include_metadata)
    
    # Create dataloaders
    # num_workers is 0 on Windows to prevent multiprocessing/hanging issues;
    # on Linux/Docker containers it is 4 for better throughput.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=_NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=_NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader
