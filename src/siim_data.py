import torch
import pandas as pd
import cv2
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SIIMPneumothoraxDataset(Dataset):
    def __init__(self, csv_path, img_dir="data/siim/png", transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['ImageId']}.png")
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback if image not found on disk
            import numpy as np
            img = np.zeros((224, 224), dtype=np.uint8)
            
        # Convert to PIL RGB Image
        image = Image.fromarray(img).convert("RGB")
        
        if self.transform:
            img_tensor = self.transform(image)
        else:
            # Default fallback transformation
            fallback_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = fallback_transform(image)
            
        label = torch.tensor(row["has_pneumo"], dtype=torch.float32).unsqueeze(0)
        return img_tensor, label
