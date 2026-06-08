import os
import torch
import pandas as pd
import numpy as np

# Apply NumPy 2.0 monkeypatch before importing Flower
if not hasattr(np, 'float_'):
    np.float_ = np.float64

import flwr as fl
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data import PneumothoraxDataset
from src.active_learning import get_feedback_dataset

class PneumoFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, epochs: int = 1, batch_size: int = 4, lr: float = 1e-4, 
                 max_grad_norm: float = 1.0, noise_multiplier: float = 0.1):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        
        # Identify trainable named parameters (requires_grad=True)
        # This restricts parameter synchronization strictly to LoRA adapters + classifier head
        self.trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    def get_parameters(self, config):
        # Return only the trainable parameters
        return [p.detach().cpu().numpy() for name, p in self.trainable_params]

    def set_parameters(self, parameters):
        # Load parameters back into the model's trainable weights
        for (_, p), val in zip(self.trainable_params, parameters):
            p.data = torch.from_numpy(val).to(p.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Load feedback data
        df = get_feedback_dataset()
        if len(df) == 0:
            # Fallback to train.csv to allow test rounds if no feedback exists
            train_csv = os.path.join("data", "train.csv")
            if os.path.exists(train_csv):
                df = pd.read_csv(train_csv).head(8)
            else:
                # If everything is missing, create a dummy dataframe to prevent crash
                df = pd.DataFrame({
                    "ImagePath": ["dicoms/image_000.dcm"],
                    "Label": [0],
                    "Sex": [0],
                    "Age": [40]
                })

        # Set up dataloader
        # Check model_type for normalization
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        dataset = PneumothoraxDataset(df, "data", transform=transform, include_metadata=False)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        
        optimizer = torch.optim.Adam([p for _, p in self.trainable_params], lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        self.model.train()
        device = next(self.model.parameters()).device
        
        total_loss = 0.0
        samples_count = 0
        
        for epoch in range(self.epochs):
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(x).squeeze(-1)
                loss = loss_fn(logits, y)
                
                loss.backward()
                
                # DP-SGD Approximation Step:
                # 1. Clip gradients of trainable parameters
                trainable_tensors = [p for _, p in self.trainable_params]
                torch.nn.utils.clip_grad_norm_(trainable_tensors, max_norm=self.max_grad_norm)
                
                # 2. Add Gaussian noise to clipped gradients
                with torch.no_grad():
                    for p in trainable_tensors:
                        if p.grad is not None:
                            noise = torch.randn_like(p.grad) * (self.noise_multiplier * self.max_grad_norm / self.batch_size)
                            p.grad.add_(noise)
                            
                optimizer.step()
                
                total_loss += loss.item() * len(x)
                samples_count += len(x)
                
        avg_loss = total_loss / max(samples_count, 1)
        return self.get_parameters(config={}), samples_count, {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        df = get_feedback_dataset()
        if len(df) == 0:
            train_csv = os.path.join("data", "train.csv")
            if os.path.exists(train_csv):
                df = pd.read_csv(train_csv).head(4)
            else:
                df = pd.DataFrame({
                    "ImagePath": ["dicoms/image_000.dcm"],
                    "Label": [0],
                    "Sex": [0],
                    "Age": [40]
                })

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        dataset = PneumothoraxDataset(df, "data", transform=transform, include_metadata=False)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model.eval()
        device = next(self.model.parameters()).device
        
        total_loss = 0.0
        samples_count = 0
        correct = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                logits = self.model(x).squeeze(-1)
                loss = loss_fn(logits, y)
                
                total_loss += loss.item() * len(x)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y).sum().item()
                samples_count += len(x)
                
        avg_loss = total_loss / max(samples_count, 1)
        accuracy = correct / max(samples_count, 1)
        
        return float(avg_loss), samples_count, {"accuracy": float(accuracy)}
