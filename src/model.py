import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC

class PneumothoraxClassifier(pl.LightningModule):
    """
    PyTorch Lightning module implementing ResNet-50 binary classification.
    """
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        
        import os
        local_weights_path = os.path.join("models", "pretrained", "resnet50.pth")
        
        # 1. Try loading from local weights file if exists
        loaded_locally = False
        if os.path.exists(local_weights_path):
            try:
                self.resnet = models.resnet50(weights=None)
                state_dict = torch.load(local_weights_path, map_location="cpu")
                self.resnet.load_state_dict(state_dict)
                loaded_locally = True
                print(f"Loaded ResNet-50 weights locally from {local_weights_path}")
            except Exception as e:
                print(f"Error loading local weights from {local_weights_path}: {e}. Retrying online...")
        
        # 2. If not loaded locally, try online download
        if not loaded_locally:
            try:
                try:
                    self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                except AttributeError:
                    self.resnet = models.resnet50(pretrained=True)
                print("Downloaded and loaded ResNet-50 weights from PyTorch Hub.")
            except Exception as e:
                print(f"Offline or network error loading ResNet-50: {e}. Falling back to random initialization.")
                self.resnet = models.resnet50(weights=None)
            
        # Modify the classification head for binary classification (1 output feature)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 1)
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self.loss_fn(logits, y)
        
        # Update metrics
        self.train_auroc.update(logits, y.long())
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self.loss_fn(logits, y)
        
        # Update metrics
        self.val_auroc.update(logits, y.long())
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
