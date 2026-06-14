import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC

class PneumothoraxClassifier(pl.LightningModule):
    """
    PyTorch Lightning module implementing ResNet-50 binary classification.

    Improvements:
    - pos_weight in BCEWithLogitsLoss for class-imbalance correction.
    - AdamW + CosineAnnealingLR scheduler.
    - Gradient clipping handled by Trainer (max_grad_norm=1.0).
    """
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        debias: bool = False,
        debias_weight: float = 1.0,
        pos_weight: float = 4.0,  # negative/positive ratio for class imbalance
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.debias = debias
        self.debias_weight = debias_weight

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

        # Initialize debiasing head if enabled
        if self.debias:
            from src.fairness import AdversarialDebiasHead
            self.debias_head = AdversarialDebiasHead(input_dim=in_features, hidden_dim=256)

        # Loss function with class-imbalance correction via pos_weight
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        # Metrics
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()

    def forward(self, x):
        return self.resnet(x)

    def forward_features(self, x):
        # Extract features right before the fc layer
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, sex = batch
        else:
            x, y = batch[:2]
            sex = None

        features = self.forward_features(x)
        logits = self.resnet.fc(features).squeeze(-1)
        loss = self.loss_fn(logits, y.float())

        if self.debias and sex is not None:
            sex_logits = self.debias_head(features).squeeze(-1)
            adv_loss = nn.BCEWithLogitsLoss()(sex_logits, sex.float())
            self.log("train_adv_loss", adv_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss = loss + self.debias_weight * adv_loss

        # Update metrics
        self.train_auroc.update(logits, y.long())

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, sex = batch
        else:
            x, y = batch[:2]

        logits = self(x).squeeze(-1)
        loss = self.loss_fn(logits, y.float())

        # Update metrics
        self.val_auroc.update(logits, y.long())

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 15,
            eta_min=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_auroc"
            }
        }
