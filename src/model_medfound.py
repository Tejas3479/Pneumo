import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

class MedicalFoundationClassifier(pl.LightningModule):
    """
    PyTorch Lightning module implementing a medical foundation model backbone
    (BioViL-T or CheXzero/CLIP) with LoRA fine-tuning for chest X-ray binary classification.
    """
    def __init__(self, model_name: str = "microsoft/Biovil-T", num_labels: int = 1, lr: float = 1e-4, use_lora: bool = True, debias: bool = False, debias_weight: float = 1.0, pos_weight: float = 4.0, weight_decay: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.debias = debias
        self.debias_weight = debias_weight

        # 1. Load model backbone based on model name
        if "biovil" in model_name.lower():
            # Load microsoft/Biovil-T vision module
            self.model = AutoModel.from_pretrained(model_name)
            if hasattr(self.model, "vision_model"):
                self.vision_model = self.model.vision_model
            else:
                self.vision_model = self.model
            hidden_size = getattr(self.vision_model.config, "hidden_size", 768)
        elif "chexzero" in model_name.lower() or "clip" in model_name.lower():
            # Load CLIP vision model backbone
            from transformers import CLIPModel
            clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.model = clip.vision_model
            self.vision_model = self.model
            hidden_size = getattr(self.vision_model.config, "hidden_size", 768)
        else:
            # Fallback to standard ViT
            self.model = AutoModel.from_pretrained(model_name)
            self.vision_model = self.model
            hidden_size = getattr(self.vision_model.config, "hidden_size", 768)

        # 2. Add classification linear head
        self.classifier = nn.Linear(hidden_size, num_labels)

        # 3. Setup LoRA
        if use_lora:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none"
            )
            self.vision_model = get_peft_model(self.vision_model, peft_config)
            print(f"LoRA adapters successfully registered on {model_name} vision backbone.")

        # Initialize debiasing head if enabled
        if self.debias:
            from src.fairness import AdversarialDebiasHead
            self.debias_head = AdversarialDebiasHead(input_dim=hidden_size, hidden_dim=256)

        # Loss function with class-imbalance correction via pos_weight
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        # Metrics
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()

    def forward_features(self, x):
        outputs = self.vision_model(x)
        
        # Extract features (CLS token or pooled output)
        if hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif isinstance(outputs, tuple):
            features = outputs[0][:, 0, :] if outputs[0].ndim == 3 else outputs[0]
        else:
            features = outputs
        return features

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, sex = batch
        else:
            x, y = batch[:2]
            sex = None
            
        features = self.forward_features(x)
        logits = self.classifier(features).squeeze(-1)
        loss = self.loss_fn(logits, y.float())
        
        if self.debias and sex is not None:
            sex_logits = self.debias_head(features).squeeze(-1)
            adv_loss = nn.BCEWithLogitsLoss()(sex_logits, sex.float())
            self.log("train_adv_loss", adv_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss = loss + self.debias_weight * adv_loss
            
        self.train_auroc.update(logits, y.long())
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
        
        self.val_auroc.update(logits, y.long())
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
            T_max=self.trainer.max_epochs if self.trainer else 20,
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
