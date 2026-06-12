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
    def __init__(self, model_name: str = "microsoft/Biovil-T", num_labels: int = 1, lr: float = 1e-4, use_lora: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr

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

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()

    def forward(self, x):
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
            
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self.loss_fn(logits, y)
        
        self.train_auroc.update(logits, y.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self.loss_fn(logits, y)
        
        self.val_auroc.update(logits, y.long())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
