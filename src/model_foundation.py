import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model

class ViTPneumothoraxClassifier(pl.LightningModule):
    """
    PyTorch Lightning module implementing a ViT-B/16 foundation model (LoRA tuned)
    for Chest X-Ray Pneumothorax binary classification.

    Improvements over baseline:
    - pos_weight: BCEWithLogitsLoss class-imbalance correction (negative/positive ratio).
    - CosineAnnealingLR scheduler: smooth LR decay prevents overfitting.
    - Gradient clipping (max_norm=1.0) via trainer flag — declared here for docs.
    - AdamW instead of Adam: weight decay for regularisation.
    """
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        r: int = 16,
        lora_alpha: int = 16,
        debias: bool = False,
        debias_weight: float = 1.0,
        pos_weight: float = 4.0,  # default: 4:1 negative-to-positive ratio
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.debias = debias
        self.debias_weight = debias_weight

        import os
        local_dir = os.path.join("models", "pretrained", "vit-base-patch16-224-in21k")
        config = None
        model = None

        # 1. Try loading from local weights directory
        if os.path.exists(local_dir):
            try:
                config = ViTConfig.from_pretrained(local_dir, local_files_only=True)
                config.num_labels = 1
                config.output_hidden_states = True
                config.output_attentions = True
                model = ViTForImageClassification.from_pretrained(
                    local_dir,
                    config=config,
                    ignore_mismatched_sizes=True,
                    local_files_only=True
                )
                print(f"Loaded ViT model locally from {local_dir}")
            except Exception as e:
                print(f"Error loading local weights from {local_dir}: {e}. Retrying online...")

        # 2. If not loaded, try downloading online
        if model is None:
            try:
                config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
                config.num_labels = 1
                config.output_hidden_states = True
                config.output_attentions = True
                model = ViTForImageClassification.from_pretrained(
                    "google/vit-base-patch16-224-in21k",
                    config=config,
                    ignore_mismatched_sizes=True
                )
                print("Downloaded and loaded ViT model from Hugging Face Hub.")
            except Exception as e:
                print(f"Offline or network error loading ViT from Hugging Face Hub: {e}. Falling back to random initialization.")

        # 3. Fallback to random initialization if both failed
        if model is None:
            config = ViTConfig(
                num_labels=1,
                output_hidden_states=True,
                output_attentions=True
            )
            model = ViTForImageClassification(config)
            print("Initialized ViT model with random weights.")

        # Setup LoRA configuration
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"]  # The linear classification head
        )

        # Wrap model with LoRA adapters
        self.resnet_or_vit = get_peft_model(model, peft_config)

        if self.debias:
            from src.fairness import AdversarialDebiasHead
            self.debias_head = AdversarialDebiasHead(input_dim=768, hidden_dim=256)

        # Loss function with class-imbalance correction via pos_weight
        # pos_weight > 1 increases recall for minority positive class
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        # Metrics
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()

    def forward(self, x):
        outputs = self.resnet_or_vit(x)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        if self.debias:
            x, y, sex = batch
        else:
            x, y = batch[:2]
            sex = None

        if self.debias and sex is not None:
            # We must output hidden states to extract CLS token
            outputs = self.resnet_or_vit(x, output_hidden_states=True)
            logits = outputs.logits.squeeze(-1)
            loss = self.loss_fn(logits, y)

            # Extract CLS token from the last layer's hidden states
            cls_representation = outputs.hidden_states[-1][:, 0, :]
            debias_logits = self.debias_head(cls_representation, alpha=1.0).squeeze(-1)
            debias_loss = nn.BCEWithLogitsLoss()(debias_logits, sex)

            total_loss = loss + self.debias_weight * debias_loss

            # Log separate loss components
            self.log("train_clf_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train_debias_loss", debias_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            self.train_auroc.update(logits, y.long())
            self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return total_loss
        else:
            logits = self(x).squeeze(-1)
            loss = self.loss_fn(logits, y)

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
        loss = self.loss_fn(logits, y)

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
        # CosineAnnealing: smoothly decays LR to near-zero over T_max epochs
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
