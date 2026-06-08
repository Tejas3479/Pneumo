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
    """
    def __init__(self, lr: float = 1e-4, r: int = 16, lora_alpha: int = 16, debias: bool = False, debias_weight: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.debias = debias
        self.debias_weight = debias_weight
        
        # Configure the HuggingFace ViT base model
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        config.num_labels = 1
        config.output_hidden_states = True
        config.output_attentions = True  # Required for attention-based Grad-CAM
        
        # Load the base classifier
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            config=config,
            ignore_mismatched_sizes=True
        )
        
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
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        
        # Hooks variables for Grad-CAM
        self.attention_map = None
        self.attention_grad = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers forward and backward hooks to capture attention maps and their gradients.
        """
        target_layer = None
        # Safely locate the last self-attention layer of ViT
        try:
            target_layer = self.resnet_or_vit.base_model.model.vit.encoder.layer[-1].attention.attention
        except AttributeError:
            try:
                target_layer = self.resnet_or_vit.vit.encoder.layer[-1].attention.attention
            except AttributeError:
                pass
                
        if target_layer is not None:
            def forward_hook_with_grad(module, input, output):
                # output: (context_layer, attention_probs) if output_attentions=True
                if isinstance(output, tuple) and len(output) > 1:
                    self.attention_map = output[1]
                    if self.attention_map.requires_grad:
                        # Register tensor hook to capture gradients w.r.t attention probs
                        h = self.attention_map.register_hook(self._save_grad)
                        
            h_forward = target_layer.register_forward_hook(forward_hook_with_grad)
            self._hook_handles.append(h_forward)
            print("ViT Attention Grad-CAM hooks successfully registered.")
        else:
            print("Warning: Could not register self-attention hooks on ViT backbone.")

    def _save_grad(self, grad):
        self.attention_grad = grad

    def forward(self, x):
        # Clear previous hook results before forward pass
        self.attention_map = None
        self.attention_grad = None
        
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_destroy(self):
        """Clean up registered hook handles to avoid memory leaks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
