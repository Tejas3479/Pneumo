import os
import sys
import shutil
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data import get_dataloaders
from src.model_foundation import ViTPneumothoraxClassifier
from src.model import PneumothoraxClassifier
from export_onnx import export_onnx_model

def evaluate_model(model_class, ckpt_path, val_loader):
    """
    Evaluates a model checkpoint on validation data and returns validation AUROC.
    """
    print(f"Evaluating model checkpoint: {ckpt_path} ...")
    model = model_class.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to("cpu")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            
            logits = model(x)
            # ViT or ResNet logits shape handling
            if len(logits.shape) > 1 and logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            else:
                probs = torch.sigmoid(logits.squeeze()).numpy()
                if probs.ndim == 0:
                    probs = np.array([probs])
            
            all_preds.extend(probs)
            all_targets.extend(y.numpy())
            
    val_auroc = roc_auc_score(all_targets, all_preds)
    print(f"Validation AUROC: {val_auroc:.4f}")
    return val_auroc

def main():
    metrics_path = "models/metrics.json"
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Cannot determine model type.")
        sys.exit(1)
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    model_type = metrics.get("model_type", "vit")
    print(f"Detected model type: {model_type}")
    
    if model_type == "vit":
        model_class = ViTPneumothoraxClassifier
    else:
        model_class = PneumothoraxClassifier
        
    candidate_ckpt = "models/candidate.ckpt"
    production_ckpt = "models/production.ckpt"
    
    if not os.path.exists(candidate_ckpt):
        print(f"Error: Candidate model checkpoint not found at {candidate_ckpt}")
        sys.exit(1)
        
    # Initialize dataloaders
    print("Loading validation dataset...")
    # Small batch size for faster CPU validation evaluation
    _, val_loader = get_dataloaders(
        csv_file="data/train.csv",
        data_dir="data",
        batch_size=16,
        seed=42,
        model_type=model_type,
        include_metadata=False
    )
    
    candidate_auroc = evaluate_model(model_class, candidate_ckpt, val_loader)
    
    # Check if production model exists
    if not os.path.exists(production_ckpt):
        print("No existing production model found. Promoting candidate model automatically.")
        should_promote = True
        production_auroc = 0.0
    else:
        production_auroc = evaluate_model(model_class, production_ckpt, val_loader)
        # Strictly greater comparison as per requirements
        should_promote = candidate_auroc > production_auroc
        
    if should_promote:
        print(f"Candidate model (AUROC: {candidate_auroc:.4f}) outperforms production (AUROC: {production_auroc:.4f}). Promoting candidate!")
        
        # 1. Copy candidate to production checkpoints
        shutil.copy(candidate_ckpt, production_ckpt)
        shutil.copy(candidate_ckpt, "models/best.ckpt")
        shutil.copy(candidate_ckpt, "models/best_seed_0.ckpt")
        print("Updated production model checkpoints.")
        
        # 2. Trigger ONNX exports
        weights_path = "models/model_weights.npy"
        print("Exporting production model to ONNX...")
        export_onnx_model(model_type, production_ckpt, weights_path, "models/production.onnx")
        
        # Overwrite standard model files so live API loads the updated version
        shutil.copy("models/production.onnx", "models/model.onnx")
        shutil.copy("models/production.onnx", "models/model_0.onnx")
        print("Promoted ONNX files successfully.")
        sys.exit(0)
    else:
        print(f"Candidate model (AUROC: {candidate_auroc:.4f}) does not outperform production (AUROC: {production_auroc:.4f}). Keeping current production model.")
        sys.exit(1)

if __name__ == "__main__":
    main()
