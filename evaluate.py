import os
import argparse
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

from src.data import get_dataloaders
from src.model import PneumothoraxClassifier
from src.model_foundation import ViTPneumothoraxClassifier

MODEL_CLASSES = {
    "resnet": PneumothoraxClassifier,
    "vit": ViTPneumothoraxClassifier,
}

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained PneumoDetect AI Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train.csv and dicoms/")
    parser.add_argument("--checkpoint", type=str, default="models/best.ckpt", help="Path to PyTorch Lightning checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for validation")
    parser.add_argument("--model_type", type=str, default="vit", choices=["resnet", "vit"], help="Type of model architecture")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        # Fallback search if filename was appended with epoch/step by ModelCheckpoint
        checkpoint_dir = os.path.dirname(args.checkpoint)
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")] if os.path.exists(checkpoint_dir) else []
        if ckpt_files:
            args.checkpoint = os.path.join(checkpoint_dir, ckpt_files[0])
            print(f"Specified checkpoint not found. Auto-detected: {args.checkpoint}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")

    # Load validation data
    csv_path = os.path.join(args.data_dir, "train.csv")
    
    # Check if model type is supported
    model_type = args.model_type.lower()
    if model_type not in MODEL_CLASSES:
        print(f"Error: Unknown model type '{args.model_type}'. Supported types: {list(MODEL_CLASSES.keys())}")
        sys.exit(1)
        
    _, val_loader = get_dataloaders(csv_path, args.data_dir, batch_size=args.batch_size, model_type=model_type)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint} as model_type: {model_type}")
    model_class = MODEL_CLASSES[model_type]
    model = model_class.load_from_checkpoint(args.checkpoint)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_logits = []
    all_targets = []

    print("Running evaluation on validation set...")
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x).squeeze(-1)
            all_logits.extend(logits.cpu().numpy())
            all_targets.extend(y.numpy())

    all_logits = np.array(all_logits)
    all_targets = np.array(all_targets)
    
    # Compute sigmoid probabilities
    probabilities = 1.0 / (1.0 + np.exp(-all_logits))
    predictions = (probabilities > 0.5).astype(int)

    # Metrics calculation
    # Handing cases where validation set has only one class present to avoid roc_auc_score crash
    try:
        auroc = roc_auc_score(all_targets, probabilities)
    except ValueError:
        auroc = float('nan')
        print("Warning: Only one class present in validation targets. AUROC is undefined.")

    accuracy = accuracy_score(all_targets, predictions)
    f1 = f1_score(all_targets, predictions, zero_division=0)
    cm = confusion_matrix(all_targets, predictions)

    print("\n" + "="*40)
    print("             EVALUATION REPORT")
    print("="*40)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation Samples: {len(all_targets)}")
    print(f"AUROC:              {auroc:.4f}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"F1 Score:           {f1:.4f}")
    print("-"*40)
    print("Confusion Matrix:")
    print(f"   True Negatives (TN):  {cm[0, 0] if cm.shape == (2,2) else 'N/A'}")
    print(f"   False Positives (FP): {cm[0, 1] if cm.shape == (2,2) else 'N/A'}")
    print(f"   False Negatives (FN): {cm[1, 0] if cm.shape == (2,2) else 'N/A'}")
    print(f"   True Positives (TP):  {cm[1, 1] if cm.shape == (2,2) else 'N/A'}")
    print("="*40)

if __name__ == "__main__":
    main()
