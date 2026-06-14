import os
import argparse
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils import set_seed
from src.data import get_dataloaders

# Constant list of random seeds for the ensemble
SEEDS = [42, 123, 456, 789, 1011]

def main():
    parser = argparse.ArgumentParser(description="PneumoDetect AI Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train.csv and dicoms/")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--model_type", type=str, default="vit", choices=["resnet", "vit", "medfound"], help="Type of model architecture")
    parser.add_argument("--dataset_type", type=str, default="mock", choices=["mock", "siim"], help="Type of dataset")
    parser.add_argument("--medfound_model", type=str, default="microsoft/Biovil-T", help="Hugging Face model ID for medical foundation model")
    parser.add_argument("--num_models", type=int, default=1, help="Number of models to train for the ensemble (1-5)")
    parser.add_argument("--debias", action="store_true", help="Enable adversarial debiasing during training")
    parser.add_argument("--debias_weight", type=float, default=1.0, help="Weight factor for debiasing loss")
    parser.add_argument("--pos_weight", type=float, default=4.0, help="BCEWithLogitsLoss positive class weight for class-imbalance correction (negative/positive ratio)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay for regularization")
    parser.add_argument("--production", action="store_true", help="Promote directly to production")
    args = parser.parse_args()

    # Create models output directory
    os.makedirs("models", exist_ok=True)

    if args.dataset_type.lower() == "siim":
        csv_path = os.path.join(args.data_dir, "siim", "train.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"SIIM-ACR train CSV not found at {csv_path}. Please run prepare_siim.py first."
            )
    else:
        csv_path = os.path.join(args.data_dir, "train.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Dataset CSV not found at {csv_path}. Please run generate_mock_data.py first."
            )

    # Determine model class dynamically
    if args.model_type.lower() == "vit":
        print("Using Vision Transformer (ViT-B/16) model architecture with LoRA.")
        from src.model_foundation import ViTPneumothoraxClassifier
        model_class = ViTPneumothoraxClassifier
    elif args.model_type.lower() == "medfound":
        print(f"Using Medical Foundation ({args.medfound_model}) model architecture with LoRA.")
        from src.model_medfound import MedicalFoundationClassifier
        model_class = MedicalFoundationClassifier
    else:
        print("Using ResNet-50 model architecture.")
        from src.model import PneumothoraxClassifier
        model_class = PneumothoraxClassifier

    num_runs = min(max(args.num_models, 1), len(SEEDS))

    for idx in range(num_runs):
        seed = SEEDS[idx]
        print(f"\n==========================================")
        print(f" TRAINING MODEL {idx+1}/{num_runs} (SEED: {seed})")
        print(f"==========================================")

        # Set seed for this training run
        set_seed(seed)

        # Initialize loaders using custom normalization configurations
        print("Initializing Datasets and DataLoaders...")
        train_loader, val_loader = get_dataloaders(
            csv_file=csv_path,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seed=seed,
            model_type=args.model_type,
            include_metadata=args.debias,
            dataset_type=args.dataset_type,
            medfound_model=args.medfound_model
        )

        # Initialize model
        if args.model_type.lower() == "vit":
            model = model_class(lr=args.lr, weight_decay=args.weight_decay, debias=args.debias, debias_weight=args.debias_weight, pos_weight=args.pos_weight)
        elif args.model_type.lower() == "medfound":
            model = model_class(model_name=args.medfound_model, lr=args.lr, weight_decay=args.weight_decay, use_lora=True, debias=args.debias, debias_weight=args.debias_weight, pos_weight=args.pos_weight)
        else:
            model = model_class(lr=args.lr, weight_decay=args.weight_decay, debias=args.debias, debias_weight=args.debias_weight, pos_weight=args.pos_weight)

        # Configure TensorBoard logger
        logger = TensorBoardLogger(save_dir="logs", name=f"pneumodetect_{args.model_type}_seed_{seed}")

        # Configure MLflow logger
        from mlops.mlflow_tracking import get_mlflow_logger
        mlflow_logger = get_mlflow_logger(run_name=f"run_seed_{seed}_{args.model_type}")

        # Configure EarlyStopping and ModelCheckpoint
        early_stop_callback = EarlyStopping(
            monitor="val_auroc",
            patience=5,
            mode="max",
            verbose=True
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath="models",
            filename=f"best_seed_{idx}",
            monitor="val_auroc",
            mode="max",
            save_top_k=1,
            verbose=True
        )

        callbacks = [early_stop_callback, checkpoint_callback]
        if args.debias:
            from src.fairness import FairnessLoggingCallback
            callbacks.append(FairnessLoggingCallback(val_loader))

        # Initialize Trainer with both loggers and gradient clipping
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            logger=[logger, mlflow_logger],
            callbacks=callbacks,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=1.0,          # Prevent exploding gradients
            gradient_clip_algorithm="norm",  # Clip by L2 norm
            enable_progress_bar=True
        )

        # Start training
        print(f"Starting training run {idx+1}...")
        trainer.fit(model, train_loader, val_loader)
        print(f"Training run {idx+1} complete! Checkpoint saved as models/best_seed_{idx}.ckpt")

        # Clean up hooks inside lightning module if implemented
        if hasattr(model, "on_destroy"):
            model.on_destroy()

    # Extract and save training metrics
    metrics_to_save = {}
    for k, v in trainer.callback_metrics.items():
        metrics_to_save[k] = float(v.item()) if hasattr(v, "item") else float(v)
    metrics_to_save["model_type"] = args.model_type
    if args.model_type.lower() == "medfound":
        metrics_to_save["medfound_model"] = args.medfound_model
    metrics_to_save["dataset_type"] = args.dataset_type
    metrics_to_save["epochs"] = args.epochs
    metrics_to_save["lr"] = args.lr
    metrics_to_save["batch_size"] = args.batch_size
    metrics_to_save["debias"] = args.debias

    with open(os.path.join("models", "metrics.json"), "w") as f:
        import json
        json.dump(metrics_to_save, f, indent=4)
    print("Saved metrics to models/metrics.json")

    # Post-process checkpoints
    src_path = os.path.join("models", "best_seed_0.ckpt")
    if os.path.exists(src_path):
        target_name = "production.ckpt" if args.production else "candidate.ckpt"
        dst_path = os.path.join("models", target_name)
        shutil.copy(src_path, dst_path)
        print(f"Copied best checkpoint to {dst_path}")
        
        # If production is explicitly set, copy to standard filenames as well
        if args.production:
            shutil.copy(src_path, os.path.join("models", "best.ckpt"))
            shutil.copy(src_path, os.path.join("models", "best_seed_0.ckpt"))
            print("Copied production checkpoint to best.ckpt and best_seed_0.ckpt")

    print("\nAll training runs complete.")

if __name__ == "__main__":
    main()
