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
    parser.add_argument("--model_type", type=str, default="vit", choices=["resnet", "vit"], help="Type of model architecture")
    parser.add_argument("--num_models", type=int, default=1, help="Number of models to train for the ensemble (1-5)")
    parser.add_argument("--debias", action="store_true", help="Enable adversarial debiasing during training")
    parser.add_argument("--debias_weight", type=float, default=1.0, help="Weight factor for debiasing loss")
    args = parser.parse_args()

    # Create models output directory
    os.makedirs("models", exist_ok=True)

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
            include_metadata=args.debias
        )

        # Initialize model
        if args.model_type.lower() == "vit":
            model = model_class(lr=args.lr, debias=args.debias, debias_weight=args.debias_weight)
        else:
            model = model_class(lr=args.lr)

        # Configure TensorBoard logger
        logger = TensorBoardLogger(save_dir="logs", name=f"pneumodetect_{args.model_type}_seed_{seed}")

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
        if args.model_type.lower() == "vit" and args.debias:
            from src.fairness import FairnessLoggingCallback
            callbacks.append(FairnessLoggingCallback(val_loader))

        # Initialize Trainer
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            logger=logger,
            callbacks=callbacks,
            accelerator="auto",
            devices="auto",
            enable_progress_bar=True
        )

        # Start training
        print(f"Starting training run {idx+1}...")
        trainer.fit(model, train_loader, val_loader)
        print(f"Training run {idx+1} complete! Checkpoint saved as models/best_seed_{idx}.ckpt")

        # Clean up hooks inside lightning module if implemented
        if hasattr(model, "on_destroy"):
            model.on_destroy()

    # Post-process checkpoints: if only training a single model, copy best_seed_0 to best.ckpt
    if num_runs == 1:
        src_path = os.path.join("models", "best_seed_0.ckpt")
        dst_path = os.path.join("models", "best.ckpt")
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"Copied single model checkpoint to {dst_path}")

    print("\nAll training runs complete.")

if __name__ == "__main__":
    main()
