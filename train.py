import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils import set_seed
from src.data import get_dataloaders
from src.model import PneumothoraxClassifier

def main():
    parser = argparse.ArgumentParser(description="PneumoDetect AI Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train.csv and dicoms/")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    args = parser.parse_args()

    # Create models output directory
    os.makedirs("models", exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)

    csv_path = os.path.join(args.data_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset CSV not found at {csv_path}. Please run generate_mock_data.py first."
        )

    # Get DataLoaders
    print("Initializing Datasets and DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        csv_file=csv_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=42
    )

    # Initialize model
    print("Loading PneumothoraxClassifier (ResNet-50)...")
    model = PneumothoraxClassifier(lr=args.lr)

    # Configure TensorBoard logger
    logger = TensorBoardLogger(save_dir="logs", name="pneumodetect")

    # Configure Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_auroc",
        patience=5,
        mode="max",
        verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="best",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        verbose=True
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True
    )

    # Start training
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training finished! Best checkpoint saved under models/best.ckpt")

if __name__ == "__main__":
    main()
