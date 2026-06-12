import argparse
import os
import pandas as pd
from src.data import generate_mock_shipment_data, enrich_with_external_data, train_val_test_split
from src.features import FeatureEngineering
from src.model import ShipmentDelayPredictor
from src.utils import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--model_path", default="models/xgb_model.pkl")
    parser.add_argument("--feature_path", default="models/feature_eng.pkl")
    args = parser.parse_args()

    # Create models output directory
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    logger.info("Generating mock data...")
    df = generate_mock_shipment_data(num_samples=args.samples)
    df = enrich_with_external_data(df)
    train_df, val_df, test_df = train_val_test_split(df)

    logger.info("Building features...")
    fe = FeatureEngineering()
    y_train = train_df["ActualDelay"]
    fe.fit(train_df, y_train)
    X_train = fe.transform(train_df)
    X_val = fe.transform(val_df)
    y_val = val_df["ActualDelay"]

    logger.info("Training model...")
    model = ShipmentDelayPredictor()
    model.fit(X_train, y_train, X_val, y_val)

    logger.info(f"Saving model to {args.model_path} and feature engineering to {args.feature_path}...")
    model.save(args.model_path)
    fe.save(args.feature_path)
    logger.info("Training pipeline complete.")

if __name__ == "__main__":
    main()
