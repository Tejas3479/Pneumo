import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data import generate_mock_shipment_data, enrich_with_external_data, train_val_test_split
from src.features import FeatureEngineering
from src.model import ShipmentDelayPredictor
from src.utils import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/xgb_model.pkl")
    parser.add_argument("--feature_path", default="models/feature_eng.pkl")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    logger.info("Loading model and feature engineering pipeline...")
    model = ShipmentDelayPredictor.load(args.model_path)
    fe = FeatureEngineering.load(args.feature_path)

    logger.info("Generating test dataset...")
    df = generate_mock_shipment_data(num_samples=args.samples)
    df = enrich_with_external_data(df)
    _, _, test_df = train_val_test_split(df)
    
    X_test = fe.transform(test_df)
    y_test = test_df["ActualDelay"]

    logger.info("Evaluating...")
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    logger.info(f"MAE: {mae:.3f}")
    logger.info(f"RMSE: {rmse:.3f}")
    logger.info(f"R2: {r2:.3f}")

if __name__ == "__main__":
    main()
