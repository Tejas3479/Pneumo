import argparse
import os
import numpy as np
import pandas as pd
from src.data import load_shipment_data, enrich_shipments, split_train_test
from src.features import build_features
from src.model import SupplyChainDelayPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Directory containing CSV data files")
    parser.add_argument("--model_path", default="models/xgb_model.json", help="Path to the trained model")
    args = parser.parse_args()

    print("Loading model...")
    if not os.path.exists(args.model_path):
        print(f"Model file {args.model_path} not found. Please train the model first.")
        return

    model = SupplyChainDelayPredictor.load(args.model_path)

    print("Loading test data...")
    shipments_path = os.path.join(args.data_dir, "shipments.csv")
    ports_path = os.path.join(args.data_dir, "ports.csv")
    suppliers_path = os.path.join(args.data_dir, "suppliers.csv")
    external_path = os.path.join(args.data_dir, "external_factors.csv")

    if not all(os.path.exists(p) for p in [shipments_path, ports_path, suppliers_path, external_path]):
        print("Data files not found. Please run generate_mock_data.py first.")
        return

    shipments_df = load_shipment_data(shipments_path)
    enriched_df = enrich_shipments(shipments_df, ports_path, suppliers_path, external_path)

    print("Recreating data splits...")
    _, X_test_raw, _, y_test = split_train_test(
        enriched_df, target="actual_delay_days", test_size=0.2, random_state=42
    )

    print("Building test features...")
    X_test = build_features(X_test_raw)

    print("Generating predictions...")
    preds = model.predict(X_test)
    mean_preds, lower_preds, upper_preds = model.predict_with_uncertainty(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Compute coverage of uncertainty intervals
    coverage = np.mean((y_test >= lower_preds) & (y_test <= upper_preds)) * 100

    print("\n================ EVALUATION RESULTS ================")
    print(f"Mean Absolute Error (MAE):    {mae:.4f} days")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} days")
    print(f"R-squared (R2) Score:          {r2:.4f}")
    print(f"90% Prediction Interval Coverage: {coverage:.2f}%")
    print("====================================================\n")

if __name__ == "__main__":
    main()
