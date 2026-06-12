import argparse
import os
import pandas as pd
from src.data import load_shipment_data, enrich_shipments, split_train_test
from src.features import build_features
from src.model import SupplyChainDelayPredictor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Directory containing CSV data files")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials for Optuna")
    parser.add_argument("--model_out", default="models/xgb_model.json", help="Path to save the trained model")
    args = parser.parse_args()

    # Create models output directory
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    print("Loading and enriching data...")
    shipments_path = os.path.join(args.data_dir, "shipments.csv")
    ports_path = os.path.join(args.data_dir, "ports.csv")
    suppliers_path = os.path.join(args.data_dir, "suppliers.csv")
    external_path = os.path.join(args.data_dir, "external_factors.csv")

    if not all(os.path.exists(p) for p in [shipments_path, ports_path, suppliers_path, external_path]):
        print("Data files not found. Please run generate_mock_data.py first.")
        return

    shipments_df = load_shipment_data(shipments_path)
    enriched_df = enrich_shipments(shipments_df, ports_path, suppliers_path, external_path)

    print("Splitting data into train and test sets...")
    X_train_raw, X_val_raw, y_train, y_val = split_train_test(
        enriched_df, target="actual_delay_days", test_size=0.2, random_state=42
    )

    print("Building features...")
    X_train = build_features(X_train_raw)
    X_val = build_features(X_val_raw)

    best_params = None

    if args.tune:
        print("Running Optuna hyperparameter optimization...")
        import optuna
        # Suppress optuna logs a bit
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "objective": "reg:squarederror",
                "eval_metric": "mae",
                "random_state": 42,
                "verbosity": 0
            }
            # Train a temporary model just for the mean prediction
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return mean_absolute_error(y_val, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.n_trials)
        best_params = study.best_params
        best_params.update({
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "random_state": 42,
            "verbosity": 0
        })
        print(f"Best hyperparameters found: {best_params}")

    print("Training final quantile models...")
    model = SupplyChainDelayPredictor(params=best_params)
    model.fit(X_train, y_train, X_val, y_val)

    print(f"Saving models to {args.model_out}...")
    model.save(args.model_out)

    # Validate model
    preds = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, preds)
    print(f"Validation MAE: {val_mae:.4f}")
    print("Training complete.")

if __name__ == "__main__":
    main()
