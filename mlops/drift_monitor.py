import os
import json
import sqlite3
import datetime
import requests
import numpy as np

BASELINE_STATS_PATH = "models/baseline_stats.json"

def compute_baseline_stats(csv_path: str = 'data/train.csv'):
    """
    Computes pixel mean and standard deviation distributions from the training dataset.
    Saves results to models/baseline_stats.json.
    """
    print(f"Computing training baseline statistics from {csv_path}...")
    if not os.path.exists(csv_path):
        print("Training CSV not found. Returning empty baseline.")
        return {"means": [], "stds": []}

    import pandas as pd
    df = pd.read_csv(csv_path)
    means = []
    stds = []
    
    # Process up to 500 images for baseline speed/efficiency
    sample_df = df.sample(min(len(df), 500), random_state=42)
    
    for _, row in sample_df.iterrows():
        path = os.path.join(os.path.dirname(csv_path), row['ImagePath'])
        if os.path.exists(path):
            try:
                import pydicom
                ds = pydicom.dcmread(path)
                pixels = ds.pixel_array.astype(np.float32)
                min_val, max_val = pixels.min(), pixels.max()
                if max_val - min_val > 0:
                    pixels = (pixels - min_val) / (max_val - min_val)
                else:
                    pixels = np.zeros_like(pixels)
                means.append(float(pixels.mean()))
                stds.append(float(pixels.std()))
            except Exception:
                try:
                    from PIL import Image
                    img = Image.open(path).convert('L')
                    pixels = np.array(img).astype(np.float32) / 255.0
                    means.append(float(pixels.mean()))
                    stds.append(float(pixels.std()))
                except Exception:
                    pass
                    
    baseline_stats = {
        "means": means,
        "stds": stds
    }
    
    os.makedirs(os.path.dirname(BASELINE_STATS_PATH), exist_ok=True)
    with open(BASELINE_STATS_PATH, "w") as f:
        json.dump(baseline_stats, f, indent=4)
        
    print(f"Saved baseline statistics to {BASELINE_STATS_PATH}")
    return baseline_stats

def compute_psi(expected, actual, bins: int = 10) -> float:
    """
    Calculates the Population Stability Index (PSI) between expected and actual distributions.
    """
    expected = np.array(expected)
    actual = np.array(actual)
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Bin boundaries from 0.0 to 1.0 (since images are normalized in [0, 1])
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Convert to fractions
    expected_pcts = expected_counts / len(expected)
    actual_pcts = actual_counts / len(actual)
    
    # Apply Laplace smoothing to avoid log(0) or division by zero
    eps = 1e-4
    expected_pcts = np.where(expected_pcts == 0, eps, expected_pcts)
    actual_pcts = np.where(actual_pcts == 0, eps, actual_pcts)
    
    # Re-normalize to sum to 1.0
    expected_pcts /= expected_pcts.sum()
    actual_pcts /= actual_pcts.sum()
    
    # Compute PSI formula
    psi_val = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return float(psi_val)

def check_data_drift():
    """
    Loads baseline training statistics, queries the last 24 hours of predictions,
    computes PSI drift values, and fires webhook alerts if drift exceeds 0.25.
    """
    if os.path.exists(BASELINE_STATS_PATH):
        with open(BASELINE_STATS_PATH, "r") as f:
            baseline = json.load(f)
    else:
        baseline = compute_baseline_stats()
        
    baseline_means = baseline.get("means", [])
    baseline_stds = baseline.get("stds", [])
    
    # Fetch actual predictions from the last 24 hours
    actual_means = []
    actual_stds = []
    
    db_path = "data/audit_ledger.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Calculate time 24 hours ago in ISO format
        twenty_four_hours_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat()
        
        try:
            cursor.execute("SELECT mean_pixel, std_pixel FROM audit_ledger WHERE timestamp >= ?", (twenty_four_hours_ago,))
            rows = cursor.fetchall()
            for r in rows:
                actual_means.append(r[0])
                actual_stds.append(r[1])
        except Exception as e:
            print(f"Error reading audit ledger: {e}")
        finally:
            conn.close()
            
    # Compute PSI
    psi_mean = compute_psi(baseline_means, actual_means)
    psi_std = compute_psi(baseline_stds, actual_stds)
    
    # Check drift threshold (significant drift is > 0.25)
    drift_detected = (psi_mean > 0.25) or (psi_std > 0.25)
    
    webhook_url = os.getenv("DRIFT_WEBHOOK_URL")
    if drift_detected and webhook_url:
        payload = {
            "alert": "CRITICAL_DATA_DRIFT_DETECTED",
            "psi_mean_pixel": psi_mean,
            "psi_std_pixel": psi_std,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        try:
            requests.post(webhook_url, json=payload, timeout=5)
            print("Drift webhook alert posted successfully.")
        except Exception as ex:
            print(f"Failed to post drift webhook alert: {ex}")
            
    return {
        "psi_mean": psi_mean,
        "psi_std": psi_std,
        "drift_detected": drift_detected,
        "actual_samples_count": len(actual_means)
    }

if __name__ == "__main__":
    # If run directly, compute baseline statistics
    compute_baseline_stats()
