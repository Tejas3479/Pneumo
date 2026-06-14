import os
import sqlite3

# Enable Write-Ahead Logging (WAL) and busy timeout on sqlite3 connections
if sqlite3.connect.__name__ != "sqlite3_connect_wal":
    _original_sqlite3_connect = sqlite3.connect
    def sqlite3_connect_wal(database, timeout=30.0, *args, **kwargs):
        conn = _original_sqlite3_connect(database, timeout=timeout, *args, **kwargs)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        return conn
    sqlite3.connect = sqlite3_connect_wal
import datetime
import hashlib
import json
import hmac
import numpy as np

DB_PATH = "data/audit_ledger.db"

def get_ledger_secret_key() -> bytes:
    key = os.getenv("LEDGER_SECRET_KEY", "default-pneumodetect-secret-key")
    if key == "default-pneumodetect-secret-key":
        import warnings
        warnings.warn(
            "SECURITY WARNING: LEDGER_SECRET_KEY is using the insecure default value. "
            "Set LEDGER_SECRET_KEY environment variable to a long random string before production deployment.",
            UserWarning, stacklevel=1
        )
    return key.encode("utf-8")

def compute_row_hash(content_str: str) -> str:
    key = get_ledger_secret_key()
    return hmac.new(key, content_str.encode("utf-8"), hashlib.sha256).hexdigest()

def init_audit_db():
    """
    Initializes the SQLite database for the tamper-evident prediction audit ledger.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_hash TEXT NOT NULL,
            probability REAL NOT NULL,
            prediction TEXT NOT NULL,
            mean_pixel REAL NOT NULL,
            std_pixel REAL NOT NULL,
            previous_hash TEXT,
            row_hash TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now')),
            psi_mean REAL NOT NULL,
            psi_std REAL NOT NULL,
            alert_flag INTEGER DEFAULT 0,
            samples_count INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction_audit(image_bytes: bytes, probability: float, prediction: str, mean_pixel: float, std_pixel: float) -> str:
    """
    Logs a prediction run to the audit ledger database with row-hash chaining.
    """
    init_audit_db()
    
    timestamp = datetime.datetime.utcnow().isoformat()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    
    # Retrieve the row_hash of the last inserted row
    cursor.execute("SELECT row_hash FROM audit_ledger ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    previous_hash = row[0] if row else "0"
    
    # Pre-determine the next auto-increment id for hash consistency
    cursor.execute("SELECT MAX(id) FROM audit_ledger")
    max_id_row = cursor.fetchone()
    next_id = (max_id_row[0] + 1) if (max_id_row and max_id_row[0] is not None) else 1
    
    # Construct block string and compute chained hash
    content_str = f"{next_id}|{timestamp}|{image_hash}|{probability:.6f}|{prediction}|{mean_pixel:.6f}|{std_pixel:.6f}|{previous_hash}"
    row_hash = compute_row_hash(content_str)
    
    # Insert ledger record
    cursor.execute("""
        INSERT INTO audit_ledger (timestamp, image_hash, probability, prediction, mean_pixel, std_pixel, previous_hash, row_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, image_hash, probability, prediction, mean_pixel, std_pixel, previous_hash, row_hash))
    
    conn.commit()
    conn.close()
    return row_hash

def verify_audit_trail():
    """
    Scans the entire database, validates row hashes, and detects any data tampering.
    Returns (isValid, list of mismatched row IDs).
    """
    init_audit_db()
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, image_hash, probability, prediction, mean_pixel, std_pixel, previous_hash, row_hash FROM audit_ledger ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    
    mismatches = []
    expected_previous_hash = "0"
    
    for row in rows:
        row_id, timestamp, image_hash, probability, prediction, mean_pixel, std_pixel, previous_hash, row_hash = row
        
        # Verify hash link chain
        if previous_hash != expected_previous_hash:
            mismatches.append(row_id)
            # Recover chain to continue scanning
            expected_previous_hash = row_hash
            continue
            
        # Recompute row hash
        content_str = f"{row_id}|{timestamp}|{image_hash}|{probability:.6f}|{prediction}|{mean_pixel:.6f}|{std_pixel:.6f}|{previous_hash}"
        computed_hash = compute_row_hash(content_str)
        
        if computed_hash != row_hash:
            mismatches.append(row_id)
            
        expected_previous_hash = row_hash
        
    return len(mismatches) == 0, mismatches

def generate_model_card() -> str:
    """
    Generates a Google-standard compliant model_card.md using model metrics.
    """
    metrics_path = "models/metrics.json"
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    # Load data sizes if available
    train_size = "N/A"
    val_size = "N/A"
    if os.path.exists("data/train.csv"):
        try:
            import pandas as pd
            df = pd.read_csv("data/train.csv")
            train_size = str(len(df))
            if os.path.exists("data/val_indices.npy"):
                val_indices = np.load("data/val_indices.npy")
                val_size = str(len(val_indices))
                train_size = str(len(df) - len(val_indices))
        except Exception:
            pass
            
    model_type = metrics.get("model_type", "vit").upper()
    val_auroc = metrics.get("val_auroc", metrics.get("val_auroc_epoch", 0.0))
    val_loss = metrics.get("val_loss", metrics.get("val_loss_epoch", 0.0))
    
    # Read fairness metrics from metrics json
    dp_diff = metrics.get("val_demographic_parity_diff", metrics.get("val_demographic_parity_diff_epoch", "N/A"))
    eo_diff = metrics.get("val_equal_opportunity_diff", metrics.get("val_equal_opportunity_diff_epoch", "N/A"))
    
    # Convert numerical values to strings
    dp_diff_str = f"{dp_diff:.4f}" if isinstance(dp_diff, float) else str(dp_diff)
    eo_diff_str = f"{eo_diff:.4f}" if isinstance(eo_diff, float) else str(eo_diff)
    
    now_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    card_content = f"""# Model Card — PneumoDetect AI

This Model Card details the operational parameters, intended clinical use cases, demographic fairness metrics, and training specifications of the PneumoDetect AI system.

## Model Details
- **Developer:** PneumoDetect AI Team
- **Model Date:** {now_str}
- **Model Type:** Vision Transformer (ViT-B/16) with LoRA Adapters / ResNet-50 Ensemble
- **Version:** 1.0.0
- **License:** Research-only medical evaluation license.
- **Reference:** Google Model Card Toolkit standard schema.

## Intended Use
- **Primary Intended Uses:** Assist clinical radiologists in rapid secondary screening of chest radiographs for signs of Pneumothorax (collapsed lung).
- **Primary Intended Users:** Licensed radiologists, emergency clinicians, and pulmonology specialists.
- **Out-of-Scope Use Cases:** Diagnosis of non-radiological images, multi-abnormality chest pathology screening (without secondary verification), or autonomous diagnostic decision-making without clinician supervision.

## Factors
- **Demographic Factors:** Biological Sex (Male/Female), Patient Age.
- **Clinical/Imaging Factors:** Modality (Digital Radiography DX / Computed Radiography CR), scanner resolution, chest orientation (PA or AP).

## Metrics
- **Performance Evaluation Metric:** Area Under the Receiver Operating Characteristic curve (AUROC), Validation Binary Cross-Entropy Loss.
- **Fairness Metrics:** Demographic Parity Difference (DPD), Equal Opportunity Difference (EOD) evaluated across Patient Sex.

## Training Data
- **Dataset Source:** Synthetic clinical-grade Chest X-Ray database.
- **Training Set Size:** {train_size} images.
- **Target Label Balance:** Approximately balanced positive (pneumothorax) and negative cases.

## Evaluation Data
- **Evaluation Set Size:** {val_size} images.
- **Validation Split Strategy:** Reproducible index splits stored in validation indices cache.

## Quantitative Analyses
| Metric | Value | Target Threshold | Status |
|---|---|---|---|
| Validation AUROC | {val_auroc:.4f} | >= 0.8500 | {"PASS" if val_auroc >= 0.85 else "EVAL"} |
| Validation Loss | {val_loss:.4f} | < 0.5000 | {"PASS" if val_loss < 0.5 else "EVAL"} |
| Demographic Parity Difference | {dp_diff_str} | < 0.1000 | {"PASS" if (isinstance(dp_diff, float) and dp_diff < 0.1) else "WARNING"} |
| Equal Opportunity Difference | {eo_diff_str} | < 0.1000 | {"PASS" if (isinstance(eo_diff, float) and eo_diff < 0.1) else "WARNING"} |

## Ethical Considerations & Limitations
- The model is not approved for autonomous diagnosis. It must be utilized strictly as a diagnostic aid.
- High demographic differences indicate potential biases in clinical subset distributions; the model should be retrained using debiased adversarial heads or updated training weights if parity metrics exceed 0.1.

---
Generated automatically by PneumoDetect Regulatory Pipeline on {now_str}.
"""
    
    card_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_card.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content)
        
    print(f"Generated Model Card at {card_path}")
    return card_path

if __name__ == "__main__":
    # If run directly, generate model card
    generate_model_card()
