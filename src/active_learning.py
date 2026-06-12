import os
import sqlite3

# Enable Write-Ahead Logging (WAL) and busy timeout on sqlite3 connections
_original_sqlite3_connect = sqlite3.connect
def sqlite3_connect_wal(database, timeout=30.0, *args, **kwargs):
    conn = _original_sqlite3_connect(database, timeout=timeout, *args, **kwargs)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    return conn
sqlite3.connect = sqlite3_connect_wal
import pandas as pd

DB_PATH = os.path.join("data", "active_learning.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE,
            prediction_prob REAL,
            prediction_label INTEGER,
            clinician_label INTEGER,
            sex INTEGER,
            age INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'flagged'
        )
    """)
    conn.commit()
    conn.close()

def log_flagged_prediction(image_path: str, prediction_prob: float, sex: int = None, age: int = None):
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    prediction_label = 1 if prediction_prob >= 0.5 else 0
    # Ensure sex and age are converted to appropriate types or None
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO feedback_samples (image_path, prediction_prob, prediction_label, sex, age, status)
            VALUES (?, ?, ?, ?, ?, 'flagged')
        """, (image_path, prediction_prob, prediction_label, sex, age))
        conn.commit()
    except Exception as e:
        print(f"Error logging flagged prediction: {e}")
    finally:
        conn.close()

def save_clinician_feedback(image_path: str, clinician_label: int):
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM feedback_samples WHERE image_path = ?", (image_path,))
        row = cursor.fetchone()
        if row:
            cursor.execute("""
                UPDATE feedback_samples
                SET clinician_label = ?, status = 'corrected'
                WHERE image_path = ?
            """, (clinician_label, image_path))
        else:
            # If not flagged previously, insert fresh corrected sample
            cursor.execute("""
                INSERT INTO feedback_samples (image_path, clinician_label, status)
                VALUES (?, ?, 'corrected')
            """, (image_path, clinician_label))
        conn.commit()
    except Exception as e:
        print(f"Error saving clinician feedback: {e}")
    finally:
        conn.close()

def get_feedback_dataset():
    """
    Query all corrected feedback samples. Returns a pandas DataFrame
    formatted to match train.csv (with columns ImagePath, Label, Sex, Age).
    """
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    df = pd.read_sql_query("""
        SELECT image_path AS ImagePath, clinician_label AS Label, sex AS Sex, age AS Age FROM feedback_samples
        WHERE status = 'corrected'
    """, conn)
    conn.close()
    
    # Fill NaN values for Sex and Age if they were not stored
    if not df.empty:
        df['Sex'] = df['Sex'].fillna(0).astype(int)
        df['Age'] = df['Age'].fillna(40).astype(int)
    return df
