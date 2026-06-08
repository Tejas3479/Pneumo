import os
import sqlite3
import pandas as pd
from src.active_learning import init_db, log_flagged_prediction, save_clinician_feedback, get_feedback_dataset, DB_PATH

def test_active_learning_db(tmp_path):
    # Temporarily override DB_PATH to use a test database in pytest temp directory
    test_db = os.path.join(tmp_path, "test_active_learning.db")
    import src.active_learning
    original_db_path = src.active_learning.DB_PATH
    src.active_learning.DB_PATH = test_db
    
    try:
        # Initialize
        src.active_learning.init_db()
        assert os.path.exists(test_db)
        
        # Log flagged prediction
        src.active_learning.log_flagged_prediction("dicoms/test_image.dcm", 0.55, sex=1, age=45)
        
        # Verify it exists in db
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT image_path, prediction_prob, prediction_label, status, sex, age FROM feedback_samples")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "dicoms/test_image.dcm"
        assert row[1] == 0.55
        assert row[2] == 1  # 0.55 >= 0.5 -> 1
        assert row[3] == "flagged"
        assert row[4] == 1
        assert row[5] == 45
        
        # Clinician feedback is empty initially
        df_empty = src.active_learning.get_feedback_dataset()
        assert len(df_empty) == 0
        
        # Save clinician feedback
        src.active_learning.save_clinician_feedback("dicoms/test_image.dcm", 0)
        
        # Verify status is updated to corrected
        cursor.execute("SELECT clinician_label, status FROM feedback_samples WHERE image_path = ?", ("dicoms/test_image.dcm",))
        updated_row = cursor.fetchone()
        assert updated_row[0] == 0
        assert updated_row[1] == "corrected"
        
        # Verify get_feedback_dataset loads it
        df = src.active_learning.get_feedback_dataset()
        assert len(df) == 1
        assert df.iloc[0]['ImagePath'] == "dicoms/test_image.dcm"
        assert df.iloc[0]['Label'] == 0
        assert df.iloc[0]['Sex'] == 1
        assert df.iloc[0]['Age'] == 45
        
        # Test saving clinician feedback for a new image that wasn't flagged
        src.active_learning.save_clinician_feedback("dicoms/unflagged_image.dcm", 1)
        df2 = src.active_learning.get_feedback_dataset()
        assert len(df2) == 2
        
        conn.close()
        
    finally:
        # Restore DB_PATH
        src.active_learning.DB_PATH = original_db_path
