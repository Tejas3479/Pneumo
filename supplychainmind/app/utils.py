import pandas as pd
from fastapi import UploadFile
import io

def parse_uploaded_csv(file: UploadFile) -> pd.DataFrame:
    """
    Reads the uploaded CSV file and returns it as a Pandas DataFrame.
    """
    content = file.file.read()
    # If the file object was read, we should reset the file pointer or just wrap it in StringIO/BytesIO
    df = pd.read_csv(io.BytesIO(content))
    return df
