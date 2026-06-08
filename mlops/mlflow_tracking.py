import os
import mlflow
from pytorch_lightning.loggers import MLFlowLogger

from pathlib import Path

# Configure MLflow tracking URI to write to local mlruns directory
MLRUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlruns"))
tracking_uri = Path(MLRUNS_DIR).as_uri()
mlflow.set_tracking_uri(tracking_uri)

def get_mlflow_logger(experiment_name: str = "pneumodetect", run_name: str = None) -> MLFlowLogger:
    """
    Initializes and returns a PyTorch Lightning MLFlowLogger pointing to the local tracking directory.
    """
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri
    )
