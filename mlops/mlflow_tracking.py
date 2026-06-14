import os
import mlflow
from pytorch_lightning.loggers import MLFlowLogger
from pathlib import Path

# Configure MLflow tracking URI:
# 1. Use MLFLOW_TRACKING_URI env var if set (points to Docker MLflow server or remote)
# 2. Fall back to local mlruns/ directory
_env_uri = os.getenv("MLFLOW_TRACKING_URI", "")

if _env_uri:
    tracking_uri = _env_uri
    print(f"[MLflow] Using remote tracking URI: {tracking_uri}")
else:
    MLRUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlruns"))
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    tracking_uri = Path(MLRUNS_DIR).as_uri()
    print(f"[MLflow] Using local tracking directory: {MLRUNS_DIR}")

mlflow.set_tracking_uri(tracking_uri)


def get_mlflow_logger(experiment_name: str = "pneumodetect", run_name: str = None) -> MLFlowLogger:
    """
    Initializes and returns a PyTorch Lightning MLFlowLogger.
    Respects MLFLOW_TRACKING_URI environment variable for remote/Docker MLflow servers.
    """
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri
    )
