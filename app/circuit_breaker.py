import os
import io
import pybreaker
import numpy as np
from PIL import Image

# Global circuit breaker
# fail_max is the number of consecutive failures before opening the circuit
# reset_timeout is the number of seconds to wait before trying to close the circuit again
inference_breaker = pybreaker.CircuitBreaker(
    fail_max=int(os.getenv("CB_FAIL_MAX", 5)),
    reset_timeout=int(os.getenv("CB_RESET_TIMEOUT", 30))
)

def fallback_prediction(image_bytes: bytes) -> dict:
    """
    Generates a graceful, degraded structured response when the primary AI model is offline
    or the circuit breaker is open.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        pixels = np.array(img).astype(np.float32) / 255.0
        mean_val = float(pixels.mean())
    except Exception:
        mean_val = 0.5

    return {
        "probability": 0.5,
        "uncertainty": 1.0,
        "prediction": "UNKNOWN (FALLBACK)",
        "heatmap_base64": "",
        "counterfactual_base64": "",
        "text_justification": (
            "Primary AI model is currently offline (Circuit Breaker open). "
            f"Service is operating in degraded fallback mode. Rule-based pixel mean: {mean_val:.4f}."
        ),
        "tcav_scores": {},
        "image_path": None
    }

def predict_with_breaker(model_manager, image_bytes: bytes) -> dict:
    """
    Executes prediction using the ModelManager under the protection of the circuit breaker.
    Automatically falls back to a rule-based prediction if the circuit is open or throws errors.
    """
    try:
        return inference_breaker.call(model_manager.predict, image_bytes)
    except Exception as e:
        print(f"[Circuit Breaker] Handled model failure/open state: {e}")
        return fallback_prediction(image_bytes)
