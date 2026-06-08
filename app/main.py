import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.models import ModelManager

app = FastAPI(
    title="PneumoDetect AI",
    description="Medical-grade Chest X-ray Pneumothorax detection served via FastAPI and ONNX.",
    version="1.0.0"
)

# Enable CORS for cross-origin client interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager initialized lazily
_model_manager = None

def get_model_manager():
    """
    Lazy loader for ModelManager to prevent API server startup failures
    if model training and export have not been executed yet.
    """
    global _model_manager
    if _model_manager is None:
        model_path = os.path.join("models", "model.onnx")
        weights_path = os.path.join("models", "model_weights.npy")
        try:
            _model_manager = ModelManager(model_path=model_path, weights_path=weights_path)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model files not available. Please run training and ONNX export. Error: {str(e)}"
            )
    return _model_manager

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serves the single-page dashboard HTML from app/static/index.html.
    """
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(
            status_code=404, 
            detail="Frontend index.html file not found in app/static/."
        )
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded chest X-ray image (DICOM, PNG, or JPEG),
    runs model prediction with Grad-CAM overlays, and returns results.
    """
    manager = get_model_manager()
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        result = manager.predict(image_bytes)
        return result
    except HTTPException as he:
        raise he
    except ValueError as ve:
        # Invalid input files/extensions
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Internal model/runtime issues
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
