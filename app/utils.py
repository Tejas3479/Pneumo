import os
import io
import pydicom
from PIL import Image
import numpy as np

def preprocess_image(image_bytes: bytes, model_type: str = None, medfound_model: str = None):
    """
    Decodes image bytes (DICOM or standard formats like PNG/JPEG),
    normalizes the pixel array, and formats it as a batch tensor [1, 3, 224, 224] for model evaluation.
    Also returns the resized RGB PIL image for Grad-CAM blending.
    Uses MODEL_TYPE env var as fallback to set mean/std values.
    """
    # Read MODEL_TYPE environment variable if not explicitly passed
    if model_type is None:
        model_type = os.getenv("MODEL_TYPE", "vit").lower()
    else:
        model_type = model_type.lower()
        
    if medfound_model is None:
        medfound_model = os.getenv("MEDFOUND_MODEL", "microsoft/Biovil-T").lower()
    else:
        medfound_model = medfound_model.lower()
    
    # 1. Attempt decoding as DICOM
    try:
        f = io.BytesIO(image_bytes)
        ds = pydicom.dcmread(f, force=True)
        # Attempt to access pixel data to check validity
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Apply rescale intercept/slope if they exist
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope) if ds.RescaleSlope is not None else 1.0
            intercept = float(ds.RescaleIntercept) if ds.RescaleIntercept is not None else 0.0
            pixel_array = pixel_array * slope + intercept
            
        # Normalize to [0, 1] range
        min_val = pixel_array.min()
        max_val = pixel_array.max()
        if max_val - min_val > 0:
            pixel_array = (pixel_array - min_val) / (max_val - min_val)
        else:
            pixel_array = np.zeros_like(pixel_array)
            
        # Convert monochrome uint8 to PIL RGB
        pixel_array_uint8 = (pixel_array * 255.0).astype(np.uint8)
        img = Image.fromarray(pixel_array_uint8).convert("RGB")
    except Exception as e:
        # 2. Fallback to standard formats (PNG/JPEG)
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e_inner:
            raise ValueError(
                f"Failed to parse input bytes as DICOM or standard image format. "
                f"DICOM error: {e}. PIL error: {e_inner}"
            )

    # 3. Resize to 224x224
    img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
    
    # 4. Standard normalization based on model type
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    
    if model_type == "vit":
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    elif model_type == "medfound":
        if "chexzero" in medfound_model or "clip" in medfound_model:
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        else:
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    else:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
    img_normalized = (img_np - mean) / std
    
    # 5. Transform layout from HWC to CHW and add batch dimension: [1, 3, 224, 224]
    img_chw = img_normalized.transpose(2, 0, 1)
    batch_img = np.expand_dims(img_chw, axis=0)
    
    return batch_img, img_resized
