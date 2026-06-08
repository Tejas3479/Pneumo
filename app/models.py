import os
import io
import base64
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

from app.utils import preprocess_image

class ModelManager:
    """
    Singleton-style manager to load and run ONNX inference and generate Grad-CAM heatmaps.
    """
    def __init__(self, model_path: str = "models/model.onnx", weights_path: str = "models/model_weights.npy"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at {model_path}. Train and export the model first.")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"FC weights file not found at {weights_path}. Train and export the model first.")
        
        # Load ONNX Inference Session (runs on CPU)
        # Using CPU provider explicitly for portability
        self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Load fully connected weights (bias and weight matrices)
        self.model_weights = np.load(weights_path, allow_pickle=True).item()

    def predict(self, image_bytes: bytes):
        """
        Runs the preprocessed image through the ONNX model, computes the binary classification,
        calculates the Grad-CAM activation map, overlays it on the image, and returns the result.
        """
        # Preprocess input image
        batch_img, img_resized = preprocess_image(image_bytes)
        
        # Run model session
        input_name = self.sess.get_inputs()[0].name
        logits, feature_map = self.sess.run(None, {input_name: batch_img})
        
        # Calculate positive class probability (sigmoid activation)
        prob = 1.0 / (1.0 + np.exp(-logits[0, 0]))
        prediction_label = "POSITIVE" if prob > 0.5 else "NEGATIVE"
        
        # --- Grad-CAM Calculation ---
        # Weight shape is (1, 2048). We use the first class weight vector (positive class).
        # Bias shape is (1,)
        weight = self.model_weights["weight"][0]  # Shape: (2048,)
        bias = self.model_weights["bias"]        # Shape: (1,)
        
        # Feature map shape: (1, 2048, 7, 7). Extract single batch sample: (2048, 7, 7)
        fm = feature_map[0]
        
        # Transpose feature map to (7, 7, 2048) to match the channel axis for dot product
        fm_transposed = fm.transpose(1, 2, 0)
        
        # Compute CAM: dot product of (7, 7, 2048) and (2048,) weights, plus class bias
        cam = np.dot(fm_transposed, weight) + bias[0]
        
        # Apply ReLU activation (discard negative features)
        cam = np.maximum(cam, 0)
        
        # Normalize between [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        # Resize activation map from 7x7 back to original input image size (224x224)
        cam_resized = cv2.resize(cam, (224, 224))
        
        # Scale to [0, 255] and apply JET colormap
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        # OpenCV operates in BGR, convert colormap to RGB
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend original resized image and heatmap with equal weight (0.5 opacity each)
        img_np = np.array(img_resized)
        blended_image = cv2.addWeighted(img_np, 0.5, heatmap_rgb, 0.5, 0)
        
        # Encode the blended image to JPEG and convert to base64 string for API response
        blended_pil = Image.fromarray(blended_image)
        buffer = io.BytesIO()
        blended_pil.save(buffer, format="JPEG")
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "probability": float(prob),
            "prediction": prediction_label,
            "heatmap_base64": f"data:image/jpeg;base64,{heatmap_base64}"
        }
