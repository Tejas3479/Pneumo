import os
import io
import base64
import numpy as np
import cv2
import torch
from PIL import Image

from app.utils import preprocess_image
from src.xai import generate_clinical_justification, generate_counterfactual_numpy

class ModelManager:
    """
    Hybrid inference manager coordinating ONNX ensembles for high-speed predictions
    and PyTorch on CPU for transformer-based Grad-CAM explainability.
    """
    def __init__(self, models_dir: str = "models", model_type: str = None):
        self.models_dir = models_dir
        self.model_type = model_type or os.getenv("MODEL_TYPE", "vit").lower()
        
        # Load ONNX ensemble sessions (CPU provider)
        from src.uncertainty import EnsemblePredictor
        self.ensemble = EnsemblePredictor(models_dir=models_dir)
        
        # Load TCAV Concept Activation Vector classifiers
        from src.xai import TCAVEngine
        tcav_path = os.path.join(models_dir, "tcav_classifiers.pkl")
        self.tcav_engine = TCAVEngine(classifiers_pkl_path=tcav_path)
        
        # Lazy-loaded PyTorch model placeholder
        self.pytorch_model = None

    def _get_pytorch_model(self):
        """
        Lazy-loads the PyTorch checkpoint on CPU to keep memory footprint low
        until XAI explanations are requested.
        """
        if self.pytorch_model is None:
            print(f"Lazy-loading PyTorch model ({self.model_type}) on CPU for explanation generation...")
            
            if self.model_type == "vit":
                from src.model_foundation import ViTPneumothoraxClassifier
                ckpt_path = os.path.join(self.models_dir, "best_seed_0.ckpt")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(self.models_dir, "best.ckpt")
                    
                if os.path.exists(ckpt_path):
                    self.pytorch_model = ViTPneumothoraxClassifier.load_from_checkpoint(ckpt_path)
                else:
                    self.pytorch_model = ViTPneumothoraxClassifier()
            else:
                from src.model import PneumothoraxClassifier
                ckpt_path = os.path.join(self.models_dir, "best.ckpt")
                if os.path.exists(ckpt_path):
                    self.pytorch_model = PneumothoraxClassifier.load_from_checkpoint(ckpt_path)
                else:
                    self.pytorch_model = PneumothoraxClassifier()
                    
            self.pytorch_model.eval()
            self.pytorch_model.to(torch.device("cpu"))
            
        return self.pytorch_model

    def predict(self, image_bytes: bytes):
        """
        Runs predictions using ONNX ensembles, computes Grad-CAM heatmaps (ONNX-based for ResNet,
        PyTorch-based for ViT), generates inpainting counterfactuals, and maps TCAV scores.
        """
        # Preprocess input image
        batch_img, img_resized = preprocess_image(image_bytes)
        
        # 1. Run ONNX Ensemble inference
        prob, uncertainty, logits, cls_or_fm = self.ensemble.predict_ensemble(batch_img)
        prediction_label = "POSITIVE" if prob > 0.5 else "NEGATIVE"
        
        # 2. Generate model explainability (Grad-CAM & TCAV)
        if self.model_type == "vit":
            # PyTorch Attention Grad-CAM for ViT
            model = self._get_pytorch_model()
            image_tensor = torch.tensor(batch_img, dtype=torch.float32)
            
            from src.xai import ViTAttentionGradCAM
            explainer = ViTAttentionGradCAM(model)
            heatmap, _ = explainer.generate_heatmap(image_tensor)
            
            # Compute TCAV scores using the cls_token output from ONNX
            # cls_or_fm is the CLS token of shape [1, 768]
            tcav_scores = self.tcav_engine.compute_tcav_scores(cls_or_fm[0])
            
        else:
            # Traditional NumPy Grad-CAM for ResNet
            weights_path = os.path.join(self.models_dir, "model_weights.npy")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"FC weights file not found at {weights_path}.")
                
            model_weights = np.load(weights_path, allow_pickle=True).item()
            weight = model_weights["weight"][0]  # Shape: (2048,)
            bias = model_weights["bias"]        # Shape: (1,)
            
            # cls_or_fm is the feature map of shape [1, 2048, 7, 7]
            fm = cls_or_fm[0]
            cam = np.dot(fm.transpose(1, 2, 0), weight) + bias[0]
            cam = np.maximum(cam, 0)
            
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 0:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
                
            heatmap = cv2.resize(cam, (224, 224))
            tcav_scores = {"Pleural Line": 0.0, "Rib Shadow": 0.0, "Mediastinum": 0.0}

        # 3. Generate Clinical AI Text Justification
        narrative = generate_clinical_justification(prob, prediction_label, heatmap)

        # 4. Generate Saliency-based Counterfactual Image
        counterfactual_pil = generate_counterfactual_numpy(img_resized, heatmap)

        # 5. Blend original image with the heatmap
        img_np = np.array(img_resized)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(img_np, 0.5, heatmap_rgb, 0.5, 0)
        
        # 6. Encode outputs to Base64
        blended_pil = Image.fromarray(blended)
        buffer_heat = io.BytesIO()
        blended_pil.save(buffer_heat, format="JPEG")
        heatmap_base64 = base64.b64encode(buffer_heat.getvalue()).decode("utf-8")
        
        buffer_cf = io.BytesIO()
        counterfactual_pil.save(buffer_cf, format="JPEG")
        counterfactual_base64 = base64.b64encode(buffer_cf.getvalue()).decode("utf-8")

        return {
            "probability": float(prob),
            "uncertainty": float(uncertainty) if uncertainty is not None else None,
            "prediction": prediction_label,
            "heatmap_base64": f"data:image/jpeg;base64,{heatmap_base64}",
            "counterfactual_base64": f"data:image/jpeg;base64,{counterfactual_base64}",
            "text_justification": narrative,
            "tcav_scores": tcav_scores
        }
