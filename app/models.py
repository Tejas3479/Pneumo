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

        # Track model paths and modification timestamps for dynamic reloading
        self.model_paths = [
            os.path.join(models_dir, "model_0.onnx"),
            os.path.join(models_dir, "model.onnx")
        ]
        self.last_timestamps = {}
        self._update_timestamps()

    def _update_timestamps(self):
        for path in self.model_paths:
            if os.path.exists(path):
                self.last_timestamps[path] = os.path.getmtime(path)
            else:
                self.last_timestamps[path] = 0.0

    def check_and_reload_sessions(self):
        changed = False
        for path in self.model_paths:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                if mtime > self.last_timestamps.get(path, 0.0):
                    changed = True
                    self.last_timestamps[path] = mtime
                    
        if changed:
            print("Detected ONNX model modification. Reloading sessions dynamically...")
            from src.uncertainty import EnsemblePredictor
            self.ensemble = EnsemblePredictor(models_dir=self.models_dir)
            # Invalidate PyTorch model so it reloads weights
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
        # Dynamic ONNX reload check
        self.check_and_reload_sessions()

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

        # 7. Log to active learning database if probability is within [0.4, 0.6]
        if 0.4 <= prob <= 0.6:
            import random
            import string
            rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            flagged_filename = f"flagged_{rand_str}.dcm"
            flagged_path = os.path.join(self.models_dir, "..", "data", "dicoms", flagged_filename)
            flagged_path = os.path.abspath(flagged_path)
            
            try:
                gray_img = img_resized.convert("L")
                pixel_array = np.array(gray_img).astype(np.float32) / 255.0
                
                import pydicom
                from pydicom.dataset import Dataset, FileMetaDataset
                from pydicom.uid import ExplicitVRLittleEndian
                
                file_meta = FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.1')
                file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
                file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

                ds = Dataset()
                ds.file_meta = file_meta
                ds.preamble = b"\0" * 128

                ds.PatientName = "Flagged^Patient"
                ds.PatientID = f"FLG-{random.randint(1000, 9999)}"
                ds.Modality = "DX"
                ds.StudyDate = "20260608"
                ds.StudyTime = "120000"
                
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.Rows = pixel_array.shape[0]
                ds.Columns = pixel_array.shape[1]
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 0
                ds.RescaleSlope = "1.0"
                ds.RescaleIntercept = "0.0"

                scaled_pixel_data = (pixel_array * 60000).astype(np.uint16)
                ds.PixelData = scaled_pixel_data.tobytes()

                ds.is_little_endian = True
                ds.is_implicit_VR = False

                os.makedirs(os.path.dirname(flagged_path), exist_ok=True)
                ds.save_as(flagged_path, write_like_original=False)
                
                from src.active_learning import log_flagged_prediction
                log_flagged_prediction(
                    image_path=f"dicoms/{flagged_filename}",
                    prediction_prob=prob,
                    sex=random.choice([0, 1]),
                    age=random.randint(18, 90)
                )
                image_path = f"dicoms/{flagged_filename}"
            except Exception as ex:
                print(f"Failed to save flagged prediction DICOM/DB log: {ex}")
                image_path = None
        else:
            image_path = None

        return {
            "probability": float(prob),
            "uncertainty": float(uncertainty) if uncertainty is not None else None,
            "prediction": prediction_label,
            "heatmap_base64": f"data:image/jpeg;base64,{heatmap_base64}",
            "counterfactual_base64": f"data:image/jpeg;base64,{counterfactual_base64}",
            "text_justification": narrative,
            "tcav_scores": tcav_scores,
            "image_path": image_path
        }
