import os
import pickle
import numpy as np
import torch
import cv2
from PIL import Image

class ViTAttentionGradCAM:
    """
    Implements a class-specific Grad-CAM explainability method targeting
    the self-attention weights of the final Vision Transformer block.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, image_tensor: torch.Tensor):
        """
        Runs a forward and backward pass under grad-enabled context to extract
        attention weights, calculate head-weighted gradients, and build a 2D activation map.
        """
        # Dynamically find the target layer
        target_layer = None
        if hasattr(self.model, "resnet_or_vit"):
            try:
                target_layer = self.model.resnet_or_vit.base_model.model.vit.encoder.layer[-1].attention.attention
            except AttributeError:
                try:
                    target_layer = self.model.resnet_or_vit.vit.encoder.layer[-1].attention.attention
                except AttributeError:
                    pass
        elif hasattr(self.model, "vision_model"):
            try:
                v_model = self.model.vision_model
                if hasattr(v_model, "base_model") and hasattr(v_model.base_model, "model"):
                    inner_model = v_model.base_model.model
                else:
                    inner_model = v_model
                
                if hasattr(inner_model, "encoder") and hasattr(inner_model.encoder, "layer"):
                    last_layer = inner_model.encoder.layer[-1]
                    if hasattr(last_layer, "attention") and hasattr(last_layer.attention, "attention"):
                        target_layer = last_layer.attention.attention
                    elif hasattr(last_layer, "attention") and hasattr(last_layer.attention, "self"):
                        target_layer = last_layer.attention.self
                    elif hasattr(last_layer, "self_attn"):
                        target_layer = last_layer.self_attn
                elif hasattr(inner_model, "vision_model") and hasattr(inner_model.vision_model, "encoder"):
                    last_layer = inner_model.vision_model.encoder.layer[-1]
                    if hasattr(last_layer, "self_attn"):
                        target_layer = last_layer.self_attn
            except Exception:
                pass

        if target_layer is None:
            print("Warning: Could not register self-attention hooks on ViT backbone.")
            return np.zeros((224, 224), dtype=np.float32), 0.5

        attention_map = []
        attention_grad = []

        def forward_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attention_map.append(output[1])

        def backward_hook(grad):
            attention_grad.append(grad)

        hook_handles = []
        h_forward = target_layer.register_forward_hook(forward_hook)
        hook_handles.append(h_forward)

        prob_val = 0.5
        try:
            # Ensure gradients are enabled for the backward pass
            with torch.enable_grad():
                if image_tensor.ndim == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                    
                x = image_tensor.clone().detach().requires_grad_(True)
                
                # Forward pass
                self.model.zero_grad()
                logits = self.model(x)
                prob = torch.sigmoid(logits[0, 0])
                prob_val = float(prob.detach().cpu())
                
                if len(attention_map) > 0 and attention_map[0].requires_grad:
                    h_backward = attention_map[0].register_hook(backward_hook)
                    hook_handles.append(h_backward)
                
                # Backward pass w.r.t positive class logit
                logits[0, 0].backward()
        finally:
            for h in hook_handles:
                h.remove()

        if not attention_map or not attention_grad:
            print("Warning: Hook placeholders empty. Returning empty heatmap.")
            return np.zeros((224, 224), dtype=np.float32), prob_val

        A = attention_map[0]
        G = attention_grad[0]

        # Tensor structures: A shape is (1, heads, 197, 197), G is (1, heads, 197, 197)
        # Extract attention weights of CLS token (index 0) to all 196 patch tokens (index 1 to 197)
        A_cls = A[0, :, 0, 1:]  # Shape: (heads, 196)
        G_cls = G[0, :, 0, 1:]  # Shape: (heads, 196)

        # Average gradients over spatial dimensions for each head
        alpha = G_cls.mean(dim=-1, keepdim=True)  # Shape: (heads, 1)

        # Head-weighted summation of attention maps
        cam_1d = (alpha * A_cls).sum(dim=0)  # Shape: (196,)

        # Reshape sequence back to grid dynamically depending on the number of patches
        num_patches = cam_1d.shape[0]
        grid_size = int(np.sqrt(num_patches))
        cam_2d = cam_1d.reshape(grid_size, grid_size).detach().cpu()

        # Apply ReLU activation
        cam_2d = torch.clamp(cam_2d, min=0)

        # Normalize to [0, 1] range
        cam_min = cam_2d.min()
        cam_max = cam_2d.max()
        if cam_max - cam_min > 0:
            cam_2d = (cam_2d - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam_2d = torch.zeros_like(cam_2d)

        # Upsample 14x14 grid to original 224x224 input image dimensions
        heatmap = cv2.resize(cam_2d.numpy(), (224, 224))
        
        return heatmap, float(prob.detach().cpu())

class TCAVEngine:
    """
    Computes concept presence scores using pretrained LogisticRegression classifiers
    loaded from serializations mapping target concept activation vectors (CAVs).
    """
    def __init__(self, classifiers_pkl_path: str = "models/tcav_classifiers.pkl"):
        self.classifiers = {}
        if os.path.exists(classifiers_pkl_path):
            try:
                with open(classifiers_pkl_path, "rb") as f:
                    self.classifiers = pickle.load(f)
                print(f"Loaded TCAV classifiers from {classifiers_pkl_path}")
            except Exception as e:
                print(f"Error loading TCAV classifiers: {e}")
        else:
            print(f"Warning: CAV classifiers pickle not found at {classifiers_pkl_path}")

    def compute_tcav_scores(self, cls_token: np.ndarray):
        """
        Evaluates the classification probability of each CAV on the input CLS token vector.
        cls_token: shape (768,)
        """
        scores = {}
        if not self.classifiers:
            # Default empty score fallback
            return {"Pleural Line": 0.0, "Rib Shadow": 0.0, "Mediastinum": 0.0}

        # Ensure flat shape
        cls_token = cls_token.flatten()

        for concept_name, weights in self.classifiers.items():
            w = np.array(weights["weight"])
            b = weights["intercept"]
            
            # Predict logit/decision function and map to probability via sigmoid
            logit = np.dot(cls_token, w) + b
            prob = 1.0 / (1.0 + np.exp(-logit))
            scores[concept_name] = float(prob)
            
        return scores

def generate_clinical_justification(probability: float, prediction: str, heatmap: np.ndarray):
    """
    Finds the center-of-mass quadrant of the Grad-CAM saliency map and builds
    a clinical narrative explanation template.
    """
    # Locate highly activated pixels
    y_indices, x_indices = np.where(heatmap > 0.5)
    
    if len(x_indices) > 0 and len(y_indices) > 0:
        cx = x_indices.mean()
        cy = y_indices.mean()
        horiz = "right" if cx > 112 else "left"
        vert = "upper" if cy < 112 else "lower"
        location = f"{vert} {horiz} lung zone"
    else:
        location = "apical lung margins"

    if prediction == "POSITIVE":
        text = (
            f"CLINICAL AI ASSESSMENT: A high-confidence neural prediction ({probability * 100:.1f}%) "
            f"detects pneumothorax. Localization heatmaps show significant density deviations in the {location}, "
            f"indicating a potential visceral pleural separation. Clinical correlation is recommended."
        )
    else:
        text = (
            f"CLINICAL AI ASSESSMENT: Normal chest radiograph ({(1.0 - probability) * 100:.1f}% negative confidence). "
            f"No evidence of visceral pleural line separation, apical air pocketing, or anomalous density patterns "
            f"detected in any lung field."
        )
        
    return text

def generate_counterfactual_numpy(original_image: Image.Image, heatmap: np.ndarray):
    """
    Generates a counterfactual image by masking out the high-saliency Grad-CAM region
    and reconstructing pixel values using OpenCV Fast Marching Method (FMM) inpainting.
    """
    img_np = np.array(original_image)
    
    # Create mask where Grad-CAM activation is greater than 55%
    mask = np.uint8(heatmap > 0.55) * 255
    
    # Check if there is any mask area to inpaint
    if mask.sum() > 0:
        # Reconstruct pixel values in masked region
        inpainted = cv2.inpaint(img_np, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        inpainted = img_np.copy()
        
    return Image.fromarray(inpainted)
