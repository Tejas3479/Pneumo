import os
import tempfile
import pickle
import numpy as np
import torch
import pytest
from PIL import Image

from src.model_foundation import ViTPneumothoraxClassifier
from src.xai import ViTAttentionGradCAM, generate_counterfactual_numpy, TCAVEngine

def test_vit_attention_grad_cam():
    """
    Tests that running the ViTAttentionGradCAM generate_heatmap method registers
    the self-attention hooks, runs a backward pass, and outputs a normalized
    (224, 224) numpy heatmap with min=0.0 and max=1.0.
    """
    model = ViTPneumothoraxClassifier()
    model.eval()

    explainer = ViTAttentionGradCAM(model)
    x = torch.randn(1, 3, 224, 224)

    # Generate the heatmap
    heatmap, prob = explainer.generate_heatmap(x)

    # Assert shape and normalization properties
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    # Allow small precision tolerances
    assert heatmap.max() <= 1.0 + 1e-6
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

def test_inpainting_counterfactual():
    """
    Tests that the inpainting counterfactual generator successfully alters
    pixels inside highly activated regions of a mock image (producing an MSE > 0).
    """
    # Create mock 224x224 RGB image (Pillow)
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    img_np = np.array(img)
    
    # Create mock activation heatmap (high values in the center)
    heatmap = np.zeros((224, 224), dtype=np.float32)
    y, x = np.ogrid[:224, :224]
    mask = (x - 112)**2 + (y - 112)**2 < 40**2
    heatmap[mask] = 0.9  # Saliency value above 0.55 threshold
    
    # Modify center pixels in original image so inpainting has something to replace
    img_np[mask] = [255, 255, 255]
    img_modified = Image.fromarray(img_np)

    # Run inpainting counterfactual
    counterfactual = generate_counterfactual_numpy(img_modified, heatmap)
    cf_np = np.array(counterfactual)

    # Assert inpainting replaced the bright white spot with surrounding grey, altering pixels
    mse = np.mean((img_np - cf_np) ** 2)
    assert mse > 0.0
    assert counterfactual.size == (224, 224)

def test_tcav_engine_scores():
    """
    Tests that the TCAVEngine correctly deserializes classifiers and evaluates
    concept scores from mock CLS representations.
    """
    # Create temporary pkl file containing mock CAV classifiers
    with tempfile.TemporaryDirectory() as tmp_dir:
        pkl_path = os.path.join(tmp_dir, "tcav_classifiers.pkl")
        
        # Mock CAV classifiers coefficients (shape 768)
        mock_weights = {
            "Pleural Line": {
                "weight": [0.1] * 768,
                "intercept": 0.5
            },
            "Rib Shadow": {
                "weight": [-0.2] * 768,
                "intercept": -0.2
            }
        }
        
        with open(pkl_path, "wb") as f:
            pickle.dump(mock_weights, f)
            
        # Instantiate engine using temporary pkl
        engine = TCAVEngine(classifiers_pkl_path=pkl_path)
        
        # Create dummy cls_token (shape 768)
        cls_token = np.ones(768)
        
        scores = engine.compute_tcav_scores(cls_token)
        
        # Verify keys and range
        assert "Pleural Line" in scores
        assert "Rib Shadow" in scores
        assert 0.0 <= scores["Pleural Line"] <= 1.0
        assert 0.0 <= scores["Rib Shadow"] <= 1.0
        
        # Expected Pleural Line score calculation:
        # logit = sum(1.0 * 0.1) + 0.5 = 76.8 + 0.5 = 77.3
        # sigmoid(77.3) should be very close to 1.0
        assert scores["Pleural Line"] > 0.99
