import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import pytest

from src.uncertainty import EnsemblePredictor

class TinyModel(nn.Module):
    """A tiny module to compile rapidly for ONNX serving unit tests."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 224 * 224, 1)
        self.fc_features = nn.Linear(3 * 224 * 224, 768)

    def forward(self, x):
        # Flatten input [batch, 3, 224, 224] -> [batch, 3*224*224]
        flat = x.view(x.size(0), -1)
        logits = self.linear(flat)
        cls_token = self.fc_features(flat)
        return logits, cls_token

def test_ensemble_predictor_calculations():
    # Setup temporary directory to save mock ONNX models
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a tiny model and export to ONNX
        model = TinyModel()
        model.eval()
        
        example_input = torch.randn(1, 3, 224, 224)
        
        # We will write two identical mock sessions under different filenames
        onnx_path_0 = os.path.join(tmp_dir, "model_0.onnx")
        onnx_path_1 = os.path.join(tmp_dir, "model_1.onnx")
        
        for path in [onnx_path_0, onnx_path_1]:
            torch.onnx.export(
                model,
                example_input,
                path,
                export_params=True,
                opset_version=15,
                input_names=["input"],
                output_names=["logits", "cls_token"],
                dynamic_axes={
                    "input": {0: "batch"},
                    "logits": {0: "batch"},
                    "cls_token": {0: "batch"}
                }
            )
            
        # Test Case 1: Multi-model Ensemble (2 models loaded)
        predictor_ens = EnsemblePredictor(models_dir=tmp_dir)
        assert predictor_ens.is_ensemble() is True
        assert len(predictor_ens.sessions) == 2
        
        # Run inference
        batch_img = np.random.randn(1, 3, 224, 224).astype(np.float32)
        mean_prob, uncertainty, mean_logits, mean_sec_out = predictor_ens.predict_ensemble(batch_img)
        
        # Verify predictions
        assert isinstance(mean_prob, float)
        assert 0.0 <= mean_prob <= 1.0
        # Since the two models are identical, the predictions are identical, so std should be 0.0
        assert isinstance(uncertainty, float)
        assert uncertainty == 0.0
        assert mean_logits.shape == (1, 1)
        assert mean_sec_out.shape == (1, 768)
        
        # Test Case 2: Single-model Fallback (1 model loaded)
        # Delete model_1.onnx so only model_0.onnx remains
        os.remove(onnx_path_1)
        
        predictor_single = EnsemblePredictor(models_dir=tmp_dir)
        assert predictor_single.is_ensemble() is False
        assert len(predictor_single.sessions) == 1
        
        # Run inference
        mean_prob_s, uncertainty_s, _, _ = predictor_single.predict_ensemble(batch_img)
        
        assert isinstance(mean_prob_s, float)
        # Verify fallback value: uncertainty must be None (null) if single model
        assert uncertainty_s is None
