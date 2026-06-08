import os
import tempfile
import torch
import numpy as np
import pytest
import onnxruntime as ort

from src.model import PneumothoraxClassifier
from export_onnx import ResNetModelWrapper as ModelWrapper

def test_onnx_parity():
    # Setup temporary export path
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, "model.onnx")
        
        # Instantiate model (no checkpoint required, standard weights)
        model = PneumothoraxClassifier()
        model.eval()
        
        # Wrap the model
        wrapper = ModelWrapper(model)
        wrapper.eval()
        
        # Create dummy input: 1 sample, 3 channels, 224x224
        x = torch.randn(1, 3, 224, 224)
        
        # Run PyTorch inference
        with torch.no_grad():
            logits_pt, fm_pt = wrapper(x)
            
        # Export to ONNX
        torch.onnx.export(
            wrapper,
            x,
            onnx_path,
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits", "feature_map"]
        )
        
        assert os.path.exists(onnx_path)
        
        # Load ONNX Session
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Run ONNX inference
        inputs = {"input": x.numpy()}
        logits_onnx, fm_onnx = sess.run(None, inputs)
        
        # Compare shapes
        assert logits_pt.shape == logits_onnx.shape
        assert fm_pt.shape == fm_onnx.shape
        
        # Compare numerical values (tolerance: 1e-4)
        np.testing.assert_allclose(logits_pt.detach().numpy(), logits_onnx, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(fm_pt.detach().numpy(), fm_onnx, rtol=1e-3, atol=1e-4)
        
        print("ONNX and PyTorch parity test passed successfully!")
