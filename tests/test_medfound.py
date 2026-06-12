import os
import tempfile
import torch
import numpy as np
import pytest
import onnxruntime as ort

from src.model_medfound import MedicalFoundationClassifier
from export_onnx import MedFoundWrapper

def test_medical_foundation_classifier():
    # Test BioViL-T (should use microsoft/Biovil-T configuration)
    model = MedicalFoundationClassifier(model_name="microsoft/Biovil-T", use_lora=True)
    model.eval()
    
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 1)
    
    # Test CheXzero/CLIP configuration
    model_clip = MedicalFoundationClassifier(model_name="openai/clip-vit-base-patch32", use_lora=False)
    model_clip.eval()
    logits_clip = model_clip(x)
    assert logits_clip.shape == (2, 1)

def test_medfound_training_validation_steps():
    model = MedicalFoundationClassifier(model_name="microsoft/Biovil-T", use_lora=True)
    
    # Dummy batch
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([1.0, 0.0])
    batch = (x, y)
    
    loss_train = model.training_step(batch, 0)
    loss_val = model.validation_step(batch, 0)
    
    assert isinstance(loss_train, torch.Tensor)
    assert isinstance(loss_val, torch.Tensor)
    
    # Test optimizer configuration
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

def test_medfound_onnx_export():
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, "medfound_model.onnx")
        
        # Instantiate model
        model = MedicalFoundationClassifier(model_name="microsoft/Biovil-T", use_lora=True)
        model.eval()
        
        # Wrap model
        wrapper = MedFoundWrapper(model)
        wrapper.eval()
        
        # Input sample
        x = torch.randn(1, 3, 224, 224)
        
        # Run PyTorch inference
        with torch.no_grad():
            logits_pt, cls_pt = wrapper(x)
            
        # Export to ONNX
        torch.onnx.export(
            wrapper,
            x,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits", "cls_token"]
        )
        
        assert os.path.exists(onnx_path)
        
        # Load ONNX Session
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Run ONNX inference
        inputs = {"input": x.numpy()}
        logits_onnx, cls_onnx = sess.run(None, inputs)
        
        # Compare shapes
        assert logits_pt.shape == logits_onnx.shape
        assert cls_pt.shape == cls_onnx.shape
        
        # Compare numerical values
        np.testing.assert_allclose(logits_pt.detach().numpy(), logits_onnx, rtol=1e-3, atol=5e-4)
        np.testing.assert_allclose(cls_pt.detach().numpy(), cls_onnx, rtol=1e-3, atol=5e-4)
