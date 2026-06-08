import torch
import pytest
from src.model import PneumothoraxClassifier

def test_classifier_forward():
    # Instantiate the model
    model = PneumothoraxClassifier(lr=1e-4)
    model.eval()

    # Create dummy input batch: shape (4, 3, 224, 224)
    x = torch.randn(4, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
        
    # Check output shape: (batch_size, 1)
    assert logits.shape == (4, 1)

def test_classifier_step():
    # Instantiate model
    model = PneumothoraxClassifier(lr=1e-4)
    
    # Mock a batch
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([1.0, 0.0], dtype=torch.float32)
    batch = (x, y)
    
    # Test training step
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar loss
    assert not torch.isnan(loss)
    
    # Test validation step
    val_loss = model.validation_step(batch, 0)
    assert isinstance(val_loss, torch.Tensor)
    assert val_loss.ndim == 0
    assert not torch.isnan(val_loss)
