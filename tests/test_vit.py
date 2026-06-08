import torch
import pytest
from src.model_foundation import ViTPneumothoraxClassifier

def test_vit_forward_pass():
    """
    Tests that a dummy batch input of [batch, 3, 224, 224] running through
    the ViTPneumothoraxClassifier module outputs predictions of shape [batch, 1].
    """
    model = ViTPneumothoraxClassifier()
    model.eval()

    # Create mock batch: shape [4, 3, 224, 224]
    x = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (4, 1)
    assert not torch.isnan(logits).any()

def test_vit_lora_parameter_freezing():
    """
    Verifies that all transformer backbone parameters are frozen (requires_grad = False),
    and only LoRA layers and classification head have requires_grad = True.
    """
    model = ViTPneumothoraxClassifier()

    trainable_params_count = 0
    frozen_params_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params_count += 1
            # Assert that the trainable param belongs to LoRA or classification head
            assert "lora_" in name or "classifier" in name
        else:
            frozen_params_count += 1

    assert trainable_params_count > 0
    assert frozen_params_count > 0
    print(f"LoRA Freezing verified. Trainable: {trainable_params_count}, Frozen: {frozen_params_count}")
