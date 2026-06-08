import torch
import numpy as np
from src.fairness import FairnessAudit, AdversarialDebiasHead, grad_reverse

def test_fairness_audit():
    # Toy dataset: 
    # subgroup 0: y_true=[1, 1, 0, 0], y_pred_prob=[0.9, 0.8, 0.1, 0.2] (predicted label: [1, 1, 0, 0] -> positive rate: 0.5)
    # subgroup 1: y_true=[1, 1, 0, 0], y_pred_prob=[0.1, 0.2, 0.9, 0.8] (predicted label: [0, 0, 1, 1] -> positive rate: 0.5)
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.1, 0.2, 0.9, 0.8])
    sex = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    metrics = FairnessAudit.compute_metrics(y_true, y_pred_prob, sex)
    
    assert metrics["demographic_parity_difference"] == 0.0
    assert metrics["equal_opportunity_difference"] == 1.0
    assert metrics["tpr_subgroup_0"] == 1.0
    assert metrics["tpr_subgroup_1"] == 0.0

def test_gradient_reversal_layer():
    x = torch.randn(2, 5, requires_grad=True)
    alpha = 0.5
    y = grad_reverse(x, alpha)
    
    loss = y.sum()
    loss.backward()
    
    expected_grad = torch.ones_like(x) * -0.5
    assert torch.allclose(x.grad, expected_grad)

def test_adversarial_debias_head():
    head = AdversarialDebiasHead(input_dim=10, hidden_dim=5)
    x = torch.randn(2, 10, requires_grad=True)
    out = head(x, alpha=1.0)
    assert out.shape == (2, 1)
    
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
