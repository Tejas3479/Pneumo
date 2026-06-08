import torch
import torch.nn as nn
from torch.autograd import Function
import pytorch_lightning as pl
import numpy as np

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient reversal: multiply gradients by -alpha
        return grad_output * -ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

class AdversarialDebiasHead(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, alpha=1.0):
        # Apply GRL
        x_reversed = grad_reverse(x, alpha)
        return self.net(x_reversed)

class FairnessAudit:
    @staticmethod
    def compute_metrics(y_true, y_pred_prob, sex):
        """
        Compute demographic parity and equal opportunity differences.
        y_true, y_pred_prob, sex should be numpy arrays of shape (N,)
        y_pred_prob are binary probabilities, thresholded at 0.5 to get predictions.
        """
        y_pred = (y_pred_prob >= 0.5).astype(int)
        sex = sex.astype(int)
        y_true = y_true.astype(int)
        
        # Group indices
        idx_a0 = (sex == 0)
        idx_a1 = (sex == 1)
        
        # Demographic Parity: P(Y_hat=1 | A=0) and P(Y_hat=1 | A=1)
        prob_a0 = np.mean(y_pred[idx_a0]) if np.sum(idx_a0) > 0 else 0.0
        prob_a1 = np.mean(y_pred[idx_a1]) if np.sum(idx_a1) > 0 else 0.0
        demographic_parity_diff = abs(prob_a0 - prob_a1)
        
        # Equality of Opportunity: P(Y_hat=1 | A=0, Y=1) and P(Y_hat=1 | A=1, Y=1)
        idx_a0_y1 = (sex == 0) & (y_true == 1)
        idx_a1_y1 = (sex == 1) & (y_true == 1)
        
        tpr_a0 = np.mean(y_pred[idx_a0_y1]) if np.sum(idx_a0_y1) > 0 else 0.0
        tpr_a1 = np.mean(y_pred[idx_a1_y1]) if np.sum(idx_a1_y1) > 0 else 0.0
        equal_opportunity_diff = abs(tpr_a0 - tpr_a1)
        
        return {
            "demographic_parity_difference": float(demographic_parity_diff),
            "equal_opportunity_difference": float(equal_opportunity_diff),
            "tpr_subgroup_0": float(tpr_a0),
            "tpr_subgroup_1": float(tpr_a1),
            "selection_rate_subgroup_0": float(prob_a0),
            "selection_rate_subgroup_1": float(prob_a1)
        }

class FairnessLoggingCallback(pl.Callback):
    def __init__(self, val_loader):
        super().__init__()
        self.val_loader = val_loader

    def on_validation_epoch_end(self, trainer, pl_module):
        # Collect predictions, true labels, and sex attributes from val_loader
        was_training = pl_module.training
        pl_module.eval()
        
        y_trues = []
        y_pred_probs = []
        sexes = []
        
        device = pl_module.device
        
        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 3:
                    x, y, sex = batch
                else:
                    x, y = batch[:2]
                    sex = torch.zeros_like(y)
                
                # Move to device
                x = x.to(device)
                
                # Forward pass
                logits = pl_module(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = logits.squeeze(-1)
                
                probs = torch.sigmoid(logits)
                
                y_trues.append(y.cpu().numpy())
                y_pred_probs.append(probs.cpu().numpy())
                sexes.append(sex.numpy())
                
        if was_training:
            pl_module.train()
            
        if len(y_trues) == 0:
            return
            
        y_true_all = np.concatenate(y_trues)
        y_pred_prob_all = np.concatenate(y_pred_probs)
        sex_all = np.concatenate(sexes)
        
        metrics = FairnessAudit.compute_metrics(y_true_all, y_pred_prob_all, sex_all)
        
        # Log to TensorBoard/Trainer logger
        for k, v in metrics.items():
            pl_module.log(f"val_fairness_{k}", v, on_epoch=True, prog_bar=False, logger=True)
            
        print(f"\n--- Validation Fairness Metrics (Epoch {trainer.current_epoch}) ---")
        print(f"Demographic Parity Diff: {metrics['demographic_parity_difference']:.4f}")
        print(f"Equal Opportunity Diff:  {metrics['equal_opportunity_difference']:.4f}")
        print("------------------------------------------------------\n")
