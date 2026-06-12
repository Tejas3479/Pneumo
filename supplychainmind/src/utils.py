import random
import numpy as np

def set_seed(seed=42):
    """
    Sets standard random and numpy seeds for model and pipeline reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
