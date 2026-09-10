# -*- coding: utf-8 -*-
"""
utils/helper_functions.py
Common helper utilities for data normalization, device detection, and seeding.
"""

import os
import random
import numpy as np
import torch

def get_device():
    """Returns CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_scientific(val, decimals=2):
    """Formats floating point values into clean scientific notation."""
    return f"{val:.{decimals}e}"

def ensure_dir(path):
    """Ensures that the directory for a given path exists."""
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
