"""
POS rPPG method implementation
"""
import numpy as np

def pos_method(rgb_signals):
    # rgb_signals: shape (N, 3)
    mean_rgb = np.mean(rgb_signals, axis=0)
    normalized = rgb_signals / (mean_rgb + 1e-8)
    S = normalized[:, 1] - normalized[:, 2]
    S = S - np.mean(S)
    return S
