"""
CHROM rPPG method implementation
"""
import numpy as np

def chrom_method(rgb_signals):
    # rgb_signals: shape (N, 3)
    X = 3 * rgb_signals[:, 0] - 2 * rgb_signals[:, 1]
    Y = 1.5 * rgb_signals[:, 0] + rgb_signals[:, 1] - 1.5 * rgb_signals[:, 2]
    S = X / (Y + 1e-8)
    S = S - np.mean(S)
    return S
