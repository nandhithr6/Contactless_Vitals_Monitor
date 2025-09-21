"""
Green channel averaging baseline
"""
import numpy as np

def green_method(rgb_signals):
    # rgb_signals: shape (N, 3)
    S = rgb_signals[:, 1]  # Green channel
    S = S - np.mean(S)
    return S
