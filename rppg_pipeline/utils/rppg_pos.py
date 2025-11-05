"""
Real-time POS algorithm and BPM estimation for rPPG
CPU-friendly, uses NumPy + SciPy only.
"""
import numpy as np
from scipy.signal import butter, filtfilt, welch


def bandpass_filter(x, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, x)


def pos_from_frame_buffer(frames_roi):
    """
    frames_roi: array/list (N, H, W, 3) BGR (OpenCV)
    returns: rppg_signal (N,) float32
    """
    if len(frames_roi) == 0:
        return np.zeros(0, dtype=np.float32)
    acc = np.array([fr.mean(axis=(0, 1)) for fr in frames_roi], dtype=np.float32)  # (N,3) BGR
    rgb = acc[:, ::-1]  # BGR->RGB
    # classic POS steps
    X = rgb.T  # 3 x N
    meanX = X.mean(axis=1, keepdims=True)
    Xn = X / (meanX + 1e-8)
    P = np.array([[0, 1, -1],
                  [-2, 1, 1]], dtype=np.float32)  # 2x3
    S = P @ Xn  # 2 x N
    stds = S.std(axis=1) + 1e-8
    alpha = stds[0] / stds[1]
    h = S[0] - alpha * S[1]
    rppg = h - h.mean()
    return rppg.astype(np.float32)


def estimate_bpm_from_signal(sig, fs, band=(0.7, 4.0)):
    """Estimate BPM using Welch PSD peak in physiological band."""
    if sig is None or len(sig) < max(32, fs):
        return None
    sig_f = bandpass_filter(sig, fs, low=band[0], high=band[1])
    f, Pxx = welch(sig_f, fs=fs, nperseg=min(256, len(sig_f)))
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        return None
    fsel, Psel = f[mask], Pxx[mask]
    # quadratic interpolation around the max bin for sub-bin precision
    k = np.argmax(Psel)
    if 0 < k < len(Psel) - 1:
        y0, y1, y2 = Psel[k-1], Psel[k], Psel[k+1]
        denom = (y0 - 2*y1 + y2)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            f_peak = fsel[k] + delta * (fsel[1] - fsel[0])
        else:
            f_peak = fsel[k]
    else:
        f_peak = fsel[k]
    bpm = float(f_peak * 60.0)
    return bpm
