"""
Metrics for rPPG: HR, HRV, SNR
"""
import numpy as np
from utils.signal_tools import compute_fft


def compute_hr(signal, fs, band=(0.7, 4.0)):
    freqs, fft_vals = compute_fft(signal, fs)
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    if len(idx) == 0:
        return 0
    peak_freq = freqs[idx][np.argmax(fft_vals[idx])]
    hr = peak_freq * 60  # BPM
    return hr

def compute_hrv(signal):
    # Simple HRV: std of RR intervals
    peaks = np.diff(np.where(signal > np.mean(signal))[0])
    if len(peaks) < 2:
        return 0
    rr_intervals = peaks
    hrv = np.std(rr_intervals)
    return hrv

def compute_snr(signal):
    signal_power = np.mean(signal ** 2)
    noise_power = np.var(signal - np.mean(signal))
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    return snr
