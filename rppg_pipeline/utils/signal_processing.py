"""
Advanced signal processing for rPPG HR estimation
"""
import numpy as np
from scipy.signal import butter, filtfilt, welch, detrend


def bandpass_filter(signal, fs, low=0.7, high=4.0, order=2):
    b, a = butter(order, [low/(0.5*fs), high/(0.5*fs)], btype='band')
    return filtfilt(b, a, signal)


def update_buffer(buffer, new_data, window_size, step_size):
    buffer = np.concatenate([buffer, new_data])
    if len(buffer) > window_size:
        buffer = buffer[-window_size:]
    return buffer


def estimate_hr(signal, fs, band=(0.7, 4.0)):
    # Detrend and normalize
    signal = detrend(signal)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    # Bandpass filter
    signal = bandpass_filter(signal, fs, band[0], band[1])
    # Welch's PSD
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)), scaling='density')
    # Limit to HR band
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    if len(idx) == 0:
        return 0
    band_freqs = freqs[idx]
    band_psd = psd[idx]
    # Interpolate peak
    peak_idx = np.argmax(band_psd)
    if 0 < peak_idx < len(band_psd)-1:
        # Quadratic interpolation
        y0, y1, y2 = band_psd[peak_idx-1:peak_idx+2]
        x0, x1, x2 = band_freqs[peak_idx-1:peak_idx+2]
        denom = (y0 - 2*y1 + y2)
        if denom != 0:
            peak_freq = x1 + 0.5 * (y0 - y2) / denom * (x2 - x0) / 2
        else:
            peak_freq = band_freqs[peak_idx]
    else:
        peak_freq = band_freqs[peak_idx]
    hr = peak_freq * 60  # BPM
    return hr


def smooth_hr(hr_values, method='ma', alpha=0.2, window=5):
    hr_values = np.array(hr_values)
    if method == 'ma':
        if len(hr_values) < window:
            return hr_values[-1]
        return np.mean(hr_values[-window:])
    elif method == 'ema':
        smoothed = [hr_values[0]]
        for h in hr_values[1:]:
            smoothed.append(alpha * h + (1-alpha) * smoothed[-1])
        return smoothed[-1]
    else:
        return hr_values[-1]
