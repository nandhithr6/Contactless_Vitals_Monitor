"""
Advanced signal processing for rPPG HR estimation
"""
import numpy as np
from scipy.signal import butter, filtfilt, welch, detrend


def bandpass_filter(signal, fs, low=0.7, high=4.0, order=2):
    b, a = butter(order, [low/(0.5*fs), high/(0.5*fs)], btype='band')
    return filtfilt(b, a, signal)
from scipy.signal import find_peaks


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

# --- Robust BPM utilities ---
def clip_bpm(bpm, min_bpm=45, max_bpm=180):
    if bpm is None:
        return None
    return float(np.clip(bpm, min_bpm, max_bpm))

def rate_limit_bpm(prev_bpm, new_bpm, dt_sec, max_change_per_sec=8.0):
    if new_bpm is None:
        return prev_bpm
    if prev_bpm is None or dt_sec is None or dt_sec <= 0:
        return new_bpm
    max_step = max_change_per_sec * dt_sec
    delta = new_bpm - prev_bpm
    if abs(delta) <= max_step:
        return new_bpm
    return prev_bpm + np.sign(delta) * max_step

def reject_outlier_bpm(new_bpm, history, sigma=3.5):
    """Return None if new_bpm is an outlier vs recent history using median/MAD."""
    if new_bpm is None:
        return None
    if history is None or len(history) < 5:
        return new_bpm
    recent = np.array(history[-10:], dtype=float)
    med = np.median(recent)
    mad = np.median(np.abs(recent - med)) + 1e-8
    if np.abs(new_bpm - med) > sigma * 1.4826 * mad:
        return None
    return new_bpm


def compute_snr_psd(signal, fs, band=(0.7, 4.0), peak_bw_hz=0.3):
    """Estimate SNR in dB using Welch PSD within band: power in +/-peak_bw/2 around peak vs rest.
    Returns float dB.
    """
    if signal is None or len(signal) < max(32, fs):
        return -np.inf
    from scipy.signal import welch
    f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        return -np.inf
    fsel, Psel = f[mask], Pxx[mask]
    k = np.argmax(Psel)
    f0 = fsel[k]
    # define peak band
    peak_mask = (fsel >= (f0 - peak_bw_hz/2)) & (fsel <= (f0 + peak_bw_hz/2))
    sig_pow = np.trapz(Psel[peak_mask], fsel[peak_mask]) + 1e-12
    noise_pow = np.trapz(Psel[~peak_mask], fsel[~peak_mask]) + 1e-12
    return 10.0 * np.log10(sig_pow / noise_pow)
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
