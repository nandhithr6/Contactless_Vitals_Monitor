"""
Signal processing utilities for rPPG
"""
import numpy as np
from scipy.signal import detrend, butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq


def bandpass_filter(signal, fs, low, high):
    b, a = butter(2, [low/(0.5*fs), high/(0.5*fs)], btype='band')
    return filtfilt(b, a, signal)

def preprocess_signal(signal, fs, band=(0.7, 4.0)):
    signal = detrend(signal)
    signal = bandpass_filter(signal, fs, band[0], band[1])
    return signal

def compute_fft(signal, fs):
    N = len(signal)
    freqs = rfftfreq(N, 1/fs)
    fft_vals = np.abs(rfft(signal))
    return freqs, fft_vals

def detect_peaks(signal, distance=30):
    peaks, _ = find_peaks(signal, distance=distance)
    return peaks
