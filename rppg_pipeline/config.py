"""
Configuration file for rPPG pipeline hyperparameters
"""

FRAME_RATE = 30
ROI_SIZE = (100, 50)  # width, height
BANDPASS_RANGE = (0.7, 4.0)  # Hz, typical HR range
FFT_WINDOW_SEC = 10
PEAK_DETECTION_MIN_DIST = 30  # frames
SKIN_HSV_LOWER = (0, 48, 80)
SKIN_HSV_UPPER = (20, 255, 255)

# BPM constraints and smoothing knobs
MIN_BPM = 45
MAX_BPM = 180
# Max allowed change rate in BPM per second (slew-rate limiter)
MAX_BPM_SLEW_PER_SEC = 8
# Outlier rejection sensitivity (higher = more tolerant)
OUTLIER_SIGMA = 3.5

# Optional lightweight model refinement (PyTorch)
USE_TORCH_REFINE = True  # if torch missing, code will fallback automatically
MODEL_REFINER_PATH = "models/refiner.pth"  # optional; leave missing to skip
MODEL_HR_PATH = None  # optional BPM regressor; leave None to skip

# Signal quality gate
SNR_MIN_DB = 0.0  # require at least 0 dB SNR to accept an update
