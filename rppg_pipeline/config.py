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
