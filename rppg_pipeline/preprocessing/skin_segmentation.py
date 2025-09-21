"""
Skin segmentation using HSV color space
"""
import cv2
import numpy as np
from config import SKIN_HSV_LOWER, SKIN_HSV_UPPER

def skin_mask(frame):
    if frame is None or frame.size == 0:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(SKIN_HSV_LOWER, dtype=np.uint8)
    upper = np.array(SKIN_HSV_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask
