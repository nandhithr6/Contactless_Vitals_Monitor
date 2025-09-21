"""
ROI extraction for forehead/cheeks
"""
import numpy as np
import cv2

# Assumes face bounding box: (x, y, w, h)
def extract_forehead_roi(frame, face_bbox, roi_size=(100, 50)):
    x, y, w, h = face_bbox
    roi_x = x + w//2 - roi_size[0]//2
    roi_y = y + int(0.15*h)
    roi = frame[roi_y:roi_y+roi_size[1], roi_x:roi_x+roi_size[0]]
    return roi

def average_roi_signal(roi, mask=None):
    if mask is not None:
        roi = cv2.bitwise_and(roi, roi, mask=mask)
    mean_signal = np.mean(roi, axis=(0, 1))
    return mean_signal
