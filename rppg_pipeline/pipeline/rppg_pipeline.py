"""
rPPG pipeline orchestrator
"""
import numpy as np
from preprocessing.face_detection import FaceDetector
from preprocessing.skin_segmentation import skin_mask
from preprocessing.roi_extraction import extract_forehead_roi, average_roi_signal
from models.green import green_method
from models.chrom import chrom_method
from models.pos import pos_method
from utils.signal_tools import preprocess_signal
from config import FRAME_RATE, ROI_SIZE, BANDPASS_RANGE

class RPPGPipeline:
    def __init__(self, method='green'):
        self.face_detector = FaceDetector()
        self.method = method
        self.rgb_buffer = []

    def process_frame(self, frame):
        faces = self.face_detector.detect(frame)
        if len(faces) == 0:
            return 0, np.zeros(1)
        face_bbox = faces[0]
        roi = extract_forehead_roi(frame, face_bbox, ROI_SIZE)
        if roi is None or roi.size == 0:
            return 0, np.zeros(1)
        mask = skin_mask(roi)
        if mask is None:
            return 0, np.zeros(1)
        mean_rgb = average_roi_signal(roi, mask)
        self.rgb_buffer.append(mean_rgb)
        if len(self.rgb_buffer) < FRAME_RATE * 2:
            return 0, np.zeros(1)
        rgb_signals = np.array(self.rgb_buffer[-FRAME_RATE*2:])
        if self.method == 'green':
            rppg_signal = green_method(rgb_signals)
        elif self.method == 'chrom':
            rppg_signal = chrom_method(rgb_signals)
        elif self.method == 'pos':
            rppg_signal = pos_method(rgb_signals)
        else:
            rppg_signal = green_method(rgb_signals)
        rppg_signal = preprocess_signal(rppg_signal, FRAME_RATE, BANDPASS_RANGE)
        return None, rppg_signal
