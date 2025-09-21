"""
Modern GUI overlay for rPPG webcam feed
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class HRGui:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.hr_history = []  # List of (timestamp, hr)
        self.signal_history = []  # List of rPPG signal values
        self.bg_color = (30, 30, 30)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.status = "Initializing..."
        self.snr = None
        self.face_bbox = None

    def update_hr(self, hr):
        now = datetime.now().strftime('%H:%M:%S')
        self.hr_history.append((now, hr))
        if len(self.hr_history) > 4:
            self.hr_history = self.hr_history[-4:]

    def update_signal(self, signal):
        self.signal_history.append(signal)
        if len(self.signal_history) > 600:
            self.signal_history = self.signal_history[-600:]

    def update_status(self, status):
        self.status = status

    def update_snr(self, snr):
        self.snr = snr

    def update_face_bbox(self, bbox):
        self.face_bbox = bbox

    def render_overlay(self, frame, hr, calculating=False):
        overlay = frame.copy()
        sidebar_w = 250
        cv2.rectangle(overlay, (0,0), (sidebar_w, self.height), (50,50,60), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        # Draw HR and time
        if calculating:
            cv2.putText(frame, "HR: Calculating...", (20, 50), self.font, 1.2, (0,255,255), 3)
        else:
            cv2.putText(frame, f"HR: {int(hr)} BPM", (20, 50), self.font, 1.2, (0,255,0), 3)
        cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (20, 90), self.font, 0.8, (200,200,255), 2)
        # Draw last 4 minutes table
        start_y = 130
        cv2.putText(frame, "Last 4 min HR:", (20, start_y), self.font, 0.7, (255,255,255), 2)
        for i, (ts, hr_val) in enumerate(self.hr_history[::-1]):
            cv2.putText(frame, f"{ts}: {int(hr_val)} BPM", (20, start_y+30*(i+1)), self.font, 0.7, (180,220,255), 2)
        # Draw SNR
        if self.snr is not None:
            cv2.putText(frame, f"Signal Quality: {self.snr:.1f} dB", (20, start_y+150), self.font, 0.7, (0,255,255), 2)
        # Draw status bar
        cv2.putText(frame, f"Status: {self.status}", (20, self.height-30), self.font, 0.8, (255,255,0), 2)
        # Draw face ROI
        if self.face_bbox is not None:
            x, y, w, h = self.face_bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        return frame

    def plot_signal_graph(self):
        plt.clf()
        plt.plot(self.signal_history, color='b', linewidth=2)
        plt.title('rPPG Signal (Raw/Smoothed)')
        plt.xlabel('Frame')
        plt.ylabel('Signal')
        plt.grid(True)
        plt.tight_layout()
        plt.pause(0.001)
