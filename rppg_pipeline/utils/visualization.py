"""
Visualization utilities for rPPG
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay_hr_on_frame(frame, hr):
    text = f"HR: {int(hr)} BPM"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

hr_fig = None
hr_ax = None

def plot_hr_realtime(hr_history):
    global hr_fig, hr_ax
    if hr_fig is None or hr_ax is None:
        plt.ion()
        hr_fig, hr_ax = plt.subplots()
    hr_ax.clear()
    hr_ax.plot(hr_history, color='r')
    hr_ax.set_title('Heart Rate (BPM)')
    hr_ax.set_xlabel('Frame')
    hr_ax.set_ylabel('HR')
    plt.pause(0.001)


# --- OpenCV-only overlays (non-blocking) ---
def draw_bpm_overlay(frame, bpm, pos=(10, 30), color=(0, 255, 0)):
    txt = f"BPM: {bpm:.1f}" if (bpm is not None and bpm > 0) else "BPM: --"
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def draw_scroll_plot(frame, signal_buffer, rect=(10, 400, 500, 120)):
    fh, fw = frame.shape[:2]
    x, y, w, h = rect
    # Clamp to frame bounds
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > fw:
        w = max(1, fw - x)
    if y + h > fh:
        h = max(1, fh - y)
    plot = np.zeros((h, w, 3), dtype=np.uint8) + 30
    if signal_buffer is not None and len(signal_buffer) > 1:
        s = np.array(signal_buffer, dtype=np.float32)
        # normalize to [0, 1] for drawing
        s = (s - s.min()) / (s.ptp() + 1e-8)
        xs = np.linspace(0, w - 1, len(s)).astype(np.int32)
        ys = (h - 1 - (s * (h - 1))).astype(np.int32)
        for i in range(len(s) - 1):
            cv2.line(plot, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), (200, 200, 200), 1)
    frame[y:y + h, x:x + w] = plot
