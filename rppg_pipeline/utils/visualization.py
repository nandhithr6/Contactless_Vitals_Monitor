"""
Visualization utilities for rPPG
"""
import cv2
import matplotlib.pyplot as plt


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
