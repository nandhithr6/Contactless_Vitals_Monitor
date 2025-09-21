import cv2
import numpy as np
from pipeline.rppg_pipeline import RPPGPipeline
from utils.visualization import plot_hr_realtime, overlay_hr_on_frame
from utils.signal_processing import update_buffer, estimate_hr, smooth_hr
from utils.gui import HRGui
from config import FRAME_RATE
def main():
    cap = cv2.VideoCapture(0)
    pipeline = RPPGPipeline()
    gui = HRGui()
    hr_minute_buffer = []
    signal_buffer = np.zeros(FRAME_RATE * 12)
    window_size = FRAME_RATE * 10  # 10 seconds
    step_size = FRAME_RATE // 2    # 1 second, 50% overlap
    frame_count = 0
    last_hr = None
    minute_hr_accum = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Get face bbox for overlay
        faces = pipeline.face_detector.detect(frame)
        gui.update_face_bbox(faces[0] if len(faces) > 0 else None)
        # Status bar
        if len(faces) == 0:
            gui.update_status("No face detected")
        else:
            gui.update_status("Face detected")
        _, rppg_signal = pipeline.process_frame(frame)
        gui.update_signal(rppg_signal)
        signal_buffer = update_buffer(signal_buffer, rppg_signal, window_size, step_size)
        frame_count += 1
        calculating = len(signal_buffer) < window_size
        # Signal quality (SNR)
        if not calculating:
            snr = 10 * np.log10(np.mean(rppg_signal ** 2) / (np.var(rppg_signal - np.mean(rppg_signal)) + 1e-8))
            gui.update_snr(snr)
        if not calculating and frame_count % step_size == 0:
            hr = estimate_hr(signal_buffer, FRAME_RATE)
            hr = smooth_hr(hr_minute_buffer + [hr], method='ema', alpha=0.2)
            last_hr = hr
            hr_minute_buffer.append(hr)
            minute_hr_accum.append(hr)
            if len(hr_minute_buffer) > 240:
                hr_minute_buffer = hr_minute_buffer[-240:]
        # Update sidebar every minute
        if frame_count % (FRAME_RATE * 60) == 0 and minute_hr_accum:
            avg_minute_hr = np.mean(minute_hr_accum)
            gui.update_hr(avg_minute_hr)
            minute_hr_accum = []
        overlay = gui.render_overlay(frame, last_hr if last_hr is not None else 0, calculating=calculating)
        cv2.imshow('rPPG HR Monitor', overlay)
        gui.plot_signal_graph()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    main()
