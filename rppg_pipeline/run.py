import cv2
import numpy as np
from preprocessing.roi_extraction import extract_forehead_roi
from pipeline.rppg_pipeline import RPPGPipeline
from utils.visualization import draw_bpm_overlay, draw_scroll_plot
from utils.signal_processing import smooth_hr, clip_bpm, rate_limit_bpm, reject_outlier_bpm
from utils.rppg_pos import pos_from_frame_buffer, estimate_bpm_from_signal
from models.model_inference import load_model, refine_bpm_with_model, load_refiner, refine_signal_with_model
from config import (
    FRAME_RATE,
    ROI_SIZE,
    MIN_BPM,
    MAX_BPM,
    MAX_BPM_SLEW_PER_SEC,
    OUTLIER_SIGMA,
    USE_TORCH_REFINE,
    MODEL_REFINER_PATH,
    MODEL_HR_PATH,
    SNR_MIN_DB,
)
from utils.signal_processing import compute_snr_psd
import time
def main():
    cap = cv2.VideoCapture(0)
    pipeline = RPPGPipeline()  # reuse face detector

    # FPS handling
    fs = cap.get(cv2.CAP_PROP_FPS)
    if not fs or fs != fs or fs < 1:
        fs = FRAME_RATE
    fs = int(fs)

    # POS buffers
    BUFFER_SECONDS = 12
    frames_buf = []  # list of ROI frames
    bpm_hist = []    # for smoothing
    rppg_draw = []   # for scroll plot

    # Optional tiny model (will return None if torch not present)
    hr_model = load_model(path=MODEL_HR_PATH, device='cpu') if USE_TORCH_REFINE else None
    refiner = load_refiner(path=MODEL_REFINER_PATH, device='cpu') if USE_TORCH_REFINE else None

    last_bpm = None
    last_update_ts = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face/ROI
        faces = pipeline.face_detector.detect(frame)
        if len(faces) == 0:
            # draw placeholders and show
            draw_bpm_overlay(frame, None)
            draw_scroll_plot(frame, rppg_draw[-fs*BUFFER_SECONDS:] if rppg_draw else [])
            cv2.imshow('rPPG', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi = extract_forehead_roi(frame, faces[0], ROI_SIZE)
        if roi is None or roi.size == 0:
            draw_bpm_overlay(frame, None)
            draw_scroll_plot(frame, rppg_draw[-fs*BUFFER_SECONDS:] if rppg_draw else [])
            cv2.imshow('rPPG', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frames_buf.append(roi.copy())
        # keep buffer length ~ BUFFER_SECONDS
        max_len = int(BUFFER_SECONDS * fs)
        if len(frames_buf) > max_len:
            frames_buf = frames_buf[-max_len:]

        bpm_to_show = None
        if len(frames_buf) >= max(4*fs, 60):  # need at least a few seconds
            rppg = pos_from_frame_buffer(frames_buf)
            # optional refiner improves SNR
            rppg_refined = refine_signal_with_model(refiner, rppg)
            if rppg_refined is not None:
                rppg = rppg_refined
            rppg_draw.extend(rppg[-fs:])  # push last 1s section for drawing
            rppg_draw = rppg_draw[-max_len:]
            # signal quality gate
            snr_db = compute_snr_psd(rppg, fs)
            if snr_db < SNR_MIN_DB:
                bpm_to_show = last_bpm
                draw_bpm_overlay(frame, bpm_to_show)
                draw_scroll_plot(frame, rppg_draw[-max_len:])
                cv2.imshow('rPPG', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            bpm_est = estimate_bpm_from_signal(rppg, fs)
            # optional HR regressor
            bpm_refined = refine_bpm_with_model(hr_model, rppg, fs)
            bpm = bpm_refined if (bpm_refined is not None and bpm_refined > 0) else bpm_est
            if bpm is not None and bpm > 0:
                # clamp to physiologic range
                bpm = clip_bpm(bpm, MIN_BPM, MAX_BPM)
                # outlier rejection against recent history
                bpm_robust = reject_outlier_bpm(bpm, bpm_hist, sigma=OUTLIER_SIGMA)
                if bpm_robust is None:
                    bpm_robust = last_bpm if last_bpm is not None else bpm
                # smoothing (EMA)
                bpm_smoothed = smooth_hr(bpm_hist + [bpm_robust], method='ema', alpha=0.25)
                # rate limit based on elapsed time
                now = time.time()
                dt = (now - last_update_ts) if last_update_ts else 1.0
                bpm_limited = rate_limit_bpm(last_bpm, bpm_smoothed, dt, max_change_per_sec=MAX_BPM_SLEW_PER_SEC)
                last_update_ts = now
                last_bpm = bpm_limited
                bpm_hist.append(bpm_limited)
                bpm_hist = bpm_hist[-120:]
                bpm_to_show = bpm_limited

        draw_bpm_overlay(frame, bpm_to_show if bpm_to_show is not None else last_bpm)
        draw_scroll_plot(frame, rppg_draw[-max_len:] if rppg_draw else [])

        cv2.imshow('rPPG', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    main()
