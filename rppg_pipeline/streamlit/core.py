# streamlit_demo/core.py
"""
Improved core rPPG processor with robust ME-rPPG ONNX adapter + model-output stabilization.
"""

import time
import threading
from collections import deque
import io
import os
import json
import csv

import cv2
import numpy as np
from scipy.signal import butter, lfilter, welch
import onnxruntime as ort
import psutil
import pandas as pd
import math
from rppg_pipeline.models.onnx_inference import get_model, onnx_step


# ---------------- signal helpers ----------------
def bandpass_iir(sig, fs, low=0.7, high=4.0, order=2):
    nyq = 0.5 * fs
    lown, highn = low/nyq, high/nyq
    b, a = butter(order, [lown, highn], btype='band')
    return lfilter(b, a, sig)

def estimate_bpm_welch_improved(sig, fs, low=0.8, high=3.5):
    sig = np.asarray(sig)
    if sig.size < 4:
        return None, 0.0, None, None
    try:
        filtered = bandpass_iir(sig, fs, low=low, high=high)
    except Exception:
        filtered = sig
    f, P = welch(filtered, fs=fs, nperseg=min(512, len(filtered)))
    mask = (f >= low) & (f <= high)
    if not mask.any():
        return None, 0.0, None, None
    fsel, Psel = f[mask], P[mask]
    if Psel.sum() <= 0:
        return None, 0.0, None, None
    idx = np.argmax(Psel)
    freq = fsel[idx]
    peak_power = Psel[idx]
    conf = float(peak_power / (Psel.sum() + 1e-12))
    return float(freq*60.0), conf, freq, peak_power

def pos_from_buffer(frames_roi):
    arr = np.array([f.mean(axis=(0,1)) for f in frames_roi], dtype=np.float32)  # N x 3 BGR
    if arr.shape[0] < 2:
        return arr[:,0] if arr.shape[0] else np.zeros((0,))
    rgb = arr[:, ::-1]  # BGR -> RGB
    X = rgb.T  # 3 x N
    meanX = X.mean(axis=1, keepdims=True)
    Xn = X / (meanX + 1e-8)
    P = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)
    S = P.dot(Xn)
    stds = S.std(axis=1)
    if stds[1] < 1e-8:
        h = S[0]
    else:
        alpha = stds[0] / (stds[1] + 1e-8)
        h = S[0] - alpha * S[1]
    h = h - h.mean()
    return h

# ---------------- ONNX helpers ----------------
def load_onnx_session(path, cpu_threads=None):
    try:
        sess_opts = ort.SessionOptions()
        if cpu_threads is None:
            cpu_threads = max(1, psutil.cpu_count(logical=False))
        sess_opts.intra_op_num_threads = cpu_threads
        sess = ort.InferenceSession(path, sess_opts, providers=['CPUExecutionProvider'])
        return sess
    except Exception as e:
        print("[core] ONNX load failed:", e)
        return None

def load_state_json(path):
    with open(path, 'r') as f:
        j = json.load(f)
    state = {}
    for k, v in j.items():
        try:
            arr = np.array(v, dtype=np.float32)
        except Exception:
            arr = np.asarray(v)
        state[k] = arr
    return state

def build_face_input(roi_bgr, H=36, W=36):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (W, H))
    arr = roi_resized.astype(np.float32) / 255.0
    arr = arr.reshape(1, 1, H, W, 3)
    return arr

def me_rppg_step(sess, state_dict, face_patch_bgr, verbose=False):
    if sess is None:
        return None, state_dict, {"error":"no_session"}
    try:
        face_in = build_face_input(face_patch_bgr, H=36, W=36)
    except Exception as e:
        return None, state_dict, {"error": f"face_prep_failed: {e}"}
    inputs = sess.get_inputs()
    feed={}
    feed[inputs[0].name] = face_in.astype(np.float32)
    for inp in inputs[1:]:
        name = inp.name
        # get expected shape, convert None/dynamic -> 1
        expected_shape = []
        for d in inp.shape:
            try:
                expected_shape.append(int(d))
            except Exception:
                expected_shape.append(1)
        expected_shape = tuple(expected_shape)
        if name in state_dict:
            arr = np.asarray(state_dict[name], dtype=np.float32)
            if arr.shape == expected_shape:
                feed[name] = arr
            else:
                if arr.size == np.prod(expected_shape):
                    feed[name] = arr.reshape(expected_shape).astype(np.float32)
                else:
                    feed[name] = np.zeros(expected_shape, dtype=np.float32)
                    if verbose:
                        print(f"[ME] state shape mismatch {name}: {arr.shape} != {expected_shape} -> zeros")
        else:
            feed[name] = np.zeros(expected_shape, dtype=np.float32)
            if verbose:
                print(f"[ME] missing state key {name} -> zeros {expected_shape}")
    try:
        outputs = sess.run(None, feed)
    except Exception as e:
        if verbose:
            print("[ME-adapter] sess.run failed:", e)
        return None, state_dict, {"error": str(e)}
    bpm_val = None
    dbg_raw = None
    try:
        out0 = np.asarray(outputs[0]).flatten()
        dbg_raw = out0.tolist()
        if out0.size > 0:
            v = float(out0[0])
            # Interpret:
            if 0.2 <= v <= 4.0:
                bpm_val = float(v * 60.0)
            elif 10.0 <= v <= 300.0:
                bpm_val = float(v)
            else:
                # borderline: if between 4 and 10 maybe it's scaled; treat as raw and try to convert if plausible
                if 4.0 < v <= 10.0:
                    # assume it's freq in deci-Hz? unlikely; set None to be safe
                    bpm_val = None
                else:
                    bpm_val = None
    except Exception:
        bpm_val = None
    # update state
    new_state = dict(state_dict) if state_dict is not None else {}
    input_names_rest = [inp.name for inp in inputs[1:]]
    for i, name in enumerate(input_names_rest):
        out_idx = i + 1
        if out_idx < len(outputs):
            new_state[name] = np.asarray(outputs[out_idx], dtype=np.float32)
        else:
            if name not in new_state:
                # fallback zeros
                try:
                    expected_shape = tuple(int(d) if isinstance(d, (int, np.integer)) else 1 for d in sess.get_inputs()[i+1].shape)
                except Exception:
                    expected_shape = (1,)
                new_state[name] = np.zeros(expected_shape, dtype=np.float32)
    debug = {"out0_raw_sample": dbg_raw, "out0_shape": outputs[0].shape if len(outputs)>0 else None}
    return bpm_val, new_state, debug

# ---------------- smoothing + model stabilizer ----------------
class BPMSmoother:
    def __init__(self, median_k=5, ema_alpha=0.25, max_rate_bpm_per_s=12.0):
        self.median_k = median_k
        self.ema_alpha = ema_alpha
        self.max_rate = max_rate_bpm_per_s
        self.hist = deque(maxlen=max(1, median_k))
        self.ema = None
        self.last_ts = None
    def update(self, bpm_candidate, ts=None):
        if bpm_candidate is None:
            return self.ema
        now = ts if ts is not None else time.time()
        self.hist.append(float(bpm_candidate))
        med = float(np.median(np.array(self.hist)))
        if self.ema is None:
            self.ema = med
            self.last_ts = now
            return self.ema
        dt = max(1e-6, now - (self.last_ts or now))
        max_delta = self.max_rate * dt
        target = (1 - self.ema_alpha) * self.ema + self.ema_alpha * med
        delta = target - self.ema
        if abs(delta) > max_delta:
            delta = math.copysign(max_delta, delta)
        self.ema = float(self.ema + delta)
        self.last_ts = now
        return self.ema
    def reset(self):
        self.hist.clear(); self.ema = None; self.last_ts = None

# ---------------- safe diagnostic print (module-level) ----------------
def _safe_print_model_diag(raw0_val, bpm_model_val, bpm_pos_val, conf_pos_val):
    try:
        if raw0_val is None:
            print(f"[MODEL DIAG] raw=None pos_bpm={bpm_pos_val} conf={conf_pos_val}")
            return
        cand = None
        rv = None
        try:
            rv = float(raw0_val)
            if 0.05 <= rv <= 6.0:
                cand = rv * 60.0
            elif 10.0 <= rv <= 300.0:
                cand = rv
        except Exception:
            cand = None
        if cand is None:
            print(f"[MODEL DIAG] raw={raw0_val} candidate=None pos_bpm={bpm_pos_val} conf={conf_pos_val}")
        else:
            if bpm_model_val is not None:
                print(f"[MODEL DIAG] raw={rv:.4f} candidate_bpm={cand:.2f} bpm_model={float(bpm_model_val):.2f} pos_bpm={bpm_pos_val} conf={conf_pos_val}")
            else:
                print(f"[MODEL DIAG] raw={rv:.4f} candidate_bpm={cand:.2f} bpm_model=None pos_bpm={bpm_pos_val} conf={conf_pos_val}")
    except Exception as _e:
        print("[MODEL DIAG] print failed:", _e)

# ---------------- Processor ----------------
class Processor:
    def __init__(self, onnx_path=None, state_json=None, camera_idx=0, buffer_seconds=12, target_fps=30, roi_size=(128,128)):
        self.onnx_path = onnx_path
        self.state_json = state_json
        self.camera_idx = int(camera_idx)
        self.buffer_seconds = int(buffer_seconds)
        self.target_fps = int(target_fps)
        self.roi_size = tuple(roi_size)
        self.buf_size = max(8, int(self.buffer_seconds * self.target_fps))
        self.frames_buf = deque(maxlen=self.buf_size)
        self.rppg_buf = deque(maxlen=self.buf_size)
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_face = None

        self.onnx_sess = None
        self.me_state = {}
        if self.onnx_path and os.path.exists(self.onnx_path):
            self.onnx_sess = load_onnx_session(self.onnx_path)
        if self.state_json and os.path.exists(self.state_json):
            try:
                self.me_state = load_state_json(self.state_json)
            except Exception as e:
                print("[core] failed loading state json:", e)
                self.me_state = {}

        # initialize ONNX model helper (keeps internal state)
        try:
            self.onnx_model = get_model()
        except Exception:
            self.onnx_model = None

        self.cam = None
        self.thread_cam = None
        self.thread_proc = None
        self.latest_frame = None
        self.latest_metrics = {}
        self.logging = False
        self.log_rows = []
        self.log_lock = threading.Lock()

        self.bpm_smoother = BPMSmoother(median_k=5, ema_alpha=0.25, max_rate_bpm_per_s=12.0)

        # model stability tracker
        self.model_raw_recent = deque(maxlen=8)   # last raw outputs (float)
        self.model_recent_ts = deque(maxlen=8)
        self.model_log_path = "me_onnx_model_log.csv"
        # create header if not exists
        if not os.path.exists(self.model_log_path):
            try:
                with open(self.model_log_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["ts", "raw_out0", "interpreted_bpm"])
            except Exception:
                pass

        # calibration factor for model BPM
        self.model_calibration_factor = 1.0
        self.last_pos_confidence = 0.0
        self.last_pos_value = None

    def start(self):
        self.stop_flag.clear()
        self.cam = cv2.VideoCapture(self.camera_idx)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.thread_cam = threading.Thread(target=self._cam_worker, daemon=True)
        self.thread_cam.start()
        self.thread_proc = threading.Thread(target=self._proc_worker, daemon=True)
        self.thread_proc.start()

    def _cam_worker(self):
        while not self.stop_flag.is_set():
            ret, frame = self.cam.read()
            if not ret:
                time.sleep(0.01); continue
            with self.lock:
                self.latest_raw = frame.copy()
            time.sleep(0)

    def _proc_worker(self):
        fps_timer = time.time(); frames_count = 0; measured_fps = self.target_fps
        while not self.stop_flag.is_set():
            with self.lock:
                frame = getattr(self, 'latest_raw', None)
            if frame is None:
                time.sleep(0.01); continue
            frames_count += 1
            if time.time() - fps_timer >= 1.0:
                measured_fps = frames_count / (time.time() - fps_timer)
                fps_timer = time.time(); frames_count = 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80,80))
            if len(faces)>0:
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                x,y,w,h = faces[0]; self.last_face=(x,y,w,h)
            elif self.last_face is not None:
                x,y,w,h = self.last_face
            else:
                x,y,w,h = 0,0,0,0

            if w>0 and h>0:
                rx = int(x + 0.15*w); ry = int(y + 0.12*h)
                rw = int(w*0.7); rh = int(h*0.35)
                roi = frame[ry:ry+rh, rx:rx+rw]
                if roi.size==0:
                    roi_small = cv2.resize(frame, self.roi_size)
                else:
                    roi_small = cv2.resize(roi, self.roi_size)
            else:
                roi_small = np.zeros((self.roi_size[1], self.roi_size[0], 3), dtype=np.uint8)

            self.frames_buf.append(roi_small.copy())

            bpm_pos=None; conf_pos=0.0; bpm_model=None; fused_bpm=None; raw0=None
            if len(self.frames_buf) >= max(8, int(2*measured_fps)):
                pos_sig = pos_from_buffer(list(self.frames_buf))
                # debug POS signal stats (helps spot stuck POS)
                try:
                    if len(pos_sig) > 0:
                        print(f"[POS DEBUG] len={len(pos_sig)} mean={np.mean(pos_sig):.4f} std={np.std(pos_sig):.4f}")
                except Exception:
                    pass

                self.rppg_buf.clear()
                for v in pos_sig: self.rppg_buf.append(float(v))
                pos_est, conf, _, _ = estimate_bpm_welch_improved(np.array(list(self.rppg_buf)), fs=max(1, int(measured_fps)))
                if pos_est is not None:
                    bpm_pos=pos_est; conf_pos=conf

                # ------------------- NEW ONNX call using onnx_inference helper -------------------
                bpm_model = None
                raw0 = None
                roi_rgb = cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB)
                try:
                    raw0, outs_map = onnx_step(roi_rgb)
                except Exception as e:
                    raw0, outs_map = None, {}
                    print("[ME-ONNX] onnx_step failed:", e)

                if raw0 is not None:
                    try:
                        self.model_raw_recent.append(float(raw0))
                        self.model_recent_ts.append(time.time())
                    except Exception:
                        pass
                    # Interpret model raw to BPM safely
                    try:
                        v = float(raw0)
                        if 0.05 <= v <= 6.0:
                            bpm_model = v * 60.0
                        elif 10.0 <= v <= 300.0:
                            bpm_model = v
                        else:
                            bpm_model = None
                    except Exception:
                        bpm_model = None
                # ------------------- end new onnx call -------------------

            # ------------------ Fusion & model auto-calibration ------------------
            # keep last-pos values for safe printing/fallbacks
            self.last_pos_confidence = float(conf_pos) if conf_pos is not None else float(getattr(self, "last_pos_confidence", 0.0))
            self.last_pos_value = float(bpm_pos) if bpm_pos is not None else getattr(self, "last_pos_value", None)

            # prepare recent model BPMs array
            try:
                recent = np.array([float(v) for v in self.model_raw_recent if v is not None], dtype=np.float32)
                recent_bpm_list = []
                for r in recent:
                    if 0.05 <= r <= 6.0:
                        recent_bpm_list.append(r * 60.0)
                    else:
                        recent_bpm_list.append(r)
                recent_bpm = np.array(recent_bpm_list, dtype=np.float32) if len(recent_bpm_list)>0 else np.array([], dtype=np.float32)
            except Exception:
                recent_bpm = np.array([], dtype=np.float32)

            # update calibration factor if we have stable recent model outputs and valid POS
            try:
                if recent_bpm.size >= 8 and bpm_pos is not None:
                    med_model = float(np.median(recent_bpm))
                    med_pos = float(bpm_pos)
                    if med_model > 20 and 30 <= med_pos <= 220:
                        target_cf = med_pos / (med_model + 1e-12)
                        self.model_calibration_factor = 0.92 * float(self.model_calibration_factor) + 0.08 * float(target_cf)
            except Exception:
                pass

            # apply calibration
            calibrated_model_bpm = None
            if bpm_model is not None:
                try:
                    calibrated_model_bpm = float(bpm_model) * float(self.model_calibration_factor)
                except Exception:
                    calibrated_model_bpm = bpm_model

            # model stability measure (std of recent BPMs)
            model_stable = False
            model_std = 999.0
            try:
                if recent_bpm.size >= 6:
                    model_std = float(np.std(recent_bpm))
                    model_stable = (model_std < 6.0)
            except Exception:
                pass

            # weights
            w_pos = 0.0
            w_model = 0.0
            if bpm_pos is not None:
                w_pos = float(np.clip(conf_pos * 3.0, 0.05, 1.0))
            if calibrated_model_bpm is not None:
                w_model = float(np.clip(0.2 + (0.8 * (1.0 if model_stable else 0.0)), 0.05, 1.0))

            fused_bpm = None
            if (w_pos + w_model) > 0.0:
                numer = 0.0
                if bpm_pos is not None:
                    numer += w_pos * float(bpm_pos)
                if calibrated_model_bpm is not None:
                    numer += w_model * float(calibrated_model_bpm)
                fused_bpm = float(numer / (w_pos + w_model))
            else:
                fused_bpm = calibrated_model_bpm if calibrated_model_bpm is not None else bpm_pos

            # final display smoothing
            display_raw = fused_bpm if fused_bpm is not None else (calibrated_model_bpm if calibrated_model_bpm is not None else bpm_pos)
            display_bpm = self.bpm_smoother.update(display_raw, ts=time.time()) if display_raw is not None else self.bpm_smoother.update(None, ts=time.time())
            # ------------------ end fusion ------------------

            # overlay
            overlay = frame.copy()
            if w>0 and h>0:
                cv2.rectangle(overlay,(x,y),(x+w,y+h),(200,200,0),2)
                cv2.rectangle(overlay,(rx,ry),(rx+rw,ry+rh),(0,200,0),2)
            bpm_text = f"BPM: {display_bpm:.1f}" if display_bpm is not None else "BPM: --"
            col = (0,200,0) if conf_pos > 0.15 else (0,180,220) if conf_pos>0.06 else (0,120,255)
            cv2.putText(overlay, bpm_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
            cv2.putText(overlay, f"POS: {bpm_pos:.1f}" if bpm_pos is not None else "POS: --", (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
            cv2.putText(overlay, f"MODEL: {calibrated_model_bpm:.1f}" if calibrated_model_bpm is not None else "MODEL: --", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
            cv2.putText(overlay, f"FUSED: {fused_bpm:.1f}" if fused_bpm is not None else "FUSED: --", (20,125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
            cv2.putText(overlay, f"CONF: {conf_pos:.2f}", (20,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
            cpu = psutil.cpu_percent(interval=None)
            cv2.putText(overlay, f"CPU: {cpu:.0f}%", (20,175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
            cv2.putText(overlay, f"FPS: {measured_fps:.1f}", (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)

            with self.lock:
                self.latest_frame = overlay
                self.latest_metrics = {
                    "pos_bpm": bpm_pos, "model_bpm": calibrated_model_bpm, "fused_bpm": fused_bpm,
                    "display_bpm": display_bpm, "confidence": conf_pos, "cpu": cpu, "fps": measured_fps, "ts": time.time()
                }

            if self.logging:
                row = {"timestamp": time.time(), "pos_bpm": bpm_pos, "model_bpm": calibrated_model_bpm, "fused_bpm": fused_bpm,
                       "display_bpm": display_bpm, "confidence": conf_pos, "cpu": cpu, "fps": measured_fps}
                with self.log_lock:
                    self.log_rows.append(row)

            # diagnostics (safe)
            _safe_print_model_diag(raw0, calibrated_model_bpm, bpm_pos, conf_pos)

            # Debug print for POS unsticking
            try:
                print(f"[POS ROI Mean] {np.mean(roi_small):.4f}")
            except Exception:
                pass

            time.sleep(0.01)

    def get_frame_and_info(self, encode_jpeg=True):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            metrics = dict(self.latest_metrics)
        if frame is None:
            return None, {}
        if encode_jpeg:
            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                return None, metrics
            return buf.tobytes(), metrics
        else:
            return frame, metrics

    def start_logging(self):
        with self.log_lock:
            self.log_rows = []
            self.logging = True

    def stop_logging(self):
        with self.log_lock:
            self.logging = False
            df = pd.DataFrame(self.log_rows)
            buf = io.BytesIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            return buf

    def stop(self):
        self.stop_flag.set()
        try:
            if self.cam is not None:
                self.cam.release()
        except Exception:
            pass
