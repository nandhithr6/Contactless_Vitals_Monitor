# rppg_pipeline/streamlit/onnx_probe.py
import sys, os, time, json, numpy as np, cv2
import onnxruntime as ort

def load_sess(path):
    try:
        return ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print("Failed loading ONNX:", e); raise

def try_run(sess, inp_name, arr):
    try:
        out = sess.run(None, {inp_name: arr.astype(np.float32)})
        a = np.asarray(out[0])
        return {"ok": True, "shape_in": list(arr.shape), "shape_out": list(a.shape),
                "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
                "sample": a.flatten()[:8].tolist()}
    except Exception as e:
        return {"ok": False, "error": str(e), "shape_in": list(arr.shape)}

def build_candidates_from_frames(frames, H=36, W=36):
    """
    frames: list of BGR frames (H_src,W_src,3) latest first or oldest first
    returns dictionary of candidate arrays keyed by (T, norm_mode)
    Norm modes:
      - 'uint8' : raw uint8 values
      - '01'    : /255.0
      - 'mean'  : per-frame mean subtracted (per-channel)
      - 'z'     : global zscore across frames/channels
    Candidate shapes tried: (1, T, H, W, 3)
    """
    cand = {}
    # ensure frames as numpy
    frames_np = []
    for f in frames:
        try:
            r = cv2.resize(f, (W, H))
        except Exception:
            r = np.zeros((H,W,3), dtype=np.uint8)
        frames_np.append(r)
    frames_np = np.stack(frames_np, axis=0)  # (T_src, H, W, 3)
    for T in (1, 4, 8, 12):
        if frames_np.shape[0] < T:
            continue
        seq = frames_np[-T:]  # last T frames
        # candidate normalizations
        # uint8
        cand[(T, 'uint8')] = seq.reshape(1, T, H, W, 3).astype(np.float32)
        # /255
        cand[(T, '01')] = (seq.astype(np.float32)/255.0).reshape(1, T, H, W, 3)
        # per-frame mean subtract (per channel)
        seqf = seq.astype(np.float32)
        seq_ms = np.empty_like(seqf)
        for i in range(seqf.shape[0]):
            for c in range(3):
                ch = seqf[i,:,:,c]
                seq_ms[i,:,:,c] = ch - ch.mean()
        cand[(T, 'mean')] = seq_ms.reshape(1, T, H, W, 3)
        # global zscore
        flat = seqf.reshape(-1).astype(np.float32)
        if flat.std() > 1e-6:
            z = (seqf - flat.mean())/flat.std()
        else:
            z = seqf
        cand[(T, 'z')] = z.reshape(1, T, H, W, 3)
    return cand

def capture_frames_from_camera(n=12, cam_idx=0, wait=0.05):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError("camera open fail idx="+str(cam_idx))
    frames=[]
    for i in range(n):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
        time.sleep(wait)
    cap.release()
    return frames

def main(model_path, frames_dir=None, cam_idx=0):
    sess = load_sess(model_path)
    print("Session loaded. Inputs:")
    for i in sess.get_inputs():
        print("  ", i.name, i.shape, i.type)
    inp_name = sess.get_inputs()[0].name
    print("Using first input:", inp_name)

    # prepare frames
    frames=[]
    if frames_dir:
        # read image files sorted
        files = sorted([os.path.join(frames_dir,f) for f in os.listdir(frames_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        for f in files:
            frames.append(cv2.imread(f))
        if len(frames) == 0:
            print("No frames found in dir:", frames_dir)
    if len(frames) == 0:
        print("Capturing frames from camera", cam_idx)
        frames = capture_frames_from_camera(n=16, cam_idx=cam_idx, wait=0.05)
    print("Frames collected:", len(frames))

    # build candidates using H,W from inspector hint; use 36x36 first
    cand = build_candidates_from_frames(frames, H=36, W=36)
    results = {}
    for key, arr in cand.items():
        print("Trying key", key, "shape", arr.shape, " dtype", arr.dtype)
        r = try_run(sess, inp_name, arr)
        print(" ->", r)
        results[str(key)] = r

    # save results
    out_json = "onnx_probe_results.json"
    with open(out_json, "w") as f:
        json.dump({"model":model_path, "results": results}, f, indent=2)
    print("Saved diagnostic results to", out_json)
    print("If you want, paste the JSON here and I will interpret the best candidate.")
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python onnx_probe.py path/to/model.onnx [frames_dir] [cam_idx]")
        sys.exit(1)
    model = sys.argv[1]
    frames_dir = sys.argv[2] if len(sys.argv)>2 else None
    cam_idx = int(sys.argv[3]) if len(sys.argv)>3 else 0
    main(model, frames_dir, cam_idx)
