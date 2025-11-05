# rppg_pipeline/streamlit/onnx_state_test.py
import sys, os, json, numpy as np, cv2
import onnxruntime as ort

def load_state(path):
    if not os.path.exists(path):
        print("[test] state.json not found:", path)
        return {}
    with open(path,'r') as f:
        j=json.load(f)
    # convert values to numpy arrays
    out={}
    for k,v in j.items():
        try:
            arr = np.array(v, dtype=np.float32)
        except Exception:
            arr = np.asarray(v)
        out[k]=arr
    return out

def open_cam_get_face_patch(cam_idx=0, target_H=36, target_W=36, wait_frames=6):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError("camera open failed idx="+str(cam_idx))
    frames=[]
    for i in range(wait_frames):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    if len(frames)==0:
        raise RuntimeError("no frames captured")
    # use middle frame for face detect
    frame = frames[len(frames)//2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80,80))
    if len(faces)==0:
        # fallback: center crop
        h,w = frame.shape[:2]
        cx,cy = w//2, h//2
        s = min(h,w)//3
        x,y,wc,hc = cx-s, cy-s, s*2, s*2
        patch = frame[max(0,y):y+hc, max(0,x):x+wc]
    else:
        x,y,wc,hc = faces[0]
        rx = int(x + 0.15*wc); ry = int(y + 0.12*hc)
        rw = int(wc*0.7); rh = int(hc*0.35)
        patch = frame[ry:ry+rh, rx:rx+rw]
    if patch is None or patch.size==0:
        patch = cv2.resize(frame, (target_W, target_H))
    patch = cv2.resize(patch, (target_W,target_H))
    return patch

def main(model_path, state_json_path=None, cam_idx=0):
    print("Loading model:", model_path)
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inputs = sess.get_inputs()
    print("Model inputs (in order):")
    for i,inp in enumerate(inputs):
        print(i, inp.name, inp.shape, inp.type)
    outputs = sess.get_outputs()
    print("Model outputs (in order):")
    for j,out in enumerate(outputs):
        print(j, out.name, out.shape, out.type)

    # load state_json if provided
    state = {}
    if state_json_path:
        state = load_state(state_json_path)
        print("Loaded state.json keys:", list(state.keys())[:20], " total:", len(state.keys()))
    else:
        print("No state_json_path provided; state empty.")

    # prepare face patch
    print("Capturing face patch from camera", cam_idx)
    patch = open_cam_get_face_patch(cam_idx=cam_idx)
    # convert to RGB and normalize to [0,1], reshape to (1,1,H,W,3)
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    face_input = patch_rgb.reshape(1,1,patch_rgb.shape[0], patch_rgb.shape[1], 3)
    print("Face input shape:", face_input.shape, "min/max:", face_input.min(), face_input.max())

    # build feed dict exactly matching model input names
    feed={}
    # map the FIRST input to face_input
    feed[inputs[0].name] = face_input.astype(np.float32)
    # for others: try to find same-named key in state.json; else try default mapping with truncated names
    missing = []
    for inp in inputs[1:]:
        name = inp.name
        if name in state:
            arr = state[name].astype(np.float32)
            # if shape mismatch attempt reshape if same count, else warn
            expected = []
            for d in inp.shape:
                try:
                    expected.append(int(d))
                except Exception:
                    expected.append(1)
            expected = tuple(expected)
            if arr.shape != expected:
                if arr.size == np.prod(expected):
                    feed[name] = arr.reshape(expected).astype(np.float32)
                    print(f"[feed] reshaped state {name} from {arr.shape} -> {expected}")
                else:
                    print(f"[feed] shape mismatch for {name}: state {arr.shape} != expected {expected} -> filling zeros")
                    feed[name] = np.zeros(expected, dtype=np.float32)
            else:
                feed[name] = arr
        else:
            # try to find key by suffix or partial match
            found_key = None
            for k in state.keys():
                if k.endswith(name) or name.endswith(k) or (name in k) or (k in name):
                    found_key = k; break
            if found_key:
                arr = state[found_key].astype(np.float32)
                expected = tuple(int(d) if (isinstance(d,(int,np.integer))) else 1 for d in inp.shape)
                if arr.size == np.prod(expected):
                    feed[name] = arr.reshape(expected).astype(np.float32)
                    print(f"[feed] matched {name} <- {found_key} reshaped {arr.shape} -> {expected}")
                else:
                    print(f"[feed] matched {name} <- {found_key} but size mismatch {arr.shape} vs {expected} -> zeros")
                    feed[name] = np.zeros(expected, dtype=np.float32)
            else:
                # no matching key: fill zeros
                expected = tuple(int(d) if (isinstance(d,(int,np.integer))) else 1 for d in inp.shape)
                feed[name] = np.zeros(expected, dtype=np.float32)
                missing.append((name, expected))
    print("Number of non-face inputs filled:", len(inputs)-1, "missing unique:", len(missing))
    if missing:
        print("Some inputs had no matching state.json key and were zero-filled (first 10):", missing[:10])

    # run single inference
    print("Running single sess.run with feed dict keys:", list(feed.keys())[:10], "...")
    try:
        outs = sess.run(None, feed)
    except Exception as e:
        print("sess.run FAILED:", e)
        return

    print("sess.run OK. Outputs summary:")
    saved = {}
    for i, out in enumerate(outs):
        a = np.asarray(out)
        print(i, "name:", (outputs[i].name if i < len(outputs) else f"out_{i}"), "shape:", a.shape, "min/max/mean:", float(a.min()), float(a.max()), float(a.mean()))
        # print first 8 values
        flat = a.flatten()
        print("  sample:", flat[:8].tolist())
        saved[f"out_{i}"] = a
    # save outputs for inspection
    np.savez("me_onnx_probe_outputs.npz", **saved)
    print("Saved outputs to me_onnx_probe_outputs.npz")
    print("If sess.run returned a scalar in outputs[0], that's candidate BPM or freq. Inspect outputs[0] above.")
    return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python onnx_state_test.py path/to/model.onnx [path/to/state.json] [cam_idx]")
        sys.exit(1)
    model = sys.argv[1]
    state = sys.argv[2] if len(sys.argv) > 2 else None
    cam = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(model, state, cam)
