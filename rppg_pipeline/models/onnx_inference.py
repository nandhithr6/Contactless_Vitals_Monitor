import json, os, numpy as np, onnxruntime as ort

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")
_STATE_PATH = os.path.join(os.path.dirname(__file__), "state.json")

class MERPPGModel:
    def __init__(self, model_path=_MODEL_PATH, state_path=_STATE_PATH, provider=None):
        sess_opts = ort.SessionOptions()
        self.sess = ort.InferenceSession(model_path, sess_options=sess_opts)
        self.input_meta = {i.name: i for i in self.sess.get_inputs()}
        self.output_meta = [o.name for o in self.sess.get_outputs()]
        self.state = {}
        if os.path.exists(state_path):
            import json
            with open(state_path, "r") as f:
                j = json.load(f)
            for k, v in j.items():
                self.state[k] = np.asarray(v, dtype=np.float32)

    def _prep_face(self, img):
        import cv2
        f = img.astype(np.float32)
        if f.max() > 1.5:
            f /= 255.0
        if f.shape[:2] != (36, 36):
            f = cv2.resize(f, (36, 36))
        return f.reshape(1, 1, 36, 36, 3).astype(np.float32)

    def step(self, face_img):
        face = self._prep_face(face_img)
        feed = {}
        for name in self.input_meta:
            if name == "arg_0.1" or name.startswith("arg_0"):
                feed[name] = face
            elif name in self.state:
                feed[name] = self.state[name]
            else:
                shape = [1 if isinstance(s, str) or s == 0 else s for s in self.input_meta[name].shape]
                feed[name] = np.zeros(shape, dtype=np.float32)
        outs = self.sess.run(None, feed)
        for i, name in enumerate(self.output_meta):
            if name in self.state:
                self.state[name] = outs[i]
        raw0 = None
        try:
            arr = np.asarray(outs[0])
            if arr.size == 1:
                raw0 = float(arr)
        except Exception:
            pass
        return raw0, {self.output_meta[i]: outs[i] for i in range(len(outs))}

_singleton = None
def get_model():
    global _singleton
    if _singleton is None:
        _singleton = MERPPGModel()
    return _singleton

def onnx_step(face_img):
    m = get_model()
    return m.step(face_img)