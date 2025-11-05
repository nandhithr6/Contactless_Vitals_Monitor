# rppg_pipeline/models/onnx_inference.py
"""
Stateful ME-rPPG ONNX wrapper.
Loads model.onnx + state.json once. Keeps state arrays in memory and updates them each step.
Provides get_model() singleton and a step(face_img) method.
"""

import os
import json
import numpy as np
import onnxruntime as ort

# default paths relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")
STATE_PATH = os.path.join(os.path.dirname(__file__), "state.json")

class MERPPGModel:
    def __init__(self, model_path=MODEL_PATH, state_path=STATE_PATH, provider=None):
        sess_opts = ort.SessionOptions()
        # avoid spamming logs
        sess_opts.log_severity_level = 3
        if provider is None:
            self.sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=['CPUExecutionProvider'])
        else:
            self.sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=provider)

        # collect I/O metadata
        self.inputs_meta = {inp.name: inp for inp in self.sess.get_inputs()}
        self.outputs_meta = [out.name for out in self.sess.get_outputs()]

        # load state.json once and convert to numpy arrays
        self.state = {}
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    j = json.load(f)
                for k, v in j.items():
                    try:
                        arr = np.asarray(v, dtype=np.float32)
                        self.state[k] = arr
                    except Exception:
                        # leave None to be zero-filled later
                        self.state[k] = None
            except Exception:
                # fallback: empty state
                self.state = {}
        else:
            # no state.json - initialize empty
            self.state = {}

    def _prepare_face(self, face_img):
        """face_img: HxWx3 BGR or RGB, uint8 or float. Return float32 RGB normalized [0,1] shape (1,1,36,36,3)."""
        import cv2
        arr = np.asarray(face_img)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        # assume BGR from OpenCV, convert to RGB
        if arr.shape[2] == 3:
            # if values look like 0..255 (uint8) then convert, else assume already float [0..1]
            if arr.dtype == np.uint8 or arr.max() > 1.5:
                arr = arr[..., ::-1]  # BGR->RGB
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr[..., ::-1].astype(np.float32)
        # resize to 36x36 if necessary
        if arr.shape[0] != 36 or arr.shape[1] != 36:
            arr = (cv2.resize(arr, (36, 36))).astype(np.float32)
        arr = arr.reshape(1, 1, 36, 36, 3).astype(np.float32)
        return arr

    def step(self, face_img):
        """
        Run one step of ONNX model.
        face_img: HxWx3 BGR or RGB (OpenCV style)
        Returns: (raw_out0_scalar_or_None, outputs_dict)
        and updates self.state in-place using model outputs where applicable.
        """
        face = self._prepare_face(face_img)

        # build feed dict
        feed = {}
        for name, meta in self.inputs_meta.items():
            # first input is face (arg_0.1 or similar)
            if name.startswith("arg_0") or name == "arg_0.1" or name.lower().startswith("input"):
                feed[name] = face.astype(np.float32)
                continue
            # if we have that key in state and it's not None, use it
            if name in self.state and self.state[name] is not None:
                feed[name] = np.asarray(self.state[name], dtype=np.float32)
                continue
            # else zero-fill according to expected shape (replace dynamic dims with 1)
            shape = []
            for d in meta.shape:
                try:
                    d_int = int(d)
                    if d_int <= 0:
                        shape.append(1)
                    else:
                        shape.append(d_int)
                except Exception:
                    shape.append(1)
            try:
                feed[name] = np.zeros(tuple(shape), dtype=np.float32)
            except Exception:
                feed[name] = np.array(0.0, dtype=np.float32)

        # run model
        outs = self.sess.run(None, feed)

        # map outputs to names
        out_map = {nm: outs[idx] for idx, nm in enumerate(self.outputs_meta)}

        # update state: if any output name matches an input/state key, update it
        for k in list(self.state.keys()):
            if k in out_map:
                try:
                    self.state[k] = np.asarray(out_map[k], dtype=np.float32)
                except Exception:
                    pass

        # try to extract outputs[0] as scalar raw0
        raw0 = None
        try:
            first = outs[0]
            arr = np.asarray(first)
            if arr.size == 1:
                raw0 = float(arr.flatten()[0])
        except Exception:
            raw0 = None

        return raw0, out_map

# singleton accessor
_SINGLETON = None
def get_model():
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = MERPPGModel()
    return _SINGLETON

def onnx_step(face_img):
    m = get_model()
    return m.step(face_img)
