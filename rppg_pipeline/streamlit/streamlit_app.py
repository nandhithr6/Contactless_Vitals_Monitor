# streamlit_demo/streamlit_app.py
"""
Run with:
    streamlit run streamlit_demo/streamlit_app.py

This UI starts/stops the Processor, allows uploading an ONNX or using the existing ONNX in repo,
starts/stops logging and downloads CSV of logs.
"""

import streamlit as st
import numpy as np
import tempfile
import time
from core import Processor
import os

st.set_page_config(layout="wide", page_title="rPPG Streamlit Demo")

st.title("rPPG Live demo — POS + ONNX refinement (CPU)")

# Sidebar
st.sidebar.header("Settings & Controls")
camera_idx = st.sidebar.number_input("Camera index", value=0, min_value=0, max_value=5, step=1)
onnx_upload = st.sidebar.file_uploader("Upload ONNX model (optional)", type=['onnx'])
use_repo_model = st.sidebar.checkbox("Use ONNX from repo (rppg_pipeline/models/model.onnx)", value=True)
buffer_seconds = st.sidebar.slider("Buffer seconds (latency vs stability)", 6, 24, 12)
fps = st.sidebar.number_input("Target FPS", value=30, min_value=5, max_value=60, step=1)
start_button = st.sidebar.button("Start demo")
stop_button = st.sidebar.button("Stop demo")
start_log = st.sidebar.button("Start logging")
stop_log = st.sidebar.button("Stop logging & download")

# session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'onnx_temp' not in st.session_state:
    st.session_state.onnx_temp = None

# If user uploaded ONNX, save to temp and use that
if onnx_upload is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx")
    tmp.write(onnx_upload.getvalue()); tmp.flush()
    st.session_state.onnx_temp = tmp.name
    st.sidebar.success("Uploaded ONNX saved to temp file.")

# determine model path
repo_model_path = os.path.join("rppg_pipeline", "models", "model.onnx")
selected_model = None
if st.session_state.onnx_temp is not None:
    selected_model = st.session_state.onnx_temp
elif use_repo_model and os.path.exists(repo_model_path):
    selected_model = repo_model_path
else:
    selected_model = None

col1, col2 = st.columns((2,1))
with col1:
    img_placeholder = st.empty()
    st.write("Live video with overlays")
with col2:
    fps_metric = st.metric("FPS", "--")
    cpu_metric = st.metric("CPU %", "--")
    fused_metric = st.metric("Fused BPM", "--")
    pos_metric = st.metric("POS BPM", "--")
    model_metric = st.metric("Model BPM", "--")
    st.write("Confidence")
    conf_bar = st.progress(0)

# Start demo
if start_button:
    # create processor
    st.session_state.processor = Processor(onnx_path=selected_model, camera_idx=camera_idx,
                                           buffer_seconds=buffer_seconds, target_fps=fps)
    st.session_state.processor.start()
    st.success("Processor started. If onnx selected, it's loaded (may take a sec).")

if stop_button:
    if st.session_state.processor is not None:
        st.session_state.processor.stop()
        st.session_state.processor = None
        st.success("Processor stopped.")

# Logging control
if start_log:
    if st.session_state.processor is not None:
        st.session_state.processor.start_logging()
        st.sidebar.success("Logging started.")
    else:
        st.sidebar.warning("Start processor first.")

if stop_log:
    if st.session_state.processor is not None:
        csvbuf = st.session_state.processor.stop_logging()
        st.sidebar.download_button("Download CSV", csvbuf, file_name="rppg_logs.csv", mime="text/csv")
    else:
        st.sidebar.warning("No active logging found.")

# Main display loop
def update_display():
    if st.session_state.processor is None:
        # show black image
        img_placeholder.image(np.zeros((480,640,3), dtype=np.uint8), channels="BGR")
        return
    img_bytes, metrics = st.session_state.processor.get_frame_and_info()
    if img_bytes is None:
        return
    img_placeholder.image(img_bytes, channels="BGR")
    if metrics:
        fps_metric.metric("FPS", f"{metrics.get('fps',0):.1f}")
        cpu_metric.metric("CPU %", f"{metrics.get('cpu',0):.0f}")
        fused_bpm = metrics.get('fused_bpm', None)
        pos_bpm = metrics.get('pos_bpm', None)
        model_bpm = metrics.get('model_bpm', None)
        fused_metric.metric("Fused BPM", f"{fused_bpm:.1f}" if fused_bpm is not None else "--")
        pos_metric.metric("POS BPM", f"{pos_bpm:.1f}" if pos_bpm is not None else "--")
        model_metric.metric("Model BPM", f"{model_bpm:.1f}" if model_bpm is not None else "--")
        conf = metrics.get('confidence', 0.0)
        conf_bar.progress(min(100, int(conf*100)))

# Streamlit runs top-to-bottom. To keep updating, re-run periodically.
while True:
    update_display()
    time.sleep(0.06)
