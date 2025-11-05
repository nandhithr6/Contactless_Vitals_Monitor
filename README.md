# 🩺 Contactless Vitals Monitor

A real-time **contactless heart-rate monitor** using **remote photoplethysmography (rPPG)**.  
Built with **OpenCV**, **ONNXRuntime**, and **Streamlit** — runs on CPU using webcam input.

---

## ⚙️ Features
- Face detection + ROI extraction (OpenCV)
- POS-based rPPG signal estimation  
- ONNX ME-rPPG model inference  
- Dual-stream **fusion** (POS + Model) with auto-calibration  
- Smoothed & stable BPM output  
- Streamlit UI for live video and vitals overlay

---

## 🧠 Tech Stack
- Python 3.10+
- Streamlit
- OpenCV
- NumPy, SciPy
- ONNXRuntime
- psutil, pandas

---

pip install -r requirements.txt
streamlit run rppg_pipeline/streamlit/streamlit_app.py
