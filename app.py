import streamlit as st
import cv2
import tempfile
import os

st.set_page_config(page_title="RoadSense India", layout="wide")

st.title("RoadSense India")
st.markdown("""
Welcome to **RoadSense India**.

This application will analyze recorded road videos, detect pothole candidates, add visible traffic context, remove duplicate sightings, and produce a repair/inspection-priority report.

**ML analysis is not connected yet.**
""")

uploaded_file = st.file_uploader("Upload a road video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.write(f"**File name:** {uploaded_file.name}")
    st.write(f"**File size:** {uploaded_file.size / (1024 * 1024):.2f} MB")

    # Save uploaded file to a temporary file to read with OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name

    try:
        cap = cv2.VideoCapture(temp_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            st.write(f"**Resolution:** {width}x{height}")
            st.write(f"**FPS:** {fps:.2f}")
            st.write(f"**Duration:** {duration:.2f} seconds")
        cap.release()
    finally:
        os.remove(temp_path)

    st.video(uploaded_file)
