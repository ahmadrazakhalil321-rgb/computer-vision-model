import tempfile
import os
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="YOLO Vision AI — Object & Scene Detector",
    page_icon="🤖",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Cache YOLO model loading to optimize performance
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

st.title("🤖 YOLO Vision AI — Object Detection for Images & Videos")
st.write("Upload a picture or video file to detect objects with bounding boxes drawn directly onto your input media.")

# Sidebar Configuration
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.3, step=0.05)

# Universal Media Uploader
uploaded_file = st.file_uploader(
    "Upload Media File (Images: JPG, PNG | Videos: MP4, MOV, AVI)",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
)

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    if file_type.startswith("image"):
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input Image", use_column_width=True)
            
        with col2:
            if st.button("Run YOLO Detection", type="primary"):
                with st.spinner("Processing image with YOLO..."):
                    results = model(image, conf=confidence_threshold)
                    res_plotted = results[0].plot()
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    
                    st.image(res_rgb, caption="Detected Objects Output", use_column_width=True)
                    
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        st.success(f"Detected {len(boxes)} object(s) successfully!")
                    else:
                        st.warning("No objects detected above the confidence threshold.")

    else:
        st.video(uploaded_file)
        
        if st.button("Run YOLO Video Object Tracking", type="primary"):
            with st.spinner("Processing video frames with YOLO... (This may take a moment)"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()
                
                cap = cv2.VideoCapture(tfile.name)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if fps == 0:
                    fps = 30

                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = model(frame, conf=confidence_threshold, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    out.write(annotated_frame)
                    frame_count += 1
                    
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()
                out.release()
                os.unlink(tfile.name)

                st.success("Video processing complete!")
                st.video(output_path)
