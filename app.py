# app.py
import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 Brain Tumor Detection AI")
st.markdown("### Upload an MRI scan to detect brain tumors")

# Load the model
@st.cache_resource
def load_model():
    try:
        # Load your .pt file directly
        model = YOLO('brain_tumor_model.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is not None:
    st.success("✅ AI Model loaded successfully!")

# File upload
uploaded_file = st.file_uploader(
    "Choose a brain MRI image", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None and model is not None:
    # Display columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original MRI")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🔍 AI Analysis")
        
        if st.button("Detect Tumors", type="primary"):
            with st.spinner("Analyzing MRI..."):
                # Run YOLO detection
                results = model(image)
                
                # Get image with bounding boxes
                result_image = results[0].plot()
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                # Display result
                st.image(result_image_rgb, use_column_width=True)
                
                # Show detection info
                detections = results[0].boxes
                
                if len(detections) > 0:
                    st.error(f"🚨 **TUMOR DETECTED** - {len(detections)} region(s) found")
                    
                    for i, det in enumerate(detections):
                        conf = det.conf.item()
                        st.write(f"Region {i+1}: {conf:.1%} confidence")
                else:
                    st.success("✅ **NO TUMOR DETECTED**")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This AI detects brain tumors in MRI scans using YOLOv8.")
    
    st.header("⚠️ Disclaimer")
    st.warning("For research purposes only. Consult doctors for diagnosis.")
    
    if model is not None:
        st.header("📊 Model Info")
        st.write("Type: YOLOv8")
        st.write("Input size: 640x640")
        st.write("Trained for: Brain tumor detection")