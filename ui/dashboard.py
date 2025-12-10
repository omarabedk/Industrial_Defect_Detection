import streamlit as st
from PIL import Image
import requests
import io
import os

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/detect_defect/"

st.title("Industrial Defect Detection")
st.write("Upload an image to detect defects and view Grad-CAM heatmaps.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Button to detect defects
    if st.button("Detect Defects"):
        with st.spinner("Processing..."):
            files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
            
            try:
                response = requests.post(API_URL, files=files)
                response.raise_for_status()
                result = response.json()
                
                # Open mask and Grad-CAM from returned file paths
                mask_path = result["mask"]
                gradcam_path = result["gradcam"]
                
                if os.path.exists(mask_path) and os.path.exists(gradcam_path):
                    mask = Image.open(mask_path)
                    gradcam = Image.open(gradcam_path)
                    
                    st.image(mask, caption="Predicted Mask", use_column_width=True)
                    st.image(gradcam, caption="Grad-CAM Heatmap", use_column_width=True)
                else:
                    st.error("Output files not found.")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")
