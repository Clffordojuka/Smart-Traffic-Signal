import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import os
import pandas as pd
import altair as alt

st.set_page_config(page_title="Smart Traffic Signal", layout="wide")
st.title("🚦 Smart Traffic Management System")

# Upload image
uploaded_file = st.file_uploader("Upload a Traffic Image", type=["jpg", "png", "jpeg"])

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8l.pt")  

model = load_model()

if uploaded_file:
    # Decode uploaded file directly without saving to disk
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Failed to load the image. It might be corrupted or unreadable.")
    else:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.success("✅ Image uploaded. Processing...")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run YOLO detection
        results = model(image_rgb, verbose=False, conf=0.15, iou=0.4)[0] 

        # Draw detections
        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_rgb, f"Class: {cls}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display processed image
        st.image(image_rgb, caption="Processed Image", use_container_width=True)

        # Download processed image
        processed_img_name = "processed_image.jpg"
        processed_img_path = os.path.join(tempfile.gettempdir(), processed_img_name)
        cv2.imwrite(processed_img_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        with open(processed_img_path, "rb") as file:
            st.download_button(label="📥 Download Processed Image",
                               data=file,
                               file_name=processed_img_name,
                               mime="image/jpeg")