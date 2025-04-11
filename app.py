import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import os
import pandas as pd
import altair as alt
import json
import tempfile
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

st.set_page_config(page_title="Smart Traffic Signal", layout="wide")
st.title("🚦 Smart Traffic Management System")

@st.cache_resource
def init_firebase():
    
    # Load Firebase config from Streamlit secrets
    firebase_config = dict(st.secrets["firebase"])
    
    # Fix private key formatting
    private_key = firebase_config["private_key"]
    
    # Ensure proper PEM formatting
    if "\\n" in private_key:
        firebase_config["private_key"] = private_key.replace("\\n", "\n")
    
    # Create temporary credentials file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(firebase_config, f, ensure_ascii=False)
        temp_path = f.name
    
    # Initialize Firebase
    cred = credentials.Certificate(temp_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://smart-traffic-system-6efc1-default-rtdb.firebaseio.com/'
    })

init_firebase()

# Upload image
uploaded_file = st.file_uploader("Upload a Traffic Image", type=["jpg", "png", "jpeg"])

@st.cache_resource
def load_model():
    return YOLO("yolov8l.pt")

model = load_model()

if uploaded_file:
    # Process image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Failed to load the image. It might be corrupted or unreadable.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        results = model(image_rgb, verbose=False, conf=0.15, iou=0.4)[0]
        
        # Initialize counters
        vehicle_count = 0
        detection_data = []

        # Process detections
        for box in results.boxes:
            cls = int(box.cls[0])
            # Vehicle classes in COCO dataset: 2(car), 3(motorcycle), 5(bus), 7(truck)
            if cls in [2, 3, 5, 7]:
                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Store detection data
                detection_data.append({
                    "class": model.names[cls],
                    "confidence": float(box.conf[0]),
                    "position": [int(x1), int(y1), int(x2), int(y2)]
                })

                # Draw bounding box
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_rgb, 
                           f"{model.names[cls]} {box.conf[0]:.2f}", 
                           (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)

        # Push to Firebase
        try:
            ref = db.reference('/detections')
            new_entry = ref.push({
                'vehicle_count': vehicle_count,
                'timestamp': datetime.now().isoformat(),
                'detections': detection_data
            })
            st.success("✅ Data successfully pushed to Firebase!")
        except Exception as e:
            st.error(f"🔥 Firebase Error: {str(e)}")

        # Display processed image
        with col2:
            st.image(image_rgb, caption="Processed Image", use_container_width=True)
            st.subheader(f"🚗 Detected Vehicles: {vehicle_count}")

        # Download processed image
        processed_img_name = "processed_image.jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            cv2.imwrite(temp_file.name, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            with open(temp_file.name, "rb") as file:
                st.download_button(
                    label="📥 Download Processed Image",
                    data=file,
                    file_name=processed_img_name,
                    mime="image/jpeg"
                )

        # Show historical data
        st.subheader("Historical Traffic Data")
        try:
            hist_ref = db.reference('/detections')
            historical_data = hist_ref.get()
            
            if historical_data:
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': data['timestamp'],
                        'vehicle_count': data['vehicle_count']
                    } 
                    for key, data in historical_data.items() if data
                ])
                
                # Create chart
                chart = alt.Chart(df).mark_line().encode(
                    x='timestamp:T',
                    y='vehicle_count:Q',
                    tooltip=['timestamp', 'vehicle_count']
                ).properties(width=800, height=400)
                
                st.altair_chart(chart)
            else:
                st.info("No historical data found in Firebase")
                
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
