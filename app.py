import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import altair as alt
import time

st.set_page_config(page_title="Smart Traffic Signal", layout="wide")
st.title("🚦 Smart Traffic Management System (Live Webcam)")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")

model = load_model()

# Initialize webcam
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
lane_width = frame_width // 3

# Vehicle classes for YOLO detection
vehicle_classes = [2, 3, 5, 7]
signal_colors = {'Green': (0, 255, 0), 'Red': (0, 0, 255), 'Yellow': (0, 255, 255)}

stframe = st.empty()
chart_placeholder = st.empty()
stop_button = st.button("Stop Webcam")

prev_active_lane = -1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or stop_button:
        break

    results = model(frame, verbose=False)[0]
    lane_counts = [0, 0, 0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            lane_idx = min(cx // lane_width, 2)
            lane_counts[lane_idx] += 1

    max_lane = np.argmax(lane_counts)
    lane_signals = ['Red', 'Red', 'Red']
    lane_signals[max_lane] = 'Green'

    # Draw signals and text
    for i in range(3):
        cx = lane_width * i + lane_width // 2
        cv2.circle(frame, (cx, 30), 15, signal_colors[lane_signals[i]], -1)
        cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}", (lane_width * i + 10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    overlay_text = f"Active Lane: {max_lane+1} ({lane_signals[max_lane]})"
    cv2.putText(frame, overlay_text, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Real-time traffic bar chart
    lane_df = pd.DataFrame({'Lane': ['1', '2', '3'], 'Vehicles': lane_counts})
    bar_chart = alt.Chart(lane_df).mark_bar().encode(
        x='Lane', y='Vehicles', color=alt.Color('Lane', scale=alt.Scale(scheme='category10'))
    ).properties(width=400, height=300)
    chart_placeholder.altair_chart(bar_chart, use_container_width=True)

    # Show current frame in app
    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Live Traffic Feed", use_container_width=True)
    
cap.release()
st.success("✅ Webcam stopped!")