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

# Upload video
uploaded_file = st.file_uploader("Upload a Traffic Video", type=["mp4", "avi"])

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")

model = load_model()

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(uploaded_file)

    st.success("✅ Video uploaded. Processing...")

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    lane_width = frame_width // 3

    # Output video setup
    output_path = os.path.join(tempfile.gettempdir(), "processed_output.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    vehicle_classes = [2, 3, 5, 7]
    signal_colors = {'Green': (0, 255, 0), 'Red': (0, 0, 255), 'Yellow': (0, 255, 255)}

    frame_count = 0
    prev_active_lane = -1
    snapshot_frames = []

    stframe = st.empty()  # For real-time frame display
    chart_placeholder = st.empty()  # For live bar chart

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 200:  # Limit frames for speed
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

        # Snapshot on lane change
        if max_lane != prev_active_lane:
            snapshot_frames.append(frame.copy())
            prev_active_lane = max_lane

        # Draw signals and text
        for i in range(3):
            cx = lane_width * i + lane_width // 2
            cv2.circle(frame, (cx, 30), 15, signal_colors[lane_signals[i]], -1)
            cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}", (lane_width * i + 10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        overlay_text = f"Active Lane: {max_lane+1} ({lane_signals[max_lane]})"
        cv2.putText(frame, overlay_text, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Write output video
        out.write(frame)

        # Real-time traffic bar chart
        lane_df = pd.DataFrame({'Lane': ['1', '2', '3'], 'Vehicles': lane_counts})
        bar_chart = alt.Chart(lane_df).mark_bar().encode(
            x='Lane', y='Vehicles', color=alt.Color('Lane', scale=alt.Scale(scheme='category10'))
        ).properties(width=400, height=300)
        chart_placeholder.altair_chart(bar_chart, use_container_width=True)

        # Show current frame in app
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Frame {frame_count}", use_container_width=True)

        frame_count += 1

    cap.release()
    out.release()

    st.success("✅ Processing complete! Download or view below:")

    with open(output_path, "rb") as file:
        btn = st.download_button(label="📥 Download Processed Video",
                                 data=file,
                                 file_name="ai_traffic_signal_output.avi",
                                 mime="video/avi")

    st.video(output_path)

    # Show snapshots of lane changes
    if snapshot_frames:
        st.subheader("📸 Lane Change Snapshots")
        for idx, snap in enumerate(snapshot_frames):
            st.image(cv2.cvtColor(snap, cv2.COLOR_BGR2RGB), caption=f"Snapshot {idx+1}", use_container_width=True)
            st.download_button(label=f"Download Snapshot {idx+1}",
                               data=cv2.imencode('.jpg', snap)[1].tobytes(),
                               file_name=f"snapshot_{idx+1}.jpg",
                               mime='image/jpeg')
