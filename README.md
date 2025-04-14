# 🚦 Smart Traffic Management System

An intelligent traffic monitoring web application that uses **YOLOv8** object detection to identify and count vehicles in traffic images. Built with **Streamlit**, integrated with **Firebase Realtime Database** for cloud storage, and enhanced with **Altair** visualizations for historical insights.

---

## 📌 Features

- 🔍 **Real-time Vehicle Detection**: Detects cars, motorcycles, buses, and trucks from uploaded images using YOLOv8.
- 💾 **Firebase Integration**: Stores vehicle count and detection metadata (class, confidence, bounding boxes) in Firebase Realtime Database.
- 📊 **Historical Data Visualization**: Displays historical traffic patterns using Altair line charts.
- 📸 **Image Processing and Download**: Annotated image with bounding boxes can be downloaded directly.
- 🧠 **Automatic Class Filtering**: Focuses on COCO vehicle classes only for relevant traffic analysis.

---

## 🛠️ Tech Stack

| Component     | Technology           |
|---------------|----------------------|
| Frontend      | Streamlit            |
| Backend       | Python, OpenCV       |
| AI Model      | YOLOv8 (Ultralytics) |
| Cloud Storage | Firebase Realtime DB |
| Visualization | Altair, Pandas       |

---

## 📂 File Structure

```
smart-traffic-app/
│
├── app.py                  # Main Streamlit application
├── .streamlit/
│   └── secrets.toml        # Firebase credentials (not shared publicly)
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 🚀 How It Works

1. **User uploads a traffic image**.
2. **YOLOv8** detects vehicle objects (car, motorcycle, bus, truck).
3. Detected objects are:
   - Counted and annotated on the image
   - Saved to Firebase with a timestamp
4. A real-time line chart shows **vehicle count trends** over time.
5. Processed image can be downloaded for reference.

---

## 🔧 Firebase Configuration

Store your Firebase credentials in `.streamlit/secrets.toml` as:

```toml
[firebase]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\\n....\\n-----END PRIVATE KEY-----\\n"
client_email = "your-client-email@your-project-id.iam.gserviceaccount.com"
client_id = "your-client-id"
...
```

✅ Secrets are loaded securely using Streamlit Cloud’s secret manager.

---

## ✅ Running Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/Clffordojuka/Smart-Traffic-Signal.git
   cd smart-traffic-app
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   or .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add secrets**:
   Save your Firebase credentials to `.streamlit/secrets.toml`.

5. **Run the app**:
   ```bash
   streamlit run app.py
   ```

---

## 📈 Example Output

- Original and Processed Image display
- Detected classes (car, motorcycle, etc.)
- Bounding boxes and confidence scores
- Realtime chart: 🚗 vehicle counts over time

## 🧪 Sample Detection Output

**Screenshot Image:**

![Uploaded image alongside Detected vechicles](results\sample_image_detected.PNG)

**A video of complete process from image upload to fetched historical data from firebase:**

![A whole process video](results\sample_video.mp4)

## Firebase sample database

![Car attributes](results\car_detected.PNG)

![Truck attribute](results\truck_detected.PNG)

![Bus attributes](results\bus_detected.PNG)

---

## 🛡️ Error Handling

- 🚫 Missing or invalid Firebase credentials
- ❌ Image upload failure (e.g., corrupted files)
- 🔥 Firebase write or read errors
- ℹ️ Feedback and logs provided within the app

---

## 📦 Dependencies

```
streamlit
opencv-python
ultralytics
pandas
altair
numpy<2
torch==2.2.0
torchvision==0.17.0
torchaudio==2.2.0
Pillow
requests
matplotlib
pytz
firebase-admin
```

Use this to generate `requirements.txt`:
```bash
pip freeze > requirements.txt
```

---

## 💡 Future Improvements

- 📹 Video upload & live stream detection
- 🗺️ GPS metadata for geo-tagging detections
- 🔄 Auto-refreshing dashboard
- 🔔 Real-time traffic alert system

---

## 👨‍💻 Author

**Clifford Ojuka**  
🔗 [GitHub](https://github.com/Clffordojuka) | 🐦 [Twitter](https://x.com/TmKojuka) | 🌐 [Portfolio](https://clifford-portfolio.vercel.app)

---
