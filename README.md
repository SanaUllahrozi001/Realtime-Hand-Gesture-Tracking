# ✋ Realtime Hand Gesture Tracking using OpenCV & MediaPipe

This project demonstrates **real-time Virtual Writing and track Hand** using your webcam, powered by **OpenCV** and **MediaPipe**.  
It can detect hand movements, recognize hand and track movement like , and even allow interactive writing using finger tracking.

---

## 🎥 Demo

https://github.com/SanaUllahrozi001/Realtime-Hand-Gesture-Tracking/assets/your-demo-video-link

*(Replace this link with your uploaded demo video once available)*

---

## 🧠 Features

✅ Real-time hand and finger tracking  
✅ Write or draw on screen using finger movements ✍️  
✅ Option to toggle “Writing Mode” by pressing the **W** key  
✅ Works with both Laptop Webcam and Mobile Camera (via IP Webcam)  

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenCV** – for image processing & camera access  
- **MediaPipe** – for hand landmarks and gesture recognition  
- **NumPy** – for matrix and image operations  

---

## ⚙️ Installation

### 1️⃣ Clone this Repository
```bash
git clone https://github.com/SanaUllahrozi001/Realtime-Hand-Gesture-Tracking.git
cd Realtime-Hand-Gesture-Tracking
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
conda create -n handtrack python=3.10
conda activate handtrack
```

### 3️⃣ Install Dependencies
```bash
pip install opencv-python mediapipe numpy
```

---

## 🚀 Run the Project

### ▶️ For Laptop Webcam
```bash
python handtrack.py
```

### 📱 For Mobile Camera
1. Install **IP Webcam** app (Android)  
2. Start the camera and copy the stream URL (e.g., `http://192.168.1.10:8080/video`)
3. Replace this line in your code:
   ```python
   cap = cv2.VideoCapture("http://192.168.1.10:8080/video")
   ```

---

## 🧩 Hand Gestures Supported

| Gesture | Meaning |
|----------|----------|
| 👍 | Thumbs Up (Good / Like) |
| ✌️ | Victory / Peace |
| ✋ | Open Palm |
| 👊 | Fist |
| ☝️ | Finger Writing Mode |

---

## 🧑‍💻 Author

**Sana Ullah**  
📧 Email: [sanaullahrozi001@gmail.com]  
🌐 GitHub: [github.com/SanaUllahrozi001](https://github.com/SanaUllahrozi001)  

---

## ⭐ Contribute

If you’d like to improve this project:
1. Fork the repo 🍴  
2. Create a new branch  
3. Submit a pull request 🚀  

---

## 🏁 Future Enhancements

- Add **object detection** using YOLOv8  
- Integrate **face detection** and **age estimation**  
- Control system actions (e.g., volume, brightness) using hand gestures  

---

## 📜 License

This project is open-source under the **MIT License**.
