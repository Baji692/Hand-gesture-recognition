# ✋ Hand Gesture Recognition & Control

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)  
![Mediapipe](https://img.shields.io/badge/Mediapipe-Hands-red)  
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A **Hand Gesture Recognition System** built using **Mediapipe**, **OpenCV**, and **Deep Learning**.  
Recognizes static & dynamic gestures in realtime and demonstrates **Human–Machine Interaction** by controlling system **volume** with hand gestures.

---

## 📌 Features
- 🎥 **Realtime Gesture Recognition** (via webcam).  
- 🖼️ **Static Image Prediction** (upload photo & detect gesture).  
- ✋ Multiple Gestures Supported:  
  - Palm (Open Hand)  
  - Thumbs Up  
  - Pointing (Index)  
  - Peace ✌️  
  - Rock Sign 🤘  
  - Call Me 🤙  
  - OK Sign 👌  
  - Fist ✊  
- 🔢 **Finger Counting** (open vs. closed fingers).  
- 🔊 **Volume Control** with thumb–index distance (synced with system).  
- 📡 Easily extendable to **media, IoT, or gaming control**.  

---

## 📂 Project Directory Structure
```
HAND GESTURE PROJECT
│── data/
│   └── raw/
│       └── dataset.csv
│
│── images/
│
│── models/
│   ├── gesture_model.h5          # Trained deep learning model
│   ├── label_encoder.joblib      # Encoded gesture labels
│   └── scaler.joblib             # Feature scaler
│
│── scripts/
│   ├── collect_gestures.py       # Script to collect gesture images
│   ├── hand_features.py          # Extracts hand landmarks/features
│   ├── predict_image.py          # Predicts gesture from image
│   ├── preprocess_dataset.py     # Preprocesses dataset for training
│   ├── realtime_predict.py       # Real-time gesture prediction
│   ├── train_dl.py               # Trains deep learning model
│   └── volume_control.py         # Controls system volume using gestures
│
│── requirements.txt              # Python dependencies
│── X_train.npy                   # Training features
│── y_train.npy                   # Training labels
│── X_test.npy                    # Test features
│── y_test.npy                    # Test labels
```

---

## 🚀 Installation
**1. Clone this repo:**
- git clone https://github.com/yourusername/hand-gesture-volume-control.git
- cd hand-gesture-volume-control

**2. Install dependencies:**
- pip install -r requirements.txt

---

## 📂 Dataset Collection
Custom dataset created using **webcam auto-capture**:
- Press **A** → start capturing frames (500 samples per gesture).  
- Press **Q** → quit.  
- Separate datasets for **Right Hand** and **Left Hand**.  
- Stored as **NumPy landmark arrays** from Mediapipe’s 21 hand landmarks.  

---

## 🧠 Algorithms & Techniques
- **Hand Landmark Detection** → Mediapipe Hands (21 3D landmarks).  
- **Feature Extraction** → Normalized landmark positions, angles, distances.  
- **Finger Counting** → Rule-based (open/closed detection).  
- **Classifier** → Deep Neural Network (Keras Sequential model).  
- **Action Mapping** → LabelEncoder maps gestures → actions.  
- **System Volume Control** → [pycaw](https://github.com/AndreMiras/pycaw) controls Windows audio.  

---

## 🏋️ Model Training
- Input: 63 features (x, y, z for 21 landmarks).  
- Model:  
  - Dense layers with ReLU  
  - Softmax output for gesture classification  
- Loss: Categorical Crossentropy  
- Optimizer: Adam  
- Output Files:  
  - `models/gesture_model.h5`  
  - `label_encoder.joblib`  

---

## ⚙️ Requirements
- Python **3.9+**  
- Libraries:
  ```bash
  pip install opencv-python mediapipe tensorflow scikit-learn joblib pycaw comtypes numpy

---

## 🚀 Usage
**1️⃣ Collect Dataset**
- python scripts/collect_data.py --gesture palm_right
- (repeat for all gestures & both hands)

**2️⃣ Preprocess**
- python scripts/preprocess.py

**3️⃣ Train**
- python scripts/train_model.py

**4️⃣ Realtime Prediction (Webcam)**
- python scripts/realtime_predict.py

**5️⃣ Static Image Prediction**
- python scripts/predict_image.py

**6️⃣ Volume Control with Gestures**
- python scripts/volume_control.py

---

## 📈 Workflow
- **Data Collection → Preprocessing → Model Training → Gesture Prediction → Volume Control**
- **Uses hand landmarks as input features.**
- **Trained deep learning model predicts gesture → maps to volume change.**

---

## 🔮 Future Enhancements

- **🎵 Media Controls → Play, Pause, Next, Previous.**
- **🖱️ Air Mouse → Cursor & clicks using gestures.**
- **🌐 IoT → Smart home device control.**
- **🧏 Sign Language → Real-time translation into text/speech.**
- **🕹️ Gesture-Based Games.**

---

## 📜 Conclusion

- This project shows how **computer vision + deep learning** enable **real-time human–machine interaction.**
- It serves as a strong base for applications in **accessibility, IoT, robotics, AR/VR, and sign language recognition.**
