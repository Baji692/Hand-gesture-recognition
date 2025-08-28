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

##🚀 Usage
1️⃣ Collect Dataset
