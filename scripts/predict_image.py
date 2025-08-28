import absl.logging
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
from hand_features import mp_landmarks_to_np, normalize_landmarks, count_fingers

# Suppress TensorFlow / Mediapipe warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gesture_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

# Load model and encoders
model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def predict_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.5) as hands:

        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            print("⚠️ No hands detected in the image.")
            return

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            pts = mp_landmarks_to_np(hand_landmarks)
            norm = normalize_landmarks(pts).flatten().reshape(1, -1)
            norm = scaler.transform(norm)

            # Predict gesture
            probs = model.predict(norm, verbose=0)[0]
            idx = np.argmax(probs)
            gesture = encoder.inverse_transform([idx])[0]
            conf = probs[idx]

            # Finger counting
            fingers = count_fingers(pts)
            open_count = sum(fingers)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Overlay prediction
            label = f"{handedness.classification[0].label} {gesture} ({conf:.2f})"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.putText(img, f"Fingers: {open_count} {fingers}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            print(f"👉 {handedness.classification[0].label} hand")
            print(f"   Gesture: {gesture} (conf: {conf:.2f})")
            print(f"   Fingers open: {open_count}, States: {fingers}")

        # Show result
    cv2.imshow("Prediction", img)

    # Force window to the front
    cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
    cv2.imshow("Prediction", img)
    cv2.setWindowProperty("Prediction", cv2.WND_PROP_TOPMOST, 1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Open Tkinter file dialog
    root = Tk()
    root.withdraw()  # hide the small tkinter root window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    root.destroy()

    if file_path:
        predict_image(file_path)
    else:
        print("⚠️ No file selected.")
