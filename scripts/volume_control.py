import cv2
import mediapipe as mp
import numpy as np

# Windows volume control (pycaw)
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Setup system volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_system_volume_percent():
    """Read system volume (0–100) using scalar interface."""
    return int(volume.GetMasterVolumeLevelScalar() * 100)

def set_system_volume_percent(percent):
    """Set system volume using scalar (0.0–1.0)."""
    percent = np.clip(percent, 0, 100) / 100.0
    volume.SetMasterVolumeLevelScalar(percent, None)

def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for selfie-view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Process with Mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Thumb tip and index tip
                    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # Convert to pixel coordinates
                    x1, y1 = int(thumb.x * w), int(thumb.y * h)
                    x2, y2 = int(index.x * w), int(index.y * h)

                    # Draw
                    cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
                    cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Distance between fingers
                    dist = np.linalg.norm([x2 - x1, y2 - y1])

                    # Map distance → volume percentage (0–100)
                    vol_percent = np.interp(dist, [30, 200], [0, 100])
                    set_system_volume_percent(vol_percent)

            # Always show actual system volume
            vol_percent = get_system_volume_percent()

            # --- Draw Volume Bar ---
            bar_x = w - 100
            bar_y1 = 100
            bar_y2 = h - 100

            # Outline of bar
            cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + 50, bar_y2), (255, 255, 255), 2)

            # Fill bar
            filled_y = int(bar_y2 - (vol_percent / 100) * (bar_y2 - bar_y1))
            cv2.rectangle(frame, (bar_x, filled_y), (bar_x + 50, bar_y2), (0, 255, 0), -1)

            # Volume text
            cv2.putText(frame, f"{vol_percent}%", (bar_x - 10, bar_y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Hand Gesture Volume Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
