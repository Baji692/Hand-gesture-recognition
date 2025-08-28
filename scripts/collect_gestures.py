import argparse
import csv
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

# Paths
CSV_PATH = Path("data/raw/dataset.csv")
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

HEADER = (
    [f"x{i}" for i in range(21)] +
    [f"y{i}" for i in range(21)] +
    [f"z{i}" for i in range(21)] +
    ["label"]
)

def ensure_header():
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)

def write_sample(pts, label):
    row = pts[:,0].tolist() + pts[:,1].tolist() + pts[:,2].tolist() + [label]
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Auto-capture MediaPipe hand landmarks.")
    parser.add_argument("--gesture", required=True,
                        help="Gesture name (e.g., palm, thumbs_up, rock, call, pointing, peace, ok).")
    parser.add_argument("--hand", required=True, choices=["left", "right"],
                        help="Which hand you are recording (left/right).")
    parser.add_argument("--samples", type=int, default=500, help="Number of frames to auto-capture when pressing A.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    args = parser.parse_args()

    label = f"{args.hand}_{args.gesture}"

    ensure_header()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print(f"▶ Ready to collect '{label}'.")
    print("Press A to start auto-capture, Q to quit this window.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        recording = False
        saved = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

                if recording and saved < args.samples:
                    write_sample(pts, label)
                    saved += 1
                    print(f"Saved {saved}/{args.samples}")
                    time.sleep(0.002)

            # HUD
            htxt = f"Gesture: {label} | Press 'A' to start ({args.samples} frames), 'Q' to quit."
            cv2.putText(frame, htxt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if recording:
                cv2.putText(frame, f"Recording... {saved}/{args.samples}",
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                if saved >= args.samples:
                    cv2.putText(frame, "Done! Press Q to quit or A to record again.",
                                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    recording = False

            cv2.imshow("Collect Gestures", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('a'), ord('A')) and not recording:
                saved = 0
                recording = True
                print(f"▶ Auto-capturing {args.samples} frames for '{label}'...")

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Finished capturing for '{label}'. Data appended to {CSV_PATH}")

if __name__ == "__main__":
    main()
