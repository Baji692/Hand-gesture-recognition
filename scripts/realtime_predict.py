# scripts/realtime_predict.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
from tensorflow.keras.models import load_model

from hand_features import mp_landmarks_to_np, normalize_landmarks, count_fingers

# Paths
MODEL_PATH = "models/gesture_model.h5"
ENCODER_PATH = "models/label_encoder.joblib"
SCALER_PATH = "models/scaler.joblib"

# Load model + encoder + scaler
model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# Mediapipe
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# smoothing deques (store full class labels like 'right_palm')
smooth_left = deque(maxlen=5)
smooth_right = deque(maxlen=5)

# helper: infer gesture from finger states + geometry (returns base gesture e.g. 'palm', 'fist', 'ok', or None)
def infer_from_fingers(states, pts):
    # states: [thumb, index, middle, ring, pinky] (1=open, 0=closed)
    s = states
    total = sum(s)

    # exact matches first
    if s == [1,1,1,1,1]:
        return "palm"
    if s == [0,0,0,0,0]:
        return "fist"
    if s == [0,1,0,0,0]:
        return "pointing"
    if s == [0,1,1,0,0]:
        return "peace"
    if s[1] == 1 and s[4] == 1 and total == 2:
        return "rock"
    if s[0] == 1 and s[4] == 1 and total == 2:
        return "call"
    if s[0] == 1 and total == 1:
        # thumb only => thumbs_up (note: could be rotated, but good heuristic)
        return "thumbs_up"

    # OK sign detection: thumb and index tip are very close AND the other three are open
    # pts is (21,3) normalized coords
    try:
        thumb_tip = pts[4]
        index_tip = pts[8]
        thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
        # threshold tuned for normalized coords ~ 0.04..0.06
        if thumb_index_dist < 0.05 and s[2] == 1 and s[3] == 1 and s[4] == 1:
            return "ok"
    except Exception:
        pass

    # fallback none
    return None

# top-k helper for debug
def topk_labels(probs, k=3):
    idxs = np.argsort(probs)[::-1][:k]
    return [(encoder.inverse_transform([i])[0], float(probs[i])) for i in idxs]

def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # mirror for natural interaction
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands_detector.process(rgb)

        # build batch for all detected hands (reduce model predict overhead)
        items = []  # tuples (hand_landmarks, handedness_label, pts_np, norm_flat)
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hlab = handedness.classification[0].label  # 'Left' or 'Right'
                pts = mp_landmarks_to_np(hand_lms)         # (21,3)
                norm = normalize_landmarks(pts).flatten()  # (63,)
                items.append((hand_lms, hlab, pts, norm))

        # predict in batch if any hands
        if items:
            X = np.stack([it[3] for it in items], axis=0)
            Xs = scaler.transform(X)
            probs_batch = model.predict(Xs, verbose=0)  # shape (n_hands, n_classes)
        else:
            probs_batch = []

        # iterate hands and decide label
        for i, (hand_lms, hlab, pts, norm) in enumerate(items):
            probs = probs_batch[i]
            pred_idx = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])
            pred_label_full = encoder.inverse_transform([pred_idx])[0]  # e.g. 'right_fist' or 'left_palm'
            # model base (without handedness prefix) if label uses "left_" or "right_"
            if "_" in pred_label_full:
                _, pred_base = pred_label_full.split("_", 1)
            else:
                pred_base = pred_label_full

            # finger states + rule inference
            states = count_fingers(pts, handedness=hlab)  # returns list [0/1,...]
            rule_base = infer_from_fingers(states, pts)   # e.g. 'fist' or None

            # construct rule_full if available (prefix with hand)
            rule_full = f"{hlab.lower()}_{rule_base}" if rule_base else None

            # Decide final label using override logic:
            # - If rule_full exists and in encoder classes and model_conf < 0.7 => use rule
            # - If rule_full exists and equals model base => use model (confidence consistent)
            # - Else if model_conf >= 0.7 => trust model
            # - Else if rule_full exists in encoder classes => use rule
            # - Else Unknown
            final_label = "Unknown"
            if pred_conf >= 0.7:
                final_label = pred_label_full
            else:
                if rule_full and rule_full in encoder.classes_:
                    final_label = rule_full
                elif pred_conf >= 0.4:
                    final_label = pred_label_full
                else:
                    final_label = "Unknown"

            # Append to smoothing deque (only append meaningful labels)
            if final_label != "Unknown":
                if hlab == "Left":
                    smooth_left.append(final_label)
                    smoothed = max(set(smooth_left), key=smooth_left.count)
                else:
                    smooth_right.append(final_label)
                    smoothed = max(set(smooth_right), key=smooth_right.count)
            else:
                # fallback to last smoothed if available
                if hlab == "Left":
                    smoothed = max(set(smooth_left), key=smooth_left.count) if smooth_left else "Unknown"
                else:
                    smoothed = max(set(smooth_right), key=smooth_right.count) if smooth_right else "Unknown"

            # bbox + drawing (lightweight)
            coords = [(int(lm.x * W), int(lm.y * H)) for lm in hand_lms.landmark]
            x_min, y_min = min([c[0] for c in coords]), min([c[1] for c in coords])
            x_max, y_max = max([c[0] for c in coords]), max([c[1] for c in coords])
            cv2.rectangle(frame, (x_min - 5, y_min - 5), (x_max + 5, y_max + 5), (0,255,0), 2)

            # draw landmarks (small circles)
            for (cx, cy) in coords:
                cv2.circle(frame, (cx, cy), 3, (0,255,0), -1)

            # overlay text
            cv2.putText(frame, f"{hlab} {smoothed} ({pred_conf:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Open:{sum(states)} {states}", (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # DEBUG: print top3 model guesses and rule candidate
            top3 = topk_labels(probs, k=3)
            print(f"[DEBUG] hand={hlab} model_top3={top3} rule={rule_full} final={final_label}")

        cv2.imshow("Realtime Gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
