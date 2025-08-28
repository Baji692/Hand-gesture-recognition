import numpy as np

# Landmark indices (MediaPipe Hands reference)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

THUMB_IP = 3
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18


def mp_landmarks_to_np(hand_landmarks):
    """Convert MediaPipe NormalizedLandmarkList to numpy array (21,3)."""
    pts = np.array([[lm.x, lm.y, lm.z]
                   for lm in hand_landmarks.landmark], dtype=np.float32)
    return pts


def normalize_landmarks(pts):
    """Normalize landmarks: translate wrist to origin & scale."""
    pts = pts - pts[WRIST]  # move wrist to origin
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    return pts


def count_fingers(pts, handedness="Right"):
    """
    Count open fingers given 21 landmarks (numpy array).
    Returns [thumb, index, middle, ring, pinky] as 1=open, 0=closed.
    Uses distance and relative positions for thumb.
    """

    fingers = []

    # --- Thumb logic ---
    # Compare distance between thumb tip and wrist vs thumb IP and wrist
    dist_tip_wrist = np.linalg.norm(pts[THUMB_TIP] - pts[WRIST])
    dist_ip_wrist = np.linalg.norm(pts[THUMB_IP] - pts[WRIST])

    # If tip is much farther than IP → thumb is extended
    thumb_open = 1 if dist_tip_wrist > dist_ip_wrist * 1.2 else 0
    fingers.append(thumb_open)

    # --- Other fingers ---
    fingers.append(1 if pts[INDEX_TIP][1] < pts[INDEX_PIP][1] else 0)
    fingers.append(1 if pts[MIDDLE_TIP][1] < pts[MIDDLE_PIP][1] else 0)
    fingers.append(1 if pts[RING_TIP][1] < pts[RING_PIP][1] else 0)
    fingers.append(1 if pts[PINKY_TIP][1] < pts[PINKY_PIP][1] else 0)

    return fingers
