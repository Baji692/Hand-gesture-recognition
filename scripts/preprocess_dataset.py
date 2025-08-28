from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

RAW_CSV = Path("data/raw/dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def normalize_row(xs, ys, zs):
    pts = np.stack([xs, ys, zs], axis=1)  # (21,3)
    wrist = pts[0].copy()
    translated = pts - wrist
    ranges = translated[:, :2].max(axis=0) - translated[:, :2].min(axis=0)
    scale = float(np.max(ranges)) if np.max(ranges) > 1e-6 else 1.0
    scaled = translated / scale
    return scaled.reshape(-1)  # (63,)

def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_CSV}. Collect data first.")

    df = pd.read_csv(RAW_CSV)
    # Expect columns: x0..x20,y0..y20,z0..z20,label
    X_list, y = [], []
    for _, row in df.iterrows():
        xs = np.array([row[f"x{i}"] for i in range(21)], dtype=np.float32)
        ys = np.array([row[f"y{i}"] for i in range(21)], dtype=np.float32)
        zs = np.array([row[f"z{i}"] for i in range(21)], dtype=np.float32)
        X_list.append(normalize_row(xs, ys, zs))
        y.append(row["label"])

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N,63)
    y = np.array(y, dtype=str)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # Save to project root (matches your previous layout)
    np.save("X_train.npy", X_train)
    np.save("X_test.npy",  X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy",  y_test)

    print("✅ Preprocess complete.")
    print("Classes:", list(le.classes_))
    print("X shape:", X_scaled.shape)

if __name__ == "__main__":
    main()
