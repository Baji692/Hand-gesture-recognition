from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    X_train = np.load("X_train.npy")
    X_test  = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test  = np.load("y_test.npy")

    input_dim = X_train.shape[1]  # 63
    num_classes = int(np.max(y_train)) + 1

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.35),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
        keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "gesture_model.h5"),
                                        monitor="val_accuracy", save_best_only=True)
    ]

    print("🚀 Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=120,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test accuracy: {test_acc:.3f}")

    model.save(MODEL_DIR / "gesture_model.h5")
    print("💾 Saved model to models/gesture_model.h5")

if __name__ == "__main__":
    main()
