import os
import cv2
import numpy as np
import tensorflow as tf

# --------------------------------------------------
# PATHS (ABSOLUTE, WINDOWS SAFE)
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODULE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_PATH = os.path.join(
    EMOTION_MODULE_DIR,
    "checkpoints",
    "emotion_epoch_01.keras"
)

LABELS_PATH = os.path.join(
    EMOTION_MODULE_DIR,
    "checkpoints",
    "labels.txt"
)


class EmotionModel:
    def __init__(self):
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        if not os.path.isfile(LABELS_PATH):
            raise FileNotFoundError(
                f"labels.txt not found at:\n{LABELS_PATH}\n"
                "Run create_labels.py once."
            )

        self.model = tf.keras.models.load_model(MODEL_PATH)

        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f if line.strip()]

        output_classes = self.model.output_shape[-1]
        if len(self.labels) != output_classes:
            raise ValueError(
                f"Labels ({len(self.labels)}) != model outputs ({output_classes})"
            )

        print("[EmotionModel] Loaded model + labels")

    # --------------------------------------------------
    def preprocess(self, face):
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)   # (48,48,1)
        face = np.expand_dims(face, axis=0)    # (1,48,48,1)

        return face

    def predict(self, face):
        blob = self.preprocess(face)
        preds = self.model.predict(blob, verbose=0)[0]

        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        return self.labels[idx], confidence
