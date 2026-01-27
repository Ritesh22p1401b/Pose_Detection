import os
import cv2
import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "..", "checkpoints", "emotion_epoch_01.keras"
)

LABELS_PATH = os.path.join(
    BASE_DIR, "..", "checkpoints", "labels.txt"
)


class EmotionModel:
    def __init__(self):
        # Load trained Keras model
        self.model = tf.keras.models.load_model(MODEL_PATH)

        # Load labels (must be 40)
        with open(LABELS_PATH, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        if len(self.labels) != 40:
            raise ValueError(
                f"Expected 40 emotion labels, got {len(self.labels)}"
            )

        # Optional safety check
        output_classes = self.model.output_shape[-1]
        if output_classes != 40:
            raise ValueError(
                f"Model outputs {output_classes} classes, expected 40"
            )

    def preprocess(self, face):
        """
        IMPORTANT:
        This MUST match training preprocessing.
        Change ONLY if you trained differently.
        """
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)   # (48, 48, 1)
        face = np.expand_dims(face, axis=0)    # (1, 48, 48, 1)

        return face

    def predict(self, face):
        blob = self.preprocess(face)

        preds = self.model.predict(blob, verbose=0)
        preds = preds[0]  # shape: (40,)

        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        return self.labels[idx], confidence
