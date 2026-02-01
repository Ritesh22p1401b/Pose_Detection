import os
import cv2
import numpy as np
import tensorflow as tf


class GenderAgeAdapter:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            base_dir, "model", "utkface_age_gender_final.h5"
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"[GenderAgeAdapter ERROR] Model not found: {model_path}"
            )

        print("[GenderAgeAdapter] Loading model...")
        self.model = tf.keras.models.load_model(
            model_path,
            compile=False   # 🔴 CRITICAL FIX FOR KERAS 3.x
        )
        print("[GenderAgeAdapter] Model loaded")

        self.gender_map = {0: "Male", 1: "Female"}

    # --------------------------------------------------
    def preprocess(self, face_img):
        if face_img is None or face_img.size == 0:
            return None

        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype("float32") / 255.0
        return np.expand_dims(face_img, axis=0)

    # --------------------------------------------------
    def predict(self, face_img):
        try:
            inp = self.preprocess(face_img)
            if inp is None:
                return None, "Unknown"

            preds = self.model.predict(inp, verbose=0)

            # Multi-output model
            if isinstance(preds, (list, tuple)) and len(preds) == 2:
                age_pred, gender_pred = preds
            else:
                age_pred = preds[:, 0:1]
                gender_pred = preds[:, 1:]

            age = int(np.clip(age_pred[0][0], 0, 120))

            if gender_pred.shape[-1] == 1:
                gender_idx = int(gender_pred[0][0] > 0.5)
            else:
                gender_idx = int(np.argmax(gender_pred[0]))

            gender = self.gender_map.get(gender_idx, "Unknown")
            return age, gender

        except Exception as e:
            print("[GenderAgeAdapter ERROR]", e)
            return None, "Unknown"
