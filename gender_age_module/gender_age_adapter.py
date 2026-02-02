import os
import numpy as np
import tensorflow as tf
import cv2   # ✅ now available because test venv is used

class GenderAgeAdapter:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            base_dir, "model", "utkface_age_gender_final.h5"
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Gender model not found: {model_path}")

        self.model = tf.keras.models.load_model(model_path, compile=False)

    def preprocess(self, face_img):
        if face_img is None or face_img.size == 0:
            return None

        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype("float32") / 255.0
        return np.expand_dims(face_img, axis=0)

    def predict(self, face_img):
        inp = self.preprocess(face_img)
        if inp is None:
            return "Unknown"

        preds = self.model.predict(inp, verbose=0)
        prob = preds[0][0]

        return "Female" if prob >= 0.5 else "Male"
