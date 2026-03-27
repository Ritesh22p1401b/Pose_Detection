# import os
# import numpy as np
# import tensorflow as tf
# import cv2   # ✅ now available because test venv is used

# class GenderAgeAdapter:
#     def __init__(self):
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         model_path = os.path.join(
#             base_dir, "model", "utkface_age_gender_final.h5"
#         )

#         if not os.path.isfile(model_path):
#             raise FileNotFoundError(f"Gender model not found: {model_path}")

#         self.model = tf.keras.models.load_model(model_path, compile=False)

#     def preprocess(self, face_img):
#         if face_img is None or face_img.size == 0:
#             return None

#         face_img = cv2.resize(face_img, (224, 224))
#         face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#         face_img = face_img.astype("float32") / 255.0
#         return np.expand_dims(face_img, axis=0)

#     def predict(self, face_img):
#         inp = self.preprocess(face_img)
#         if inp is None:
#             return "Unknown"

#         preds = self.model.predict(inp, verbose=0)
#         prob = preds[0][0]

#         return "Female" if prob >= 0.5 else "Male"

import os
import numpy as np
import tensorflow as tf
import cv2


class GenderAgeAdapter:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            base_dir, "model", "utkface_age_gender_final.h5"
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Gender model not found: {model_path}")

        self.model = tf.keras.models.load_model(model_path, compile=False)

        # Debug flag (set True once to inspect outputs)
        self.debug = False

    def preprocess(self, face_img):
        if face_img is None or face_img.size == 0:
            return None

        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype("float32") / 255.0

        return np.expand_dims(face_img, axis=0)

    def extract_gender_prob(self, preds):
        """
        Handles different model output formats:
        - Sigmoid: [[0.7]]
        - Softmax: [[0.3, 0.7]]
        - Multi-output: [age_pred, gender_pred]
        """

        # Case 1: multi-output model
        if isinstance(preds, list):
            preds = preds[-1]  # assume last output is gender

        preds = np.array(preds)

        # Case 2: sigmoid (single value)
        if preds.shape[-1] == 1:
            return float(preds[0][0])

        # Case 3: softmax (2 classes)
        elif preds.shape[-1] == 2:
            return float(preds[0][1])  # index 1 = female

        else:
            raise ValueError(f"Unexpected prediction shape: {preds.shape}")

    def predict(self, face_img):
        inp = self.preprocess(face_img)
        if inp is None:
            return "Unknown"

        preds = self.model.predict(inp, verbose=0)

        if self.debug:
            print("Raw preds:", preds)

        try:
            prob = self.extract_gender_prob(preds)
        except Exception as e:
            print("Prediction error:", e)
            return "Unknown"

        # 🔥 Improved decision logic (VERY IMPORTANT)
        if prob >= 0.6:
            return "Female"
        elif prob <= 0.4:
            return "Male"
        else:
            return "Uncertain"