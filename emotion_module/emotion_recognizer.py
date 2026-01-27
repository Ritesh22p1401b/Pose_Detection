import cv2
import numpy as np
from tensorflow.keras.models import load_model
from emotion_module.emotion_labels import EMOTION_LABELS

class EmotionRecognizer:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 48, 48, 1)

        preds = self.model.predict(reshaped, verbose=0)[0]
        idx = np.argmax(preds)

        return EMOTION_LABELS[idx], float(preds[idx])
