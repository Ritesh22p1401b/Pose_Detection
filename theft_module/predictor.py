# theft_module/predictor.py

import os
from ultralytics import YOLO
import torch


class TheftPredictor:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", "best.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = YOLO(model_path)
        self.threshold = 0.50

    def predict(self, frame):
        results = self.model(frame, device=self.device)

        if not results:
            return "nothing"

        result = results[0]

        # Classification models use .probs
        if result.probs is None:
            return "nothing"

        probs = result.probs
        top1 = probs.top1
        confidence = float(probs.top1conf)

        label = self.model.names[top1]

        # Confidence rejection
        if confidence < self.threshold:
            return "nothing"

        return label
