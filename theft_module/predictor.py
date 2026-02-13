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

        # Disable verbose logs
        self.model = YOLO(model_path, verbose=False)

        self.threshold = 0.30

        # 🔥 Frame skipping logic
        self.frame_counter = 0
        self.skip_interval = 5   # run every 5th frame
        self.last_label = "nothing"

    def predict(self, frame):
        self.frame_counter += 1

        # If not the 5th frame → return cached result
        if self.frame_counter % self.skip_interval != 0:
            return self.last_label

        # Run classification on this frame
        results = self.model(frame, device=self.device, verbose=False)

        if not results:
            self.last_label = "nothing"
            return self.last_label

        result = results[0]

        if result.probs is None:
            self.last_label = "nothing"
            return self.last_label

        probs = result.probs
        top1 = probs.top1
        confidence = float(probs.top1conf)
        label = self.model.names[top1]

        if confidence < self.threshold:
            self.last_label = "nothing"
        else:
            self.last_label = label

        return self.last_label
