import torch
import cv2
import numpy as np
import os
import torch.nn.functional as F

from models.mobilenetv3_age import AgeMobileNetV3

AGE_LABELS = [
    "0-5", "6-10", "11-15", "16-20",
    "21-25", "26-30", "31-35", "36-40",
    "41-45", "46-50", "51-60", "60+"
]

class AgePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", "age_mobilenetv3_final.pth")

        self.model = AgeMobileNetV3(num_classes=12)

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, face):
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = np.transpose(face, (2, 0, 1))
        face = torch.tensor(face).unsqueeze(0).to(self.device)
        return face

    def predict(self, face):
        input_tensor = self.preprocess(face)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        return AGE_LABELS[pred_class]
