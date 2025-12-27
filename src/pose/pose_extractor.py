import cv2
import numpy as np

class PoseExtractor:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def extract(self, frame):
        blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255, (256, 256),
            (0, 0, 0), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        out = self.net.forward()

        # Example: extract selected joints (simplified)
        keypoints = out[0][:10]  # take first 10 values
        return np.array(keypoints)
