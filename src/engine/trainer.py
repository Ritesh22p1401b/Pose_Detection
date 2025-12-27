import cv2
import numpy as np
from pathlib import Path

from src.pose.pose_extractor import PoseExtractor
from src.models.gait_encoder import GaitEncoder

class Trainer:
    def __init__(self):
        self.pose = PoseExtractor("models/pose.onnx")
        self.encoder = GaitEncoder()

    def build_profile(self, video_paths, person_id):
        embeddings = []

        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            skeletons = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                sk = self.pose.extract(frame)
                if sk is not None:
                    skeletons.append(sk)

            cap.release()

            emb = self.encoder.encode(np.array(skeletons))
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            raise RuntimeError("No gait data extracted")

        profile = np.mean(embeddings, axis=0)

        Path("data/profiles").mkdir(parents=True, exist_ok=True)
        path = f"data/profiles/{person_id}.npy"
        np.save(path, profile)

        return path
