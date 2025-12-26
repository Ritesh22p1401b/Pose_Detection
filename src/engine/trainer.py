import numpy as np
from pathlib import Path
from src.models.gait_encoder import GaitEncoder
from src.config import PROFILE_DIR
import cv2

class Trainer:
    def __init__(self):
        self.encoder = GaitEncoder()

    def build_profile(self, video_path, person_id="person_01"):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        embedding = self.encoder.encode_video(frames)
        if embedding is None:
            raise RuntimeError("Failed to extract gait features")

        Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)
        profile_path = Path(PROFILE_DIR) / f"{person_id}.npy"
        np.save(profile_path, embedding)

        return profile_path
