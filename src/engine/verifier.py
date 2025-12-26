import numpy as np
from src.models.gait_encoder import GaitEncoder
from src.models.similarity import cosine_similarity
from src.config import SIMILARITY_THRESHOLD
import cv2

class Verifier:
    def __init__(self, profile_path):
        self.encoder = GaitEncoder()
        self.profile = np.load(profile_path)

    def verify_frames(self, frames):
        emb = self.encoder.encode_video(frames)
        if emb is None:
            return 0.0, False

        score = float(cosine_similarity(emb, self.profile))
        return score, score >= SIMILARITY_THRESHOLD
