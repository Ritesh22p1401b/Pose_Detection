import numpy as np
from gait.pose.pose_extractor import PoseExtractor
from gait.models.gait_encoder import GaitEncoder

class Verifier:
    def __init__(self, profile_path):
        self.profile = np.load(profile_path)
        self.pose = PoseExtractor()
        self.encoder = GaitEncoder()

    def verify_frames(self, frames):
        skeletons = []

        for f in frames:
            sk = self.pose.extract(f)
            if sk is not None:
                skeletons.append(sk)

        emb = self.encoder.encode(np.array(skeletons))
        if emb is None:
            return 0.0, False

        score = float(
            np.dot(emb, self.profile) /
            (np.linalg.norm(emb) * np.linalg.norm(self.profile))
        )

        return score, score > 0.75
