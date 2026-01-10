import numpy as np
import torch
import torch.nn.functional as F

from src.models.model import GaitModel


class GaitMatcher:
    def __init__(self, model_path, num_joints, num_people=1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        checkpoint = torch.load(model_path, map_location=device)

        self.model = GaitModel(num_joints, num_people).to(device)
        self.model.load_state_dict(checkpoint["model_state"], strict=False)
        self.model.eval()

    def embed(self, skeleton_seq):
        x = torch.tensor(skeleton_seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb, _, _ = self.model(x)
        return emb.squeeze(0).cpu().numpy()

    def build_reference(self, embeddings):
        ref = np.mean(embeddings, axis=0)
        return ref

    def match(self, emb, ref, threshold=0.75):
        score = np.dot(emb, ref) / (
            np.linalg.norm(emb) * np.linalg.norm(ref)
        )
        return score > threshold, score
