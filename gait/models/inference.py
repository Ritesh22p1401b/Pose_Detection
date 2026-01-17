import os
import torch
import numpy as np
import torch.nn.functional as F

from models import GaitModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 100
THRESHOLD = 0.75


class GaitInference:
    def __init__(self, model_path, num_joints, num_people):
        self.model = GaitModel(num_joints, num_people)
        self.model.load_state_dict(
            torch.load(model_path, map_location=DEVICE)
        )
        self.model.to(DEVICE)
        self.model.eval()

        self.gallery = {}

    def _prepare(self, seq):
        if seq.shape[0] > SEQ_LEN:
            seq = seq[:SEQ_LEN]
        else:
            pad = np.zeros((SEQ_LEN - seq.shape[0], seq.shape[1], 2))
            seq = np.concatenate([seq, pad], axis=0)

        seq = torch.tensor(seq, dtype=torch.float32)
        seq = seq.unsqueeze(0).to(DEVICE)
        return seq

    def embed(self, seq):
        x = self._prepare(seq)
        with torch.no_grad():
            emb, _, _ = self.model(x)
        return emb.squeeze(0)

    def build_gallery(self, dataset_dir):
        for fname in os.listdir(dataset_dir):
            if not fname.endswith(".npy"):
                continue

            pid = fname.replace(".npy", "")
            seq = np.load(os.path.join(dataset_dir, fname))

            emb = self.embed(seq)
            self.gallery[pid] = emb

    def identify(self, seq):
        emb = self.embed(seq)

        best_id = None
        best_score = 0.0

        for pid, ref_emb in self.gallery.items():
            score = F.cosine_similarity(emb, ref_emb, dim=0).item()
            if score > best_score:
                best_score = score
                best_id = pid

        found = best_score > THRESHOLD
        return best_id, best_score, found
