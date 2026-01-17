import torch
import numpy as np
import torch.nn.functional as F
from gait.models.model import GaitModel


class GaitMatcher:
    def __init__(self, model_path: str, num_joints: int):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        self.model = GaitModel(num_joints=num_joints, num_people=1).to(self.device)

        # Remove classification head
        filtered_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith("id_head")
        }

        self.model.load_state_dict(filtered_state, strict=False)
        self.model.eval()

        # ---- NEW ----
        self.ref_mean = None
        self.ref_std = None

    # ---------------- EMBEDDING ----------------
    def embed(self, skeleton_seq: np.ndarray) -> np.ndarray:
        x = torch.tensor(
            skeleton_seq, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            emb, _, _ = self.model(x)

        emb = emb.squeeze(0).cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)   # 🔥 NORMALIZATION

    # ---------------- REFERENCE BUILD ----------------
    def build_reference(self, embeddings: list[np.ndarray]) -> np.ndarray:
        embeddings = [e / (np.linalg.norm(e) + 1e-8) for e in embeddings]
        ref = np.mean(embeddings, axis=0)

        # ---- CALIBRATION (IMPORTANT) ----
        scores = [
            np.dot(e, ref) / (np.linalg.norm(ref) + 1e-8)
            for e in embeddings
        ]

        self.ref_mean = float(np.mean(scores))
        self.ref_std = float(np.std(scores) + 1e-6)

        return ref

    # ---------------- MATCH ----------------
    def match(self, emb: np.ndarray, ref: np.ndarray):
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        ref = ref / (np.linalg.norm(ref) + 1e-8)

        score = float(np.dot(emb, ref))

        # 🔥 ADAPTIVE THRESHOLD (KEY FIX)
        threshold = self.ref_mean - 2.0 * self.ref_std

        found = score > threshold
        return found, score
