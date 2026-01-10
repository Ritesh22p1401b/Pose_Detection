import torch
import numpy as np
import torch.nn.functional as F

from src.models.model import GaitModel


class GaitMatcher:
    """
    Uses a trained gait model as a FEATURE EXTRACTOR
    (biometric verification, not classification)
    """

    def __init__(self, model_path: str, num_joints: int):
        """
        model_path : path to gait_model.pth
        num_joints : MUST match training joint count (e.g. 12)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---------------- Load checkpoint ----------------
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle both save formats:
        # 1) torch.save(model.state_dict())
        # 2) torch.save({"model_state": model.state_dict(), ...})
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        # ---------------- Build model ----------------
        # num_people=1 because we DO NOT use classification head
        self.model = GaitModel(num_joints=num_joints, num_people=1).to(self.device)

        # ---------------- Remove classification head ----------------
        # We only want LSTM + embedding layers
        filtered_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith("id_head")
        }

        # strict=False allows missing id_head weights safely
        self.model.load_state_dict(filtered_state, strict=False)
        self.model.eval()

    # --------------------------------------------------
    # EMBEDDING EXTRACTION
    # --------------------------------------------------
    def embed(self, skeleton_sequence: np.ndarray) -> np.ndarray:
        """
        skeleton_sequence : (T, J, 2)
        returns           : (embedding_dim,)
        """
        if skeleton_sequence.ndim != 3:
            raise ValueError("Skeleton sequence must be (T, J, 2)")

        x = torch.tensor(
            skeleton_sequence,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # (1, T, J, 2)

        with torch.no_grad():
            embedding, _, _ = self.model(x)

        return embedding.squeeze(0).cpu().numpy()

    # --------------------------------------------------
    # BUILD REFERENCE PROFILE
    # --------------------------------------------------
    def build_reference(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """
        embeddings : list of embedding vectors
        returns    : averaged reference embedding
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")

        return np.mean(np.stack(embeddings, axis=0), axis=0)

    # --------------------------------------------------
    # MATCH AGAINST REFERENCE
    # --------------------------------------------------
    def match(
        self,
        embedding: np.ndarray,
        reference: np.ndarray,
        threshold: float = 0.75
    ) -> tuple[bool, float]:
        """
        embedding : test embedding
        reference : stored reference embedding
        threshold : cosine similarity threshold

        returns   : (found, score)
        """
        emb = torch.tensor(embedding)
        ref = torch.tensor(reference)

        score = F.cosine_similarity(emb, ref, dim=0).item()
        found = score >= threshold

        return found, score
