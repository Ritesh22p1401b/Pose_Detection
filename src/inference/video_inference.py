import torch
import joblib
import numpy as np

from src.models.model import GaitModel
from src.inference.skeleton_extractor import SkeletonExtractor
from src.inference.gender_features import extract_gender_features


# -------- PATHS --------
GAIT_MODEL_PATH = "checkpoints/gait_model.pth"
GENDER_MODEL_PATH = "checkpoints/gender_model.pkl"
# -----------------------


def main(video_source=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------- Load skeleton --------
    extractor = SkeletonExtractor()
    skeleton_seq = extractor.extract_from_video(video_source)

    if len(skeleton_seq) == 0:
        print("No person detected")
        return

    # -------- Load gait model --------
    checkpoint = torch.load(GAIT_MODEL_PATH, map_location=device)
    num_joints = skeleton_seq.shape[1]
    num_people = checkpoint["num_people"]

    model = GaitModel(num_joints, num_people).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Prepare tensor
    x = torch.tensor(skeleton_seq).unsqueeze(0).to(device)

    with torch.no_grad():
        _, id_logits, _ = model(x)
        pred_id = torch.argmax(id_logits, dim=1).item()

    print(f"PERSON FOUND → ID: {pred_id}")

    # -------- Gender inference --------
    gender_model = joblib.load(GENDER_MODEL_PATH)
    gender_features = extract_gender_features(skeleton_seq)
    gender_pred = gender_model.predict(gender_features)[0]

    gender = "Male" if gender_pred == 0 else "Female"
    print(f"GENDER → {gender}")


if __name__ == "__main__":
    # 0 = webcam
    # or pass video path as string
    main(0)
