import os
import numpy as np
from src.models.inference import GaitInference

MODEL_PATH = "checkpoints/gait_model.pth"
DATASET_DIR = "datasets/skeleton_train_norm/train"

# Load one sample to infer joint count
sample_file = next(
    f for f in os.listdir(DATASET_DIR) if f.endswith(".npy")
)
sample = np.load(os.path.join(DATASET_DIR, sample_file))

num_joints = sample.shape[1]
num_people = len(
    [f for f in os.listdir(DATASET_DIR) if f.endswith(".npy")]
)

# Initialize inference
infer = GaitInference(
    model_path=MODEL_PATH,
    num_joints=num_joints,
    num_people=num_people
)

# Build gallery from normalized skeletons
infer.build_gallery(DATASET_DIR)

# Test on a known sequence
test_seq = np.load(os.path.join(DATASET_DIR, sample_file))

pid, score, found = infer.identify(test_seq)

print("IDENTITY:", pid)
print("SCORE:", round(score, 3))
print("FOUND:", found)
