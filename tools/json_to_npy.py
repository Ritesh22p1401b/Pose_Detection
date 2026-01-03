import os
import json
import numpy as np
from tqdm import tqdm

# -------- CONFIG --------
INPUT_DIR = "datasets/kinetics_val"
OUTPUT_DIR = "datasets/skeleton_train/val"

NUM_JOINTS = 17   # COCO format (change to 25 if needed)
# ------------------------


def extract_pose(json_path):
    """
    Extract skeleton array of shape (T, J, 2)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if "data" not in data:
        return None

    frames = []

    for frame in data["data"]:
        if "skeleton" not in frame or len(frame["skeleton"]) == 0:
            continue

        person = frame["skeleton"][0]

        if "pose" not in person:
            continue

        pose = np.array(person["pose"], dtype=np.float32)

        # (J*3,) -> (J, 3)
        pose = pose.reshape(-1, 3)

        # keep (x, y)
        pose = pose[:NUM_JOINTS, :2]

        frames.append(pose)

    if len(frames) == 0:
        return None

    return np.stack(frames)  # (T, J, 2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    print(f"Found {len(files)} training JSON files")

    for fname in tqdm(files, desc="Converting kinetics_train"):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(
            OUTPUT_DIR, fname.replace(".json", ".npy")
        )

        try:
            skeleton = extract_pose(in_path)
            if skeleton is None:
                continue

            np.save(out_path, skeleton)

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")


if __name__ == "__main__":
    main()
