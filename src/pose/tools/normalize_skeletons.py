import os
import numpy as np
from tqdm import tqdm

# -------- CONFIG --------
INPUT_DIR = "datasets/skeleton_train/val"
OUTPUT_DIR = "datasets/skeleton_train_norm/val"

EPS = 1e-6
# ------------------------


def normalize_sequence(seq):
    """
    seq: (T, J, 2)  where J may be 12, 17, or 25
    """

    T, J, _ = seq.shape
    seq = seq.astype(np.float32)

    # ---- Dynamic joint mapping ----
    # Works for 12, 17, 25 joints
    hip_candidates = [j for j in [8, 9, 11, 12] if j < J]
    ankle_candidates = [j for j in [10, 11, 15, 16] if j < J]
    shoulder_candidates = [j for j in [5, 6] if j < J]

    if len(hip_candidates) >= 2:
        hip_center = seq[:, hip_candidates[:2]].mean(axis=1)
    else:
        hip_center = seq.mean(axis=1)

    seq = seq - hip_center[:, None, :]

    if len(shoulder_candidates) >= 2 and len(ankle_candidates) >= 2:
        shoulder_center = seq[:, shoulder_candidates].mean(axis=1)
        ankle_center = seq[:, ankle_candidates[:2]].mean(axis=1)
        body_height = np.linalg.norm(shoulder_center - ankle_center, axis=1)
        scale = np.mean(body_height)
    else:
        scale = np.mean(np.linalg.norm(seq, axis=2))

    if scale < EPS:
        scale = 1.0

    seq = seq / scale
    return seq


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npy")]
    print(f"Normalizing {len(files)} skeleton sequences")

    skipped = 0

    for fname in tqdm(files, desc="Normalizing"):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)

        try:
            seq = np.load(in_path)

            if seq.ndim != 3 or seq.shape[2] != 2:
                skipped += 1
                continue

            norm_seq = normalize_sequence(seq)
            np.save(out_path, norm_seq)

        except Exception as e:
            skipped += 1
            print(f"[SKIP] {fname}: {e}")

    print(f"Done. Skipped {skipped} files.")


if __name__ == "__main__":
    main()
