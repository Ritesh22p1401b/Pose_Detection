import numpy as np


def extract_gender_features(skeleton_seq):
    """
    skeleton_seq: (T, J, 2)
    returns: (1, 7)
    """

    T = skeleton_seq.shape[0]

    if T < 2:
        # Return neutral features (prevents crash)
        return np.zeros((1, 7), dtype=np.float32)

    diffs = np.diff(skeleton_seq, axis=0)
    velocity = np.linalg.norm(diffs, axis=2)

    features = [
        velocity.mean(),          # 1
        velocity.std(),           # 2
        skeleton_seq[..., 0].mean(),  # 3
        skeleton_seq[..., 1].mean(),  # 4
        skeleton_seq[..., 0].std(),   # 5
        skeleton_seq[..., 1].std(),   # 6
        velocity.max(),            # 7  ← MISSING FEATURE (FIXED)
    ]

    return np.array(features, dtype=np.float32).reshape(1, -1)
