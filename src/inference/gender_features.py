import numpy as np


def extract_gender_features(skeleton_seq):
    # skeleton_seq: (T, J, 2)
    # Simple but effective gait statistics

    diffs = np.diff(skeleton_seq, axis=0)
    velocity = np.linalg.norm(diffs, axis=2)

    features = [
        velocity.mean(),
        velocity.std(),
        skeleton_seq[:, :, 0].mean(),  # x mean
        skeleton_seq[:, :, 1].mean(),  # y mean
        skeleton_seq[:, :, 0].std(),
        skeleton_seq[:, :, 1].std(),
    ]

    return np.array(features).reshape(1, -1)
