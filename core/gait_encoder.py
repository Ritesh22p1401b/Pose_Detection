import numpy as np
from core.gait_features import extract_frame_features

def encode(sequence):
    feats = np.array([extract_frame_features(kp) for kp in sequence])
    return np.concatenate([feats.mean(axis=0), feats.std(axis=0)])
