import numpy as np

def normalize_pose(kp):
    kp = np.array(kp)

    # hip center
    center = (kp[11] + kp[12]) / 2
    kp -= center

    # scale by body height
    height = np.linalg.norm(kp[5] - kp[11]) + np.linalg.norm(kp[11] - kp[15])
    if height > 0:
        kp /= height

    return kp

def joint_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos, -1, 1))

def extract_frame_features(kp):
    return np.array([
        joint_angle(kp[11], kp[13], kp[15]),  # left knee
        joint_angle(kp[12], kp[14], kp[16])   # right knee
    ])
