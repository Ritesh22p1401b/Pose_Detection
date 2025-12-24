import cv2
import numpy as np
from core.detector import PoseDetector
from core.gait_features import normalize_pose
from core.gait_encoder import encode

def enroll(video_path, save_path):
    detector = PoseDetector()
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res = detector.track(frame)
        if res.keypoints is None:
            continue

        kp = res.keypoints.xy[0]
        sequence.append(normalize_pose(kp))

    embedding = encode(sequence)
    np.save(save_path, embedding)
    cap.release()
