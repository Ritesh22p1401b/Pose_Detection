import cv2
import mediapipe as mp
import numpy as np


# SAME JOINTS USED DURING TRAINING
JOINT_INDEXES = [
    11, 12,  # shoulders
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
    13, 14,  # elbows
    15, 16   # wrists
]


class SkeletonExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_from_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if not res.pose_landmarks:
            return None

        joints = []
        for idx in JOINT_INDEXES:
            lm = res.pose_landmarks.landmark[idx]
            joints.append([lm.x, lm.y])

        return np.array(joints, dtype=np.float32)

    def extract_from_video(self, source, max_frames=100):
        cap = cv2.VideoCapture(source)
        sequence = []

        while cap.isOpened() and len(sequence) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            joints = self.extract_from_frame(frame)
            if joints is not None:
                sequence.append(joints)

        cap.release()
        return np.array(sequence, dtype=np.float32)
