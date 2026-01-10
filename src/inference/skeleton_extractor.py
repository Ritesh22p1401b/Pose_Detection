import cv2
import mediapipe as mp
import numpy as np


class SkeletonExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_from_video(self, source, max_frames=100):
        cap = cv2.VideoCapture(source)
        sequence = []

        while cap.isOpened() and len(sequence) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)

            if res.pose_landmarks:
                joints = []
                for lm in res.pose_landmarks.landmark:
                    joints.append([lm.x, lm.y])
                sequence.append(joints)

        cap.release()
        return np.array(sequence, dtype=np.float32)
