import cv2
import numpy as np
import mediapipe as mp


class VideoSkeletonExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract(self, video_source):
        cap = cv2.VideoCapture(video_source)
        skeletons = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)

            if not res.pose_landmarks:
                continue

            joints = []
            for lm in res.pose_landmarks.landmark:
                joints.append([lm.x, lm.y])

            skeletons.append(joints)

        cap.release()
        return np.array(skeletons, dtype=np.float32)
