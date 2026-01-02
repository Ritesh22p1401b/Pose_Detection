import cv2
import numpy as np
import mediapipe as mp


class PoseExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.joints = [
            mp.solutions.pose.PoseLandmark.LEFT_HIP,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            mp.solutions.pose.PoseLandmark.LEFT_KNEE,
            mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
        ]

    def extract(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(frame_rgb)

        if not result.pose_landmarks:
            return None

        keypoints = []
        for j in self.joints:
            lm = result.pose_landmarks.landmark[j]
            keypoints.extend([lm.x, lm.y])

        return np.array(keypoints, dtype=np.float32)
