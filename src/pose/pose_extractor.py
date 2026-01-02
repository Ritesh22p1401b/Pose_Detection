import cv2
import numpy as np

# Stable MediaPipe import for Python 3.11 on Windows
from mediapipe.python.solutions.pose import Pose, PoseLandmark


class PoseExtractor:
    def __init__(self):
        self.pose = Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Key joints for gait recognition
        self.joints = [
            PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_KNEE,
            PoseLandmark.RIGHT_KNEE,
            PoseLandmark.LEFT_ANKLE,
            PoseLandmark.RIGHT_ANKLE,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER
        ]

    def extract(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(frame_rgb)

        if not result.pose_landmarks:
            return None

        keypoints = []
        for joint in self.joints:
            lm = result.pose_landmarks.landmark[joint]
            keypoints.extend([lm.x, lm.y])

        return np.array(keypoints, dtype=np.float32)
