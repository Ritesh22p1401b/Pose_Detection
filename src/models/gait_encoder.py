import cv2
import numpy as np

class GaitEncoder:
    def __init__(self):
        self.prev_gray = None

    def reset(self):
        self.prev_gray = None

    def encode_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        feature = np.mean(diff)
        return feature

    def encode_video(self, frames):
        features = []
        self.reset()

        for f in frames:
            feat = self.encode_frame(f)
            if feat is not None:
                features.append(feat)

        if not features:
            return None

        return np.array(features).mean()
