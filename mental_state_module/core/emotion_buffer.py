from collections import deque
from mental_state_module.config import MAX_SCANS


class EmotionBuffer:
    def __init__(self):
        self.first_emotion = None
        self.emotions = deque(maxlen=MAX_SCANS)

    def add(self, emotion: str):
        if self.first_emotion is None:
            self.first_emotion = emotion
        self.emotions.append(emotion)

    def is_ready(self):
        return len(self.emotions) == MAX_SCANS
