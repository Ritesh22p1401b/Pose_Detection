import cv2
from PySide6.QtCore import QThread, Signal

class VideoThread(QThread):
    frame_signal = Signal(object)

    def __init__(self, src):
        super().__init__()
        self.src = src
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.src)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_signal.emit(frame)
        cap.release()

    def stop(self):
        self.running = False
