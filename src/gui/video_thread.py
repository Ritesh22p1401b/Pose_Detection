from PySide6.QtCore import QThread, Signal
import cv2

class VideoThread(QThread):
    frame_signal = Signal(object)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_signal.emit(frame)

        cap.release()

    def stop(self):
        self.running = False
