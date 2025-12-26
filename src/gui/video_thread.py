from PySide6.QtCore import QThread, Signal
import cv2

class VideoThread(QThread):
    frame_signal = Signal(object)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.source)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_signal.emit(frame)

        self.cleanup()

    def stop(self):
        self.running = False

    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
