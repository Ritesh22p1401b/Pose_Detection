import cv2
from camera.camera_config import (
    PHONE_CAMERA_INDEX,
    LAPTOP_CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
)

class CameraManager:
    """
    Camera abstraction layer.
    This class will be reused later by face/gait modules.
    """

    def __init__(self, source="laptop"):
        self.source = source
        self.cap = None

    def open(self):
        index = (
            PHONE_CAMERA_INDEX
            if self.source == "phone"
            else LAPTOP_CAMERA_INDEX
        )

        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera '{self.source}' not available")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

        return self.cap

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
