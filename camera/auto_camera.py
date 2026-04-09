import cv2
import sys


# 🔧 Fix Windows stdout encoding issue (prevents cp1252 error)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


class AutoCamera:
    """
    Simple and Stable Camera Handler
    - Works with iVCam (index=1) or webcam (index=0)
    - Returns (ret, frame) like OpenCV
    """

    def __init__(self, index=1, width=1280, height=720):
        self.index = index
        self.width = width
        self.height = height

        print(f"[AutoCamera] Opening camera at index {index}...")

        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[AutoCamera] Camera not detected at index {index}"
            )

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print("[AutoCamera] Camera connected successfully")

    def read(self):
        """
        Returns:
            ret (bool), frame (numpy.ndarray)
        """
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            return False, None

        return True, frame

    def release(self):
        """
        Safely release camera
        """
        if self.cap:
            self.cap.release()
            self.cap = None
            print("[AutoCamera] Camera released")