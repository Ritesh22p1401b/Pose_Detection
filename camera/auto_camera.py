import cv2


class AutoCamera:
    """
    iVCam ONLY camera handler
    """

    def __init__(self, index=1, width=1280, height=720):
        """
        index=1 is default for iVCam on Windows
        """
        self.index = index
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(
                "[AutoCamera] iVCam not detected. "
                "Make sure iVCam is running and connected."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        print("[AutoCamera] iVCam connected successfully")

    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
