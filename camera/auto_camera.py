import cv2


class AutoCamera:
    """
    Automatically selects iVCam (phone webcam) if available.
    Falls back to laptop webcam if iVCam is not running.
    """

    # Common iVCam indices on Windows
    IVCAM_INDICES = [1, 2, 3, 4]

    LAPTOP_INDEX = 0

    def __init__(self):
        self.cap = None
        self.active_source = None  # "ivcam" or "laptop"

    def open(self):
        # 1️⃣ Try iVCam first
        for idx in self.IVCAM_INDICES:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.cap = cap
                self.active_source = "ivcam"
                print(f"[INFO] iVCam detected at index {idx}")
                return self.cap
            cap.release()

        # 2️⃣ Fallback to laptop webcam
        cap = cv2.VideoCapture(self.LAPTOP_INDEX, cv2.CAP_DSHOW)
        if cap.isOpened():
            self.cap = cap
            self.active_source = "laptop"
            print("[INFO] Laptop webcam detected")
            return self.cap

        raise RuntimeError(
            "No camera available. Start iVCam or enable laptop webcam."
        )

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
