import subprocess
import json
import cv2
import os
import base64
import numpy as np


class EmotionAdapter:
    """
    Windows-safe Emotion Adapter
    - No temp files
    - No stderr pipe deadlock
    - No Errno 22
    """

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        EMOTION_MODULE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

        self.python = os.path.join(
            EMOTION_MODULE_DIR,
            "emotion",
            "Scripts",
            "python.exe"
        )

        self.worker = os.path.join(
            EMOTION_MODULE_DIR,
            "emotion_code",
            "emotion_worker.py"
        )

        if not os.path.isfile(self.python):
            raise FileNotFoundError(self.python)
        if not os.path.isfile(self.worker):
            raise FileNotFoundError(self.worker)

        # ⚠️ stderr NOT piped (CRITICAL FIX)
        self.proc = subprocess.Popen(
            [self.python, self.worker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        print("[EmotionAdapter] Worker started safely")

    # --------------------------------------------------
    # SAFE PREDICT
    # --------------------------------------------------
    def predict(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return "Unknown", 0.0

        h, w = face_roi.shape[:2]
        if h < 48 or w < 48:
            return "Unknown", 0.0

        # ✅ Force contiguous memory
        face_roi = np.ascontiguousarray(face_roi, dtype=np.uint8)

        # ✅ Encode to JPEG in-memory
        ok, buf = cv2.imencode(".jpg", face_roi)
        if not ok:
            return "Unknown", 0.0

        img_b64 = base64.b64encode(buf).decode("utf-8")

        try:
            self.proc.stdin.write(
                json.dumps({"image_b64": img_b64}) + "\n"
            )
            self.proc.stdin.flush()

            response = self.proc.stdout.readline()
            if not response:
                return "Unknown", 0.0

            data = json.loads(response)
            return (
                data.get("emotion", "Unknown"),
                float(data.get("confidence", 0.0))
            )

        except Exception as e:
            print("[EmotionAdapter ERROR]", e)
            return "Unknown", 0.0

    def close(self):
        try:
            if self.proc:
                self.proc.terminate()
                self.proc = None
        except Exception:
            pass
