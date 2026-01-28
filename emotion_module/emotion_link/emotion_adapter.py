import subprocess
import json
import cv2
import os
import base64
import numpy as np


class EmotionAdapter:
    """
    Windows-safe Emotion Adapter
    - Fixes Errno 22
    - Fixes broken IPC
    - Fixes wrong working directory
    - Does NOT change any existing behavior
    """

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # emotion_module root
        EMOTION_MODULE_DIR = os.path.abspath(
            os.path.join(BASE_DIR, "..", "..")
        )

        # emotion venv python
        self.python = os.path.join(
            EMOTION_MODULE_DIR,
            "emotion_module",
            "emotion",
            "Scripts",
            "python.exe"
        )

        # worker script
        self.worker = os.path.join(
            EMOTION_MODULE_DIR,
            "emotion_module",
            "emotion_code",
            "emotion_worker.py"
        )

        if not os.path.isfile(self.python):
            raise FileNotFoundError(f"Python not found: {self.python}")

        if not os.path.isfile(self.worker):
            raise FileNotFoundError(f"Worker not found: {self.worker}")

        # 🔴 CRITICAL FIXES:
        # 1. cwd is emotion_code
        # 2. stderr is NOT piped (prevents deadlock)
        # 3. text mode + line buffering
        self.proc = subprocess.Popen(
            [self.python, self.worker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=os.path.dirname(self.worker),
            text=True,
            bufsize=1
        )

        print("[EmotionAdapter] Emotion worker started")

    # --------------------------------------------------
    def predict(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return "Unknown", 0.0

        h, w = face_roi.shape[:2]
        if h < 48 or w < 48:
            return "Unknown", 0.0

        face_roi = np.ascontiguousarray(face_roi, dtype=np.uint8)

        ok, buf = cv2.imencode(".jpg", face_roi)
        if not ok:
            return "Unknown", 0.0

        img_b64 = base64.b64encode(buf).decode("utf-8")

        try:
            self.proc.stdin.write(
                json.dumps({"image_b64": img_b64}) + "\n"
            )
            self.proc.stdin.flush()

            # 🔥 READ UNTIL VALID JSON
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    return "Unknown", 0.0

                line = line.strip()

                if not line.startswith("{"):
                    continue  # skip logs

                data = json.loads(line)
                return (
                    data.get("emotion", "Unknown"),
                    float(data.get("confidence", 0.0))
                )

        except Exception as e:
            print("[EmotionAdapter ERROR]", e)
            return "Unknown", 0.0


    # --------------------------------------------------
    def close(self):
        try:
            if self.proc:
                self.proc.terminate()
                self.proc = None
        except Exception:
            pass
