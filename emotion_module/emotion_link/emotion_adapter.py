import subprocess
import tempfile
import json
import cv2
import os


class EmotionAdapter:
    def __init__(self):
        # --------------------------------------------------
        # Resolve paths RELATIVE TO THIS FILE (CRITICAL FIX)
        # --------------------------------------------------
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(
            os.path.join(BASE_DIR, "..", "..")
        )

        # ---- emotion venv python executable ----
        self.python = os.path.join(
            PROJECT_ROOT,
            "emotion_module",
            "emotion",
            "Scripts",
            "python.exe"
        )

        # ---- emotion inference script ----
        self.script = os.path.join(
            PROJECT_ROOT,
            "emotion_module",
            "emotion_code",
            "emotion_predictor.py"
        )

        # ---- hard validation (fail fast) ----
        if not os.path.isfile(self.python):
            raise FileNotFoundError(
                f"Emotion python not found: {self.python}"
            )

        if not os.path.isfile(self.script):
            raise FileNotFoundError(
                f"Emotion script not found: {self.script}"
            )

    def predict(self, face_roi):
        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as f:
            cv2.imwrite(f.name, face_roi)
            img_path = f.name

        try:
            result = subprocess.run(
                [self.python, self.script, img_path],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr)

            data = json.loads(result.stdout)
            return data.get("emotion", "Unknown"), data.get("confidence", 0.0)

        except Exception as e:
            print("[EmotionAdapter ERROR]", e)
            return "Unknown", 0.0

        finally:
            os.remove(img_path)
