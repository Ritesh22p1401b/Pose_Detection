import subprocess
import tempfile
import json
import cv2
import os


class EmotionAdapter:
    def __init__(self):
        # ✅ emotion venv python
        self.python = os.path.abspath(
            "emotion_module/emotion/Scripts/python"
        )

        # ✅ emotion inference script
        self.script = os.path.abspath(
            "emotion_module/emotion_code/emotion_predictor.py"
        )

    def predict(self, face_roi):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, face_roi)
            img_path = f.name

        try:
            result = subprocess.run(
                [self.python, self.script, img_path],
                capture_output=True,
                text=True,
                timeout=3
            )

            data = json.loads(result.stdout)
            return data["emotion"], data["confidence"]

        except Exception as e:
            print("[EmotionAdapter ERROR]", e)
            return "Unknown", 0.0

        finally:
            os.remove(img_path)
