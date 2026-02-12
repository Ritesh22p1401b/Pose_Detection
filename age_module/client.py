import subprocess
import json
import base64
import cv2
import os

class AgeClient:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        python_exec = os.path.join(base_dir, "venv", "Scripts", "python.exe")
        server_script = os.path.join(base_dir, "server.py")

        self.process = subprocess.Popen(
            [python_exec, server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )

    def predict(self, face_roi):
        _, buffer = cv2.imencode(".jpg", face_roi)
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        request = json.dumps({"image": image_b64})

        self.process.stdin.write(request + "\n")
        self.process.stdin.flush()

        response = self.process.stdout.readline()
        data = json.loads(response.strip())

        return data.get("age", "Unknown")
