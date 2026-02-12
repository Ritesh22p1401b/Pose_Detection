import sys
import json
import base64
import numpy as np
import cv2
from predictor import AgePredictor

predictor = AgePredictor()

def decode_image(base64_str):
    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break

        request = json.loads(line.strip())
        image_b64 = request["image"]

        frame = decode_image(image_b64)
        age_label = predictor.predict(frame)

        print(json.dumps({"age": age_label}))
        sys.stdout.flush()

    except Exception:
        print(json.dumps({"age": "Unknown"}))
        sys.stdout.flush()
