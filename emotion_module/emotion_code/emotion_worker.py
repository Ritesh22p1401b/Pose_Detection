import sys
import json
import cv2
import tensorflow as tf
import base64
import numpy as np
from emotion_model import EmotionModel


# --------------------------------------------------
# CUDA / CPU AUTO CONFIGURATION
# --------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[EmotionWorker] CUDA GPU detected, using GPU", file=sys.stderr)
    except Exception as e:
        print("[EmotionWorker] GPU setup failed, using CPU:", e, file=sys.stderr)
else:
    print("[EmotionWorker] No GPU found, using CPU", file=sys.stderr)


# --------------------------------------------------
# LOAD MODEL ONCE
# --------------------------------------------------
print("[EmotionWorker] Loading emotion model...", file=sys.stderr)
model = EmotionModel()
print("[EmotionWorker] Emotion model ready", file=sys.stderr)


# --------------------------------------------------
# PERSISTENT IPC LOOP (BASE64 SAFE)
# --------------------------------------------------
for line in sys.stdin:
    try:
        msg = json.loads(line)

        # ---------- BASE64 IMAGE ----------
        if "image_b64" not in msg:
            raise ValueError("Missing image_b64 field")

        img_bytes = base64.b64decode(msg["image_b64"])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)

        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("Invalid decoded image")

        # ---------- MODEL INFERENCE ----------
        emotion, conf = model.predict(image)

        print(json.dumps({
            "emotion": emotion,
            "confidence": float(conf)
        }), flush=True)

    except Exception as e:
        print(json.dumps({
            "emotion": "Unknown",
            "confidence": 0.0,
            "error": str(e)
        }), flush=True)
