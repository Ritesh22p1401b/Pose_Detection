import sys
import json
import base64
import cv2
import numpy as np
import tensorflow as tf
from emotion_model import EmotionModel


# --------------------------------------------------
# GPU / CPU AUTO CONFIG (LOGS → STDERR)
# --------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("[EmotionWorker] GPU enabled", file=sys.stderr)
else:
    print("[EmotionWorker] CPU mode", file=sys.stderr)

# --------------------------------------------------
# LOAD MODEL ONCE (LOGS → STDERR)
# --------------------------------------------------
print("[EmotionWorker] Loading model...", file=sys.stderr)
model = EmotionModel()
print("[EmotionWorker] Emotion model ready", file=sys.stderr)

# --------------------------------------------------
# IPC LOOP (STDOUT = JSON ONLY)
# --------------------------------------------------
for line in sys.stdin:
    try:
        msg = json.loads(line)

        img_bytes = base64.b64decode(msg["image_b64"])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        emotion, conf = model.predict(image)

        # ✅ JSON ONLY
        sys.stdout.write(json.dumps({
            "emotion": emotion,
            "confidence": float(conf)
        }) + "\n")
        sys.stdout.flush()

    except Exception as e:
        sys.stdout.write(json.dumps({
            "emotion": "Unknown",
            "confidence": 0.0,
            "error": str(e)
        }) + "\n")
        sys.stdout.flush()
