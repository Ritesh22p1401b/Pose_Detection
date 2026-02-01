import sys
import os

# --------------------------------------------------
# ABSOLUTE PATH FIX (REQUIRED)
# --------------------------------------------------
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKER_DIR not in sys.path:
    sys.path.insert(0, WORKER_DIR)

# --------------------------------------------------
import pickle
import struct
import traceback
import tensorflow as tf
from gender_age_adapter import GenderAgeAdapter

# --------------------------------------------------
# CPU / GPU CONFIG
# --------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("[GenderAgeWorker] GPU enabled", file=sys.stderr)
else:
    print("[GenderAgeWorker] CPU mode", file=sys.stderr)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
try:
    print("[GenderAgeWorker] Loading model...", file=sys.stderr)
    engine = GenderAgeAdapter()
    print("[GenderAgeWorker] Model loaded successfully", file=sys.stderr)
except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# --------------------------------------------------
# IPC HELPERS
# --------------------------------------------------
def read_exact(n):
    data = b""
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

# --------------------------------------------------
# IPC LOOP
# --------------------------------------------------
while True:
    try:
        size_bytes = read_exact(4)
        if not size_bytes:
            break

        size = struct.unpack("I", size_bytes)[0]
        payload = read_exact(size)
        if payload is None:
            break

        frame = pickle.loads(payload)
        age, gender = engine.predict(frame)

        response = pickle.dumps((age, gender))
        sys.stdout.buffer.write(struct.pack("I", len(response)))
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        break
