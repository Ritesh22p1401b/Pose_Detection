import sys
import os
import pickle
import struct
import traceback
import tensorflow as tf

# --------------------------------------------------
# 🔴 ABSOLUTE, BULLETPROOF PATH FIX
# --------------------------------------------------
WORKER_FILE = os.path.abspath(__file__)
WORKER_DIR = os.path.dirname(WORKER_FILE)
MODULE_ROOT = os.path.abspath(os.path.join(WORKER_DIR))

# Add BOTH paths explicitly
for p in (WORKER_DIR, MODULE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# DEBUG (optional, can remove later)
print("[GenderAgeWorker] sys.path =", sys.path, file=sys.stderr)

# --------------------------------------------------
# CPU / GPU LOGGING
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
    from gender_age_adapter import GenderAgeAdapter
    engine = GenderAgeAdapter()
    print("[GenderAgeWorker] Model loaded successfully", file=sys.stderr)
except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
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
# IPC LOOP (stdout = BINARY ONLY)
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
