import sys
import os
import pickle
import struct
import traceback

# --------------------------------------------------
# ENSURE MODULE PATH
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from gender_age_adapter import GenderAgeAdapter
except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
try:
    print("[GenderAgeWorker] Loading gender model...")
    engine = GenderAgeAdapter()
    print("[GenderAgeWorker] Gender model loaded successfully")

    # 🔑 HANDSHAKE
    sys.stdout.buffer.write(b"READY")
    sys.stdout.buffer.flush()

except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

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
        gender = engine.predict(frame)

        response = pickle.dumps(gender)
        sys.stdout.buffer.write(struct.pack("I", len(response)))
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        break
