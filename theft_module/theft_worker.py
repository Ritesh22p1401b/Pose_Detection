# theft_module/theft_worker.py

import sys
import pickle
import struct
from predictor import TheftPredictor

predictor = TheftPredictor()

# -------- HANDSHAKE --------
sys.stdout.buffer.write(b"READY")
sys.stdout.buffer.flush()

while True:
    try:
        size_data = sys.stdin.buffer.read(4)
        if not size_data:
            break

        size = struct.unpack("I", size_data)[0]
        payload = sys.stdin.buffer.read(size)

        frame = pickle.loads(payload)

        label = predictor.predict(frame)

        response = pickle.dumps(label, protocol=pickle.HIGHEST_PROTOCOL)

        sys.stdout.buffer.write(struct.pack("I", len(response)))
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()

    except Exception:
        error = pickle.dumps("nothing", protocol=pickle.HIGHEST_PROTOCOL)
        sys.stdout.buffer.write(struct.pack("I", len(error)))
        sys.stdout.buffer.write(error)
        sys.stdout.buffer.flush()
