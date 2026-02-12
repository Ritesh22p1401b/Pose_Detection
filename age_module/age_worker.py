import sys
import pickle
import struct
from predictor import AgePredictor

predictor = AgePredictor()

# ---------------- HANDSHAKE (BINARY SAFE) ----------------
sys.stdout.buffer.write(b"READY")
sys.stdout.buffer.flush()

while True:
    try:
        # Read message size (4 bytes)
        size_data = sys.stdin.buffer.read(4)
        if not size_data:
            break

        size = struct.unpack("I", size_data)[0]

        # Read payload
        payload = sys.stdin.buffer.read(size)
        face_img = pickle.loads(payload)

        # Run prediction
        age_label = predictor.predict(face_img)

        # Send result back
        response = pickle.dumps(age_label, protocol=pickle.HIGHEST_PROTOCOL)

        sys.stdout.buffer.write(struct.pack("I", len(response)))
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()

    except Exception:
        # Return "Unknown" safely
        error = pickle.dumps("Unknown", protocol=pickle.HIGHEST_PROTOCOL)
        sys.stdout.buffer.write(struct.pack("I", len(error)))
        sys.stdout.buffer.write(error)
        sys.stdout.buffer.flush()
