import sys
import pickle
import struct
from predictor import AgePredictor

predictor = AgePredictor()

# Send handshake
sys.stdout.buffer.write(b"READY")
sys.stdout.flush()

while True:
    try:
        size_data = sys.stdin.buffer.read(4)
        if not size_data:
            break

        size = struct.unpack("I", size_data)[0]
        payload = sys.stdin.buffer.read(size)

        face_img = pickle.loads(payload)

        age_label = predictor.predict(face_img)

        response = pickle.dumps(age_label, protocol=pickle.HIGHEST_PROTOCOL)

        sys.stdout.buffer.write(struct.pack("I", len(response)))
        sys.stdout.buffer.write(response)
        sys.stdout.flush()

    except Exception:
        error = pickle.dumps("Unknown")
        sys.stdout.buffer.write(struct.pack("I", len(error)))
        sys.stdout.buffer.write(error)
        sys.stdout.flush()
