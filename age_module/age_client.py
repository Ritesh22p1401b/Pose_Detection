import subprocess
import pickle
import struct
import os
import sys

class AgeClient:
    def __init__(self):
        self.proc = None
        self._start_worker()

    def _start_worker(self):
        module_root = os.path.dirname(os.path.abspath(__file__))

        # SAME STRUCTURE AS GENDER MODULE
        python_exe = os.path.join(
            module_root,"age_module", "test", "Scripts", "python.exe"
        )

        worker_script = os.path.join(
            module_root, "age_worker.py"
        )

        if not os.path.isfile(python_exe):
            raise RuntimeError(
                "[AgeClient ERROR] age venv python not found:\n"
                f"{python_exe}"
            )

        if not os.path.isfile(worker_script):
            raise RuntimeError(
                "[AgeClient ERROR] age_worker.py not found:\n"
                f"{worker_script}"
            )

        print("[AgeClient] Starting age worker (test venv)...")

        self.proc = subprocess.Popen(
            [python_exe, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=0
        )

        # HANDSHAKE
        ready = self.proc.stdout.read(5)
        if ready != b"READY":
            raise RuntimeError(
                "[AgeClient ERROR] Age worker failed to start"
            )

        print("[AgeClient] Age worker ready ✅")

    def predict(self, face_img):
        try:
            payload = pickle.dumps(face_img, protocol=pickle.HIGHEST_PROTOCOL)

            self.proc.stdin.write(struct.pack("I", len(payload)))
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()

            size = struct.unpack("I", self.proc.stdout.read(4))[0]
            data = self.proc.stdout.read(size)

            return pickle.loads(data)

        except Exception as e:
            print("[AgeClient WARNING]", e)
            return "Unknown"
