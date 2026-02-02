import subprocess
import pickle
import struct
import os
import sys

class GenderAgeClient:
    def __init__(self):
        self.proc = None
        self._start_worker()

    def _start_worker(self):
        # Already inside gender_age_module
        module_root = os.path.dirname(os.path.abspath(__file__))

        # 🔑 Correct test venv python path
        python_exe = os.path.join(
            module_root, "test", "Scripts", "python.exe"
        )

        # 🔑 Correct worker path
        worker_script = os.path.join(
            module_root, "gender_age_worker.py"
        )

        if not os.path.isfile(python_exe):
            raise RuntimeError(
                "[GenderAgeClient ERROR] test venv python not found:\n"
                f"{python_exe}"
            )

        if not os.path.isfile(worker_script):
            raise RuntimeError(
                "[GenderAgeClient ERROR] gender_age_worker.py not found:\n"
                f"{worker_script}"
            )

        print("[GenderAgeClient] Starting gender worker (test venv)...")

        self.proc = subprocess.Popen(
            [python_exe, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=0
        )

        # 🔑 WAIT FOR HANDSHAKE
        ready = self.proc.stdout.read(5)
        if ready != b"READY":
            raise RuntimeError(
                "[GenderAgeClient ERROR] Gender worker failed to start"
            )

        print("[GenderAgeClient] Gender worker ready ✅")

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
            print("[GenderAgeClient WARNING]", e)
            return "Unknown"
