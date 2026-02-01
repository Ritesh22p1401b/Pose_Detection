import subprocess
import pickle
import struct
import os
import time


class GenderAgeClient:
    def __init__(self):
        self.proc = None
        self._start_worker()

    # --------------------------------------------------
    def _start_worker(self):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        python_exe = os.path.join(
            project_root,
            "gender_age_module",
            "test",
            "Scripts",
            "python.exe"
        )

        worker_script = os.path.join(
            project_root,
            "gender_age_module",
            "gender_age_worker.py"
        )

        if not os.path.isfile(python_exe):
            raise RuntimeError(
                "[GenderAgeClient ERROR] venv python.exe not found"
            )

        if not os.path.isfile(worker_script):
            raise RuntimeError(
                "[GenderAgeClient ERROR] Worker script not found"
            )

        self.proc = subprocess.Popen(
            [python_exe, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0
        )

        time.sleep(2.0)

        if self.proc.poll() is not None:
            raise RuntimeError(
                "[GenderAgeClient ERROR] Worker crashed on startup"
            )

        print("[GenderAgeClient] Gender & Age worker ready")

    # --------------------------------------------------
    def _worker_alive(self):
        return self.proc is not None and self.proc.poll() is None

    # --------------------------------------------------
    def predict(self, face_img):
        try:
            if not self._worker_alive():
                self._start_worker()

            payload = pickle.dumps(
                face_img, protocol=pickle.HIGHEST_PROTOCOL
            )

            self.proc.stdin.write(struct.pack("I", len(payload)))
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()

            size_bytes = self.proc.stdout.read(4)
            if not size_bytes:
                raise RuntimeError("Worker pipe closed")

            size = struct.unpack("I", size_bytes)[0]
            data = self.proc.stdout.read(size)

            return pickle.loads(data)

        except Exception as e:
            print("[GenderAgeClient WARNING]", e)
            try:
                if self.proc:
                    self.proc.kill()
            except Exception:
                pass
            self.proc = None
            return None, "Unknown"

    # --------------------------------------------------
    def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc = None
